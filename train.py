import argparse
import copy
import json
import glob
import os
import re
import time
from datetime import datetime
import numpy as np 

from collections import defaultdict, OrderedDict
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

 


from apex import amp
from apex.optimizers import FusedAdam, FusedLAMB

#from logger import Logger
from model import WaveGrad
from data import AudioDataset, MelSpectrogramFixed
from benchmark import compute_rtf
from utils import ConfigWrapper, show_message

    
def parse_args(parser):
    """
    Parse commandline arguments.
    """
    parser.add_argument('-c', '--config',       type=str, required=True, help='configure file')
    parser.add_argument('-v', '--verbose',      required=False, default=True, type=bool)
    parser.add_argument('-o', '--output',       type=str, required=True,  help='Directory to save checkpoints')
    parser.add_argument('-d', '--dataset-path', type=str, default='./',  help='Path to dataset')
    parser.add_argument('--log-file',           type=str, default=None,  help='Path to a DLLogger log file')

    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs',                type=int,         required=True,   help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int,         default=50,  help='Number of epochs per checkpoint')
    training.add_argument('--checkpoint-path',       type=str,         default=None, help='Checkpoint path to resume training')
    training.add_argument('--resume',                                  action='store_true',  help='Resume training from the last available checkpoint')
    training.add_argument('--seed',                  type=int,         default=1234,  help='Seed for PyTorch random number generators')
    training.add_argument('--amp',                                     action='store_true', help='Enable AMP')
    training.add_argument('--cuda',                                    action='store_true',   help='Run on GPU using CUDA')
    training.add_argument('--cudnn-enabled',                           action='store_true',   help='Enable cudnn')
    training.add_argument('--cudnn-benchmark',                         action='store_true',   help='Run cudnn benchmark')
    training.add_argument('--ema-decay',                   type=float, default=0, help='Discounting factor for training weights EMA')
    training.add_argument('--gradient-accumulation-steps', type=int,   default=1,   help='Training steps to accumulate gradients for')

    optimization = parser.add_argument_group('optimization setup')
    optimization.add_argument('--optimizer',                 type=str,     default='lamb',   help='Optimization algorithm')
    optimization.add_argument('-lr', '--learning-rate',      type=float,   required=True,    help='Learing rate')
    optimization.add_argument('--weight-decay',               type=float,  default=1e-6,     help='Weight decay')
    optimization.add_argument('--grad-clip-thresh',           type=float,  default=1000.0,   help='Clip threshold for gradients')
    optimization.add_argument('-bs', '--batch-size',          type=int,    required=True,    help='Batch size per GPU')
    optimization.add_argument('--warmup-steps',               type=int,    default=1000,     help='Number of steps for lr warmup')
    optimization.add_argument('--dur-predictor-loss-scale',   type=float,  default=1.0,      help='Rescale duration predictor loss')
    optimization.add_argument('--pitch-predictor-loss-scale', type=float,  default=1.0,      help='Rescale pitch predictor loss')

    #dataset = parser.add_argument_group('dataset parameters')
    #dataset.add_argument('--training-files', type=str, required=True,  help='Path to training filelist')
    #dataset.add_argument('--validation-files', type=str, required=True,  help='Path to validation filelist')

    distributed = parser.add_argument_group('distributed setup')
    distributed.add_argument('--local_rank', type=int,    default=os.getenv('LOCAL_RANK', 0),    help='Rank of the process for multiproc. Do not set manually.')
    distributed.add_argument('--world_size', type=int,    default=os.getenv('WORLD_SIZE', 1),    help='Number of processes for multiproc. Do not set manually.')
    return parser   

def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= num_gpus
    return rt


def init_distributed(args, world_size, rank):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    print("Initializing distributed training")

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(backend=('nccl' if args.cuda else 'gloo'),
                            init_method='env://')
    print("Done initializing distributed training")


def last_checkpoint(output):

    def corrupted(fpath):
        try:
            torch.load(fpath, map_location='cpu')
            return False
        except:
            print(f'WARNING: Cannot load {fpath}')
            return True

    saved = sorted(
        glob.glob(f'{output}/WaveGrad_checkpoint_*.pt'),
        key=lambda f: int(re.search('_(\d+).pt', f).group(1)))

    if len(saved) >= 1 and not corrupted(saved[-1]):
        return saved[-1]
    elif len(saved) >= 2:
        return saved[-2]
    else:
        return None


def save_checkpoint(local_rank, model, ema_model, optimizer, epoch, total_iter,
                    config, amp_run, filepath):
    if local_rank != 0:
        return
    print(f"Saving model and optimizer state at epoch {epoch} to {filepath}")
    ema_dict = None if ema_model is None else ema_model.state_dict()
    checkpoint = {'epoch': epoch,
                  'iteration': total_iter,
                  'config': config,
                  'state_dict': model.state_dict(),
                  'ema_state_dict': ema_dict,
                  'optimizer': optimizer.state_dict()}
    if amp_run:
        checkpoint['amp'] = amp.state_dict()
    torch.save(checkpoint, filepath)


def load_checkpoint(local_rank, model, ema_model, optimizer, epoch, total_iter,
                    config, amp_run, filepath, world_size):
    if local_rank == 0:
        print(f'Loading model and optimizer state from {filepath}')
    checkpoint = torch.load(filepath, map_location='cpu')
    epoch[0] = checkpoint['epoch'] + 1
    total_iter[0] = checkpoint['iteration']
    config = checkpoint['config']

    sd = {k.replace('module.', ''): v
          for k, v in checkpoint['state_dict'].items()}
    getattr(model, 'module', model).load_state_dict(sd)
    optimizer.load_state_dict(checkpoint['optimizer'])

    if amp_run:
        amp.load_state_dict(checkpoint['amp'])

    if ema_model is not None:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        



def validate(model, criterion, valset, batch_size, world_size, collate_fn,
             distributed_run, rank, batch_to_gpu, use_gt_durations=False):
    """Handles all the validation scoring and printing"""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        val_sampler = DistributedSampler(valset) if distributed_run else None
        val_loader = DataLoader(valset, num_workers=8, shuffle=False,
                                sampler=val_sampler,
                                batch_size=batch_size, pin_memory=False,
                                collate_fn=collate_fn)
        val_meta = defaultdict(float)
        val_num_frames = 0
        for i, batch in enumerate(val_loader):
            x, y, num_frames = batch_to_gpu(batch)
            y_pred = model(x, use_gt_durations=use_gt_durations)
            loss, meta = criterion(y_pred, y, is_training=False, meta_agg='sum')
            if distributed_run:
                for k,v in meta.items():
                    val_meta[k] += reduce_tensor(v, 1)
                val_num_frames += reduce_tensor(num_frames.data, 1).item()
            else:
                for k,v in meta.items():
                    val_meta[k] += v
                val_num_frames = num_frames.item()
        val_meta = {k: v / len(valset) for k,v in val_meta.items()}
        val_loss = val_meta['loss']

    if was_training:
        model.train()
    return val_loss.item(), val_meta, val_num_frames


def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate * scale


def apply_ema_decay(model, ema_model, decay):
    if not decay:
        return
    st = model.state_dict()
    add_module = hasattr(model, 'module') and not hasattr(ema_model, 'module')
    for k,v in ema_model.state_dict().items():
        if add_module and not k.startswith('module.'):
            k = 'module.' + k
        v.copy_(decay * v + (1 - decay) * st[k])
     

def run(config, args):
    
    if 'LOCAL_RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        local_rank = args.rank
        world_size = args.world_size
    distributed_run = world_size > 1

    torch.manual_seed(args.seed + local_rank)
    np.random.seed(args.seed + local_rank)

    if local_rank == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output)
 
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    
    
    if local_rank == 0:
        print("start training")
        print("args", args)
        print("config", config)

    
    #############################################
    # model
    if local_rank == 0:
        print("load model")
    model = WaveGrad(config).cuda()
    if local_rank == 0:
        print(model) 
    
    # optimizer amp config
    if local_rank == 0:    
        print("configure optimizer and amp")    
    kw = dict(lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9,
              weight_decay=args.weight_decay)
    
    if args.optimizer == 'adam':
        optimizer = FusedAdam(model.parameters(), **kw)
    elif args.optimizer == 'lamb':
        optimizer = FusedLAMB(model.parameters(), **kw)
    else:
        raise ValueError
        
    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")


    if distributed_run:
        model = DistributedDataParallel(model, device_ids=[args.local_rank],
                                        output_device=args.local_rank,
                                        find_unused_parameters=True)
    start_epoch = [1]
    start_iter  = [0]
    
    ################
    #load checkpoint 
    if local_rank == 0:
        print("checkpoint load if needed")
    
    assert args.checkpoint_path is None or args.resume is False, (
        "Specify a single checkpoint source")
    if args.checkpoint_path is not None:
        ch_fpath = args.checkpoint_path
    elif args.resume:
        ch_fpath = last_checkpoint(args.output)
    else:
        ch_fpath = None

    if ch_fpath is not None:
        load_checkpoint(local_rank, model, ema_model, optimizer, start_epoch,
                        start_iter, model_config, args.amp, ch_fpath,
                        world_size)

    start_epoch = start_epoch[0]
    total_iter = start_iter[0]
    
    
    
    # dataloader
    ##########################################################    
    if local_rank == 0:    
        print("load dataset") 
    train_dataset = AudioDataset(config, training=True)
   
    # distributed sampler 
    if distributed_run:
        train_sampler, shuffle = DistributedSampler(trainset), False
    else:
        train_sampler, shuffle = None, True
        
    train_dataloader = DataLoader(train_dataset, num_workers=1, shuffle=shuffle,
                                  sampler=train_sampler, batch_size=args.batch_size,
                                  pin_memory=False, drop_last=True )
    
    # ground truth samples 
    test_dataset = AudioDataset(config, training=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    test_batch = test_dataset.sample_test_batch(
        config.training_config.n_samples_to_test
    )
    
    # Log ground truth test batch  
    if local_rank == 0:    
        print("save truth wave and mel")
    mel_fn = MelSpectrogramFixed(
        sample_rate=config.data_config.sample_rate,
        n_fft=config.data_config.n_fft,
        win_length=config.data_config.win_length,
        hop_length=config.data_config.hop_length,
        f_min=config.data_config.f_min,
        f_max=config.data_config.f_max,
        n_mels=config.data_config.n_mels,
        window_fn=torch.hann_window
    ).cuda()    
    
    audios = {
        f'audio_{index}/gt': audio
        for index, audio in enumerate(test_batch)
    }
    specs = {
        f'mel_{index}/gt': mel_fn(audio.cuda()).cpu().squeeze()
        for index, audio in enumerate(test_batch)
    }
    
   
    
    

    
    ####### loop start
    #epoch
    
    model.train()
    val_loss = 0.0
    torch.cuda.synchronize()    
    
    if local_rank == 0:
        print("epoch start")
    for epoch in range(start_epoch, args.epochs + 1):
        tic_epoch= time.time()
        epoch_loss = 0.0
        
        if distributed_run:
            train_loader.sampler.set_epoch(epoch)        
        
        accumulated_steps = 0
        iter_loss = 0    
        epoch_iter = 0
        iteration = 0
        num_iters = len(train_loader) // args.gradient_accumulation_steps
        
        model.set_new_noise_schedule(
            init=torch.linspace,
            init_kwargs={
                'steps': config.training_config.training_noise_schedule.n_iter,
                'start': config.training_config.training_noise_schedule.betas_range[0],
                'end': config.training_config.training_noise_schedule.betas_range[1]
            }
        )

        
        for i,batch in enumerate(train_loader):
            tic_iter = time.time()
            
            model.zero_grad()                
            batch = batch.cuda()
            mels = mel_fn(batch)           
            
            # Training step
            model.zero_grad()
            loss = model.compute_loss(mels, batch)
            
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                
            if distributed_run:
                reduced_loss = reduce_tensor(loss.data, world_size).item()
            else:
                reduced_loss = loss.item()
            if np.isnan(reduced_loss):
                raise Exception("loss is NaN") 
                
            iter_loss += reduced_loss
            
            if args.amp:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.grad_clip_thresh)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_thresh)

            optimizer.step()

            toc_iter = time.time()
            dur_iter = toc_iter - tic_iter
            epoch_loss += iter_loss
            iter_loss = 0
                
            loss_stats = {
                'iter_loss': iter_loss.item(),
                'grad_norm': grad_norm.item()
            }
            if local_rank == 0:
                print("\nepoch {:4d} | iter {:>12d}  {:>3d}/{:3d} | {:3.2f}s/iter est {:d}s/epoch | losses {:>12.6f} {:>12.6f} ".format(
                        epoch, iteration, i, iter_size, dur_iter, epoch_iter, iter_loss.item(), grad_norm.item()  ) ,end='' )
            iteration += 1
        # Finished epoch
        toc_epoch = time.time()
        dur_epoch = toc_epoch - tic_epoch
        if local_rank == 0:
            print("  {:4.2f}s/epoch  ".format(iter_size, dur_epoch  )  )        
        
        
        # Test step
        if epoch % config.training_config.test_interval == 0:
            model.set_new_noise_schedule(
                init=torch.linspace,
                init_kwargs={
                    'steps': config.training_config.test_noise_schedule.n_iter,
                    'start': config.training_config.test_noise_schedule.betas_range[0],
                    'end': config.training_config.test_noise_schedule.betas_range[1]
                } )
                    
                    

    
    
        #save checkpoint 
        if (epoch > 0 and args.epochs_per_checkpoint > 0 and 
            (epoch % args.epochs_per_checkpoint == 0) and local_rank == 0):

            checkpoint_path = os.path.join(
                args.output, f"FastPitch_checkpoint_{epoch}.pt")
            save_checkpoint(local_rank, model, ema_model, optimizer, epoch,
                            total_iter, model_config, args.amp, checkpoint_path)
            if local_rank == 0:
                print("checkpoint saved")
        
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='PyTorch WaveGrad Training', 
                                     allow_abbrev=False) 
    parser = parse_args(parser)
    args, _ = parser.parse_known_args()
       
    ### additional configuration from config file 
    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))
                

    
    
    run(config, args)
