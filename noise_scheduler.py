import argparse
import json

import torch
from tqdm import tqdm

import utils
import benchmark
from model import WaveGrad
from data import AudioDataset, MelSpectrogramFixed

def load_checkpoint_org( model, filepath,):
    checkpoint = torch.load(filepath, map_location='cpu')
    valid_incompatible_unexp_keys = [
        'betas',
        'alphas',
        'alphas_cumprod',
        'alphas_cumprod_prev',
        'sqrt_alphas_cumprod',
        'sqrt_recip_alphas_cumprod',
        'sqrt_recipm1_alphas_cumprod',
        'posterior_log_variance_clipped',
        'posterior_mean_coef1',
        'posterior_mean_coef2'     ]
    checkpoint['model'] = {
        key: value for key, value in checkpoint['model'].items() \
            if key not in valid_incompatible_unexp_keys
    }
    model.load_state_dict(checkpoint['model'], strict=False)



def run(config, args):
  print(config)
  model = WaveGrad(config).cuda()
  print(f'Number of parameters: {model.nparams}'
  load_checkpoint_org(model,config.schedule_checkpoint)
  
  dataset = AudioDataset(config, training=False)
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

  iters_best_schedule, stats = benchmark.iters_schedule_grid_search(
      model=model, n_iter=6, config=config, step=1, test_batch_size=2,
      path_to_store_stats='schedules/gs_stats_{:d}iters.pt'.format(config.schedule_batch),
      verbose=True
  )
  print(config.schedule_batch)
  print(iters_best_schedule) 
  torch.save(iters_best_schedule, 'schedules/iters{:d}_best_schedule.pt'.format(config.schedule_batch) )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str)
    parser.add_argument('-i', '--iter',   required=True, type=int)    
    parser.add_argument('-b', '--schedule-batch',  required=True, type=int)   
    parser.add_argument('-f', '--schedule-checkpoint',  required=True, type=str)
    parser.add_argument('-v', '--verbose', required=False, default=True, type=bool)
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))
    
    run(config, args)
