%%file resampling.py

import os
import sys
import glob
import numpy as np
import librosa 
from librosa.core import load as wfload
from scipy.io import wavfile as wf


MAX_WAV_VALUE = 32768.0
def float_to_pcm16(signal):
    signal_float = signal * MAX_WAV_VALUE   
    signal_int = signal_float.astype(np.int16)
    return signal_int

def pcm16_to_float(signal):
    signal_float = signal.astype(np.float)  
    signal_float = signal / MAX_WAV_VALUE  
    return signal_float  


def resampling(base_dir, tar_dir, sr_org, sr_tar): 
  #sr_org = 48000
  #sr_tar = 22050
  file_extension=".wav"
  # org_path=wav_dir
  # path_mono=target_dir+"/mono" 

  for file in os.listdir(base_dir):
    if file.endswith(file_extension):
        filebody = os.path.splitext(file)[0]
        this=os.path.join(base_dir, file) 
        #print (filebody, this)
        y, sr = wfload(this, sr = sr_org, mono=False)
        y_re = librosa.resample(y, sr_org, sr_tar, res_type='polyphase'  )
        y_re_int = float_to_pcm16(y_re)
      
        save_filename =os.path.join(tar_dir,  filebody+file_extension)
        wf.write(save_filename, sr_tar, y_re_int.T )

def main(base_dir,tar_dir, sr_org, sr_tar):
  resampling(base_dir,tar_dir,sr_org, sr_tar)

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('-b', '--base_dir')
  parser.add_argument('-t', '--target_dir')
  parser.add_argument('-so', '--sr_org', type=int, default=48000 )
  parser.add_argument('-st', '--sr_tar', type=int, default=22050 )
  
  args = parser.parse_args()
  
  main(args.base_dir,args.target_dir, args.sr_org, args.sr_tar)
