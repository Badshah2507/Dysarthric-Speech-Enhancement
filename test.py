from cgi import test
import numpy as np
#import matplotlib.image as mpimg
import matplotlib.pyplot as plt
#from PIL import Image
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import torchvision
#import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, Dataset, random_split
import os
import argparse
from Unet import Generator
from Discriminator import Discriminator
import torch.optim as optim
import scipy.io as sio
from scipy.io import savemat
from tqdm import tqdm
#from torchsummary import summary
import librosa
import soundfile as sf

AUDIO_LENGTH = 100000

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

def pad(S, size):
    if(S.shape[1]<size):    
        while(1):
            S = np.concatenate((S,S),axis=1)            
            if(S.shape[1]==size):
                break
            elif(S.shape[1]>size):
                S = np.array([list(x[0:size]) for x in S])
                break 
    elif(S.shape[1]>size):
        mid = S.shape[1]/2
        l = mid - AUDIO_LENGTH
        u = mid + AUDIO_LENGTH        
        S = np.array([list(x[l:u]) for x in S])
    return S


def raw_preproc(audio_buf):
    audio_buf = audio_buf.reshape(-1,1)
    audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
    original_length = len(audio_buf)
    if original_length < AUDIO_LENGTH:
            audio_buf = np.concatenate((audio_buf, np.zeros(shape=(AUDIO_LENGTH - original_length,1))))
    elif original_length > AUDIO_LENGTH:
        audio_buf = audio_buf[0:AUDIO_LENGTH]
    
    return audio_buf


Gen = torch.load('C:\Prathamesh\DysarthiaGAN\Generator_epoch=9.pth')
Disc = torch.load('C:\Prathamesh\DysarthiaGAN\Discriminator_epoch=9.pth')

# test_file = sio.loadmat('C:\Prathamesh\Speech_enhance_data\\features\\feats_teo\medium\F02_B1_C1_M3.mat')
# test_feat = test_file['feat']
audio_path = "C:/Prathamesh/UASpeech/audio/F02/F02_B1_CW65_M5.wav"

dys_speech = librosa.load(audio_path)
print(dys_speech[0].shape)
padded_dys_speech = raw_preproc(dys_speech[0])
padded_dys_speech = preprocess(padded_dys_speech)
padded_dys_speech = padded_dys_speech.reshape(1,1,AUDIO_LENGTH)
padded_dys_speech = padded_dys_speech.float()

print(type(padded_dys_speech))
gen_out = Gen(padded_dys_speech)
print(gen_out.shape)

gen_out = gen_out.reshape(AUDIO_LENGTH).detach().numpy()

sf.write('F02_B1_CW65_M5_ens.wav', gen_out, 16000)

from scipy.io import wavfile
import noisereduce as nr
# load data
rate, data = wavfile.read("F02_B1_CW65_M5_ens.wav")
# perform noise reduction
reduced_noise = nr.reduce_noise(y=data, sr=rate)
wavfile.write("F02_B1_CW65_M5_ens.wav_reduced_noise.wav", rate, reduced_noise)