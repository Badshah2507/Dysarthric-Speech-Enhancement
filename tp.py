from email.mime import audio
import numpy as np
#import matplotlib.image as mpimg
#import matplotlib.pyplot as plt
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
# from Unet import Generator
# from Discriminator import Discriminator
from Unet1D import Generator1D as gen1d
from Discriminator1D import Discriminator as disc1d
import torch.optim as optim
import scipy.io as sio
import librosa
from torchsummary import summary


preprocess = transforms.Compose([
    transforms.ToTensor(),
])

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

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
        S = np.array([list(x[0:size]) for x in S])
    return S


class UA_corpus(Dataset):
    def __init__(self, normal_dir, dys_dir):
        self.normal_dir = os.listdir(normal_dir)
        # self.dys_dir = os.listdir(dys_dir)
        self.normal_root = normal_dir
        # self.dys_root = dys_dir

    def __len__(self):
        return len(self.dys_dir)

    def __getitem__(self, idx):
        norm_filename = self.normal_dir[idx]
        # dys_filename = self.dys_dir[idx]
        
        norm_filename = os.path.join(self.normal_root, norm_filename)
        # dys_filename = os.path.join(self.dys_root, dys_filename)
        
        normal_speech = sio.loadmat(norm_filename)
        # dys_speech = sio.loadmat(dys_filename)
        normal_speech = normal_speech['feat']
        # dys_speech = dys_speech['feat']

        padded_normal_speech = pad(normal_speech, 500)
        # padded_dys_speech = pad(dys_speech, 500)

        padded_dys_speech = preprocess(padded_dys_speech)
        # padded_normal_speech = preprocess(padded_normal_speech)
        
        #padded_dys_speech = padded_dys_speech.reshape([1,padded_dys_speech.shape[0],padded_dys_speech.shape[1]])
        #padded_normal_speech = padded_normal_speech.reshape([1,padded_normal_speech.shape[0],padded_normal_speech.shape[1]])

        sample = {'normal_speech': padded_normal_speech}#, 'dys_speech': padded_dys_speech}

        return sample


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    #parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    # parser.add_argument('-norm_dir', type=str, default='/home/201901214/Dys_GAN/audio_feature/', help='uacorpus data path')
    # parser.add_argument('-dys_dir', type=str, default='/home/201901214/Dys_GAN/dysarthia_speech_features/', help='uacorpus data path')

    parser.add_argument('-norm_dir', type=str, default='C:/Prathamesh/UASpeech/NormalWhole', help='uacorpus data path')
    parser.add_argument('-dys_dir', type=str, default='C:/Prathamesh/UASpeech/DysWhole', help='uacorpus data path')

    parser.add_argument('-epoch', type=int, default=2, help='epoch number for training')
    parser.add_argument('-img_size', type=int, default=32)
    parser.add_argument('-channels', type=int, default=1)
    parser.add_argument('-w', type=int, default=20, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-lr', type=float, default=3e-4, help='initial learning rate')
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('data loading')
    uacorpus_dataset = UA_corpus(args.norm_dir, args.dys_dir)
    train_size = int(0.8 * len(uacorpus_dataset))
    test_size = len(uacorpus_dataset) - train_size
    train_dataset, test_dataset = random_split(uacorpus_dataset, [train_size, test_size])
    uacorpus_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=args.s, num_workers=args.w, pin_memory=True)
    uacorpus_test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=args.s, num_workers=args.w, pin_memory=True)
    print('done')
