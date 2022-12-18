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

AUDIO_LENGTH = 50000

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

def raw_preproc(audio_buf):
    audio_buf = audio_buf.reshape(-1,1)
    audio_buf = (audio_buf - np.mean(audio_buf)) / np.std(audio_buf)
    original_length = len(audio_buf)
    if original_length < AUDIO_LENGTH:
            audio_buf = np.concatenate((audio_buf, np.zeros(shape=(AUDIO_LENGTH - original_length))))
    elif original_length > AUDIO_LENGTH:
        audio_buf = audio_buf[0:AUDIO_LENGTH]
    
    return audio_buf

# class UA_corpus(Dataset):
#     def __init__(self, normal_dir, dys_dir):
#         self.normal_dir = os.listdir(normal_dir)
#         self.dys_dir = os.listdir(dys_dir)
#         self.normal_root = normal_dir
#         self.dys_root = dys_dir

#     def __len__(self):
#         return len(self.dys_dir)

#     def __getitem__(self, idx):
#         norm_filename = self.normal_dir[idx]
#         dys_filename = self.dys_dir[idx]
        
#         norm_filename = os.path.join(self.normal_root, norm_filename)
#         dys_filename = os.path.join(self.dys_root, dys_filename)
        
#         normal_speech = sio.loadmat(norm_filename)
#         dys_speech = sio.loadmat(dys_filename)
#         normal_speech = normal_speech['feat']
#         dys_speech = dys_speech['feat']

#         padded_normal_speech = pad(normal_speech, 500)
#         padded_dys_speech = pad(dys_speech, 500)

#         padded_dys_speech = preprocess(padded_dys_speech)
#         padded_normal_speech = preprocess(padded_normal_speech)
        
#         #padded_dys_speech = padded_dys_speech.reshape([1,padded_dys_speech.shape[0],padded_dys_speech.shape[1]])
#         #padded_normal_speech = padded_normal_speech.reshape([1,padded_normal_speech.shape[0],padded_normal_speech.shape[1]])

#         sample = {'normal_speech': padded_normal_speech, 'dys_speech': padded_dys_speech}

#         return sample

class UA_corpus(Dataset):
    def __init__(self, normal_dir, dys_dir):
        self.normal_dir = os.listdir(normal_dir)
        self.dys_dir = os.listdir(dys_dir)
        self.normal_root = normal_dir
        self.dys_root = dys_dir

    def __len__(self):
        return len(self.dys_dir)

    def __getitem__(self, idx):
        norm_filename = self.normal_dir[idx]
        dys_filename = self.dys_dir[idx]
        
        norm_filename = os.path.join(self.normal_root, norm_filename)
        dys_filename = os.path.join(self.dys_root, dys_filename)
        
        normal_speech = librosa.load(norm_filename)
        dys_speech = librosa.load(dys_filename)

        padded_normal_speech = raw_preproc(normal_speech[0])
        padded_dys_speech = raw_preproc(dys_speech[0])

        padded_dys_speech = preprocess(padded_dys_speech)
        padded_normal_speech = preprocess(padded_normal_speech)

        sample = {'normal_speech': padded_normal_speech, 'dys_speech': padded_dys_speech}

        return sample

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") DONE
# LEARNING_RATE = 3e-4 DONE
# BATCH_SIZE = 64 DONE
# IMAGE_SIZE = 32 DONE
# CHANNELS_IMG = 1 DONE
# Z_DIM = 100 
# NUM_EPOCHS = 2 DONE
# FEATURES_DISC = 32
# FEATURES_GEN = 32

if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    #parser.add_argument('-gpu', type=bool, default=True, help='use gpu or not')
    # parser.add_argument('-norm_dir', type=str, default='/home/201901214/Dys_GAN/audio_feature/', help='uacorpus data path')
    # parser.add_argument('-dys_dir', type=str, default='/home/201901214/Dys_GAN/dysarthia_speech_features/', help='uacorpus data path')

    parser.add_argument('-norm_dir', type=str, default='C:/Prathamesh/UASpeech/NormalWhole', help='uacorpus data path')
    parser.add_argument('-dys_dir', type=str, default='C:/Prathamesh/UASpeech/DysWhole', help='uacorpus data path')

    parser.add_argument('-epoch', type=int, default=10, help='epoch number for training')
    parser.add_argument('-img_size', type=int, default=32)
    parser.add_argument('-channels', type=int, default=1)
    parser.add_argument('-w', type=int, default=32, help='number of workers for dataloader')
    parser.add_argument('-b', type=int, default=32, help='batch size for dataloader')
    parser.add_argument('-s', type=bool, default=True, help='whether shuffle the dataset')
    parser.add_argument('-lr', type=float, default=3e-4, help='initial learning rate')
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print('data loading')
    # uacorpus_dataset = UA_corpus(args.norm_dir, args.dys_dir)
    # train_size = int(0.8 * len(uacorpus_dataset))
    # test_size = len(uacorpus_dataset) - train_size
    # train_dataset, test_dataset = random_split(uacorpus_dataset, [train_size, test_size])
    # uacorpus_train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=args.s, num_workers=args.w, pin_memory=True)
    # uacorpus_test_loader = DataLoader(test_dataset, batch_size=args.b, shuffle=args.s, num_workers=args.w, pin_memory=True)
    # print('done')

    # gen = Generator().to(device)
    # disc = Discriminator(1, 32).to(device)

    gen = gen1d().to(device)
    disc = disc1d(1, 32).to(device)

    initialize_weights(gen)
    initialize_weights(disc)

    print(gen)
    print(disc)
    # summary(gen, input_size = (1, 50000), batch_size = args.b)
    # summary(disc, input_size = (1, 50000), batch_size = args.b)

    # generatorLosses = []
    # discriminatorLosses = []

    # optG = torch.optim.Adam(gen.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # optD = torch.optim.Adam(disc.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=1e-3)

    # loss = nn.BCELoss()

    # G_scheduler = optim.lr_scheduler.StepLR(optG, step_size=30, gamma=0.1)
    # D_scheduler = optim.lr_scheduler.StepLR(optD, step_size=30, gamma=0.1)

    # for epoch in range(args.epoch):

    #     G_scheduler.step(epoch)
    #     D_scheduler.step(epoch)

    #     gen.train()
    #     disc.train()

    #     for batch_idx, sample in enumerate(uacorpus_train_loader):

    #         dys_sample = sample['dys_speech']
    #         normal_sample = sample['normal_speech']
            
    #         tmpBatchSize = dys_sample.shape[0]
            
    #         true_label = torch.ones(tmpBatchSize, 1, device=device)
    #         fake_label = torch.zeros(tmpBatchSize, 1, device=device)
            
    #         dys_sample = dys_sample.float()
    #         normal_sample = normal_sample.float()
            
    #         enhanced_sample = gen(dys_sample)

    #         # Passing Normal Speech through Discriminator
    #         predictionsReal = disc(normal_sample).view(-1, 1)
    #         trueLoss = loss(predictionsReal, true_label)

    #         # Passing Enhanced Speech through Discriminator
    #         predictionsFake = disc(enhanced_sample).view(-1, 1)
    #         fakeLoss = loss(predictionsFake, fake_label)  # labels = 0
    #         discriminatorLoss = (trueLoss + fakeLoss) / 2

    #         disc.zero_grad()
    #         discriminatorLoss.backward(retain_graph=True)
    #         optD.step()  # update discriminator parameters

    #         # train generator
    #         predictionsFake = disc(enhanced_sample).view(-1, 1)
    #         generatorLoss = loss(predictionsFake, true_label)  # labels = 1
    #         optG.zero_grad()
    #         generatorLoss.backward(retain_graph=True)
    #         optG.step()

    #         optD.zero_grad()
    #         optG.zero_grad()

    #         generatorLosses.append(generatorLoss.item())
    #         discriminatorLosses.append(discriminatorLoss.item())

    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
    #             epoch, batch_idx * len(dys_sample), len(uacorpus_train_loader.dataset),
    #             100. * batch_idx / len(uacorpus_train_loader)))
        
    #     print("Epoch " + str(epoch) + " Complete")


    #     torch.save(gen, 'Generator_epoch=' + str(epoch) + '.pth')
    #     torch.save(disc, 'Discriminator_epoch=' + str(epoch) + '.pth')   

