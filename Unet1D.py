# based on vgg16

import torch
import torch.nn as nn

class Generator1D(nn.Module):

    def __init__(self):
        super(Generator1D, self).__init__()

        self.features1 = nn.Sequential(

            nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

        self.features2 = nn.Sequential(

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

        self.features3 = nn.Sequential(

            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

        self.features4 = nn.Sequential(

            nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)

        self.features5 = nn.Sequential(

            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True)
        )

        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False, return_indices=True)


        #Decoder starts here

        self.unpool1 = nn.Upsample(size=(13, 11), mode='bilinear')

        self.reconstruct1 = nn.Sequential(

            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1)
        )

        self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        # self.unpool2 = nn.Upsample(size=(27, 22), mode='bilinear')

        self.reconstruct2 = nn.Sequential(

            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(512, 256, kernel_size=3, stride=1, padding=1)
        )

        self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        # self.unpool3 = nn.Upsample(size=(54, 44), mode='bilinear')

        self.reconstruct3 = nn.Sequential(

            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(256, 128, kernel_size=3, stride=1, padding=1),
        )

        self.unpool4 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        # self.unpool4 = nn.Upsample(size=(109, 89), mode='bilinear')

        self.reconstruct4 = nn.Sequential(

            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(128, 64, kernel_size=3, stride=1, padding=1),
        )

        self.unpool5 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        # self.unpool5 = nn.Upsample(size=(218, 178), mode='bilinear')

        self.reconstruct5 = nn.Sequential(

            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ConvTranspose1d(64, 1, kernel_size=3, stride=1, padding=1)
        )


    def forward(self, x):

        p1 = self.features1(x)
        x, p1_idx = self.pool1(p1)
        p2 = self.features2(x)
        x, p2_idx = self.pool2(p2)
        # p3 = self.features3(x)
        # x, p3_idx = self.pool3(p3)
        # p4 = self.features4(x)
        # x, p4_idx = self.pool4(p4)
        # p5 = self.features5(x)
        # x, p5_idx = self.pool5(p5)
        x = self.unpool4(x, p2_idx)
        x = self.reconstruct4(x)
        # x = self.unpool5(x, p1_idx, output_size=(batch_size, 64, 218, 178))
        x = self.unpool5(x, p1_idx)
        x = self.reconstruct5(x)

        return x#, p1_idx, p2_idx#, p3_idx, p4_idx, p5_idx
