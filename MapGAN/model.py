import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nc, nc_out): 
        super(Generator, self).__init__()

        nz = 1024
        ngf = 64

        self.activatation = nn.ReLU(True)
        self.activatation_Tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)

        self.conv1 = nn.Conv2d(nc, ngf, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False)
        self.conv_bn2 = nn.BatchNorm2d(ngf * 2)

        self.conv3 = nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False)
        self.conv_bn3 = nn.BatchNorm2d(ngf * 4)

        self.conv4 = nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False)
        self.conv_bn4 = nn.BatchNorm2d(ngf * 8)

        self.conv5 = nn.Conv2d(ngf * 8, nz, 4, 1, 0, bias=False)
        self.conv_bn5 = nn.BatchNorm2d(nz)

        self.conv11 = nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False)
        self.conv_bn11 = nn.BatchNorm2d(ngf * 8)

        self.conv12 = nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=False)
        self.conv_bn12 = nn.BatchNorm2d(ngf * 4)

        self.conv13 = nn.ConvTranspose2d( ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=False)
        self.conv_bn13 = nn.BatchNorm2d(ngf * 2)

        self.conv14 = nn.ConvTranspose2d( ngf * 2 * 2, ngf, 4, 2, 1, bias=False)
        self.conv_bn14 = nn.BatchNorm2d(ngf)

        self.conv15 = nn.ConvTranspose2d( ngf * 2, nc_out, 4, 2, 1, bias=False)
        
    def forward(self, input):
        x1 = self.activatation(self.conv1(input))
        x2 = self.activatation(self.conv_bn2(self.conv2(x1)))
        x3 = self.activatation(self.conv_bn3(self.conv3(x2)))
        x4 = self.activatation(self.conv_bn4(self.conv4(x3)))
        x5 = self.activatation(self.conv_bn5(self.conv5(x4)))

        x11 = self.dropout(self.activatation(self.conv_bn11(self.conv11(x5))))
        x11 = torch.cat((x11, x4), 1)
        x12 = self.dropout(self.activatation(self.conv_bn12(self.conv12(x11))))
        x12 = torch.cat((x12, x3), 1)
        x13 = self.dropout(self.activatation(self.conv_bn13(self.conv13(x12))))
        x13 = torch.cat((x13, x2), 1)
        x14 = self.activatation(self.conv_bn14(self.conv14(x13)))
        x14 = torch.cat((x14, x1), 1)
        x15 = self.activatation_Tanh(self.conv15(x14))

        return x15     

class Discriminator(nn.Module):
    def __init__(self, nc):
        super(Discriminator, self).__init__()

        ndf = 64

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input, g_output):
        input = torch.cat((input, g_output), 1)
        return self.main(input)