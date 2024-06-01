import torch
from torchvision import transforms
from torch.autograd import Variable
import argparse
import os
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import datetime
from PIL import Image

from model import Generator, Discriminator
import sys
sys.path.append(os.path.abspath('../'))
from utils import weights_init_normal
from dataset_rplan import ImageDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=64, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='../Dataset', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--size', type=int, default=64, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--snapshot_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--resume', action='store_true', help='resume')
parser.add_argument('--iter_loss', type=int, default=500, help='average loss for n iterations')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
opt = parser.parse_args()

opt.log_path = os.path.join('output', str(datetime.datetime.now()) + '.txt')

if torch.cuda.is_available():
    opt.cuda = True

opt.resume = False

print(opt)

##### Definition of variables ######
# Networks
G = Generator(opt.input_nc, opt.output_nc)
D = Discriminator(opt.input_nc + opt.output_nc)

if opt.cuda:
    G.cuda()
    D.cuda()

G.apply(weights_init_normal)
D.apply(weights_init_normal)

# Loss function
BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()

# Optimizers
G_optimizer = torch.optim.Adam(G.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

####### resume the training process
if opt.resume:
    print('resume training:')
    G.load_state_dict(torch.load('output/G.pth'))
    D.load_state_dict(torch.load('output/D.pth'))
    G_optimizer.load_state_dict(torch.load('output/optimizer_G.pth'))
    D_optimizer.load_state_dict(torch.load('output/optimizer_D.pth'))

# Dataset loader
transforms_ = [transforms.Resize((opt.size, opt.size), Image.BICUBIC),
			   transforms.ToTensor(),
			   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, set_name = 'train'),
						batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu)

plt.ioff()
curr_iter = 0
G_losses_temp = 0
D_losses_temp = 0
G_losses = []
D_losses = []
to_pil = transforms.ToPILImage()

open(opt.log_path, 'w').write(str(opt) + '\n\n')

curr_iter = 0
for epoch in range(opt.epoch, opt.n_epochs):

    # training
    for i, (input, target) in enumerate(dataloader):

        # input & target data
        x_ = Variable(input.cuda())
        y_ = Variable(target.cuda())

        # Train discriminator with real data
        D_real_decision = D(x_, y_).squeeze()
        real_ = Variable(torch.ones(D_real_decision.size()).cuda())
        D_real_loss = BCE_loss(D_real_decision, real_)

        # Train discriminator with fake data
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        fake_ = Variable(torch.zeros(D_fake_decision.size()).cuda())
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        # Back propagation
        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train generator
        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)

        # L1 loss
        l1_loss = opt.lamb * L1_loss(gen_image, y_)

        # Back propagation
        G_loss = G_fake_loss + l1_loss
        G.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        # D_losses.append(D_loss.data[0])
        D_losses_temp += D_loss.item()

        # G_losses.append(G_loss.data[0])
        G_losses_temp += G_loss.item()

        curr_iter += 1
        if (i+1) % opt.iter_loss == 0:
            log = '[iter %d], [loss_G %.5f], ' \
                  '[loss_D %.5f]' % \
                  (curr_iter, G_loss, D_loss)
            print(log)
            open(opt.log_path, 'a').write(log + '\n')

            G_losses.append(G_losses_temp / opt.iter_loss)
            D_losses.append(D_losses_temp / opt.iter_loss)
            G_losses_temp = 0
            D_losses_temp = 0

            avg_log = '[the last %d iters], [loss_G %.5f], [D_losses %.5f], ' \
					  % (opt.iter_loss, G_losses[G_losses.__len__()-1], D_losses[D_losses.__len__()-1])
            print(avg_log)
            open(opt.log_path, 'a').write(avg_log + '\n')

            result_path = 'output'
            img_fake = 0.5 * (gen_image.detach().data + 1.0)
            save_image(img_fake, '{}/{}fake.png'.format(result_path, epoch+1, ), nrow=8)
            img_real = 0.5 * (y_.detach().data + 1.0)
            save_image(img_real, '{}/{}real.png'.format(result_path, epoch+1, ), nrow=8)
            img_input = 0.5 * (x_.detach().data + 1.0)
            save_image(img_input, '{}/{}input.png'.format(result_path, epoch+1, ), nrow=8)

    # Save models checkpoints
    torch.save(G.state_dict(), 'output/G.pth')
    torch.save(D.state_dict(), 'output/D.pth')

    torch.save(G_optimizer.state_dict(), 'output/G_optimizer.pth')
    torch.save(D_optimizer.state_dict(), 'output/D_optimizer.pth')

    if (epoch+1) % opt.snapshot_epochs == 0:
        torch.save(G.state_dict(), ('output/G_%d.pth' % (epoch+1)))
        torch.save(D.state_dict(), ('output/D_%d.pth' % (epoch+1)))

    print('Epoch:{}'.format(epoch+1))
    open(opt.log_path, 'a').write('Epoch:{}'.format(epoch+1) + '\n')

    if (epoch+1) % opt.snapshot_epochs == 0:
        plt.figure(figsize=(10, 5))
        plt.title("Generator Loss During Training")
        plt.plot(G_losses, label="G loss")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig('./output/generator.png')
        # plt.show(block=False)

        plt.figure(figsize=(10, 5))
        plt.title("Discriminator Loss During Training")
        plt.plot(D_losses, label="D loss")
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig('./output/discriminator.png')
        # plt.show(block=False)
        plt.close('all')

