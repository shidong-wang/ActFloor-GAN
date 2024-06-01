import os
import shutil
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image

import sys
sys.path.append(os.path.abspath('../'))
from utils import getBoundary, getFloorplan, getActivity
from ActFloorGAN.models_guided import Generator_B2F
from MapGAN.model import Generator as Generator_map

if torch.cuda.is_available():
    cuda = True
    device = torch.device('cuda:0')

def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

# MapGAN
input_nc = 3
output_nc = 1
model_epoch = 100
model_dir = 'trained_model'
model_path = f'{model_dir}/G_{model_epoch}.pth'
G_map = Generator_map(input_nc, output_nc)
if cuda:
    G_map.to(device)
G_map.load_state_dict(torch.load(model_path))
G_map.eval()
G_map.apply(apply_dropout)

# ActFloorGAN
input_nc = 3
output_nc = 3
model_epoch = 25
model_dir = 'trained_model'
model_path = f'{model_dir}/netG_B2A_{model_epoch}.pth'
G_B2F = Generator_B2F(input_nc, output_nc)
if cuda:
    G_B2F.to(device)
G_B2F.load_state_dict(torch.load(model_path))
G_B2F.eval()

def getInput(floorplan_path):
    
    with Image.open(floorplan_path) as temp:
        image_array = np.asarray(temp, dtype=np.uint8)

    #boundary_mask = image_array[:,:,0]
    category_mask = image_array[:,:,1]
    activity_mask = image_array[:,:,2]
    #inside_mask = image_array[:,:,3]

    transforms_fp = transforms.Compose([
        transforms.Resize((64, 64), Image.BICUBIC),
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    floorplan_3c = transforms_fp(Image.fromarray(np.uint8(getFloorplan(category_mask))))
    boundary_3c = transforms_fp(Image.fromarray(np.uint8(getBoundary(category_mask))))
    activity_mask_1c = getActivity(activity_mask)
    
    return boundary_3c, activity_mask_1c, floorplan_3c

criterion_MSE = torch.nn.MSELoss()
criterion_L1 = torch.nn.L1Loss()

def main(dataset_4c_dir, dataset_test_path, output_dir):

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    names_fp = open(dataset_test_path).read().split('\n')
    print(f'the dataset size: {len(names_fp)}')

    loss_MSE = 0
    loss_L1 = 0
    test_num = 0
    for name in names_fp:
        
        path_fp = f'{dataset_4c_dir}/{name}.png'
        if not os.path.exists(path_fp):
            print(name)
            continue

        boundary_3c, activity_mask_1c, floorplan_3c = getInput(path_fp)

        input_boundary = boundary_3c.unsqueeze(0).cuda()
        gen_map = G_map(input_boundary)
        gen_fp = G_B2F(input_boundary, gen_map)
        
        floorplan_gt = floorplan_3c.unsqueeze(0).cuda()
        loss_MSE += criterion_MSE(gen_fp.detach().data, floorplan_gt.detach().data)
        loss_L1 += criterion_L1(gen_fp.detach().data, floorplan_gt.detach().data)

        transforms_gen_fp = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            ])

        gen_fp = 0.5 * (gen_fp.detach().data + 1.0)
        gen_fp = transforms.ToPILImage()(gen_fp.squeeze(0).cpu())
        gen_fp = transforms_gen_fp(gen_fp).float()
        save_image(gen_fp, f'{output_dir}/{name}.png')

        test_num+=1
        if test_num % 1000 == 0:
            print(test_num)

    print(f'test_num: {test_num}')
    print(f'MSE: {loss_MSE/test_num}')
    print(f'L1: {loss_L1/test_num}')

    return 0

if __name__=='__main__':

    dataset_rootDir = '../Dataset'
    dataset_dir = f'{dataset_rootDir}/dataset_4c'

    set_name = 'test'
    dataset_test_path = f'{dataset_rootDir}/dataset_split/{set_name}.txt'

    output_dir = 'genRasterLayout'

    main(dataset_dir, dataset_test_path, output_dir)
    
        
