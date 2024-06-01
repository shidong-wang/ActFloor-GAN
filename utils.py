import torch
import numpy as np
import torchvision.transforms as transforms

class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def getFloorplan(mask):
    #   0     1    2    3    4    5    6    7    8     9    10   11   12
    r = [255, 156, 255, 255, 0,   65,  0,   65,  65,   128, 0,   0,   0]
    g = [0,   102, 255, 97,  255, 105, 255, 105, 105,  42,  255, 255, 255]
    b = [0,   31,  0,   0,   0,   225, 255, 225, 225,  42,  0,   0,   0]
    
    canvas_size = 256
    img_c1 = np.zeros((canvas_size, canvas_size))
    img_c2 = np.zeros((canvas_size, canvas_size))
    img_c3 = np.zeros((canvas_size, canvas_size))
        
    for i in range(13):
        img_c1[mask == i] = r[i]
        img_c2[mask == i] = g[i]
        img_c3[mask == i] = b[i]

    img_c1[mask == 17] = 0
    img_c2[mask == 17] = 0
    img_c3[mask == 17] = 0
    img_c1[mask == 16] = 0
    img_c2[mask == 16] = 0
    img_c3[mask == 16] = 0
    img_c1[mask == 15] = 127
    img_c2[mask == 15] = 127
    img_c3[mask == 15] = 127
    img_c1[mask == 14] = 0
    img_c2[mask == 14] = 0
    img_c3[mask == 14] = 0
    img_c1[mask == 13] = 255
    img_c2[mask == 13] = 255
    img_c3[mask == 13] = 255
            
    output_img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    output_img[:,:,0] = img_c1
    output_img[:,:,1] = img_c2
    output_img[:,:,2] = img_c3

    return output_img

def getBoundary(mask):
    
    canvas_size = 256
    img_c1 = np.zeros((canvas_size, canvas_size))
    img_c2 = np.zeros((canvas_size, canvas_size))
    img_c3 = np.zeros((canvas_size, canvas_size))

    img_c1[mask == 13] = 255
    img_c2[mask == 13] = 255
    img_c3[mask == 13] = 255
    img_c1[mask == 15] = 127
    img_c2[mask == 15] = 127
    img_c3[mask == 15] = 127
            
    output_img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    output_img[:,:,0] = img_c1
    output_img[:,:,1] = img_c2
    output_img[:,:,2] = img_c3

    return output_img

def getActivity(activity_mask):

    activity_mask = activity_mask.copy()
    activity_mask = torch.from_numpy(activity_mask / 255.0)
    activity_mask = transforms.ToPILImage()(activity_mask.float())
    transform_activity = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
    ])
    activity_mask = transform_activity(activity_mask)
    activity_mask = (activity_mask - 0.5) / 0.5
    activity_mask_1c = activity_mask.float()
    
    return activity_mask_1c


