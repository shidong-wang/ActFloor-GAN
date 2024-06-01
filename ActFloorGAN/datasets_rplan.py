import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from utils import getBoundary, getFloorplan, getActivity

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, set_name = 'train'):
        self.transform = transforms.Compose(transforms_)
        path_dataset_4c = f'{root}/dataset_4c'
        path_dataset_split = f'{root}/dataset_split/{set_name}.txt'
        names_fp = open(path_dataset_split).read().split('\n')
        self.floorplans = [f'{path_dataset_4c}/{name}.png' for name in names_fp]
        
    def __getitem__(self, index):

        floorplan_path = self.floorplans[index]
        with Image.open(floorplan_path) as temp:
            image_array = np.asarray(temp, dtype=np.uint8)

        #boundary_mask = image_array[:,:,0]
        category_mask = image_array[:,:,1]
        activity_mask = image_array[:,:,2]
        #inside_mask = image_array[:,:,3]

        item_A = self.transform(Image.fromarray(np.uint8(getFloorplan(category_mask))))
        item_B = self.transform(Image.fromarray(np.uint8(getBoundary(category_mask))))
        item_C = getActivity(activity_mask)

        return {'A': item_A, 'B': item_B, 'C': item_C}
    
    def __len__(self):
        return len(self.floorplans)