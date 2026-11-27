from PIL import Image
from torch.utils.data import Dataset
import os, glob
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, path, mode, transform=None):
        self.dir_path = path 
        if mode=='train':
            self.d_list = glob.glob(os.path.join(self.dir_path, '*'))
        else:
            self.d_list = glob.glob(os.path.join(self.dir_path, '*'))
        self.transform = transform

    def __len__(self):
        return len(self.d_list)

    def __getitem__(self, idx):
        img_path = self.d_list[idx] 
        image = Image.open(img_path).convert('L') # [h, w]

        if self.transform:
            image = self.transform(image)

        return image.squeeze(0)

import re
class RealDataset(Dataset):
    def __init__(self, path, transform=None):
        self.dir_path = path 
        self.d_list = glob.glob(os.path.join(self.dir_path, '*.png'))
        self.d_list.sort()
        self.transform = transform 
        
    def __len__(self):
        return len(self.d_list)

    def __getitem__(self, idx):
        img_path =  self.d_list[idx]
        image = Image.open(img_path).convert('L')
        
        image = self.transform(image)
        
        return image.squeeze(0)
        
        
        


class CenterSquareCropResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        width, height = img.size

        new_side = min(width, height)
        left = (width - new_side) // 2
        top = (height - new_side) // 2
        right = left + new_side
        bottom = top + new_side
        img = img.crop((left, top, right, bottom))

        img = img.resize((self.size, self.size), Image.LANCZOS)

        return img
        
transform = transforms.Compose([
    CenterSquareCropResize(600),
    transforms.ToTensor(),
])

