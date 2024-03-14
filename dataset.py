import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os
from zipfile import ZipFile
from PIL import Image
from skimage import color
from datasets import load_dataset
from dotenv import load_dotenv
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)
load_dotenv()

# Define a custom transformation to convert RGB to LAB color space
class RGBtoLAB(object):
    def __call__(self, img):
        # Convert the image from RGB to LAB color space
        ## one image is grayscale just repeat channels
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)

        lab_img = color.rgb2lab(img.permute(1, 2, 0).cpu().numpy()).transpose(2, 0, 1) 
        lab_img = lab_img / 100  # Scale L channel to [0, 1]
        lab_img[1:, :, :] = (lab_img[1:, :, :] + 128) / 255  # Scale a and b channels to [0, 1]
        return torch.tensor(lab_img) 
     
# Define a custom transformation to convert LAB to grayscale
class LABtoGray(object):
    def __call__(self, lab_img):
        # Convert LAB image to grayscale
        gray_img = lab_img[0:1, :, :]  # Extract the L channel (1st channel)
        return torch.tensor(gray_img)

transform_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
    RGBtoLAB()
])

transform_test = torchvision.transforms.Compose([
    torchvision.transforms.Resize((256, 256)),
    torchvision.transforms.ToTensor(),
    RGBtoLAB(),
    LABtoGray(),
])

# Define a custom dataset class for automatic batching
class ColorizationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        gray_scale = self.data[idx]['grayscale_image']
      
        return {'image': image, 'grayscale_image': gray_scale}


def check_range(tensor):
    return (tensor.min() >= 0) and (tensor.max() <= 1)


def prepare_dataset(train_size=10,test_size=10,batch_size=4):
    '''
    check out this for docs: https://huggingface.co/docs/datasets/stream

    you need to make list(output)
    the returned shapes of the images should be:
    train: [3,256,256]
    test: [1,256,256]

    returns:
    - colorization_dataloader_train: DataLoader
    - colorization_dataloader_test: DataLoader
    '''
    login_token = os.getenv('HUGGING_FACE_TOKEN')
    dataset_train = load_dataset("imagenet-1k",split='train',use_auth_token=login_token,streaming=True)
    dataset_test = load_dataset("imagenet-1k",split='test',use_auth_token=login_token,streaming=True)
    ## map resize transformation before take 
    ## TODO: 25th image is grayscale and fucks up the dataloader
    ## its because of the color transformation from skimage that doesnt work so overwrite that
    transformed_train = dataset_train.map(lambda x: {'image': transform_train(x['image']), 'grayscale_image': transform_test(x['image']),'label': torch.tensor(x['label'])})
    transformed_test = dataset_test.map(lambda x: {'image': transform_test(x['image']),'label': torch.tensor(x['label'])})
    
    ## shuffle train dataset
    # transformed_train = transformed_train.shuffle()

    ## normalization check
    normalization_check(list(transformed_train.take(2)), list(transformed_test.take(2)))
 
    print("Dataset loaded successfully")
    return transformed_train.take(train_size),transformed_test.take(test_size)
    

def prepare_dataloader(train_data,test_data,batch_size=4):
    ## prepare data loader

    list_train_data = list(train_data)
    ## filter for 1 channel images
    train_data = list(filter(lambda x: x['image'].shape[0] == 3, list_train_data))

    colorization_dataset_train = ColorizationDataset(train_data)
    colorization_dataloader_train = DataLoader(colorization_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    colorization_dataset_test = ColorizationDataset(list(test_data))
    colorization_dataloader_test = DataLoader(colorization_dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print("Data loader prepared successfully")
    return colorization_dataloader_train, colorization_dataloader_test


def normalization_check(transformed_train_list, transformed_test_list):
  
    assert check_range(transformed_train_list[0]['image']) == True
    assert check_range(transformed_test_list[0]['image']) == True

    display('All tests passed')