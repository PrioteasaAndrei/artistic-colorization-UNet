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

device = torch.device("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_built() else "cpu")
print(f"Device: {device}")

vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]

vgg_transform_direct = torchvision.transforms.Normalize(mean=vgg_mean, std=vgg_std)
vgg_transorm_inverse = torchvision.transforms.Normalize(mean=[-vgg_mean[0]/vgg_std[0], -vgg_mean[1]/vgg_std[1], -vgg_mean[2]/vgg_std[2]],
                                                          std=[1/vgg_std[0], 1/vgg_std[1], 1/vgg_std[2]])
# Define a custom transformation to convert RGB to LAB color space
class RGBtoLAB(object):
    def __call__(self, img):
        if img.shape[0] == 1:
            img = img.repeat(3,1,1)
        
        lab_img = color.rgb2lab(img.permute(1, 2, 0).cpu().numpy()).transpose(2, 0, 1) 
        lab_img[0, :, :] = lab_img[0, :, :] / 100  # Scale L channel to [0, 1]
        lab_img[1, :, :] = (lab_img[1, :, :] + 128) / 255  # Scale a and b channels to [0, 1]
        lab_img[2, :, :] = (lab_img[2, :, :] + 128) / 255 # Scale a and b channels to [0, 1]        
        return torch.tensor(lab_img) 
    

class LABtoRGB(object):
    '''
    Tested and it works fine
    '''
    def __call__(self, lab_img):
        # Scale the LAB channels back to their original ranges
        lab_img[0, :, :] = lab_img[0, :, :] * 100  # Scale L channel back to [0, 100]
        lab_img[1:, :, :] = lab_img[1:, :, :] * 255 - 128  # Scale a and b channels back to [-128, 127]
        
        lab_img = lab_img.permute(1, 2, 0)
        rgb_img = color.lab2rgb(lab_img.cpu().numpy())

        # Convert the resulting RGB image to unnormalized format (0-255)
        rgb_img_unnormalized = (rgb_img * 256).astype(np.uint8)

        return torch.tensor(rgb_img_unnormalized).permute(2, 0, 1)

# Define a custom transformation to convert LAB to grayscale
class LABtoGray(object):
    def __call__(self, lab_img):
        # Convert LAB image to grayscale
        gray_img = lab_img[0:1, :, :]  # Extract the L channel (1st channel)
        
        return torch.tensor(gray_img)


'''
normalize to mean=[0.485, 0.456, 0.406] and stdev=[0.229,0.224, 0.225]. This is expected by the pretrained VGG.
See: https://pytorch.org/vision/main/models/generated/torchvision.models.vgg19.html
'''

## TODO: add random crop rotations and other augumentations
def get_transforms(colorspace='RGB',resolution=(128,128)):
    transform_train, transform_test = None, None

    if colorspace == 'RGB':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resolution, interpolation=Image.BILINEAR),
            torchvision.transforms.ToTensor(),
            # RGBtoLAB()
            #vgg_transform_direct
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resolution, interpolation=Image.BILINEAR),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(num_output_channels=1)
            # RGBtoLAB(),
            # LABtoGray()
            #vgg_transform_direct
        ])
    elif colorspace == 'LAB':
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resolution, interpolation=Image.BILINEAR),
            torchvision.transforms.ToTensor(),
            RGBtoLAB()
            #vgg_transform_direct
        ])

        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.Resize(resolution, interpolation=Image.BILINEAR),
            torchvision.transforms.ToTensor(),
            RGBtoLAB(),
            LABtoGray()
            #vgg_transform_direct
        ])
    else:
        raise ValueError('Invalid colorspace. Please choose either RGB or LAB')
        
    return transform_train, transform_test



## tested | works fine
transform_inverse = torchvision.transforms.Compose([
    vgg_transorm_inverse,
    LABtoRGB()
    ])

# Define a custom dataset class for automatic batching
class ColorizationDataset(Dataset):
    def __init__(self, data, test=False):
        self.data = data
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]['image']
        if self.test:
            return {'image': image}
        else:
            gray_scale = self.data[idx]['grayscale_image']
            return {'image': image, 'grayscale_image': gray_scale}



def check_range(tensor):
    return (tensor.min() >= 0) and (tensor.max() <= 1)


def prepare_dataset(train_size=10,test_size=10,batch_size=4,colorspace='RGB',resolution=(128,128)):
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

    transform_train, transform_test = get_transforms(colorspace,resolution=resolution)

    login_token = os.getenv('HUGGING_FACE_TOKEN')
    dataset_train = load_dataset("imagenet-1k",split='train',use_auth_token=login_token,streaming=True)
    dataset_test = load_dataset("imagenet-1k",split='test',use_auth_token=login_token,streaming=True)
    ## map resize transformation before take 
    transformed_train = dataset_train.map(lambda x: {'image': transform_train(x['image']), 'grayscale_image': transform_test(x['image']),'label': torch.tensor(x['label'])})
    transformed_test = dataset_test.map(lambda x: {'image': transform_train(x['image']), 'grayscale_image': transform_test(x['image']),'label': torch.tensor(x['label'])})
    
    ## shuffle train dataset
    # transformed_train = transformed_train.shuffle()

    ## normalization check
    # normalization_check(list(transformed_train.take(2)), list(transformed_test.take(2)))
 
    print("Dataset loaded successfully")
    return transformed_train.take(train_size),transformed_test.take(test_size)
    

def prepare_dataloader(train_data,test_data,batch_size=4):
    '''
    function description

    returns:
    - colorization_dataloader_train: DataLoader
    - colorization_dataloader_test: DataLoader
    '''
    ## prepare data loader


    ## NOTE: remember that there are some grayscale images in the dataset that need to be filtered out
    ## NOTE: after filtering size may be different from the original size
   
    filtered_train_data = []
    filtered_test_data = []

    for entry in list(train_data):
        if entry['image'].shape[0] == 3:
            filtered_train_data.append(entry)
    
    for entry in list(test_data):
        if entry['image'].shape[0] == 3:
            filtered_test_data.append(entry)

    colorization_dataset_train = ColorizationDataset(filtered_train_data)
    colorization_dataloader_train = DataLoader(colorization_dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)

    colorization_dataset_test = ColorizationDataset(filtered_test_data)
    colorization_dataloader_test = DataLoader(colorization_dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print("Data loader prepared successfully")
    return colorization_dataloader_train, colorization_dataloader_test


def normalization_check(transformed_train_list, transformed_test_list):
  
    assert check_range(transformed_train_list[0]['image']) == True
    assert check_range(transformed_test_list[0]['image']) == True

    display('All tests passed')