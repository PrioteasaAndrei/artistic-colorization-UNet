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

torch.manual_seed(1)
load_dotenv()

# Define a custom transformation to convert RGB to LAB color space
class RGBtoLAB(object):
    def __call__(self, img):
        # Convert the image from RGB to LAB color space
        ## TODO: normalize
        lab_img = color.rgb2lab(img.permute(1, 2, 0).cpu().numpy()).transpose(2, 0, 1) 
        lab_img = lab_img / 100  # Scale L channel to [0, 1]
        lab_img[1:, :, :] = (lab_img[1:, :, :] + 128) / 255  # Scale a and b channels to [0, 1]
        return lab_img
     
# Define a custom transformation to convert LAB to grayscale
class LABtoGray(object):
    def __call__(self, lab_img):
        # Convert LAB image to grayscale
        gray_img = lab_img[0:1, :, :]  # Extract the L channel (1st channel)
        return gray_img

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


def check_range(tensor):
    return (tensor.min() >= 0) and (tensor.max() <= 1)


def prepare_dataset(train_size=10,test_size=10):
    '''
    check out this for docs: https://huggingface.co/docs/datasets/stream

    you need to make list(output)
    the returned shapes of the images should be:
    train: [3,256,256]
    test: [1,256,256]
    '''
    login_token = os.getenv('HUGGING_FACE_TOKEN')
    dataset_train = load_dataset("imagenet-1k",split='train',use_auth_token=login_token,streaming=True)
    dataset_test = load_dataset("imagenet-1k",split='test',use_auth_token=login_token,streaming=True)
    ## map resize transformation before take 
    transformed_train = dataset_train.map(lambda x: {'image': transform_train(x['image'])})
    transformed_test = dataset_test.map(lambda x: {'image': transform_test(x['image'])})
    
    ## normalization check
    normalization_check(list(transformed_train.take(2)), list(transformed_test.take(2)))

    return transformed_train.take(train_size),transformed_test.take(test_size)

def normalization_check(transformed_train_list, transformed_test_list):
  
    assert check_range(transformed_train_list[0]['image']) == True
    assert check_range(transformed_test_list[0]['image']) == True

    display('All tests passed')