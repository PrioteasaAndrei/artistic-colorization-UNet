import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import sys
import dataset

# sys.path.append("src/")

device = torch.device("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_built() else "cpu")

def plot_both(image1,image2,save_name=None):
    plt.figure(figsize=(10, 5))

    # Plot first image
    plt.subplot(1, 2, 1)
    plt.imshow(image1)
    plt.title('Original')

    # Plot second image
    plt.subplot(1, 2, 2)
    plt.imshow(image2)
    plt.title('Colorized')

    if save_name:
        plt.savefig(save_name)


    plt.show()


def compose_output(original_image, recreated_image):
    print(original_image.shape)
    print(recreated_image.shape)
    l = original_image[0,:,:]
    print("L shape",l.shape)
    return torch.cat((l.unsqueeze(0),recreated_image),0)

'''
TODO: when using lab color space we have certain artefacts that do not appear when we only use RGB images

may be from normalization in the lab space or the transformation itself

entry[image][0]
'''
def visualize(entry,model,lab=False):
    image = entry['grayscale_image']
    output = model(image.unsqueeze(0).to(device))
    output = output.detach().cpu().squeeze(0)

    if lab:
        lab2rgb = dataset.LABtoRGB()
        rgb_image = lab2rgb(entry['image'])
        ## TODO: if here entry['image'][0:1,:,:] is used, the image is not displayed correctly
        composed_lab = torch.cat((entry['grayscale_image'],output),0)
        composed_output = lab2rgb(composed_lab)


    plot_both(rgb_image.permute(1,2,0).numpy(),composed_output.permute(1,2,0).numpy())



def plot_grid(images_column1, images_column2):
    '''
    Plots a 2x8 grid of rgb images, where each row corresponds to a pair of images from the two input lists.
    '''
    num_images_column1 = len(images_column1)
    num_images_column2 = len(images_column2)
    num_rows = max(num_images_column1, num_images_column2)
    
    plt.figure(figsize=(12, 8))  # Adjust figsize as needed

    for i in range(num_rows):
        # Plot images from the first column
        if i < num_images_column1:
            plt.subplot(num_rows, 2, 2*i + 1)
            plt.imshow(images_column1[i].permute(1,2,0).numpy())
            # plt.title(f'Image {i + 1} (Column 1)')
            plt.axis('off')  # Turn off axis for better visualization
        
        # Plot images from the second column
        if i < num_images_column2:
            plt.subplot(num_rows, 2, 2*i + 2)
            plt.imshow(images_column2[i].permute(1,2,0).numpy())
            # plt.title(f'Image {i + 1} (Column 2)')
            plt.axis('off')  # Turn off axis for better visualization

    plt.tight_layout(pad=0.1)  # Adjust spacing between subplots
    plt.show()

def get_visualization_images(indexes,train_data):
    data = list(train_data)
    tbr = []
    for idx in indexes:
        tbr.append(data[idx]['image'])

    return tbr

def get_batch_outputs(images,model):
    ## batch the images
    images = torch.stack(images)
    output = model(images.to(device))
    return [output[i].detach().cpu() for i in range(output.shape[0])]