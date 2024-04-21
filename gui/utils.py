import os
import torch 
import torchvision
from model import UNetAdaiN
from PIL import Image



BATCH_SIZE = 16
RESOLUTION = (128,128)
COLORSPACE = 'RGB'
TRAIN_SIZE = 100000
VAL_SIZE = 1000
ADAIN_LATENT_SPACE_DIM = 32

device = torch.device("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_built() else "cpu")


torch.manual_seed(1)


vgg_mean = [0.485, 0.456, 0.406]
vgg_std = [0.229, 0.224, 0.225]

vgg_transform_direct = torchvision.transforms.Normalize(mean=vgg_mean, std=vgg_std)
vgg_transorm_inverse = torchvision.transforms.Normalize(mean=[-vgg_mean[0]/vgg_std[0], -vgg_mean[1]/vgg_std[1], -vgg_mean[2]/vgg_std[2]],
                                                          std=[1/vgg_std[0], 1/vgg_std[1], 1/vgg_std[2]])



def load_model_from_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model = UNetAdaiN(colorspace=COLORSPACE,adain_latent_dim=ADAIN_LATENT_SPACE_DIM, dropout_rate=0.1, verbose=False)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully from checkpoint.")
        return model.to(device)
    else:
        print("Checkpoint file does not exist.")
        return None


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