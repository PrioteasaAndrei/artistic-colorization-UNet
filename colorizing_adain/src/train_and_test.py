import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image, ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from IPython.display import Markdown, display
from net import VGG_Encoder, Decoder, Net
# from sampler import InfiniteSamplerWrapper
import yaml
import matplotlib.pyplot as plt
from colorization_dataset import prepare_dataset, prepare_dataloader

cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
# Disable OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        self.paths = list(Path(self.root).glob('*'))
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'
    


def adjust_learning_rate(cfg, optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = float(cfg["lr"]) / (1.0 + float(cfg["lr_decay"]) * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def train(network,cfg):

    # Setup directories for saving models and logs
    save_dir = Path(cfg['save_dir'])
    log_dir = Path(cfg['log_dir'])

    # Setup device: MPS for Mac, CUDA for Linux, Windows or Colab
    device = torch.device("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_built() else "cpu")
    display(Markdown(f'Using device: {device}'))
    
    # Setup tensorboard writer
    writer = SummaryWriter(log_dir=str(cfg['log_dir']))

    """ # Initialize network
    encoder = VGG_Encoder()
    decoder = Decoder()
    network = Net(encoder, decoder) """
    network.train()
    network.to(device)

    # Setup train dataset and dataloader
    train_data, test_data = prepare_dataset(train_size=cfg['train_size'], test_size=cfg['test_size'])
    train_loader, _ = prepare_dataloader(train_data, test_data, batch_size=cfg['batch_size'])

    # Setup optimizer
    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=float(cfg["lr"]))

    # For every epoch
    for epoch in tqdm(range(cfg["max_iter"])):
        running_loss = 0.
        last_loss = 0.
        adjust_learning_rate(cfg, optimizer, iteration_count=epoch)

        # For every batch
        for i, batch_data in enumerate(train_loader):
            # Unpack colour and greyscale images
            colour_images=batch_data['image']
            greyscale_images=batch_data['grayscale_image']
            
            # Move images to device
            colour_images=colour_images.to(device)
            greyscale_images=greyscale_images.to(device)

            # Zero your gradients for every batch!
            optimizer.zero_grad()

            # Forward pass: compiue loss and its gradients
            reproduced_image,loss=network([colour_images,greyscale_images])
            loss.backward()

            # Adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            if i % 10 == 9:
                last_loss = running_loss / 10 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                display(Markdown(f'reproduced_image min: {reproduced_image.min()} max: {reproduced_image.max()}') )
                running_loss = 0.

        writer.add_scalar('loss', loss.item(), epoch + 1)
        """ writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1) """

        if (epoch + 1) % cfg["save_model_interval"] == 0 or (epoch + 1) == cfg["max_iter"]:
            state_dict_decoder = net.decoder.state_dict()
            for key in state_dict_decoder.keys():
                state_dict_decoder[key] = state_dict_decoder[key].to(torch.device('cpu'))
            torch.save(state_dict_decoder, save_dir /
                    'decoder_sigmoid_iter_{:d}.pth.tar'.format(epoch + 1))
            for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
                block = getattr(net, name)
                state_dict_encoder = block.state_dict()
                for key in state_dict_encoder.keys():
                    state_dict_encoder[key] = state_dict_encoder[key].to(torch.device('cpu'))
                torch.save(state_dict_encoder, save_dir /
                        '{}_sigmoid_iter_{:d}.pth.tar'.format(name,epoch + 1))
    writer.close()


def test_model(net, cfg):
    # Set the model to evaluation mode
    net.eval()

    # Prepare the test data loader
    train_data, test_data = prepare_dataset(train_size=cfg['train_size'], test_size=cfg['test_size'])
    _, test_loader = prepare_dataloader(train_data, test_data, batch_size=3)

    # Iterate over the data loader
    for batch in test_loader:
        original_images = batch['image']
        
        # Forward pass through the network
        with torch.no_grad():
            reproduced_images = net([original_images])
            for reproduced_image in reproduced_images:
                display(Markdown(f'reproduced_image min: {reproduced_image.min()} max: {reproduced_image.max()}') )
                display(Markdown(f'reproduced_image shape: {reproduced_image.shape}'))
                
        # Convert tensors to numpy arrays and transpose to correct format for visualization
        original_images_np = original_images.permute(0, 2, 3, 1).cpu().numpy()
        reproduced_images_np = reproduced_images.permute(0, 2, 3, 1).cpu().numpy()

        # Display original and recreated images
        for i in range(original_images.size(0)):
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(original_images_np[i])
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            axes[1].imshow(reproduced_images_np[i])
            axes[1].set_title('Recreated Image')
            axes[1].axis('off')
            plt.show()