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

from sampler import InfiniteSamplerWrapper

import yaml

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



def train(net,cfg):

    save_dir = Path(cfg['save_dir'])
    content_dir=Path(cfg['content_dir'])
    style_dir=Path(cfg['style_dir'])

    device = torch.device("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_built() else "cpu")
    display(Markdown(f'Using device: {device}'))
    
    writer = SummaryWriter(log_dir=str(cfg['log_dir']))

    encoder = VGG_Encoder()
    decoder = Decoder()
    #vgg.load_state_dict(torch.load(cfg.vgg))
    #encoder = nn.Sequential([encoder.relu1,encoder.relu2,encoder.relu3,encoder.relu4])
    network = Net(encoder, decoder)
    network.train()
    network.to(device)

    content_tf = train_transform()
    style_tf = train_transform()

    content_dataset = FlatFolderDataset(cfg["content_dir"], content_tf)
    style_dataset = FlatFolderDataset(cfg["style_dir"], style_tf)

    content_iter = iter(data.DataLoader(
        content_dataset, batch_size=cfg["batch_size"],
        sampler=InfiniteSamplerWrapper(content_dataset),
        num_workers=cfg["n_threads"]))
    style_iter = iter(data.DataLoader(
        style_dataset, batch_size=cfg["batch_size"],
        sampler=InfiniteSamplerWrapper(style_dataset),
        num_workers=cfg["n_threads"]))

    optimizer = torch.optim.Adam(network.decoder.parameters(), lr=float(cfg["lr"]))

    for i in tqdm(range(cfg["max_iter"])):
        adjust_learning_rate(cfg, optimizer, iteration_count=i)
        content_images = next(content_iter).to(device)
        # display(Markdown(f'content_images: {content_images[0]}'))
        style_images = next(style_iter).to(device)
        """ _, loss_c, loss_s = network(content_images, style_images)
        loss_c = cfg["content_weight"] * loss_c
        loss_s = cfg["style_weight"] * loss_s
        loss = loss_c + loss_s """
        reproduced_image,loss=network(content_images)
        if i%50==0:
            display(Markdown(f'loss: {loss}') )
            display(Markdown(f'reproduced_image min: {reproduced_image.min()} max: {reproduced_image.max()}') )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss', loss.item(), i + 1)
        """ writer.add_scalar('loss_content', loss_c.item(), i + 1)
        writer.add_scalar('loss_style', loss_s.item(), i + 1) """

        if (i + 1) % cfg["save_model_interval"] == 0 or (i + 1) == cfg["max_iter"]:
            state_dict = net.decoder.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].to(torch.device('cpu'))
            torch.save(state_dict, save_dir /
                    'decoder_iter_{:d}.pth.tar'.format(i + 1))
    writer.close()
