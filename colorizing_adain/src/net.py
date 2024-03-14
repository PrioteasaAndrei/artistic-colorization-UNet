import torch
import torchvision
from IPython.display import Markdown, display

#from colorizing_adain.src.UNUSED_function import calc_mean_std
#from colorizing_adain.src.UNUSED_function import adaptive_instance_normalization as adain

'''
Encoder is a pretrained VGG up to relu4_1 as in the original paper (see 6.1 paper)
'''
class VGG_Encoder(torch.nn.Module):
    def __init__(self):
        super(VGG_Encoder, self).__init__()
        pretrained = torchvision.models.vgg19(pretrained=True)
        
        f = torch.nn.Sequential(*list(pretrained.features.children())[:21]).eval()
        
        # Splitting the network so we can get output of different layers
        # TODO: ADD REFLECTION PADDING LAYERS
        self.relu1 = torch.nn.Sequential(*f[:2],)
        self.relu2 = torch.nn.Sequential(*f[2:5], *f[5:7])
        self.relu3 = torch.nn.Sequential(*f[7:10],*f[10:12])
        self.relu4 = torch.nn.Sequential(*f[12:14],
                                          *f[14:16],
                                          *f[16:19],
                                           *f[19:21])

    def forward(self, x):
        out_1 = self.relu1(x)
        out_2 = self.relu2(out_1)
        out_3 = self.relu3(out_2)
        result = self.relu4(out_3)
        return out_1, out_2, out_3, result
    

''' decoder is just the second part of an Unet'''
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decode = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(512, 256, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 256, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 256, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 256, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 128, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(128, 128, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(128, 64, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(64, 64, (3, 3)),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(64, 3, (3, 3)),
            #torch.nn.Sigmoid(),
        )
    def forward(self, x):
        result = self.decode(x)
        return result
"""
decode = Decoder()
img = decode(t)
concat_img((img[:12]).detach().cpu())
"""


class Net(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        #self.enc_1 = torch.nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        #self.enc_2 = torch.nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        #self.enc_3 = torch.nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        #self.enc_4 = torch.nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_1 = torch.nn.Sequential(encoder.relu1)  # input -> relu1
        self.enc_2 = torch.nn.Sequential(encoder.relu2)  # relu1 -> relu2
        self.enc_3 = torch.nn.Sequential(encoder.relu3)  # relu2 -> relu3
        self.enc_4 = torch.nn.Sequential(encoder.relu4)  # relu3 -> relu4
        self.decoder = decoder
        #self.mse_loss = torch.nn.MSELoss()

        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
    
    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    # extract relu4_1 from input image
    def encode(self, input):
        for i in range(4):
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input
    """ 
    def calc_content_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def forward(self, content, style, alpha=1.0):
        assert 0 <= alpha <= 1
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)
        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])
        return g_t,loss_c, loss_s """
    
    def forward(self, images):
        # Forget about colorization for the moment: test on recreating colour images

        if len(images)==2:          # Train time
            colour_images = images[0]
            grayscale_images = images[1]
            encoded = self.encode(colour_images)
            reproduced_image=self.decoder(encoded)
            loss = torch.nn.functional.mse_loss(colour_images, reproduced_image)
            return reproduced_image,loss
        elif len(images)==1:        # Test time
            grayscale_images = images[0]
            encoded = self.encode(grayscale_images)
            reproduced_image=self.decoder(encoded)
            return reproduced_image
        else:
            raise ValueError("Wrong argument passed to formward method. It should be either a list of 2 batches of images for training or a list of 1 batch of image for testing.")