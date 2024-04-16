from dataset import *

'''
Encoder is a pretrained VGG up to relu4_1 as in the original paper (see 6.1 paper)
'''
class VGG_Encoder(torch.nn.Module):
    def __init__(self):
        super(VGG_Encoder, self).__init__()
        pretrained = torchvision.models.vgg19(pretrained=True)
        
        f = torch.nn.Sequential(*list(pretrained.features.children())[:21]).eval()

        self.relu1_1 = torch.nn.Sequential(*f[:2],)
        self.relu2_1 = torch.nn.Sequential(*f[2:5], *f[5:7])
        self.relu3_1 = torch.nn.Sequential(*f[7:10],*f[10:12])
        self.relu4_1 = torch.nn.Sequential(*f[12:14],
                                          *f[14:16],
                                          *f[16:19],
                                           *f[19:21])
        
        for param in self.relu1_1.parameters():
            param.requires_grad = False
        for param in self.relu2_1.parameters():
            param.requires_grad = False
        for param in self.relu3_1.parameters():
            param.requires_grad = False
        for param in self.relu4_1.parameters():
            param.requires_grad = False

    def forward(self, x):
        out_1 = self.relu1_1(x)
        out_2 = self.relu2_1(out_1)
        out_3 = self.relu3_1(out_2)
        result = self.relu4_1(out_3)
        ## print shapes
        # print("Output 1",out_1.shape)
        # print("Output 2",out_2.shape)
        # print("Output 3",out_3.shape)
        # print("Output 4",result.shape)

        return out_1, out_2, out_3, result

def mean_and_std(x):
    x = x.view(x.shape[0], x.shape[1], -1)
    mean = x.mean(dim=2) + 0.00005
    std = x.var(dim=2).sqrt()
    return mean.view(mean.shape[0], mean.shape[1], 1, 1), std.view(std.shape[0], std.shape[1], 1, 1)


''' 
decoder is just the second part of an Unet
implement skip connections (feed concat to the upsample layer)
'''
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.block1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(512, 256, (3, 3)),
            torch.nn.BatchNorm2d(256),  # BatchNorm2d added
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),

        )

        self.block2 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(512, 256, (3, 3)),
            torch.nn.BatchNorm2d(256),  # BatchNorm2d added
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 256, (3, 3)),
            torch.nn.BatchNorm2d(256),  # BatchNorm2d added
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 256, (3, 3)),
            torch.nn.BatchNorm2d(256),  # BatchNorm2d added
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 128, (3, 3)),
            torch.nn.BatchNorm2d(128),  # BatchNorm2d added
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),

        )

        self.block3 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(256, 128, (3, 3)),
            torch.nn.BatchNorm2d(128),  # BatchNorm2d added
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(128, 64, (3, 3)),
            torch.nn.BatchNorm2d(64),  # BatchNorm2d added
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),

        )

        self.block4 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            torch.nn.Conv2d(128, 64, (3, 3)),
            torch.nn.BatchNorm2d(64),  # BatchNorm2d added
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d((1, 1, 1, 1)),
            # torch.nn.Conv2d(64, 3, (3, 3)),
            ## predict only a and b channels
            torch.nn.Conv2d(64, 2, (3, 3))
        )

        ## w/ batchnorm
    #     self.decode = torch.nn.Sequential(
    #         torch.nn.ReflectionPad2d((1, 1, 1, 1)),
    #         torch.nn.Conv2d(512, 256, (3, 3)),
    #         torch.nn.BatchNorm2d(256),  # BatchNorm2d added
    #         torch.nn.ReLU(),

    #         torch.nn.Upsample(scale_factor=2, mode='nearest'),
    #         torch.nn.ReflectionPad2d((1, 1, 1, 1)),
    #         torch.nn.Conv2d(256, 256, (3, 3)),
    #         torch.nn.BatchNorm2d(256),  # BatchNorm2d added
    #         torch.nn.ReLU(),
    #         torch.nn.ReflectionPad2d((1, 1, 1, 1)),
    #         torch.nn.Conv2d(256, 256, (3, 3)),
    #         torch.nn.BatchNorm2d(256),  # BatchNorm2d added
    #         torch.nn.ReLU(),
    #         torch.nn.ReflectionPad2d((1, 1, 1, 1)),
    #         torch.nn.Conv2d(256, 256, (3, 3)),
    #         torch.nn.BatchNorm2d(256),  # BatchNorm2d added
    #         torch.nn.ReLU(),
    #         torch.nn.ReflectionPad2d((1, 1, 1, 1)),
    #         torch.nn.Conv2d(256, 128, (3, 3)),
    #         torch.nn.BatchNorm2d(128),  # BatchNorm2d added
    #         torch.nn.ReLU(),

    #         torch.nn.Upsample(scale_factor=2, mode='nearest'),
    #         torch.nn.ReflectionPad2d((1, 1, 1, 1)),
    #         torch.nn.Conv2d(128, 128, (3, 3)),
    #         torch.nn.BatchNorm2d(128),  # BatchNorm2d added
    #         torch.nn.ReLU(),
    #         torch.nn.ReflectionPad2d((1, 1, 1, 1)),
    #         torch.nn.Conv2d(128, 64, (3, 3)),
    #         torch.nn.BatchNorm2d(64),  # BatchNorm2d added
    #         torch.nn.ReLU(),

    #         torch.nn.Upsample(scale_factor=2, mode='nearest'),
    #         torch.nn.ReflectionPad2d((1, 1, 1, 1)),
    #         torch.nn.Conv2d(64, 64, (3, 3)),
    #         torch.nn.BatchNorm2d(64),  # BatchNorm2d added
    #         torch.nn.ReLU(),
    #         torch.nn.ReflectionPad2d((1, 1, 1, 1)),
    #         # torch.nn.Conv2d(64, 3, (3, 3)),
    #         ## predict only a and b channels
    #         torch.nn.Conv2d(64, 2, (3, 3)),
    # )

        
    def forward(self, x):
        '''
        Wont be used in the UNET
        '''
        x = self.block1(x)
        print("Block 1",x.shape)
        x = self.block2(x)
        print("Block 2",x.shape)
        x = self.block3(x)
        print("Block 3",x.shape)
        x = self.block4(x)
        print("Block 4",x.shape)
        return x
    
class Unet(torch.nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        self.encoder = VGG_Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        out_1, out_2, out_3, out_4 = self.encoder(x)

        decoder_block1 = self.decoder.block1(out_4)
        decoder_block2 = self.decoder.block2(torch.cat((decoder_block1, out_3), 1))
        decoder_block3 = self.decoder.block3(torch.cat((decoder_block2, out_2), 1))
        return self.decoder.block4(torch.cat((decoder_block3, out_1), 1))        

        # return self.decoder(out_4)