



import torch
from torch import nn
import torch.nn.init as init
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_built() else "cpu")

BATCH_SIZE = 16
RESOLUTION = (128,128)
COLORSPACE = 'RGB'
TRAIN_SIZE = 100000
VAL_SIZE = 1000
ADAIN_LATENT_SPACE_DIM = 32



class Adain_Encoder(nn.Module):
    def __init__(self, in_channels, out_dim):
        super(Adain_Encoder, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(512, out_dim, kernel_size=3, stride=2, padding=1)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        
        # Global average pooling
        x = self.global_avg_pool(x)        
        # Flatten
        x = x.view(x.size(0), -1)
        
        return x
    

class AdaIN(nn.Module):
    
    def __init__(self, style_dim, channels):
        super(AdaIN, self).__init__()
        self.instance_norm = nn.InstanceNorm2d(channels, affine=False)
        self.style_scale_transform = nn.Linear(style_dim, channels)
        self.style_shift_transform = nn.Linear(style_dim, channels)

        ## to ensure they learn different stuff | How tho?
        init.normal_(self.style_scale_transform.weight, mean=1.0, std=0.02)
        init.normal_(self.style_shift_transform.weight, mean=0.0, std=0.02)

        self.style_scale_transform.bias.data.fill_(1)  # Initialize scale to 1
        self.style_shift_transform.bias.data.fill_(0)  # Initialize shift to 0

    def forward(self, x, style):
        '''
        x - feature maps from the unet
        y - learned (jointly) from encoder

        return:
        same size as x
        '''
        # Normalize the input feature map
        normalized = self.instance_norm(x)
        
        # Extract style scale and shift parameters from the style vector
        scale = self.style_scale_transform(style)[:, :, None, None]
        shift = self.style_shift_transform(style)[:, :, None, None]
        
        # Apply scale and shift to the normalized feature map
        transformed = scale * normalized + shift
        
        return transformed




class UNetAdaiN(nn.Module):
    def __init__(self, colorspace="RGB", adain_latent_dim=32, dropout_rate=None, verbose=False):
        self.colorspace = colorspace
        self.dropout_rate = dropout_rate
        self.verbose = verbose
        super(UNetAdaiN, self).__init__()
        if self.colorspace == 'RGB':
            in_C = 1
            out_C = 3
        elif self.colorspace == 'LAB':
            in_C = 1
            out_C = 2
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_C, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.maxpool_1to2 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.maxpool_2to3 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.maxpool_3to4 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.conv4_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.maxpool_4to5 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )
        self.conv5_2 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
        )

        # Decoder
        self.conv_transpose_5to6 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.conv1d_fusing_5to6 = nn.Conv2d(256, 128, 1)
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )
        self.conv_transpose_6to7 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1)
        self.conv1d_fusing_6to7 = nn.Conv2d(128, 64, 1)
        self.conv7 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )
        self.conv_transpose_7to8 = nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.conv1d_fusing_7to8 = nn.Conv2d(64, 32, 1)
        self.conv8 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.conv_transpose8to9 = nn.ConvTranspose2d(16, 16, 4, stride=2, padding=1)
        self.conv1d_fusing_8to9 = nn.Conv2d(32, 16, 1)
        self.conv9 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),      # Simmetry broken here: keeps being 64 (from paper)
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.conv10 = nn.Conv2d(16, out_C, 1)

        '''
        Conv1 shape torch.Size([16, 64, 128, 128])
        Conv2 shape torch.Size([16, 128, 64, 64])
        Conv3 shape torch.Size([16, 256, 32, 32])
        Conv4 shape torch.Size([16, 512, 16, 16])
        Conv5 shape torch.Size([16, 512, 8, 8])
        '''

        self.encoder_adain1 = AdaIN(adain_latent_dim, 16)
        self.encoder_adain2 = AdaIN(adain_latent_dim, 32)
        self.encoder_adain3 = AdaIN(adain_latent_dim, 64)
        self.encoder_adain4 = AdaIN(adain_latent_dim, 128)
        self.encoder_adain5 = AdaIN(adain_latent_dim, 128)

        self.decoder_adain4 = AdaIN(adain_latent_dim, 128)
        self.decoder_adain3 = AdaIN(adain_latent_dim, 64)
        self.decoder_adain2 = AdaIN(adain_latent_dim, 32)
        self.decoder_adain1 = AdaIN(adain_latent_dim, 16)

        self.style_encoder = Adain_Encoder(in_channels=out_C, out_dim=adain_latent_dim)
       

    def forward(self,x,style_image):
        x=x.to(device)
        print(f"Input shape: {x.shape}") if self.verbose==True else None
        style_image = style_image.to(device)
        print(f"Style image shape: {style_image.shape}") if self.verbose==True else None
        style = self.style_encoder(style_image).to(device)
        print(f"Style embedding shape: {style.shape}") if self.verbose==True else None

        # Encoder
        conv1 = self.conv1(x)
        conv1 = nn.Dropout2d(p=self.dropout_rate)(conv1) if self.dropout_rate is not None else None
        conv1 = self.encoder_adain1(x,style) ## AdaIN
        print(f"Conv1 shape {conv1.shape}") if self.verbose==True else None

        maxpooled_1to2 = self.maxpool_1to2(conv1)
        conv2 = self.conv2(maxpooled_1to2)
        conv2 = nn.Dropout2d(p=self.dropout_rate)(conv2) if self.dropout_rate is not None else None
        conv2 = self.encoder_adain2(conv2,style) ## AdaIN
        print(f"Conv2 shape {conv2.shape}") if self.verbose==True else None

        maxpooled_2to3 = self.maxpool_2to3(conv2)
        conv3 = self.conv3(maxpooled_2to3)
        conv3 = nn.Dropout2d(p=self.dropout_rate)(conv3) if self.dropout_rate is not None else None
        conv3 = self.encoder_adain3(conv3,style) ## AdaIN
        print(f"Conv3 shape {conv3.shape}") if self.verbose==True else None

        maxpooled_3to4 = self.maxpool_3to4(conv3)
        conv4_1 = self.conv4(maxpooled_3to4)
        conv4_1 = nn.Dropout2d(p=self.dropout_rate)(conv4_1) if self.dropout_rate is not None else None
        conv4_1 = self.encoder_adain4(conv4_1,style) ## AdaIN
        print(f"Conv4_1 shape {conv4_1.shape}") if self.verbose==True else None
        conv4_2 = self.conv4_2(conv4_1)
        conv4_2 = nn.Dropout2d(p=self.dropout_rate)(conv4_2) if self.dropout_rate is not None else None
        conv4_2 = self.encoder_adain4(conv4_2,style) ## AdaIN
        print(f"Conv4_2 shape {conv4_2.shape}") if self.verbose==True else None

        maxpooled_4to5 = self.maxpool_4to5(conv4_2)
        conv5_1 = self.conv5(maxpooled_4to5)
        conv5_1 = nn.Dropout2d(p=self.dropout_rate)(conv5_1) if self.dropout_rate is not None else None
        conv5_1 = self.encoder_adain5(conv5_1,style) ## AdaIN
        print(f"Conv5_1 shape {conv5_1.shape}") if self.verbose==True else None
        conv5_2 = self.conv5_2(conv5_1)
        conv5_2 = nn.Dropout2d(p=self.dropout_rate)(conv5_2) if self.dropout_rate is not None else None
        conv5_2 = self.encoder_adain5(conv5_2,style) ## AdaIN
        print(f"Conv5_2 shape {conv5_2.shape}") if self.verbose==True else None


        # Decoder
        conv_transpose6 = self.conv_transpose_5to6(conv5_2)
        conv_transpose6 = self.decoder_adain4(conv_transpose6,style)
        print(f"Conv_transpose6 shape {conv_transpose6.shape}, concatenates to conv4_2") if self.verbose==True else None
        concatenation_5to6 = torch.cat((conv4_2,conv_transpose6),1)
        skip_fusion_5to6 = self.conv1d_fusing_5to6(concatenation_5to6)
        conv6 = self.conv6(skip_fusion_5to6)
        conv6 = nn.Dropout2d(p=self.dropout_rate)(conv6) if self.dropout_rate is not None else None
        print(f"Conv6 shape {conv6.shape}") if self.verbose==True else None

        conv_transpose7 = self.conv_transpose_6to7(conv6)
        conv_transpose7 = self.decoder_adain3(conv_transpose7,style)
        print(f"Conv_transpose7 shape {conv_transpose7.shape}, concatenates to conv3") if self.verbose==True else None
        concatenation_6to7 = torch.cat((conv3, conv_transpose7),1)
        skip_fusion_6to7 = self.conv1d_fusing_6to7(concatenation_6to7)
        conv7 = self.conv7(skip_fusion_6to7)
        conv7 = nn.Dropout2d(p=self.dropout_rate)(conv7) if self.dropout_rate is not None else None
        print(f"Conv7 shape {conv7.shape}") if self.verbose==True else None

        conv_transpose8 = self.conv_transpose_7to8(conv7)
        conv_transpose8 = self.decoder_adain2(conv_transpose8,style)
        print(f"Conv_transpose8 shape {conv_transpose8.shape}, concatenates to conv2") if self.verbose==True else None
        concatenation_7to8 = torch.cat((conv2, conv_transpose8),1)
        skip_fusion_7to8 = self.conv1d_fusing_7to8(concatenation_7to8)
        conv8 = self.conv8(skip_fusion_7to8)
        conv8 = nn.Dropout2d(p=self.dropout_rate)(conv8) if self.dropout_rate is not None else None
        print(f"Conv8 shape {conv8.shape}") if self.verbose==True else None

        conv_transpose9 = self.conv_transpose8to9(conv8)
        conv_transpose9 = self.decoder_adain1(conv_transpose9,style)
        print(f"Conv_transpose9 shape {conv_transpose9.shape}, concatenates to conv1") if self.verbose==True else None
        concatenation_8_to9 = torch.cat((conv1, conv_transpose9),1)
        skip_fusion_8to9 = self.conv1d_fusing_8to9(concatenation_8_to9)
        conv9 = self.conv9(skip_fusion_8to9)
        conv9 = nn.Dropout2d(p=self.dropout_rate)(conv9) if self.dropout_rate is not None else None
        print(f"Conv9 shape {conv9.shape}") if self.verbose==True else None

        output = self.conv10(conv9)
        print(f"Output shape {output.shape}") if self.verbose==True else None

        if self.colorspace == 'LAB':
            output = torch.cat((x,output),1)
            print(f"Output shape after concatenation of L channel {output.shape}") if self.verbose==True else None

        return style, output

    

    def train_model(self, train_loader, val_loader,
                    epochs=54, 
                    lr=0.0001, 
                    optimizer=torch.optim.Adam, 
                    save_path = "./model_storage/", 
                    save_name_prefix='/',
                    colorspace='RGB',
                    val_check_every=3,
                    plot_every=12,
                    plotting_samples=None,
                    latent_space_test_images=None):

        self.to(device)
        self.train()
        optimizer = optimizer(self.parameters(), lr=lr)

        #for saving progress
        best_val_loss = 99999
        loss_archive = {"training": [], "validation":[]}

        mb = master_bar(range(epochs))
        for epoch in mb:
            training_loss = 0
            for i, batch_data in progress_bar(enumerate(train_loader), total=len(train_loader), parent=mb):
                
                # Input grayscale and color image
                grayscale_images = batch_data['grayscale_image'].to(device)
                if colorspace == 'RGB':
                    colour_images = batch_data['image'].to(device)
                elif colorspace == 'LAB':
                    colour_images = batch_data['image'][:,1:,:,:].to(device)
                # Forward pass
                optimizer.zero_grad()
                _, reproduced_images = self(grayscale_images,colour_images)
                reproduced_images = reproduced_images.to(device)
                # Loss
                loss = perceptual_and_MSE_loss(reproduced_images, batch_data['image'].to(device))
               
                training_loss += loss.item()
                loss.backward()
                optimizer.step()
            
            
            # Average training loss and append to achive
            training_loss /= len(train_loader)
            loss_archive["training"].append(training_loss)            
            with torch.no_grad():
                if (epoch)%val_check_every == 0 or (epoch + 1) == epochs:
                    valdiation_loss = 0

                    for val_data in val_loader:
                        # Input grayscale and color image
                        grayscale_images = val_data['grayscale_image'].to(device)
                        if colorspace == 'RGB':
                            colour_images = val_data['image'].to(device)
                        elif colorspace == 'LAB':
                            colour_images = val_data['image'][:,1:,:,:].to(device)
                            
                        # Forward pass
                        _, reproduced_image = self(grayscale_images,colour_images)
                        reproduced_images = reproduced_images.to(device)
                       
                        loss = perceptual_and_MSE_loss(reproduced_image, val_data['image'].to(device))
                        
                        valdiation_loss += loss.item()
                        
                    valdiation_loss /= len(val_loader)
                    loss_archive["validation"].append(valdiation_loss)                


                    if valdiation_loss < best_val_loss:  # Update best validation loss and save checkpoint if best model
                        state_diction = self.state_dict()
                        best_val_loss = valdiation_loss
                        for key in state_diction.keys():
                            state_diction[key] = state_diction[key].to(torch.device('cpu'))
                        torch.save(state_diction, (save_path+save_name_prefix+f"_best_model.pth.tar"))

                    #construct df with losses
                    loss_df = pd.DataFrame({'epoch': range(0,epoch+1),  
                                            'training_loss':loss_archive['training'], 
                                            'validation_loss': loss_archive['validation']})
                    loss_df.to_csv(save_path+save_name_prefix+"_loss.csv", index=False)

                    # Plot losses
                    plt.clf()
                    plt.plot(loss_df['epoch'], loss_df['training_loss'], label='Training Loss')
                    plt.plot(loss_df['epoch'], loss_df['validation_loss'].interpolate(method='linear'), label='Validation Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training and Validation Loss')
                    plt.legend()
                    plt.savefig(save_path+save_name_prefix+'_loss_plot.png')
                    plt.clf()

                    # Plot images three triplets of images
                    if (epoch)%plot_every==0 and plotting_samples is not None:
                        if colorspace == 'RGB':
                            # 0
                            ground_truth = plotting_samples[0]['image']
                            grayscale_image = plotting_samples[0]['grayscale_image']
                            _,recreated_image = self(grayscale_image.unsqueeze(0).to(device),
                                                ground_truth.unsqueeze(0).to(device))
                            recreated_image = recreated_image.detach().cpu().squeeze(0)
                            plot_four(colored=ground_truth.permute(1,2,0).numpy(),
                                    grayscale=grayscale_image.numpy().squeeze(0),
                                    style=ground_truth.permute(1,2,0).numpy(),
                                    output=recreated_image.permute(1,2,0).numpy())
                            # 1
                            ground_truth = plotting_samples[1]['image']
                            grayscale_image = plotting_samples[1]['grayscale_image']
                            _,recreated_image = self(grayscale_image.unsqueeze(0).to(device),
                                                ground_truth.unsqueeze(0).to(device))
                            recreated_image = recreated_image.detach().cpu().squeeze(0)
                            plot_four(colored=ground_truth.permute(1,2,0).numpy(),
                                    grayscale=grayscale_image.numpy().squeeze(0),
                                    style=ground_truth.permute(1,2,0).numpy(),
                                    output=recreated_image.permute(1,2,0).numpy())
                        elif colorspace == 'LAB':
                            # 0
                            ground_truth = plotting_samples[0]['image']
                            grayscale_image = plotting_samples[0]['grayscale_image']
                            input_encoder = ground_truth[1:,:,:].unsqueeze(0).to(device)
                            lab2rgb = dataset.LABtoRGB()
                            print(f"ground_truth shape {ground_truth.shape}")
                            original_color = lab2rgb(ground_truth.to(device))

                            _,recreated_image = self(grayscale_image.unsqueeze(0).to(device),
                                                    input_encoder)
                            
                            recreated_image = lab2rgb(recreated_image.detach().cpu().squeeze(0))
                            print(f"recreated_image shape: {recreated_image.shape}")
                            plot_four(colored=original_color.permute(1,2,0).numpy(),
                                    grayscale=grayscale_image.numpy().squeeze(0),
                                    style=original_color.permute(1,2,0).numpy(),
                                    output=recreated_image.permute(1,2,0).numpy())
                            # 1
                            ground_truth = plotting_samples[1]['image']
                            grayscale_image = plotting_samples[1]['grayscale_image']
                            input_encoder = ground_truth[1:,:,:].unsqueeze(0).to(device)
                            lab2rgb = dataset.LABtoRGB()
                            print(f"ground_truth shape {ground_truth.shape}")
                            original_color = lab2rgb(ground_truth.to(device))

                            _,recreated_image = self(grayscale_image.unsqueeze(0).to(device),
                                                    input_encoder)
                            
                            recreated_image = lab2rgb(recreated_image.detach().cpu().squeeze(0))
                            print(f"recreated_image shape: {recreated_image.shape}")
                            plot_four(colored=original_color.permute(1,2,0).numpy(),
                                    grayscale=grayscale_image.numpy().squeeze(0),
                                    style=original_color.permute(1,2,0).numpy(),
                                    output=recreated_image.permute(1,2,0).numpy())
                        
                    if (epoch)%plot_every==0 and latent_space_test_images is not None:    
                        plot_histograms(latent_space_values=list(latent_space_test_images))
                        
                else:
                    loss_archive["validation"].append(np.nan)


    



