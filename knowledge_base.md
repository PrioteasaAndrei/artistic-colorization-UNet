## Colorization

Original paper [here](https://arxiv.org/pdf/2204.02980.pdf) (LOOKS REALLY PROMISING)

### Loss function

Most used loss function: MSE(original,colorized)= sum_H (sum_W (sum_C original - colorized))

Exotic loss: learn probability distribution of color and use KL loss

Conclusion from their paper: loss function doesnt do that much of a difference

### Evaluation

Report PSNR see [here](https://lightning.ai/docs/torchmetrics/stable/image/peak_signal_noise_ratio.html) (3.2 in paper)

### Architecture

The encoder architecture is identical to the CNN part of a VGG network. It allows us to start from pretrained weights initially used for ImageNet classification.

In Unet the skip connections work by concatenatic the input to the output.

### Training configuration

The training settings are described as follows:
¢ Optimizer: Adam
¢ Learning rate: 2e-5.
¢ Batch size: 16 images (10-11 GB RAM on Nvidia Titan V).
¢ All images are resized to 256 x 256 for training which enables using batches. In practice, to keep the aspect ratio, the image is resized such that the smallest dimension matches 256. If the other dimension remains larger than 256, we then apply a random crop to obtain a square image. Note that the random crop is performed using the same seed for all trainings.

More details regarding this framework are given in the other chapter Influence of Color Spaces for Deep Learning Image Colorization |Ballester et al., 2022.

## Adaptive Instance Normalization

THe trained network needs only one image that represents the style to which the image will be transferred.

Github repo is here, but its in lua :( https://github.com/xunhuang1995/AdaIN-style

See 6.2 Training here: https://arxiv.org/pdf/1703.06868v2.pdf (original paper)

Check this out: https://github.com/jcjohnson/neural-style
CHeck this out: https://www.kaggle.com/code/davidcanorosillo/adain-style

In the original paper they use encoder (vgg not all layers) -> AdaIn -> decoder 

How to treat skip connections in this case?
How to adapt loss function?

### Training details

From [this](https://web.eecs.umich.edu/~justincj/teaching/eecs442/projects/WI2021/pdfs/055.pdf) it seems that the following configuration works for style transfer using the VGG architecture and AdaIN (assume VGG is pretrained):

- epochs: 54
- dataset: 1200 style images and 1200 content images
- batch_size: 8
- style weight: 20
- input_size: 512x512
- transformation: random crop 256x256
- normalize to mean=[0.485, 0.456, 0.406] and stdev=[0.229,
  0.224, 0.225] (only needed to match the pre-processing of the pretrained VGG-19)
- Adam optimizer; lr = 1e-4
- L = Lc + lambda \* Ls; See 2.2
