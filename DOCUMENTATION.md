# Artistic colorisation with Adaptive Istance Normalization

### 🧑🏻‍🎓 Team Members

| Name and surname    |  Matric. Nr. | GitHub username  |   e-mail address   |
|:--------------------|:-------------|:-----------------|:-------------------|
| Alexander Nyamekye | <span style="color:red"> *(?)* </span>| nalexunder | alexander.nyamekye@stud.uni-heidelberg.de|
| Cristi Andrei Prioteasa | <span style="color:red"> *(?)* </span>| prio | cristi.prioteasa@stud.uni-heidelberg.de|
| Matteo Malvestiti | 4731243| Matteo-Malve | matteo.malvestiti@stud.uni-heidelberg.de|
| Jan Smolen | <span style="color:red"> *(?)* </span>| <span style="color:red"> *(?)* </span> | wm315@stud.uni-heidelberg.de|


### Advisor

Denis Zavadski: denis.zavadski@iwr.uni-heidelberg.de

***

<p align="left">
  <img src="./images/architecture_map.png" width="2000" />
</p>

## Table of contents

1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Related Work](#related_work)
4. [Approach](#approach)
5. [Conclusion](#conclusion)

# <a name="introduction"></a>1. Introduction
Our sophisticated network allows to colorise grayscale images and also to transfer a style to it, thanks to Adaptive Instance normalization layers (AdaIN) displaced both in the encoder and the decoder part of our UNet.


# <a name="motivation"></a>2. Motivation
This project has his roots in our fascination for the colorisation task: an underconstraied task that is still under research and has big room for improvements. It is mathematically impossible to deduce from the single channel of grayscale images the 3 RGB or LAB channels. The only certainty is the color intensity pixel per pixel, but it is necessary to "guess" the colors.
Deep Convolutional Neural Networks are a powerful tool to tackle this task, since they can learn to understand the semantics of the image itself.
Moreover the AdaIN encoder, if used in training on the ground truth colored image itself, can be a powerful tool to encode the colour semantics. This addition alone, consisting in a regularisation and having yet no capacity to do style transfer, lead to a great improvement in colorisation quality.

The first idea of our group was less ambitious, namely to train a UNet for colorisation and finetuining it on different style images, like paintings of certain styles, and to present the user a limited choice of styles to colorise its uploaded grayscale iamge.
Our tutor, though, immediately introduced us to Adaptive Instance normalization, which, after a careful read of the related scientific paper ([Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)), we thought to be facinating and we immediately took as primary objective.

# <a name="related_work"></a>3. Related work
Our first struggle was definitely to train a Neural Network that was able to recreate first, and then to colorise images. We started from an encoder-decoder architecture that had a pretrained VGG19 encoder, discarding the classification head. The idea was to save a lot on computational cost. Unfortunately it wasn't enough: if we froze the encoder and trained the decoder alone, we were underfitted. If we unfroze the encoder, VGG19 was simply not worth it. Indeed, it was a big network, taking a lot of comptuational resourses, but at the same time it didn't offer skip connections or the flexibility to insert AdaIN layers, not while wanting to load the pretrained weights at least.

Anyways, this was a big part of our first two weeks of work. The paper we were getting inspiration from was: [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868), but mainly its unofficial implementation from the GitHub repository [GitHub: naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN).

Our first results in recreation and colorisation came when we started following the approach described in [Image Colorization using U-Net with Skip Connections and Fusion Layer on Landscape Images](https://arxiv.org/abs/2205.12867). It was recreating manually layer by layer their UNet that we overcame our troubles.

The next step, namely the introduction of AdaIN layers for regularisation, was discussed in a meeting with our tutor and had no direct backup from scientific papers. The Adaptive Instance Normalization is done like in [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868), but with a big difference: the parameters of mean and standard deviation are learned every time from simple linear layers that act as interfaces between the latent space of the AdaIN encoder and the feature maps where normalization takes place. If you refer to the big architecture map in the beginning of this documentation, we are talking about the blue pins.
This idea of learning the AdaIN parameters comes from NVIDIA's 2019's [StyleGAN paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf).


# <a name="approach"></a>4. Approach


# <a name="conclusion"></a>5. Conclusion
