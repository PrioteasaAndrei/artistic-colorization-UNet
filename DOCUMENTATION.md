# Artistic colorisation with Adaptive Istance Normalization

### 🧑🏻‍🎓 Team Members

| Name and surname    |  Matric. Nr. | GitHub username  |   e-mail address   |
|:--------------------|:-------------|:-----------------|:-------------------|
| Alexander Nyamekye | <span style="color:red"> *(?)* </span>| nalexunder | alexander.nyamekye@stud.uni-heidelberg.de|
| Cristi Andrei Prioteasa | <span style="color:red"> 	4740844 </span>| PrioteasaAndrei | cristi.prioteasa@stud.uni-heidelberg.de|
| Matteo Malvestiti | 4731243| Matteo-Malve | matteo.malvestiti@stud.uni-heidelberg.de|
| Jan Smolen | <span style="color:red"> *(?)* </span>| <span style="color:red"> *(?)* </span> | wm315@stud.uni-heidelberg.de|


### Advisor

Denis Zavadski: denis.zavadski@iwr.uni-heidelberg.de

***


## Table of contents

1. [Introduction](#introduction)
2. [Motivation](#motivation)
3. [Related Work](#related_work)
4. [Approach](#approach)
5. [Conclusion](#conclusion)

# <a name="introduction"></a>1. Introduction

Colorization of grayscale images has been an area of significant interest in computer vision, with applications ranging from photo restoration to artistic expression. We propose an approach to colorize grayscale images in various artistic styles using a U-Net architecture enhanced with Adaptive Instance Normalization (AdaIN) layers. U-Net, known for its effectiveness in semantic segmentation tasks, provides an ideal framework for our colorization task due to its ability to capture spatial dependencies while preserving fine details. By incorporating AdaIN layers into the U-Net architecture, we introduce the capability to adaptively transfer artistic styles (here we use style to refer to color choices) from reference images to grayscale inputs. AdaIN enables the decoupling of content and style in feature representations, allowing us to leverage the content information from grayscale images while infusing them with the stylistic characteristics extracted from reference color images. This style-guided colorization approach opens up new possibilities for artistic expression, allowing users to apply various painting styles, from impressionism to surrealism, to grayscale inputs.


# <a name="motivation"></a>2. Motivation
This project has his roots in our fascination for the colorisation task: an underconstraied task that is still under research and has big room for improvements. It is mathematically impossible to deduce from the single channel of grayscale images the 3 RGB or LAB channels. The only certainty is the color intensity pixel per pixel, but it is necessary to "guess" the colors.
Deep Convolutional Neural Networks are a powerful tool to tackle this task, since they can learn to understand the semantics of the image itself.
Moreover the AdaIN encoder, if used in training on the ground truth colored image itself, can be a powerful tool to encode the colour semantics. This addition alone, consisting in a regularisation and having yet no capacity to do style transfer, lead to a great improvement in colorisation quality.

The first idea of our group was less ambitious, namely to train a UNet for colorisation and finetuining it on different style images, like paintings of certain styles, and to present the user a limited choice of styles to colorise its uploaded grayscale iamge.
Our tutor, though, immediately introduced us to Adaptive Instance normalization, which, after a careful read of the related scientific paper ([Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868)), we thought to be facinating and we immediately took as primary inspiration.

There are three big differences between our approach and the forementioned paper:
1) Our network also colorizes images, rather than just transfering style.
2) Our Adaptive Instance Normalization is applied after every block of convolution, rather than just once on the bottlenck of the UNet.
3) The mean and standard deviations used in our AdaIN are not deterministic but are learned as well.

# <a name="related_work"></a>3. Related work
Our first struggle was definitely to train a Neural Network that was able to recreate first, and then to colorise images. We started from an encoder-decoder architecture that had a pretrained VGG19 encoder, discarding the classification head. The idea was to save a lot on computational cost. Unfortunately it wasn't enough: if we froze the encoder and trained the decoder alone, we were underfitted. If we unfroze the encoder, VGG19 was simply not worth it. Indeed, it was a big network, taking a lot of comptuational resourses, but at the same time it didn't offer skip connections or the flexibility to insert AdaIN layers, not while wanting to load the pretrained weights at least.

Anyways, this was a big part of our first two weeks of work. The paper we were getting inspiration from was: [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868), together with its unofficial implementation from the GitHub repository [GitHub: naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN).

Our first results in recreation and colorisation came when we started following the approach described in [Image Colorization using U-Net with Skip Connections and Fusion Layer on Landscape Images](https://arxiv.org/abs/2205.12867). It was by manually recreating layer by layer their UNet that we overcame our troubles.

The next step, namely the introduction of AdaIN layers for regularisation, was discussed in a meeting with our tutor and had no direct backup from scientific papers. The Adaptive Instance Normalization is done like in [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868), but with a big difference: the parameters of mean and standard deviation are learned every time from simple linear layers that act as interfaces between the latent space of the AdaIN encoder and the feature maps where normalization takes place. If you refer to the big architecture map in the beginning of this documentation, we are talking about the blue pins.
This idea of learning the AdaIN parameters comes from NVIDIA's 2019's [StyleGAN paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf).


# <a name="approach"></a>4. Approach
## <a name="architecture"></a>4.1 The Architecture
<span style="color:red"> 	Change it to final one when we are done </span>
<p align="left">
  <img src="./images/architecture_map.png" width="2000" />
</p>
Let us explain our architecture step by step.

#### The baseline UNet
The main core of the NN is the central UNet. UNet is still state of the art at this day and it's particularly effective for segmenting images. This knowledge is exactly what we needed for our colorization task.
The encoder consists in convolutions with 3x3 filters, always followed by ReLu and batch normalization, which are indicated in the picture by the orange slices on the right side of the convolution blocks.
After every set of convolutions a max pooling layer halves the feature maps dimensions.
The bottleneck maintains a reasonable size: 1/16 of the original size. As we work with 128x128 images due to computational cost limitations, this means that the bottleneck feature maps have size 8x8.

The decoder doesn't exactly mirror the encoder, but present the same jumps in number of chanels and dimension of the feature maps. Upsampling is done through transpose convolution. Relu and batch normalization are applied also in the decoder as can be seen by the orange slices.

The most notable featue of the UNet are the skip connections. We have five. Every time we concatenate the respective feature maps in the encoder to the newly upscaled featuremaps of the decoder. Then we fuse them together with 1x1 convolutions.

This network alone was tested in a preliminary phase and resulted to be able to perform both the recreation of images and colorisation, although with notable overfitting problems.
Below you can see:
1) Loss curves for only UNet tasked with colorization
2) Colorization performance on a training image
3) Colorization performance on a test image (never seen before)
<p align="left">
  <img src="./images/colorization_trsize2000_valsize200_loss_plot.png" width="500" />
  <table align="right">
  <tr>
    <th>Train dataset size</th>
    <td>2000</td>
  </tr>
  <tr>
    <th>Validation dataset size</th>
    <td>200</td>
  </tr>
  <tr>
    <th>Batch size</th>
    <td>16</td>
  </tr>
  <tr>
    <th>Colorspace</th>
    <td>RGB</td>
  </tr>
  <tr>
    <th>Optimizer</th>
    <td>Adam w. Weight decay</td>
  </tr>
  <tr>
    <th>Notes:</th>
    <td>Dropout used as regul.</td>
  </tr>
</table>
  <img src="./images/colorization_training_bird.png" width="500" />
  <img src="./images/colorization_validation_elephants.png" width="500" />
</p>


The network expects input in the form of $[C,H,W]$, where the number of channels $C$ can be 2 for the LAB colorspace and 3 for the RGB colorspace. We chose to train our model using the RGB color space because the LAB tranformation introduces a number of artefacts from numerical instability of the transformation which yields poorer results.

#### The addition of Adaptive Instance Normalization
<span style="color:red"> 	Insert description of adaIN layers (formula) </span>

The AdaIN encoder runs only one per epoch, embedding the style image in a compressed latent space. After every block of convolutions, followed by ReLu activations and batch normalization, the feature maps get normalized with the mean and std of the adain interface. Since in the different stages of the UNet the feature maps have different channel sizes, a dense linear layer is put in between the AdaIN latent space and the feature maps.



## <a name="architecture"></a>4.2 Training & Dataset

#### Implicit training
During training we pass as style image the colour original image itself.
Then, we train on colorizing the grayscale image, computing the loss of the recreated image against the original colour image. The loss is a combination of MSE pixel by pixel and of <span style="color:red"> 	perceptual similarity (?) </span>.
During backpropagation the various AdaIN interfaces and the encoder get implicitly trained.

In the final version of the model, there is no training against different style images. This approach was tried and will be discussed futher ahead. It needs the definition of a new style loss and good balancing with the colorization loss. Anyways, this should not be necessary: so long as scarcity is induced in the baseline UNet, it should resort in relying on the encoded data of the style image, provided by AdaIN interface. Thus, when passing a new style image at inference time, this implicit training should be enough to transfer the style.

#### Our dataset


We decided to use the [imagenet-1k](https://huggingface.co/datasets/imagenet-1k) dataset for our training. 
As style images to test at inference phase, we sample from the [wikiart](https://huggingface.co/datasets/huggan/wikiart) dataset.
<!-- For our style colorization and later fine-tuning we use the [wikiart](https://huggingface.co/datasets/huggan/wikiart) dataset. We chose a training size of 5000 samples and train for 100 epochs using an Adam optimizer with a learning rate of $1e^{-3}$. We further fine tune on ... artistic images. -->

#### Training analysis
<span style="color:red"> 	insert graph loses from latest model here </span>

## <a name="architecture"></a>4.2 Experiments and hyperparameters

<span style="color:red"> 	I think we need to rethink this section. Maybe merge it with the previous? I would rather talk as "experiments" about the vaious different approaches we did. </span>

Since we didn't have available computing resources, we used a personal laptop with NVIDIA GeForce RTX 3050TI, 4096MiB VRAM and a M1 MacBook Air (16Gb RAM) for training. This limit our ability in terms of resolution of the input images. Training with patches of size $256 \times 256$ yielded poor results even after prologed training, but using patches of size $128 \times 128$ yields decent results which generalize.

We experiment with two types of losses: MSE against the original image and [Learned Perceptual Similarty](https://github.com/richzhang/PerceptualSimilarity), as a way to encourage the network to learn semanting colorization information about the image instead of faithfully reproducting the colors of the original image. We use a weighted sum of both losses for our training.

When using only LPIPS for our loss function, our output images present artefacts. We would need to further investigate the causes of this.

<p align="left">
  <img src="./Unet_adain/results_perceptual_only/dog_no_style.png" width="2000" />
  <img src="./Unet_adain/results_perceptual_only/magazine_no_style.png" width="2000" />
</p>
 
When using only MSE for our loss function we get better results, we obtain more meaningful results, but with a considerable greater amount of training. Thus, we settle on a weighed mean of the two which yields good results.

** TODO: insert here our best results **
<p align="left">
  <img src="./Unet_adain/results_combined_loss/frog_no_style.png" width="2000" />
  <img src="./Unet_adain/results_combined_loss/dog_no_style.png" width="2000" />
</p>


There are also images for which colorization fails completly and outputs noise:

<p align="left">
  <img src="./Unet_adain/results_combined_loss/leopard_no_style.png" width="2000" />
</p>


** TODO: insert results from style transfer ** 

# <a name="troubles"></a>5. Troubles, experiments, reflections and evolution of the project
In this section we discuss some experiments, like some alternative approaches to training, that didn't make it into the final cut, but were essential for us in order to understand the nature of the problem and of the architecture. This is a reorganized history of the thought process that led us to the final prothotype.

#### The problem that originated rethinking
In very simple words, AdaIN was neglected by the baseline UNet. After few iterations the latent space of the AdaIN encoder was all put to zero. A sign that the network was suppressing it, in favour of pure reconsruction.
The consequence is that at inference phase, when passing different style images, no difference would be observed.

In the following picture you can see the histograms of the values inside the latent space for 10 random test images, sampled from the [imagenet-1k](https://huggingface.co/datasets/imagenet-1k) dataset.
<table align="left">
  <tr>
    <th>Train dataset size</th>
    <td>50000</td>
  </tr>
  <tr>
    <th>Validation dataset size</th>
    <td>500</td>
  </tr>
  <tr>
    <th>Batch size</th>
    <td>16</td>
  </tr>
  <tr>
    <th>Epochs trained</th>
    <td>18</td>
  </tr>
  <tr>
    <th>Colorspace</th>
    <td>RGB</td>
  </tr>
  <tr>
    <th><span style="color:orange"> Latent space dimension</span></th>
    <td><span style="color:orange"> 128</span></td>
  </tr>
  
</table>
<p align="top">
  <img src="./images/LatentSpace-zeros.jpeg" width="1000" />
</p>

#### First attempted solution: training with different pairs of (context + style) images
To achieve this, we build a new dataloader that sampled pairs of context and style iamges and we fed them both to the network. The forward pass is pretty intuitive. The backward pass needed some more careful thinking though.
Our first idea was to introduce a new loss, thus distinguishing between colorization and style loss.
The colorization loss was exactly the same as before, a linear combination of MSE and lpips between recreated image and original colur image.
The style loss was obtained in the following way:
1) We stored the latent space of the AdaIN encoder, namely the embedding of the style image
2) After the forward pass, without computing the gradient, we passed the recreated image through the AdaIN encoder, thus obtaining the embedding of the recreated image
3) We computed MSE between the two latent spaces

The final loss was a weighted linear combination of the two losses.

We hoped to manually force the network to learn style transfer, but we were unfortunately stopped by a very basic inconvenience: the balancing of the two losses. Indeed, the colorization loss was of the order of magnitude of $10^{-2}$ or $10^{-3}$, while the style loss, which we rember was computed on the embdeeings obtained throught the same encoder, was much lower, woth values oscillating between $10^{-5}$ and $10^{-8}$. We realised that without additional learning techniques, we would have never found an optimal way to balance these two losses. Here is an example:

<table align="left">
  <tr>
    <th>Train dataset size</th>
    <td>5000</td>
  </tr>
  <tr>
    <th>Validation dataset size</th>
    <td>500</td>
  </tr>
  <tr>
    <th>Batch size</th>
    <td>12</td>
  </tr>
  <tr>
    <th><span style="color:orange"> combined_loss = α•color_loss + (1-α)•style_loss</span></th>
    <td><span style="color:orange"> α = 0.5</span></td>
  </tr>

<!-- Leave this empty table! Needed for alignment -->
<table align="top">
  </tr>
<!-- Leave this empty table! Needed for alignment -->

<p align="left">
  <img src="./images/combined_loss_failed_attempt.png" width="600" />
</p>



#### Second attempted: using two different optimizers
The dual-opt model was exactly what it seems from the name: at every iteration we executed two forward passes and two backward with two different optimizers and loss functions: first the colorization loss and then the style loss. More importance was given to the colorization by passing the optimizer a learning rate 10 times higher than the one of the style optimizer.

We soon experienced bad results: 
- ⁠The loss of the style optimizer, although small, wasn't converging at all. Not in the first 15 epochs at least.
- ⁠⁠The loss of the colorization optimizer was better, but polluted by the constant meaningless changes of the style optimizer.

Soon we would realise, by investigating the values of the latent space of the AdaIN encoder, that it was cast to zero by the backpropagation of the colorization optimizer, cutting out all its power to represent the style input. The style optimizer had little of sense to work on, leading just to worse and worse results.


#### Reflections that lead to the final model
The dual-opt approach was not viable in the brutal state we described and tested. One way to refine it would have been to associate with each optimizer only certain parameters, especially allowing the colorization optimizer only to backpropage on the baseline UNet.

The combined-loss approach could have been further developed with learning techniques to balance the two losses.

But actually, the takeaway from this experiments was learning how the training proceeded, how the weights were cast to zero by the colorization loss and also how we were underestimating the capabilities of the AdaIN encoder.
The whole point of leaning the adain parameters was intended for implicit training, for letting it learn the true colour semantics of the style image, withouth forcing the network exterally with different couples of content and style-images.

After a second meeting with our tutor, we came to realise that a much simpler solution could be taken, that would give justice to the learnable AdaIN original concept: we had to introduce scarsity in the baseline UNet, so that it would look into the style representation, instead of trying to recreate itself perfectly and cast all the AdaIN parameters to 0.


#### Last changes and the final model
Therefore, we rewrote the UNet to be much lighter, by reducing the number of channels to 1/4 throughout the whole network and all skip connections and AdaIN pins respectively.
One more change was needed before finally obtaining some positive news: reducing the latent space of the AdaIN encoder from size 128 to size 32.
Finally, these were the results we got:

<p align="left">
  <img src="./images/elephants_latent32.jpeg" width="800" />
</p>
In short: bad colorization capabilities still, with even some artefacts, but for the first time the latent space of the AdaIN encoder was training well and had reasonable values. You can see this in the next image, which represents histograms of the values of the embeddings of four different test images.

<p align="left">
  <img src="./images/histograms_latent32.jpeg" width="800" />
</p>


# <a name="conclusion"></a>6. Conclusion

In conclusion, we have presented a novel UNet architecture enhanced with Adaptive Instance Normalization (AdaIN) layers for grayscale image colorization in diverse artistic styles. Although it's more a draft than a finished product, we are confindent that with better resources and some more attention this could grow into a good model.
We have worked really hard, moving in unkown territory, guided by our thought process and our tireless will to test our ideas with experiments. Most of them failed, but that's how we learned how to progress.
We are really sorry for the final state of the product, we resally cared about it, as we hope our long commitment shows.
One thing is certain, under the didactic perspective, we have learned a lot from this project, and we are definitely satisfied with the choice of this topic.
