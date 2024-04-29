# Artistic colorisation with Adaptive Istance Normalization

### 🧑🏻‍🎓 Team Members

| Name and surname    |  Matric. Nr. | GitHub username  |   e-mail address   |
|:--------------------|:-------------|:-----------------|:-------------------|
| Cristi Andrei Prioteasa | 4740844| PrioteasaAndrei | cristi.prioteasa@stud.uni-heidelberg.de|
| Matteo Malvestiti | 4731243| Matteo-Malve | matteo.malvestiti@stud.uni-heidelberg.de|
| Jan Smoleń | 4734263| smolenj | wm315@stud.uni-heidelberg.de|


### Advisor

Denis Zavadski: denis.zavadski@iwr.uni-heidelberg.de

***

### Setup

```bash
conda env create -f environment.yml --name colorization
conda activate colorization
```

### Three ways to test our model:
#### 1. Use our GUI
Use the graphical interface to upload style and contet images. The images will be automatically resized to $128px\times128px$. The model takes only 400 MB, so it should work on most low-end devices. This is what an hypothetical final product would look like.
```bash
# Be sure to be in the custom environment created in Setup
cd demo/
streamlit run demo-app.py -- --model_path=models/RGB_Latent32_best_model.pth.tar
```

#### 2. Test the model with demo.ipynb
We also provide a Jupyter Notebook that will easily allow the user to generate plots such as those can be found in our [Results section](#results). The way to go to have a better comparison between multiple outputs.

#### 3. Inspecting the training and loading our datasets in main.ipynb
`implicit-UNet-AdaIN/main.ipynb` is the jupyter notebook we used to import the two datasets and to set up and train the network.
There is a lot of stuff here, but you find easy access to the datasets and let's you experiment with the training procedure.

NOTE: a Huggingface token is needed to run the training / download the dataset (this is not necessary to run the forward pass of the network). Create a .env file in the main directory and add `HUGGING_FACE_TOKEN='YOUR TOKEN'`.

### Project Structure
```
├── demo/
│   ├── checkpoints/   <--------- model checkpoint
│   ├── images/    <-------------- some content and style images for testing
│   ├── demo-app.py
│   └── demo.ipynb
└── implicit-UNet-AdaIN/
    ├── implicit_scarsity_experiments/    <--- plots and tables of loss values
    ├── samples/    <------------------------- testing samples
    ├── samples_organized/    <--------------- testing samples in a more human readable format
    └── main.ipynb

```

## Table of contents

1. [Introduction](#introduction)
2. [Results](#appendix)

For a detailed insight on the project, please refer to [Report.pdf](./Report.pdf).

# Introduction

Colorization of grayscale images has been an area of significant interest in computer vision, with applications ranging from photo restoration to artistic expression. We propose an approach to colorize grayscale images in various artistic styles using a U-Net architecture enhanced with Adaptive Instance Normalization (AdaIN) layers. U-Net, known for its effectiveness in semantic segmentation tasks, provides an ideal framework for our colorization task due to its ability to capture spatial dependencies while preserving fine details. By incorporating AdaIN layers into the U-Net architecture, we introduce the capability to adaptively transfer artistic styles (here we use style to refer to color choices) from reference images to grayscale inputs. AdaIN enables the decoupling of content and style in feature representations, allowing us to leverage the content information from grayscale images while infusing them with the stylistic characteristics extracted from reference color images. This style-guided colorization approach opens up new possibilities for artistic expression, allowing users to apply various painting styles, from impressionism to surrealism, to grayscale inputs.

# Results
Here are some examples for some random test images and random paintings. The first row always shows the colourization obtained by passing the groud truth image to the network, like in training phase. That is not a realistic scenario, since in practice the coloured image is not available. On the other hand, it provides a good baseline for comparinson.

We encourage then to experimnt yourself with any image of your choice from the [demo-app](./demo/demo-app.py) or with images from the [imagenet-1k](https://huggingface.co/datasets/imagenet-1k) and [wikiart](https://huggingface.co/datasets/huggan/wikiart) datasets with the [demo.ipynb](./demo/demo.ipynb).
<h3 align='center'> Dog and Sheeps </p>
<p align="center">
  <img src="./implicit-UNet-AdaIN/samples_organized/image_0_original.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_0_style_0.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_0_style_1.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_0_style_2.png" width="800" />
</p>

<h3 align='center'> Exotic bird </p>
<p align="center">
  <img src="./implicit-UNet-AdaIN/samples_organized/image_1_original.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_1_style_0.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_1_style_1.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_1_style_2.png" width="800" />
</p>

<h3 align='center'> Small dog in grass </p>
<p align="center">
  <img src="./implicit-UNet-AdaIN/samples_organized/image_3_original.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_3_style_0.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_3_style_1.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_3_style_2.png" width="800" />
</p>

<h3 align='center'> Nature landscape </p>
<p align="center">
  <img src="./implicit-UNet-AdaIN/samples_organized/image_5_original.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_5_style_0.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_5_style_1.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_5_style_2.png" width="800" />
</p>

<h3 align='center'> Man </p>
<p align="center">
  <img src="./implicit-UNet-AdaIN/samples_organized/image_13_original.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_13_style_0.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_13_style_1.png" width="800" />
  <img src="./implicit-UNet-AdaIN/samples_organized/image_13_style_2.png" width="800" />
</p>
