# implicit-UNet-AdaIN
The name of this folder is a summary of the main feature of our final architecture by itself: the baseline is a UNet, enhanced by Adaptive Instance Normalization layers and the AdaIN encoder is trained implicitly.
With the latter we mean that no specific loss function is defined on that branch of the network, but rather only on the baseline UNet.
Training is always done by passing as style the ground truth of the images themselves.
The gradient gets backproagated through the AdaIN encoder and thus it learns to produce a meaningful latent representation of the colorspace, that the UNet can make a use for.

The pillar of the success of this tecnique was the choice to intentionally provoke scarcity in the UNet, so that it would rely on the AdaIN layers, and thus on the colour representation of the style image as much as possible.

The `main.ipynb` notebook is structured as follows:
- Imports
- Definition of the network
- Setting of constants and hyperparameters
- Import of the dataset for content images
- Loading of the checkpoints or training call
- Import of the dataset for style images
- Testing platform

The redundancy in the choice of declaring again the network in the notebook, despite having the same in `demo/model.py` is done on purpose, so to allow the user to easily access it and modify it for further experiments.