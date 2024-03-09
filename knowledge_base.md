## Adaptive Instance Normalization

THe trained network needs only one image that represents the style to which the image will be transferred.

Github repo is here, but its in lua :( https://github.com/xunhuang1995/AdaIN-style

See 6.2 Training here: https://arxiv.org/pdf/1703.06868v2.pdf (original paper)

Check this out: https://github.com/jcjohnson/neural-style
CHeck this out: https://www.kaggle.com/code/davidcanorosillo/adain-style


In the original paper they use encoder (vgg not all layers) -> AdaIn -> decoder (vgg not all layers).

We need to do Unet encoder (which layers?) -> AdaIn -> Unet decoder (which layers?)