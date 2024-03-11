# Artistic colorization

A project by Alexander Nyamekye, Cristi Andrei Prioteasa, Matteo Malvestiti and Jan Smolen for the course "Computer vision: 3D reconstruction", held by Prof. Carsten Rother at Ruprecht Karls Universität Heidelberg in Winter Semester 2023.

Our group:
| Name and surname    |  Matric. Nr. | GitHub username  |   e-mail address   |
|:--------------------|:-------------|:-----------------|:-------------------|
| Alexander Nyamekye | <span style="color:red"> *(?)* </span>| <span style="color:red"> *(?)* </span> | alexander.nyamekye@stud.uni-heidelberg.de|
| Cristi Andrei Prioteasa | <span style="color:red"> *(?)* </span>| <span style="color:red"> *(?)* </span> | cristi.prioteasa@stud.uni-heidelberg.de|
| Matteo Malvestiti | 4731243| Matteo-Malve | matteo.malvestiti@stud.uni-heidelberg.de|
| Jan Smolen | <span style="color:red"> *(?)* </span>| <span style="color:red"> *(?)* </span> | wm315@stud.uni-heidelberg.de|

### Useful sources

Possible paintings dataset: 
- [keremberke/painting-style-classification](https://huggingface.co/datasets/keremberke/painting-style-classification?library=true) 
- [huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart)

Reference papers:
- [Colorful Image Colorization](https://arxiv.org/pdf/1603.08511v5.pdf) on Posing problem as a classification task
- [Image Colorization using U-Net with Skip Connections and Fusion Layer on Landscape Images](https://arxiv.org/abs/2205.12867) \
Note: their approach takes 60Gb in RAM for 64x64 images
- [Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization](https://arxiv.org/abs/1703.06868) ADAIN: The one given to us by the tutor.
- [Analysis of Different Losses for Deep Learning Image Colorization](https://arxiv.org/pdf/2204.02980.pdf) 

Reference Repos:
- [GitHub: xunhuang1995/AdaIN-style](https://github.com/xunhuang1995/AdaIN-style?tab=readme-ov-file) official implementation of the ADAIN paper \
Note: implementation is in lua, but we can use a wrapper to python, such as [lupa 2.0](https://pypi.org/project/lupa/) (repo: [scoder/lupa](https://github.com/scoder/lupa)) or [lunatic-python](https://github.com/bastibe/lunatic-python)
- [GitHub: naoto0804/pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) official pytorch implementation

### Report 

Latex template for report: http://cvpr2021.thecvf.com/sites/default/files/2020-09/cvpr2021AuthorKit_2.zip

Structure of typical article: 
- abstract
- introduction & related work
- method, experiments
- findings
- conclusion


