# Rethinking Minimal Sufficient Representation in Contrastive Learning
Main code for 
<br>
[**Rethinking Minimal Sufficient Representation in Contrastive Learning**](http://arxiv.org/abs/2104.14385)
<br>

Each folder provides the code and commands of its corresponding model.

## Prerequisites
- Python >= 3.6
- Pytorch >= 1.2.0 and torchvision (https://pytorch.org/)

## Datasets
All datasets are provided in `./datasets` folder. For CIFAR10, CIFAR100, STL-10, MNIST and FashionMNIST, the code automatically downloads these datasets. 

For other datasets, download them from

* **ImageNet-1k**: https://image-net.org

* **DTD**: https://www.robots.ox.ac.uk/~vgg/data/dtd/

* **CUBirds**: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

* **VGG Flower**: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/

* **Traffic Signs**: https://benchmark.ini.rub.de/gtsdb_dataset.html

* **Aircraft**: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/

and put them under their respective paths, e.g., 'datasets/DTD'.

## Training and linear evaluation
All training and linear evaluation commands are provided in `main.sh` in the folder corresponding to the model.