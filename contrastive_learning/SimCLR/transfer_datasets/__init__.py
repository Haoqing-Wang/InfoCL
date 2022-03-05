from .aircraft import Aircraft
from .cu_birds import CUBirds
from .dtd import DTD
from .fashionmnist import FashionMNIST
from .mnist import MNIST
from .traffic_sign import TrafficSign
from .vgg_flower import VGGFlower

DATASET = {
    'aircraft': Aircraft,
    'cu_birds': CUBirds,
    'dtd': DTD,
    'fashionmnist': FashionMNIST,
    'mnist': MNIST,
    'traffic_sign': TrafficSign,
    'vgg_flower': VGGFlower}