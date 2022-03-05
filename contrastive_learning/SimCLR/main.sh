# Set `model' to `orig', `RC', `LBE', `IP' and `MIB' for different models.
# Set `dataset' for different pre-training datasets.
# Set `testset' for different transfer datasets.


# Training
## SimCLR
python main.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 128 --dataset CIFAR10
python main.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 128 --dataset STL-10
python main.py --model orig --resnet resnet50 --batch_size 1024 --epochs 200 --optimizer LARS --projection_dim 128 --dataset ImageNet
## SimCLR+RC
python main.py --model RC --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 128 --lamb 1. --dataset CIFAR10
python main.py --model RC --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 128 --lamb 1. --dataset STL-10
python main.py --model RC --resnet resnet50 --batch_size 1024 --epochs 200 --optimizer LARS --projection_dim 128 --lamb 0.1 --dataset ImageNet
## SimCLR+LBE
python main.py --model LBE --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 128 --lamb 1. --dataset CIFAR10
python main.py --model LBE --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 128 --lamb 1. --dataset STL-10
python main.py --model LBE --resnet resnet50 --batch_size 1024 --epochs 200 --optimizer LARS --projection_dim 128 --lamb 1. --dataset ImageNet
## SimCLR+IP
python main.py --model IP --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 128 --lamb 5e-4 --dataset CIFAR10
## SimCLR+MIB
python main.py --model MIB --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 128 --lamb 5e-4 --dataset CIFAR10


# Linear Evaluation
python eval_lr.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --projection_dim 128 --dataset CIFAR10 --testset cu_birds
python eval_lr.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --projection_dim 128 --dataset STL-10 --testset cu_birds
python eval_lr_imagenet.py --model orig --resnet resnet50 --batch_size 1024 --epochs 200 --projection_dim 128 --dataset ImageNet --testset ImageNet --logistic_batch_size 256
python eval_lr_imagenet.py --model orig --resnet resnet50 --batch_size 1024 --epochs 200 --projection_dim 128 --dataset ImageNet --testset cu_birds --logistic_batch_size 128