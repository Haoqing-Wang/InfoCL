# Set `model' to `orig', `RC' and `LBE' for different models.
# Set `dataset' for different pre-training datasets.
# Set `testset' for different transfer datasets.


# Training
## Supervised
python main.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --dataset CIFAR10
python main.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --dataset CIFAR100
## Supervised+RC
python main.py --model RC --resnet resnet18 --batch_size 256 --epochs 200 --lamb 0.001 --dataset CIFAR10
python main.py --model RC --resnet resnet18 --batch_size 256 --epochs 200 --lamb 0.01 --dataset CIFAR100
## Supervised+LBE
python main.py --model LBE --resnet resnet18 --batch_size 256 --epochs 200 --lamb 0.1 --dataset CIFAR10
python main.py --model LBE --resnet resnet18 --batch_size 256 --epochs 200 --lamb 1. --dataset CIFAR100


# Linear Evaluation
python eval_lr.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --dataset CIFAR10 --testset cu_birds
python eval_lr.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --dataset CIFAR100 --testset cu_birds