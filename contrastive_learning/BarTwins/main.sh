# Set `model' to `orig', `RC' and `LBE' for different models.
# Set `dataset' for different pre-training datasets.
# Set `testset' for different transfer datasets.


# Training
## BarTwins
python main.py --model orig --resnet resnet18 --batch_size 128 --epochs 200 --optimizer Adam --projection_dim 1024 --dataset CIFAR10
python main.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 1024 --dataset STL-10
## BarTwins+RC
python main.py --model RC --resnet resnet18 --batch_size 128 --epochs 200 --optimizer Adam --projection_dim 1024 --lamb 1. --dataset CIFAR10
python main.py --model RC --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 1024 --lamb 1. --dataset STL-10
## BarTwins+LBE
python main.py --model LBE --resnet resnet18 --batch_size 128 --epochs 200 --optimizer Adam --projection_dim 1024 --lamb 1. --dataset CIFAR10
python main.py --model LBE --resnet resnet18 --batch_size 256 --epochs 200 --optimizer Adam --projection_dim 1024 --lamb 1. --dataset STL-10



# Linear Evaluation
python eval_lr.py --model orig --resnet resnet18 --batch_size 128 --epochs 200 --projection_dim 1024 --dataset CIFAR10 --testset cu_birds
python eval_lr.py --model orig --resnet resnet18 --batch_size 256 --epochs 200 --projection_dim 1024 --dataset STL-10 --testset cu_birds