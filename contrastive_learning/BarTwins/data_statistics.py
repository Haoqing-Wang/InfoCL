def get_data_mean_and_stdev(dataset):
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        mean = [0.5, 0.5, 0.5]
        std  = [0.5, 0.5, 0.5]
    elif dataset == 'STL-10':
        mean = [0.491, 0.482, 0.447]
        std  = [0.247, 0.244, 0.262]
    elif dataset == 'ImageNet':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif dataset == 'aircraft':
        mean = [0.486, 0.507, 0.525]
        std  = [0.266, 0.260, 0.276]
    elif dataset == 'cu_birds':
        mean = [0.483, 0.491, 0.424] 
        std  = [0.228, 0.224, 0.259]
    elif dataset == 'dtd':
        mean = [0.533, 0.474, 0.426]
        std  = [0.261, 0.250, 0.259]
    elif dataset == 'fashionmnist':
        mean = [0.348, 0.348, 0.348] 
        std  = [0.347, 0.347, 0.347]
    elif dataset == 'mnist':
        mean = [0.170, 0.170, 0.170]
        std  = [0.320, 0.320, 0.320]
    elif dataset == 'traffic_sign':
        mean = [0.335, 0.291, 0.295]
        std  = [0.267, 0.249, 0.251]
    elif dataset == 'vgg_flower':
        mean = [0.518, 0.410, 0.329]
        std  = [0.296, 0.249, 0.285]
    else:
        raise Exception('Dataset %s not supported.'%dataset)
    return mean, std

def get_data_nclass(dataset):
    if dataset == 'CIFAR10':
        nclass = 10
    elif dataset == 'CIFAR100':
        nclass = 100
    elif dataset == 'STL-10':
        nclass = 10
    elif dataset == 'ImageNet':
        nclass = 1000
    elif dataset == 'aircraft':
        nclass = 102
    elif dataset == 'cu_birds':
        nclass = 200
    elif dataset == 'dtd':
        nclass = 47
    elif dataset == 'fashionmnist':
        nclass = 10
    elif dataset == 'mnist':
        nclass = 10
    elif dataset == 'traffic_sign':
        nclass = 43
    elif dataset == 'vgg_flower':
        nclass = 102
    else:
        raise Exception('Dataset %s not supported.'%dataset)
    return nclass