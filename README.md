# Rethinking Minimal Sufficient Representation in Contrastive Learning
PyTorch implementation of
<br>
[**Rethinking Minimal Sufficient Representation in Contrastive Learning**](http://arxiv.org/abs/2104.14385)
<br>
Haoqing Wang, [Xun Guo](https://www.microsoft.com/en-us/research/people/xunguo/), [Zhi-hong Deng](http://www.cis.pku.edu.cn/jzyg/szdw/dzh.htm), [Yan Lu](https://www.microsoft.com/en-us/research/people/yanlu/)

CVPR 2022

## Abstract

Contrastive learning between different views of the data achieves outstanding success in the field of self-supervised representation learning and the learned representations are useful in broad downstream tasks. Since all supervision information for one view comes from the other view, contrastive learning approximately obtains the minimal sufficient representation which contains the shared information and eliminates the non-shared information between views. Considering the diversity of the downstream tasks, it cannot be guaranteed that all task-relevant information is shared between views. Therefore, we assume the non-shared task-relevant information cannot be ignored and theoretically prove that the minimal sufficient representation in contrastive learning is not sufficient for the downstream tasks, which causes performance degradation. This reveals a new problem that the contrastive learning models have the risk of over-fitting to the shared information between views. To alleviate this problem, we propose to increase the mutual information between the representation and input as regularization to approximately introduce more task-relevant information, since we cannot utilize any downstream task information during training. Extensive experiments verify the rationality of our analysis and the effectiveness of our method. It significantly improves the performance of several classic contrastive learning models in downstream tasks.

## Citation
If you use this code for your research, please cite our paper:
```
@inproceedings{wang2022rethinking,
  title={Rethinking Minimal Sufficient Representation in Contrastive Learning},
  author={Wang, Haoqing and Deng, Zhi-hong and Guo, Xun and Lu, Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={xx--xx},
  year={2022}
}
```

## Note
- This code is built upon the implementation from [moco](https://github.com/facebookresearch/moco) and [CLAE](https://github.com/chihhuiho/CLAE).
- The dataset, model, and code are for non-commercial research purposes only.