
## Object detection and instance segmentation

The `train_net.py` script reproduces the object detection and instance segmentation experiments.

### Instruction

1. Install [detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md).

2. Convert a pre-trained model to detectron2's format:
   ```
   python convert.py pretrain/Pretrain_Name.pt pretrain/Pretrain_Name.pkl
   ```

3. Put datasets (Pascal VOC and COCO) under "./datasets" directory,
   following the [directory structure](https://github.com/facebookresearch/detectron2/tree/master/datasets)
	 requried by detectron2.

4. Run training:
   ```
   python train_net.py --config-file configs/pascal_voc_R50_C4.yaml --num-gpus 8 MODEL.WEIGHTS pretrain/Pretrain_Name.pkl OUTPUT_DIR output/Exp_Name
   python train_net.py --config-file configs/coco_R50_C4_2x.yaml --num-gpus 8 MODEL.WEIGHTS pretrain/Pretrain_Name.pkl OUTPUT_DIR output/Exp_Name
   ```