# yolov3-channel-pruning
# Update:
补充prune_utils.py. 性能待测。

# requirements:

+ numpy>=1.13
+ tensorboardX
```
pip install tensorboardX
```
or
```
git clone https://github.com/lanpa/tensorboardX && cd tensorboardX && python setup.py install
```
+ albumentations
```
pip install albumentations
```
or
```
conda install -c conda-forge imgaug
conda install albumentations -c albumentations
```
+ terminaltables
```
pip install terminaltables
```
+ tqdm
+ torch
+ random 
+ matplotlib
+ .......

# Run
```python train.py --model_def config/yolov3.cfg```

```python train.py --model_def config/yolov3.cfg -sr```

```python test_prune.py```

```python train.py --model_def config/prune_yolov3.cfg -pre checkpoints/prune_yolov3_ckpt.pth```

![](https://github.com/violet17/yolov3-channel-pruning/blob/master/bn_weights_hist.png?raw=true)

# Reference: 
[YOLOv3-model-pruning](https://github.com/Lam1360/YOLOv3-model-pruning)（感谢Lam1360给了很多帮助）

[yolov3-network-slimming](https://github.com/talebolano/yolov3-network-slimming)
           
[PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
