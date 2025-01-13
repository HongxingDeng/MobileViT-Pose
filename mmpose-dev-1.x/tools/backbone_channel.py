from mmcls.models.backbones.mobilevit import MobileViT
from mmcls.models.backbones.mobilenet_v2 import MobileNetV2
from mmpose.models.backbones.hrformer import HRFormer
import torch
model = MobileViT(out_indices=(0,1,2,3,4,5,6,7))
input = torch.rand(1,3,192,256)
level_output = model(input)
for level_out in level_output:
    print(tuple(level_out.shape))