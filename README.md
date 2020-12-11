# CSPPeleeNet.pytorch

Unofficial PyTorch implementation of CSPPeleeNet[1]


|  model | Params(M) | MACs(M) | top1 | top5 |
| ---- | ---- | ---- | ---- | ---- |
| PeleeNet |  2.802  | 513.876 |-|-|
| CSPPeleeNet (gamma=0.5) | 2.856 |569.063 |73.07|91.11|
| CSPPeleeNetLight (gamma=0.5)|2.431|465.704|72.05|90.56|

Usage
```
from peleenet import PeleeNet

# peleenet
net = PeleeNet(partial_ratio=1.0)

# csppeleenet (gamma=0.5)
net = PeleeNet(partial_ratio=0.5)
```

# References
[1] [CSPNet: A New Backbone that can Enhance Learning Capability of CNN](https://arxiv.org/abs/1911.11929)  
[2] [Pelee: A Real-Time Object Detection System on Mobile Devices](https://arxiv.org/abs/1804.06882)
