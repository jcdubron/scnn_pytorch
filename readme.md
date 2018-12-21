# scnn_pytorch

This is a Pytorch implementation of SCNN [ [X. Pan, J. Shi, P. Luo, X. Wang, and X. Tang, “Spatial As Deep: Spatial CNN for Traffic Scene Understanding”, AAAI2018](https://arxiv.org/abs/1712.06080) ].

A Torch7 implementation is published by the author [@XingangPan](https://github.com/XingangPan/SCNN).

## Implementation

Before running the script *test.sh*, you have to prepare the dataset provided by @XingangPan. We have added *tools/t7_to_pth.py* to convert the model (.t7) trained by the author into .pth model compatible with this project. Although the pre-trained model has achieved almost the same result as the paper mentions, self-trained model works not that well in most categories, about one percent in F1. The result except Crossroad is listed as below.

| Category     | pre-trained model | self-trained model |
| ------------ | ----------------- | ------------------ |
| normal       | 90.60             | 89.24              |
| crowded      | 69.67             | 69.37              |
| dazzle light | 58.43             | 61.74              |
| shadow       | 67.00             | 71.05              |
| no line      | 43.39             | 42.68              |
| arrow        | 84.11             | 84.00              |
| curve        | 64.50             | 63.33              |
| night        | 66.07             | 65.55              |



