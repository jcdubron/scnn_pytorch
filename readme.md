# scnn_pytorch

This is a Pytorch implementation of SCNN [ [X. Pan, J. Shi, P. Luo, X. Wang, and X. Tang, “Spatial As Deep: Spatial CNN for Traffic Scene Understanding”, AAAI2018](https://arxiv.org/abs/1712.06080) ].

A Torch7 implementation is published by the author [@XingangPan](https://github.com/XingangPan/SCNN).

## Implementation

Before running the script *test.sh*, you have to prepare the dataset provided by @XingangPan. We have added *tools/t7_to_pth.py* to convert the model (.t7) trained by the author into .pth model compatible with this project.

The comparison of the test F1 value provided by the author in paper, the test result we obtain from the pre-trained model and what this project can achieve in test dataset is listed as below.

| Category      | paper | pre-trained model | self-trained model |
| ------------- | ----- | ----------------- | ------------------ |
| normal        | 90.6  | 90.7135           | 90.2981            |
| crowded       | 69.7  | 70.1434           | 69.5892            |
| dazzle light  | 58.4  | 59.1103           | 61.7080            |
| shadow        | 66.9  | 70.3723           | 71.5793            |
| no line       | 43.4  | 44.1308           | 44.1749            |
| arrow         | 84.1  | 85.0582           | 84.5007            |
| curve         | 64.4  | 65.4757           | 61.5954            |
| night         | 66.1  | 66.7280           | 65.3114            |
| crossroad(fp) | 1990  | 2035              | 2240               |
| total         | 71.6  | 72.1233           | 71.5177            |

