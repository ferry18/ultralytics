# PyTorch implementation of PP-LCNet

Reproduction of PP-LCNet architecture as described in [PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099.pdf) by C. Cui. T. Gao, S. Wei *et al* (2021) with the [PyTorch](pytorch.org) framework. 

The official design is implemented with [Paddle] framework, the detail [here](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.3/ppcls/arch/backbone/legendary_models/pp_lcnet.py)

## Models

| Architecture      | #Parameters | FLOPs | Top-1 Acc. (%) |
| ----------------- | :------------: | :------: | -------------------------- |
| PPLCNet_x0_25    | 1,522,960 | 18M |  |
| PPLCNet_x0_35    | 1,646,888 | 29M |  |
| PPLCNet_x0_5     | 1,881,864 | 47M |  |
| PPLCNet_x0_75    | 2,359,792 | 99M |  |
| PPLCNet_x1_0     | 2,955,816 | 161M |  |
| PPLCNet_x1_5     | 4,504,136 | 342M |  |
| PPLCNet_x2_0     | 6,526,824 | 590M |  |
| PPLCNet_x2_5     | 9,023,880 | 906M |  |