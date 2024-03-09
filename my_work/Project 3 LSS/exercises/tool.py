"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import numpy as np
import torch
import torchvision
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))


normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    """
    Sum pooling point cloud features in each voxel
    对落在同一个单元格的特征进行求和池化: 实际上这个函数并不是cumsum, 而是将重复rank位置的特征进行求和。
    @ x          : 铺平的context特征, shape为(n,c)
    @ geom_feats : 铺平的视锥点, shape为(n,4)
    @ ranks      : 每个点的rank值, shape为(n,)
    关于这个函数的理解详见: test_cumsum.py 中的演示
    Parameters
    ----------
    x : array_like, shape (N',C), N' refers to the number of points after filter
        features of point cloud.
    geom_feats : array_like, shape (N',4)
        voxel coordinates of points in BEV space, the last dimension represents batch_ix
    ranks: array_like, shape (N')
        The encoded value of the position index of B, X, Y, Z in voxel coordinates.
        Points with equal rank value are in the same batch and in the same grid. 

    Returns:
    --------
    x:  array_like, shape (N'',C), Voxel features after deduplication
    geom_feats: array_like, shape (N'',4), voxel coordinates after deduplication
    """
    # TODO: Write this function
    # 1. 求前缀和,cumsum: 对数组进行累加运算求和
    x = x.cumsum(0)
    # 2. 筛选出ranks中前后rank值不相等的位置
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    ## 因为之前索引已经进行了排序，所以这个就相当于错位比较
    kept[:-1] = (ranks[1:] != ranks[:-1])
    # 3. rank值相等的点只留下最后一个，即一个batch中的一个格子里只留最后一个点
    x, geom_feats = x[kept], geom_feats[kept]
    # 4. x后一个减前一个，还原到cumsum之前的x，此时的一个点是之前与其rank相等的点的feature的和，相当于把同一个格子的点特征进行了sum
    x = torch.cat((x[:1], x[1:] - x[:-1]))
    # 5. return
    return x, geom_feats

def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')

