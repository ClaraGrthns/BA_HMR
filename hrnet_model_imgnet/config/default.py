# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from yacs.config import CfgNode as CN


cfg = CN()

cfg.OUTPUT_DIR = ''
cfg.LOG_DIR = ''
cfg.DATA_DIR = ''


# common params for NETWORK
cfg.MODEL = CN()
cfg.MODEL.NAME = 'cls_hrnet'
cfg.MODEL.INIT_WEIGHTS = True
cfg.MODEL.PRETRAINED = ''
cfg.MODEL.NUM_JOINTS = 17
cfg.MODEL.NUM_CLASSES = 1000
cfg.MODEL.TAG_PER_JOINT = True
cfg.MODEL.TARGET_TYPE = 'gaussian'
cfg.MODEL.IMAGE_SIZE = [256, 256]  # width * height, ex: 192 * 256
cfg.MODEL.HEATMAP_SIZE = [64, 64]  # width * height, ex: 24 * 32
cfg.MODEL.SIGMA = 2
cfg.MODEL.EXTRA = CN(new_allowed=True)



def get_cfg_defaults():
    return cfg.clone()
    
def update_hrnet_cfg(cfg_filepath):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_filepath)
    return cfg.clone()


if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(cfg, file=f)

