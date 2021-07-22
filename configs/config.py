import argparse
from yacs.config import CfgNode as CN

# Configuration variables
cfg = CN()

cfg.LOGGING = CN()
cfg.LOGGING.LOGDIR = "./logging"
cfg.LOGGING.LOG_STEPS = 200

cfg.DATASETS = CN()
cfg.DATASETS.ThreeDPW = '../3DPW'

cfg.TRAIN = CN()
cfg.TRAIN.BATCH_SIZE_TRN = 16
cfg.TRAIN.BATCH_SIZE_VAL = 16
cfg.TRAIN.NUM_EPOCHS = 5
cfg.TRAIN.LEARNING_RATE = 1.0e-4
cfg.TRAIN.NUM_REQUIRED_KPS = 8

cfg.MODEL = CN()
cfg.MODEL.DIM_Z = 128
cfg.MODEL.ENCODER = 'resnet50'

cfg.LOSS = CN()
cfg.LOSS.smpl = 0
cfg.LOSS.verts = 1.
cfg.LOSS.kp_2d = 0
cfg.LOSS.kp_3d = 0


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_filepath):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_filepath)
    return cfg.clone()

