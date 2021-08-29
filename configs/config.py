from yacs.config import CfgNode as CN
#DEFAULT_YAML_PATH = "/Users/clara/Desktop/MeineProjekte/HMR_3DWP/configs/experiments/default_config.yaml"
DEFAULT_YAML_PATH = "/home/grotehans/BA_HMR/configs/experiments/default_config.yaml"
# Configuration variables
cfg = CN(new_allowed=True)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg.merge_from_file(DEFAULT_YAML_PATH)
    return cfg.clone()


def update_cfg(cfg_filepath, opts):
    print('opts:', opts)
    cfg = get_cfg_defaults()
    if cfg_filepath is not None:
        cfg.merge_from_file(cfg_filepath)
    if opts is not None:
        cfg.merge_from_list(opts)
    return cfg.clone()


