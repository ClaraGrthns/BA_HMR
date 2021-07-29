from yacs.config import CfgNode as CN
DEFAULT_YAML_PATH = "configs/experiments/default_config.yaml"
# Configuration variables
cfg = CN()
cfg.set_new_allowed(is_new_allowed=True)


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    cfg.merge_from_file(DEFAULT_YAML_PATH)
    return cfg.clone()


def update_cfg(cfg_filepath):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_filepath)
    return cfg.clone()


