import os.path as osp
from modules.utils.zarr_imgs import save_img_zarr

if __name__ == '__main__':

    # TODO: get these values from command line arguments
    root_dir = '../3DPW'
    out_dir = 'data'
    num_required_keypoints = 8

    save_img_zarr(
        root_path=root_dir,
        zarr_path=osp.join(out_dir, 'imgs_3dpw_train.zarr'),
        num_required_keypoints=num_required_keypoints,
        split='train'
    )
    save_img_zarr(
        root_path=root_dir,
        zarr_path=osp.join(out_dir, 'imgs_3dpw_valid.zarr'),
        num_required_keypoints=num_required_keypoints,
        split='valid'
    )