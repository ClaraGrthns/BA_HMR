import os.path as osp
import argparse
from modules.utils.zarr_imgs import save_img_zarr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save training and validation imgs as zarr")
    parser.add_argument('--root_path', type=str, help='path to dataset 3DPW', default='../3DPW')
    parser.add_argument('--out_dir', type=str, help='path to directory of zarr', default='data')
    parser.add_argument('--num_required_keypoints', type=int, help='minimum number of required keypoints', default=8)
    parser.add_argument('--encoder', type=str, help='Encoder Options: resnet or hrnet', default='resnet')
    args = parser.parse_args()

    if args.encoder == 'resnet':
        img_size = 224
    else:
        img_size = 256
    
    save_img_zarr(
        root_path=args.root_path,
        zarr_path=osp.join(args.out_dir, 'imgs_3dpw_train.zarr'),
        num_required_keypoints=args.num_required_keypoints,
        split='train',
        img_size=img_size,
    )
    
    save_img_zarr(
        root_path=args.root_path,
        zarr_path=osp.join(args.out_dir, 'imgs_3dpw_valid.zarr'),
        num_required_keypoints=args.num_required_keypoints,
        split='validation',
        img_size=img_size,
    )