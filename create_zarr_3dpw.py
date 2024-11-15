import os.path as osp
import argparse
from modules.utils.save_imgs import save_img_zarr_3dpw


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save training and validation imgs as zarr")
    parser.add_argument('--data_path', type=str, help='path to dataset 3DPW', default='/home/grotehans/3DPW')
    parser.add_argument('--out_dir', type=str, help='path to directory of zarr', default='/home/grotehans/BA_HMR/data')
    parser.add_argument('--num_required_keypoints', type=int, help='minimum number of required keypoints', default=8)
    parser.add_argument('--encoder', type=str, help='Encoder Options: resnet or hrnet', default='resnet')
    parser.add_argument('--padding', type=bool, help='True: Padded images', default=False)
    args = parser.parse_args()
    print(args)

    if args.encoder == 'resnet':
        img_size = 224
    else:
        img_size = 256
    
    save_img_zarr_3dpw(
        data_path=args.data_path,
        zarr_path=osp.join(args.out_dir, f'imgs_3dpw_{args.encoder}_{args.padding}_padding_train.zarr'),
        num_required_keypoints=args.num_required_keypoints,
        split='train',
        img_size=img_size,
        load_from_pkl=osp.join(args.out_dir, f'ids_imgpaths_seq/ids_paths_seq_train_min{args.num_required_keypoints}_kps.pickle'),
        padding=args.padding,
    )

    print('train zarr is done!')
   
    save_img_zarr_3dpw(
        data_path=args.data_path,
        zarr_path=osp.join(args.out_dir, f'imgs_3dpw_{args.encoder}_{args.padding}_padding_valid.zarr'),
        num_required_keypoints=args.num_required_keypoints,
        split='validation',
        img_size=img_size,
        load_from_pkl=osp.join(args.out_dir, f'ids_imgpaths_seq/ids_paths_seq_validation_min{args.num_required_keypoints}_kps.pickle'),
        padding=args.padding,
    )
    print('validation zarr is done!')