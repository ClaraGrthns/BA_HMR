import os.path as osp
import argparse
from modules.utils.save_imgs import save_img_zarr_h36m
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save training and validation imgs as zarr")
    parser.add_argument('--data_path', type=str, help='path to folder of Human 3.6', default='/home/grotehans/H36M/')
    parser.add_argument('--subject_list', type=str, help='1to11, 1to8 or 9to11', default='1to8')
    parser.add_argument('--out_dir', type=str, help='path to directory of zarr', default='/home/grotehans/H36M/')
    parser.add_argument('--fitting_thr', type=int, help='fitting threshhold in mm for h36m gt and joints from smpl-mesh', default=25)
    parser.add_argument('--encoder', type=str, help='Encoder Options: resnet or hrnet', default='resnet')
    args = parser.parse_args()
    print(args)

    if args.encoder == 'resnet':
        img_size = 224
    else:
        img_size = 256
    if args.subject_list == '1to11':
        subject_list = [1, 5, 6, 7, 8, 9, 11]
    elif args.subject_list == '1to8':
        subject_list = [1, 5, 6, 7, 8]
    elif args.subject_list == '9to11':
        subject_list = [9, 11]
    else:
        subject_list = [1]

    rand_id = random.randint(0, 1000000)

    save_img_zarr_h36m(
        data_path=args.data_path,
        zarr_path=osp.join(args.data_path, 'img_zarr', f'imgs_h36m_{args.encoder}_thr{args.fitting_thr}_{args.subject_list}subj_{rand_id}.zarr'),
        img_size=img_size,
        subject_list=subject_list,
        fitting_thr=args.fitting_thr,
        load_from_pkl=osp.join(args.data_path, f'datalist_h36m_thr{args.fitting_thr}_{args.subject_list}subj.pickle'),
    )
    
    print('zarr is done!')
   