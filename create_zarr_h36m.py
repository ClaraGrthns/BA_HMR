import os.path as osp
import argparse
from modules.utils.save_imgs import save_img_zarr_h36m


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Save training and validation imgs as zarr")
    parser.add_argument('--annot_dir', type=str, help='path to annotations of Human 3.6', default='/home/grotehans/H36M/annotations')
    parser.add_argument('--img_dir', type=str, help='path to images of Human 3.6', default='/home/grotehans/H36M/images')
    parser.add_argument('--subject_list', type=str, help='1to11, 1to8 or 9to11', default='1to11')
    parser.add_argument('--out_dir', type=str, help='path to directory of zarr', default='/home/grotehans/H36M')
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
    else:
        subject_list = [9, 11]
    
    save_img_zarr_h36m(
        annot_dir=args.annot_dir,
        img_dir=args.img_dir,
        zarr_path=osp.join(args.out_dir, f'imgs_h36m_{args.encoder}_{args.fitting_thr}_thr_{args.subject_list}_subj.zarr'),
        img_size=img_size,
        subject_list=subject_list,
        fitting_thr=args.fitting_thr,
        load_from_pkl=osp.join(args.annot_dir, f'datalist_h36m_thr25_1to11subj.pickle'),
    )
    
    print('zarr is done!')
   