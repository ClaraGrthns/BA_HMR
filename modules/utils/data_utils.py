import os
import os.path as osp
import pickle as pkl
import zarr


def get_relevant_keypoints(keypoints, threshold=0.3):
    return [(x,y) for (x,y,relevance) in (keypoints).transpose(1,0) if relevance>threshold]


def get_ids_imgspaths_seq(root_path:str="../3DPW",
                          out_dir:str="./data/ids_imgpaths",
                          split:str='train',
                          num_required_keypoints:int=8,
                          load_from_pkl=None,
                          store_as_pkl=False):
    if load_from_pkl is not None:
        with open(load_from_pkl , "rb") as fp:
            ids_imgpaths_seq = pkl.load(fp)
    else: 
        sequence_path = osp.join(root_path, 'sequenceFiles', split)
        seq_names = [seq_name.split('.pkl')[0] for seq_name in sorted(os.listdir(sequence_path))]
        num_required_keypoints = num_required_keypoints
        image_paths = []
        person_ids = []
        sequences = {}
        for seq_name in seq_names:
            ## loop through all sequences and filter out those where kps are missing
            img_dir = osp.join(root_path, 'imageFiles', seq_name)
            seq_file_name = os.path.join(sequence_path, f'{seq_name}.pkl')
            with open(seq_file_name, 'rb') as f:
                seq = pkl.load(f, encoding='latin1')
            sequences[seq_name] = seq
            num_people = len(seq['poses'])
            for img_idx, img_name in enumerate(sorted(os.listdir(img_dir))):
                ## for each person: save img_path 
                image_path = osp.join(img_dir,img_name)
                for person_id in range(num_people):
                    pose2d = seq['poses2d'][person_id][img_idx]
                    relevant_poses2d = get_relevant_keypoints(pose2d)
                    if len(relevant_poses2d) >= num_required_keypoints:
                        image_paths.append(image_path)
                        person_ids.append(person_id)
        ids_imgpaths_seq = {'person_ids': person_ids, 'image_paths': image_paths, 'sequences': sequences}
        if store_as_pkl:
            with open(osp.join(out_dir, f'ids_paths_seq_{split}_min{num_required_keypoints}_kps_ids_img_list.pickle'), 'wb') as fp:
                pkl.dump(ids_imgpaths_seq, fp)
    return ids_imgpaths_seq

class AverageMeter(object):
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count