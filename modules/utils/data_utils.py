import os
import os.path as osp
import pickle as pkl
import zarr
import random 

#Source: https://stackoverflow.com/questions/3352737/how-to-randomly-partition-a-list-into-n-nearly-equal-parts
def rand_partition (list_in, n):
    rand_chunks = list_in.copy()
    random.shuffle(rand_chunks)
    return [rand_chunks[i::n] for i in range(n)]


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
            with open(osp.join(out_dir, f'ids_paths_seq_{split}_min{num_required_keypoints}_kps.pickle'), 'wb') as fp:
                pkl.dump(ids_imgpaths_seq, fp)
    return ids_imgpaths_seq


def get_chunks_seq(data_path:str="../3DPW",
                   out_dir:str="./data/img_list_paths_seq",
                   split:str='train',
                   img_list:list=None,
                   num_required_keypoints:int=8,
                   len_chunks=8,
                   load_from_pkl=None,
                   store_as_pkl=False):
    
    ## With initialising a new loader: get new random chunks from img_list
    if img_list is not None:
        return [chunk for path_list in img_list for chunk in rand_partition(path_list, len(path_list)//len_chunks)]
    ## Load from pickle file, when initiliasing the dataset
    elif load_from_pkl is not None:
        with open(load_from_pkl , "rb") as fp:
            chunk_dict = pkl.load(fp)
        img_list = chunk_dict['img_list']
        sequences = chunk_dict['sequences']
        img_paths = chunk_dict['img_paths']    
    else:
        sequences = {}
        img_paths = []
        sequence_path = osp.join(data_path, 'sequenceFiles', split)
        seq_names = [seq_name.split('.pkl')[0] for seq_name in sorted(os.listdir(sequence_path))]
        img_dict = {}
        img_indices = 0
        for seq_name in seq_names:
            ## loop through all sequences and filter out those where kps are missing
            img_dir = osp.join(data_path, 'imageFiles', seq_name)
            seq_file_name = os.path.join(sequence_path, f'{seq_name}.pkl')
            with open(seq_file_name, 'rb') as f:
                seq = pkl.load(f, encoding='latin1')
            sequences[seq_name] = seq
            num_people = len(seq['poses'])
            for img_idx, img_name in enumerate(sorted(os.listdir(img_dir))):
                ## for each person: save img_path 
                img_path = osp.join(img_dir,img_name)
                for person_id in range(num_people):
                    pose2d = seq['poses2d'][person_id][img_idx]
                    relevant_poses2d = get_relevant_keypoints(pose2d)
                    if len(relevant_poses2d) >= num_required_keypoints:
                        #Save for each sequence and each person separately in dict key
                        img_dict[f'{seq_name}_{person_id}'] = img_dict.get(f'{seq_name}_{person_id}', [])
                        img_dict[f'{seq_name}_{person_id}'].append([img_path, person_id, img_indices])
                        img_indices += 1
                        img_paths.append(img_path)
        img_list = list(img_dict.values())
        if store_as_pkl:
            chunk_dict = {'img_list': img_list, 'img_paths': img_paths, 'sequences': sequences}
            with open(osp.join(out_dir, f'img_list_paths_seq_{split}_min{num_required_keypoints}_kps.pickle'), 'wb') as fp:
                pkl.dump(chunk_dict, fp)
        
    #create n chunks of length len_chunks for each sequence and each person: num_seq x num_chunks x len_chunks  
    chunks = [chunk for path_list in img_list for chunk in rand_partition(path_list, len(path_list)//len_chunks)]
    return chunks, img_list, img_paths, sequences



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