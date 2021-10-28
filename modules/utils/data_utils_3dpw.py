import os
import os.path as osp
import pickle as pkl
import torch
from pathlib import Path
from datetime import date
import random 
from .data_utils import rand_partition

 


def get_relevant_keypoints(keypoints, threshold=0.3):
    return [(x,y) for (x,y,relevance) in (keypoints).transpose(1,0) if relevance>threshold]


def get_ids_imgspaths_seq(data_path:str="../3DPW",
                          out_dir:str="./data/ids_imgpaths",
                          split:str='train',
                          num_required_keypoints:int=8,
                          load_from_pkl=None,
                          store_as_pkl=False
                         )-> dict:

    ''' Yields a dictionary with the list of person ids, corresponding list of image paths and all sequences of 3DPW.

    Depending on the minimum number of keypoints specified, each image is assigned to a list of person ids and the image path.
    If the path to a pickle file is specified, the dictionary is loaded from the pickle file. 
    The dictionary can be saved as a pickle file.
        Args: 
           data_path:
               Path to dataset directory 
           out_dir: 
               Saves directory as pickle for given filepath if 'store_as_pkl' is True
           split:
               Specifies dataset split: 'train', 'validation' or 'test'
           num_required_keypoints: 
               Minimum number of required 2d keypoints
           load_from_pkl: 
               If filepath is specified, the dictionary is loaded from the pickle file.
           store_as_pkl: 
               Saves directory as pickle if 'out_dir' is not none.
        Returns:
            Dictionary, containing person ids, corresponding image paths and sequences
    '''
    if load_from_pkl is not None:
        with open(load_from_pkl , "rb") as fp:
            ids_imgpaths_seq = pkl.load(fp)
    else: 
        sequence_path = osp.join(data_path, 'sequenceFiles', split)
        seq_names = [seq_name.split('.pkl')[0] for seq_name in sorted(os.listdir(sequence_path))]
        image_paths = []
        person_ids = []
        seq_name_to_seq = {}
        for seq_name in seq_names:
            ## loop through all sequences and filter out those where kps are missing
            img_dir = osp.join(data_path, 'imageFiles', seq_name)
            seq_file_name = os.path.join(sequence_path, f'{seq_name}.pkl')
            with open(seq_file_name, 'rb') as f:
                seq = pkl.load(f, encoding='latin1')
            seq_name_to_seq[seq_name] = seq
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
        ids_imgpaths_seq = {'person_ids': person_ids, 'image_paths': image_paths, 'sequences': seq_name_to_seq}
        if store_as_pkl and out_dir is not None:
            with open(osp.join(out_dir, f'ids_paths_seq_{split}_min{num_required_keypoints}_kps.pickle'), 'wb') as fp:
                pkl.dump(ids_imgpaths_seq, fp)
    return ids_imgpaths_seq


def get_chunks_img_paths_list_seq(data_path:str="../3DPW",
                   out_dir:str="./data/img_seqs_list_paths_seq",
                   split:str='train',
                   img_seqs_list:list=None,
                   num_required_keypoints:int=8,
                   len_chunks=8,
                   load_from_pkl=None,
                   store_as_pkl=False
                  )-> tuple:
    ''' Yields a tuple, chunks of img_seqs_list of each sequence and person, img_seqs_list, list of all img paths, sequences
    
    Depending on the minimum number of keypoints specified, 
    If the path to a pickle file is specified, the dictionary is loaded from the pickle file. 
    The dictionary can be saved as a pickle file.
    Args: 
        data_path:
            Path to dataset directory 
        out_dir: 
            Saves directory {'img_seqs_list': img_seqs_list, 'img_paths': img_paths, 'sequences': seq_name_to_seq} as pickle
            for given filepath if 'store_as_pkl' is True
        split:
            Specifies dataset split: 'train', 'validation' or 'test'
        num_required_keypoints: 
            Minimum number of required 2d keypoints
        len_chunks:
            Length of random sublists 
        img_seqs_list: 
            If filepath is specified, the dict {'img_seqs_list': img_seqs_list, 'img_paths': img_paths, 'sequences': seq_name_to_seq}
            is loaded from the pickle file.
        store_as_pkl: 
            Saves directory  {'img_seqs_list': img_seqs_list, 'img_paths': img_paths, 'sequences': seq_name_to_seq} as pickle
            if 'out_dir' is not none.
    Returns:
        chunks: a list of random sub-lists with image path, person_id and img_id. each chunk is a subset of a sub-list in img_seqs_list
        img_seqs_list: a list of sub-lists with image path, person_id and img_id for each sequence and corresponding person
        img_paths: a list of image paths
        seq_name_to_seq: dictionary that maps sequence name to sequences
    '''
    seq_name_to_seq = img_paths = None
    ## With initialising a new loader: get new random chunks from img_seqs_list
    if img_seqs_list is None:
        ## Load from pickle file, when initiliasing the dataset
        if load_from_pkl is not None:
            with open(load_from_pkl , "rb") as fp:
                chunk_dict = pkl.load(fp)
            img_seqs_list = chunk_dict['img_seqs_list']
            seq_name_to_seq = chunk_dict['sequences']
            img_paths = chunk_dict['img_paths']    
        else:
            seq_name_to_seq = {}
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
                seq_name_to_seq[seq_name] = seq
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
            img_seqs_list = list(img_dict.values())
            if store_as_pkl:
                chunk_dict = {'img_seqs_list': img_seqs_list, 'img_paths': img_paths, 'sequences': seq_name_to_seq}
                with open(osp.join(out_dir, f'img_seqs_list_paths_seq_{split}_min{num_required_keypoints}_kps.pickle'), 'wb') as fp:
                    pkl.dump(chunk_dict, fp)        
    #create n chunks of length len_chunks for each sequence and each person: num_seq x num_chunks x len_chunks  
    chunks = [chunk for path_list in img_seqs_list for chunk in rand_partition(path_list, len(path_list)//len_chunks, len_chunks)]
    return chunks, img_seqs_list, img_paths, seq_name_to_seq





