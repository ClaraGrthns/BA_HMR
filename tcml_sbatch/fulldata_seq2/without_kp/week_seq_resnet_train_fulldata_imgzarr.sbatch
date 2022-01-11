#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=seqwb8

#resources:

#SBATCH --cpus-per-task=9
# the job can use and see 12 CPUs (from max 24).

#SBATCH --partition=week
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=26G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=6-00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=jobs/seq/train/seqwb8.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=jobs/seq/train/seqwb8.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=clara.grotehans@student.uni-tuebingen.de
# your mail address

#SBATCH 

####
#b) copy all needed data to the jobs scratch folder
####
mkdir /scratch/$SLURM_JOB_ID/H36M
cp -r /home/grotehans/H36M/img_zarr /scratch/$SLURM_JOB_ID/H36M/
echo copied h36m image zarr!
cp -R /home/grotehans/H36M/data_pickle/seq_datalist_h36m_thr25_1to8subj.pickle /scratch/$SLURM_JOB_ID/
cp -R /home/grotehans/H36M/data_pickle/seq_datalist_h36m_thr25_9to11subj.pickle /scratch/$SLURM_JOB_ID/
echo copied datapickle!

cp -R /home/grotehans/3DPW/data_pickle/img_seqs_list_paths_seq /scratch/$SLURM_JOB_ID/
cp -R /home/grotehans/3DPW/img_zarr/imgs_3dpw_train.zarr /scratch/$SLURM_JOB_ID/
cp -R /home/grotehans/3DPW/img_zarr/imgs_3dpw_valid.zarr /scratch/$SLURM_JOB_ID/
echo copied zarr img and pickle files!

ls /scratch/$SLURM_JOB_ID/
du -sh /scratch/$SLURM_JOB_ID/H36M/img_zarr
ls /scratch/$SLURM_JOB_ID/H36M/img_zarr
####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
####

singularity exec --nv /common/singularityImages/TCML-Cuda11_2_TF2_6_0_PT_1_10_0.simg \
/home/grotehans/miniconda3/envs/ba-env/bin/python -u /home/grotehans/BA_HMR/start_train_seq_fulldata.py \
--cfg /home/grotehans/BA_HMR/configs/experiments/seq_zarr_resnet34_ep50.yaml \
--opts "DATASETS.THREEDPW" /scratch/$SLURM_JOB_ID/3DPW/ \
'OUT_DIR' ./checkpoints/seq_fulldata/week/batchsize8.2 \
"H36M.LOAD_SEQ_DATALIST.TRN" /scratch/$SLURM_JOB_ID/seq_datalist_h36m_thr25_1to8subj.pickle \
"H36M.LOAD_SEQ_DATALIST.VAL" /scratch/$SLURM_JOB_ID/seq_datalist_h36m_thr25_9to11subj.pickle \
"H36M.LOAD_FROM_ZARR" /scratch/$SLURM_JOB_ID/H36M/img_zarr/imgs_h36m_resnet_thr25 \
"THREEDPW.LOAD_FROM_ZARR.TRN" /scratch/$SLURM_JOB_ID/imgs_3dpw_train.zarr \
"THREEDPW.LOAD_FROM_ZARR.VAL" /scratch/$SLURM_JOB_ID/imgs_3dpw_valid.zarr \
"THREEDPW.LOAD_CHUNKS_SEQ.TRN" /scratch/$SLURM_JOB_ID/img_seqs_list_paths_seq/img_seqs_list_paths_seq_train_min8_kps.pickle \
"THREEDPW.LOAD_CHUNKS_SEQ.VAL" /scratch/$SLURM_JOB_ID/img_seqs_list_paths_seq/img_seqs_list_paths_seq_validation_min8_kps.pickle \
"DATASETS.H36M" /scratch/$SLURM_JOB_ID/H36M \
"LOGGING.LOGDIR" logging/seq_fulldata_zarr_resnet34_ep50/week/batchsize8.2 \
"TRAIN.BATCH_SIZE_TRN" 8


echo DONE!

