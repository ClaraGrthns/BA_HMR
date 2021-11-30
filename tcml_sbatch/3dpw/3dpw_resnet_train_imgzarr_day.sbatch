#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=3dpw2

#resources:

#SBATCH --cpus-per-task=12
# the job can use and see 12 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=4G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=12:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=jobs/train/job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=jobs/train/job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=clara.grotehans@student.uni-tuebingen.de
# your mail address

#SBATCH 

####
#b) copy all needed data to the jobs scratch folder
####
cp -R /home/grotehans/3DPW/data_pickle/ids_imgpaths_seq/ /scratch/$SLURM_JOB_ID/
cp -R /home/grotehans/3DPW/img_zarr/imgs_3dpw_train.zarr /scratch/$SLURM_JOB_ID/
cp -R /home/grotehans/3DPW/img_zarr/imgs_3dpw_valid.zarr /scratch/$SLURM_JOB_ID/
echo copied zarr img and pickle files!

ls /scratch/$SLURM_JOB_ID/
####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
####

singularity exec --nv /common/singularityImages/TCML-Cuda11_2_TF2_6_0_PT_1_10_0.simg \
/home/grotehans/miniconda3/envs/ba-env/bin/python -u /home/grotehans/BA_HMR/start_train_fulldata.py \
--cfg /home/grotehans/BA_HMR/configs/experiments/fulldataset_zarr_resnet34_ep50.yaml \
--opts "DATASETS.THREEDPW" /scratch/$SLURM_JOB_ID/3DPW/ \
"DATASET_OPT" 3dpw \
'OUT_DIR' ./checkpoints/3dpw \
"THREEDPW.LOAD_FROM_ZARR.TRN" /scratch/$SLURM_JOB_ID/imgs_3dpw_train.zarr \
"THREEDPW.LOAD_FROM_ZARR.VAL" /scratch/$SLURM_JOB_ID/imgs_3dpw_valid.zarr \
"THREEDPW.LOAD_IDS_IMGPATHS_SEQ.TRN" /scratch/$SLURM_JOB_ID/ids_imgpaths_seq/ids_paths_seq_train_min8_kps.pickle \
"THREEDPW.LOAD_IDS_IMGPATHS_SEQ.VAL" /scratch/$SLURM_JOB_ID/ids_imgpaths_seq/ids_paths_seq_validation_min8_kps.pickle \
"LOGGING.LOGDIR" logging/3dpw/without_trans

echo DONE!

