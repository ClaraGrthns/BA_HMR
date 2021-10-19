#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=ba-hmr

#resources:

#SBATCH --cpus-per-task=12
# the job can use and see 12 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=20:00:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=jobs/job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=jobs/job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=clara.grotehans@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####
mkdir /scratch/$SLURM_JOB_ID/H36M
cp -R /home/grotehans/H36M/all_images.zip /scratch/$SLURM_JOB_ID/H36M
unzip /scratch/$SLURM_JOB_ID/H36M/all_images.zip
mv /scratch/$SLURM_JOB_ID/H36M/all_images /scratch/$SLURM_JOB_ID/H36M/images
cp -R /home/grotehans/H36M/data_pickle/datalist_h36m_thr25_1to8subj.pickle /scratch/$SLURM_JOB_ID/H36M

cp -R /home/grotehans/3DPW/data_pickle/ids_imgpaths_seq/ /scratch/$SLURM_JOB_ID/
cp -R /home/grotehans/3DPW/img_zarr/imgs_3dpw_train.zarr /scratch/$SLURM_JOB_ID/
cp -R /home/grotehans/3DPW/img_zarr/imgs_3dpw_valid.zarr /scratch/$SLURM_JOB_ID/

ls /scratch/$SLURM_JOB_ID/
####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####

singularity exec --nv /common/singularityImages/TCML-Cuda11_0_TF2_4_1_PT1_7_1.simg \
/home/grotehans/miniconda3/envs/ba-env/bin/python /home/grotehans/BA_HMR/start_train_fulldata.py \
--cfg /home/grotehans/BA_HMR/configs/experiments/fulldataset_zarr_resnet34_ep50.yaml \
--opts "DATASETS.THREEDPW" /scratch/$SLURM_JOB_ID/3DPW/ \
"THREEDPW.LOAD_FROM_ZARR.TRN" /scratch/$SLURM_JOB_ID/imgs_3dpw_train.zarr \
"THREEDPW.LOAD_FROM_ZARR.VAL" /scratch/$SLURM_JOB_ID/imgs_3dpw_valid.zarr \
"THREEDPW.LOAD_IDS_IMGPATHS_SEQ.TRN" /scratch/$SLURM_JOB_ID/ids_imgpaths_seq/ids_paths_seq_train_min8_kps.pickle \
"THREEDPW.LOAD_IDS_IMGPATHS_SEQ.VAL" /scratch/$SLURM_JOB_ID/ids_imgpaths_seq/ids_paths_seq_validation_min8_kps.pickle \
"DATASETS.H36M" /scratch/$SLURM_JOB_ID/H36M \
"H36M.LOAD_DATALIST.TRN" /scratch/$SLURM_JOB_ID/datalist_h36m_thr25_1to8subj.pickle

echo DONE!

