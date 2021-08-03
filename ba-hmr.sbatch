#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=ba-test

#resources:

#SBATCH --cpus-per-task=12
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=test
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=10:00
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=clara.grotehans@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####
mkdir /scratch/$SLURM_JOB_ID/3DPW
cp -R /home/grotehans/3DPW/sequenceFiles /scratch/$SLURM_JOB_ID/3DPW
#cp /home/grotehans/3DPW.zip /scratch/$SLURM_JOB_ID/
#unzip /scratch/$SLURM_JOB_ID/3DPW.zip
cp /home/grotehans/BA_HMR/data/ids_imgpaths/ids_paths_train_min8_kps_ids_img_list.pickle /scratch/$SLURM_JOB_ID/
cp -R /home/grotehans/BA_HMR/data/imgs_3dpw_train.zarr /scratch/$SLURM_JOB_ID/
cp -R /home/grotehans/BA_HMR/data/imgs_3dpw_valid.zarr /scratch/$SLURM_JOB_ID/

ls /scratch/$SLURM_JOB_ID/
####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
#cnn_minst.py <model save path> <mnist data path>
####

singularity exec --nv /common/singularityImages/TCML-Cuda11_0_TF2_4_1_PT1_7_1.simg /home/grotehans/miniconda3/envs/ba-env/bin/python /home/grotehans/BA_HMR/start_train.py --cfg /home/grotehans/BA_HMR/configs/experiments/zarr_config.yaml --opts "DATASETS.THREEDPW" /scratch/$SLURM_JOB_ID/3DPW/ "LOAD_FROM_ZARR.TRN" /scratch/$SLURM_JOB_ID/imgs_3dpw_train.zarr "LOAD_FROM_ZARR.VAL" /scratch/$SLURM_JOB_ID/imgs_3dpw_valid.zarr   
echo DONE!

