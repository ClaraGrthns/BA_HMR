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

#SBATCH --mem-per-cpu=5G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:A4000:1
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=24:00:00
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
mkdir /scratch/$SLURM_JOB_ID/H36M
cp -r /home/grotehans/H36M/images.zip /scratch/$SLURM_JOB_ID/H36M/
echo copied images.zip!
cp -r /home/grotehans/H36M/backgrounds /scratch/$SLURM_JOB_ID/H36M/
unzip -o /scratch/$SLURM_JOB_ID/H36M/images.zip -d /scratch/$SLURM_JOB_ID/H36M/
echo unziped images.zip!
cp -R /home/grotehans/H36M/data_pickle/datalist_h36m_thr25_1to8subj.pickle /scratch/$SLURM_JOB_ID/
cp -R /home/grotehans/H36M/data_pickle/datalist_h36m_thr25_9to11subj.pickle /scratch/$SLURM_JOB_ID/
ls /scratch/$SLURM_JOB_ID/
ls /scratch/$SLURM_JOB_ID/H36M/
####
#c) Execute your tensorflow code in a specific singularity container
#d) Write your checkpoints to your home directory, so that you still have them if your job fails
####
singularity exec --nv /common/singularityImages/TCML-Cuda11_2_TF2_6_0_PT_1_10_0.simg \
/home/grotehans/miniconda3/envs/ba-env/bin/python /home/grotehans/BA_HMR/start_train_fulldata.py \
--cfg /home/grotehans/BA_HMR/configs/experiments/fulldataset_zarr3dpw_resnet34_ep50.yaml \
--dataset h36m \
--opts "DATASETS.H36M" /scratch/$SLURM_JOB_ID/H36M \
"H36M.LOAD_DATALIST.TRN" /scratch/$SLURM_JOB_ID/datalist_h36m_thr25_1to8subj.pickle \
"H36M.LOAD_DATALIST.VAL" /scratch/$SLURM_JOB_ID/datalist_h36m_thr25_9to11subj.pickle \
"H36M.VAL_ON_H36M" True \



echo DONE!

