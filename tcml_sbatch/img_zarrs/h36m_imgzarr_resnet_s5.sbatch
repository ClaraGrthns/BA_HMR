#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=zarr_imgs

#resources:

#SBATCH --cpus-per-task=12
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=week
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=18G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:0
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=2-12
# the maximum time the scripts needs to run
# "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"



#SBATCH --error=jobs/h36mzarr/job.%J.err
# write the error output to job.*jobID*.err

#SBATCH --output=jobs/h36mzarr/job.%J.out
# write the standard output to job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=clara.grotehans@student.uni-tuebingen.de
# your mail address

####
#b) copy all needed data to the jobs scratch folder
####
#cp -r /home/grotehans/H36M/backgrounds /scratch/$SLURM_JOB_ID/H36M/
mkdir /scratch/$SLURM_JOB_ID/H36M
cp -r /home/grotehans/H36M/images.zip /scratch/$SLURM_JOB_ID/H36M/
echo copied images.zip!
cp -r /home/grotehans/H36M/backgrounds /scratch/$SLURM_JOB_ID/H36M/
unzip -o /scratch/$SLURM_JOB_ID/H36M/images.zip -d /scratch/$SLURM_JOB_ID/H36M/
echo unziped images.zip!
cp -R /home/grotehans/H36M/data_pickle/datalist_h36m_thr25_5to5subj.pickle /scratch/$SLURM_JOB_ID/H36M/
ls /scratch/$SLURM_JOB_ID/
ls /scratch/$SLURM_JOB_ID/H36M/

####

#c) Execute your code in a specific singularity container
####

singularity exec --nv /common/singularityImages/TCML-Cuda11_2_TF2_6_0_PT_1_10_0.simg \
/home/grotehans/miniconda3/envs/ba-env/bin/python /home/grotehans/BA_HMR/create_zarr_h36m.py --data_path /scratch/$SLURM_JOB_ID/H36M/ --subject_list 5to5

cp -r /scratch/$SLURM_JOB_ID/H36M/img_zarr/ /home/grotehans/H36M/
echo DONE!

