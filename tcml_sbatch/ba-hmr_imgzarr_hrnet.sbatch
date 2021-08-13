#!/bin/bash

####
#a) Define slurm job parameters
####

#SBATCH --job-name=zarr_imgs

#resources:

#SBATCH --cpus-per-task=12
# the job can use and see 4 CPUs (from max 24).

#SBATCH --partition=day
# the slurm partition the job is queued to.

#SBATCH --mem-per-cpu=3G
# the job will need 12GB of memory equally distributed on 4 cpus.  (251GB are available in total on one node)

#SBATCH --gres=gpu:0
#the job can use and see 1 GPUs (4 GPUs are available in total on one node)

#SBATCH --time=24:00:00
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
cp -R /home/grotehans/BA_HMR/data/ids_imgpaths_seq/ /scratch/$SLURM_JOB_ID/
ls /scratch/$SLURM_JOB_ID/


####

#c) Execute your code in a specific singularity container
####

singularity exec --nv /common/singularityImages/TCML-Cuda11_0_TF2_4_1_PT1_7_1.simg \
/home/grotehans/miniconda3/envs/ba-env/bin/python /home/grotehans/BA_HMR/create_zarr.py --encoder 'hrnet'
echo DONE!

