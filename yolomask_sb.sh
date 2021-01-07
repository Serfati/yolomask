#!/bin/bash

################################################################################################
### sbatch configuration parameters must start with #SBATCH and must precede any other commands.
### To ignore, just add another # - like so: ##SBATCH
################################################################################################

#SBATCH --partition rtx2080			### specify partition name where to run a job. main: all nodes; gtx1080: 1080 gpu card nodes; rtx2080: 2080 nodes; teslap100: p100 nodes; titanrtx: titan nodes
#SBATCH --time 0-20:30:00			### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name yolomask			### name of the job
#SBATCH --output yolomask-%J.out			### output log for running job - %J for job number
##SBATCH --mail-user=serfata@post.bgu.ac.il	### user's email for sending job status messages
##SBATCH --mail-type=ALL			### conditions for sending the email. ALL,BEGIN,END,FAIL, REQUEU, NONE

#SBATCH --gres=gpu:1				### number of GPUs, allocating more than 1 requires IT team's permission
#SBATCH --mem=40G				### ammount of RAM memory, allocating more than 60G requires IT team's permission
#SBATCH --cpus-per-task=10			### number of CPU cores, allocating more than 10G requires IT team's permission

### Print some data to output file ###
echo `date`
echo -e "\nSLURM_JOBID:\t\t" $SLURM_JOBID
echo -e "SLURM_JOB_NODELIST:\t" $SLURM_JOB_NODELIST "\n\n"

### Start your code below ####
module load anaconda				### load anaconda module (must be present when working with conda environments)
source activate yolo				### activate a conda environment, replace my_env with your conda environment
python test.py  --name yolomasktest
###python detect.py --source data/videos/
### python train.py --batch 64 --weights yolov5l_fm_opt.pt --data data/opencovid.yaml --epochs 300 --cache --img 640 --hyp hyp.finetune.yaml --name yolomask       ### this command executes jupyter lab – replace with your own command