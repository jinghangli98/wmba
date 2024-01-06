#!/bin/bash
#SBATCH --job-name=gmba
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=99:00:00
#SBATCH --array=1-10
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jil202@pitt.edu
#SBATCH --account=tibrahim
#SBATCH --cluster=htc

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Define an array of input filenames
images=(/ix1/tibrahim/jil202/wmba/img/*.nii.gz)

current_img=${images[$SLURM_ARRAY_TASK_ID - 1]}
current_name=$(echo "$current_img" | rev | cut -d'.' -f3- | rev)

echo wmba $current_img ${current_name}_wmba.csv
wmba $current_img ${current_name}_wmba.csv