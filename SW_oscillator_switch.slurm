#!/bin/bash
#SBATCH --job-name=SW_oscillator_switch
#SBATCH --time=30:00:00
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aplazar1@jhu.edu
#SBATCH --account=gsteino1

echo "Starting job $SLURM_JOBID"

ml anaconda

conda activate python_env

module list

./SW_oscillator_switch_many_sims.sh

conda deactivate

echo "Finished with job $SLURM_JOBID"
