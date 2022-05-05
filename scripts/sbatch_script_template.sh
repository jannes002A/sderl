#!/bin/bash
#SBATCH --job-name=job-name
#SBATCH --mail-user=email@addresse.com
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4096
#SBATCH --time=12:00:00
#SBATCH --qos=standard
#SBATCH --output=/package/abs/path/output/%A_%a.out
#SBATCH --error=/package/abs/path/output/%A_%a.err

python /src/sderl/algo/script.py -n ${SLURM_ARRAY_TASK_ID}
