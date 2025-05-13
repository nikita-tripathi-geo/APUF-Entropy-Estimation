#!/bin/bash
# ---------------------------------------------------------------------
#SBATCH --job-name=entropy_job
#SBATCH --mail-user=nikita.tripathi@ucalgary.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --output=%A-%a-64-ext.out
#SBATCH --array=50,100,150,200,300,400,500,750,1000,2000,3000,4000,5000,10000,20000,30000,40000,50000,60000
#SBATCH --mem=400gb
#SBATCH --time=20:00:00
#SBATCH --partition=cpu2023



# ---------------------------------------------------------------------
module load python/3.12.5

# array=50,100,150,200,300,400,500,750,1000,2000,3000,4000,5000,10000,20000,30000,40000,50000,60000
# entropy_helper.py 			d 	k 									 ell m 	 n
# python3 entropy_helper.py 65 100 200 100 50
python3 entropy_helper.py 129 $SLURM_ARRAY_TASK_ID 100 50 25

