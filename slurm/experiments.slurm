#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=03:00:00
#SBATCH --nodes=1 
#SBATCH -c2
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=64G        
#SBATCH --job-name="normalization"
#SBATCH --output=outputs/experiments/normalization.out
#SBATCH --mail-user=stefandt@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"

module purge
source /cluster/home/stefandt/anaconda3/bin/activate ppconda
python scripts/experiments.py --experiment "normalization"