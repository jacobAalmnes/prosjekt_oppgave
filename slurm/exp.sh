#!/bin/bash

# Usage: ./slurm_script.sh <model_name> <test>

# Check if the required argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <experiment>"
    exit 1
fi

EXPERIMENT=$1
OUTPUT_FILE="outputs/experiments/$EXPERIMENT.out"

# Submit the job to SLURM using a here document
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=03:00:00
#SBATCH --nodes=1 
#SBATCH -c2
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=48G        
#SBATCH --job-name="$EXPERIMENT"
#SBATCH --output=$OUTPUT_FILE
#SBATCH --mail-user=stefandt@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=\${SLURM_SUBMIT_DIR}
cd \$WORKDIR

CORES_PER_NODE=\$SLURM_CPUS_ON_NODE
TOTAL_CORES=\$((SLURM_JOB_NUM_NODES * CORES_PER_NODE))

echo "we are running from this directory: \$SLURM_SUBMIT_DIR"
echo "the name of the job is: \$SLURM_JOB_NAME"
echo "The job ID is \$SLURM_JOB_ID"
echo "The job was run on these nodes: \$SLURM_JOB_NODELIST"
echo "Number of nodes: \$SLURM_JOB_NUM_NODES"
echo "We are using \$SLURM_CPUS_ON_NODE cores"
echo "We are using \$SLURM_CPUS_ON_NODE cores per node"
echo "Total of \$TOTAL_CORES cores"

module purge
module load Anaconda3/2023.09-0
conda activate /cluster/home/stefandt/anaconda3/envs/ppconda
python scripts/experiments.py --experiment $EXPERIMENT
EOT

