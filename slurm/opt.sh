#!/bin/bash

# Usage: ./slurm_script.sh <l> <s> <n> [<E>]

# Check if the required arguments are provided
if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 <l> <s> <n> [<E>]"
    exit 1
fi

L=$1
S=$2
N=$3

if [ "$#" -eq 4 ]; then
    E=$4
    JOB_NAME="$L-$S-$N-$E"
    PYTHON_CMD="python scripts/unified5.py -l $L -s $S -n $N -e $E"
    OUTPUT_FILE="outputs/opt/$L-$S-$N-$E-julyy.out"
else
    JOB_NAME="$L-$S-$N"
    PYTHON_CMD="python scripts/unified5.py -l $L -s $S -n $N"
    OUTPUT_FILE="outputs/opt/$L-$S-$N-julyy.out"
fi


# Submit the job to SLURM using a here document
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=80:00:00
#SBATCH --nodes=1 
#SBATCH -c4
#SBATCH --ntasks-per-node=1 
#SBATCH --mem=64G        
#SBATCH --job-name="$JOB_NAME"
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
$PYTHON_CMD
EOT
