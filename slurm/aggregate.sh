#!/bin/bash

# Usage: ./slurm_script.sh <model_name> <test>

# Check if the required argument is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <model_name> <test>"
    exit 1
fi

MODEL_NAME=$1
TEST=$2

if [ "$TEST" == "1" ]; then
    TEST_STRING="test"
    PARTITION="CPUQ"
    TIME="00:05:00"
    MEM="16G"
    CORES_PER_NODE=1
    CMD_ARGS="--nrows 1024"
elif [ "$TEST" == "0" ]; then
    TEST_STRING="full"
    PARTITION="CPUQ"
    TIME="01:00:00"
    MEM="128G"
    CORES_PER_NODE=4
    CMD_ARGS=""
else
    echo "Test argument must be 0 or 1. Got: $TEST"
    exit 1
fi

IDENTIFIER="$MODEL_NAME-$TEST_STRING"
OUTPUT_FILE="jobs/outputs/aggregate/$IDENTIFIER.out"
OUTPUT_NAME="$MODEL_NAME-$TEST_STRING.csv"

# Submit the job to SLURM using a here document
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=$PARTITION
#SBATCH --account=ie-idi
#SBATCH --time=$TIME
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=$MEM
#SBATCH --job-name="agg"
#SBATCH --output="$OUTPUT_FILE"
#SBATCH --gres=gpu:0
#SBATCH --mail-user=jacobaal@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH -c$CORES_PER_NODE

WORKDIR=\${SLURM_SUBMIT_DIR}
cd \$WORKDIR

echo "we are running from this directory: \$SLURM_SUBMIT_DIR"
echo "the name of the job is: \$SLURM_JOB_NAME"
echo "The job ID is \$SLURM_JOB_ID"
echo "The job was run on these nodes: \$SLURM_JOB_NODELIST"
echo "Number of nodes: \$SLURM_JOB_NUM_NODES"
echo "We are using \$SLURM_CPUS_ON_NODE cores"
echo "We are using \$SLURM_CPUS_ON_NODE cores per node"

module purge
module load Anaconda3/2023.09-0
conda activate /cluster/home/stefandt/anaconda3/envs/ppconda
python src/scripts/aggregate.py --model_name $MODEL_NAME --test $TEST_STRING
EOT

