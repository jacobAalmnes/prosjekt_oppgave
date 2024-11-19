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
    PARTITION="GPUQ"
    TIME="00:05:00"
    MEM="64G"
    CORES_PER_NODE=2
    CMD_ARGS="--nrows 1024"
elif [ "$TEST" == "0" ]; then
    TEST_STRING="full"
    PARTITION="GPUQ"
    TIME="18:00:00"
    MEM="80G"
    CORES_PER_NODE=8
    CMD_ARGS=""
else
    echo "Test argument must be 0 or 1. Got: $TEST"
    exit 1
fi

IDENTIFIER="$MODEL_NAME-$TEST_STRING"
OUTPUT_FILE="jobs/outputs/$IDENTIFIER.out"
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
#SBATCH --job-name="$IDENTIFIER"
#SBATCH --output="$OUTPUT_FILE"
#SBATCH --gres=gpu:1
#SBATCH --mail-user=stefandt@stud.ntnu.no
#SBATCH --mail-type=ALL
#SBATCH -c$CORES_PER_NODE

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
python src/generate_embeddings.py --model_name $MODEL_NAME $CMD_ARGS --num_workers $CORES_PER_NODE --output_name $OUTPUT_NAME --non_blocking True --pin_memory True
EOT

