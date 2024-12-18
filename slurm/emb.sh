#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <model_name>"
    exit 1
fi

MODEL=$1
FILE_NAME="outputs/embeddings/best-$MODEL.out"


sbatch <<EOT
#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --account=ie-idi
#SBATCH --time=30:00:00
#SBATCH --nodes=1              # 2 compute nodes
#SBATCH --ntasks-per-node=1    # 1 mpi process each node
#SBATCH --mem=64G
#SBATCH -c4        
#SBATCH --job-name="$MODEL"
#SBATCH --output=$FILE_NAME
#SBATCH --mail-user=jacobaal@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=\${SLURM_SUBMIT_DIR}
cd \${WORKDIR}
echo "we are running from this directory: \$SLURM_SUBMIT_DIR"
echo " the name of the job is: \$SLURM_JOB_NAME"
echo "Th job ID is \$SLURM_JOB_ID"
echo "The job was run on these nodes: \$SLURM_JOB_NODELIST"
echo "Number of nodes: \$SLURM_JOB_NUM_NODES"
echo "We are using \$SLURM_CPUS_ON_NODE cores"
echo "We are using \$SLURM_CPUS_ON_NODE cores per node"
echo "Total of \$SLURM_NTASKS cores"

module purge
source /cluster/home/jacobaal/anaconda3/bin/activate ppconda
python scripts/embeddings.py --model_name $MODEL
EOT
