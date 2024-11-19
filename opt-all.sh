#!/bin/bash

# Usage: ./run-opt.sh <n> <E>

# Check if the required argument is provided
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <n> [<E>]"
    exit 1
fi

N=$1

if [ "$#" -eq 2 ]; then
    E=$2
    APPEND="$N $E"
else
    APPEND="$N"
fi



./slurm/opt.sh mtl org $APPEND
./slurm/opt.sh mtl imp $APPEND
./slurm/opt.sh mtl mix $APPEND
./slurm/opt.sh stl org $APPEND
./slurm/opt.sh stl imp $APPEND
./slurm/opt.sh stl mix $APPEND