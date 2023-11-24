#!/bin/bash
name="$1"
experiment="$2"
outputfile="output_${name}.txt"
errorfile="error_${name}.txt"

case "$experiment" in
    "1")
        rm $outputfile $errorfile & pip install . && bsub -J $name -q bio -n 12 -M 64G -R "rusage[mem=64G]" -o $outputfile -e $errorfile mpiexec -np 3 /home/meerhofj/.conda/envs/fedxgboost_mpi/bin/python tests/main.py $experiment
        ;;
    "2" | "3")
        rm $outputfile $errorfile & pip install . && bsub -J $name -q bio -n 26 -M 128G -R "rusage[mem=128G]" -o $outputfile -e $errorfile mpiexec -np 3 /home/meerhofj/.conda/envs/fedxgboost_mpi/bin/python tests/main.py $experiment
        ;;
    *)  
        echo "only experiment 1, 2, 3 exist!"
        ;;
esac