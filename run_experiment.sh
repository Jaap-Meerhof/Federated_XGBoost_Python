#!/bin/bash
name="$1"
experiment="$2"
outputfile="output_${name}${experiment}.txt"
errorfile="error_${name}${experiment}.txt"

case "$experiment" in
    "1")
        rm $outputfile $errorfile & pip install . && bsub -J $name -q bio -n 12 -M 32G -R "rusage[mem=32G]" -o $outputfile -e $errorfile mpiexec -np 5 /home/meerhofj/.conda/envs/fedxgboost_mpi/bin/python tests/main.py $experiment
        ;;
    "2" | "3")
        rm $outputfile $errorfile & pip install . && bsub -J $name -q bio -n 26 -M 128G -R "rusage[mem=128G]" -o $outputfile -e $errorfile mpiexec -np 3 /home/meerhofj/.conda/envs/fedxgboost_mpi/bin/python tests/main.py $experiment
        ;;
    *)  
        echo "only experiment 1 and 2 exist!"
        ;;
esac