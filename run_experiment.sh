#!/bin/bash
name="$1"
experiment="$2"
outputfile="output_${name}.txt"
errorfile="error_${name}.txt"

rm $outputfile $errorfile & pip install . && bsub -J $name -q bio -n 12 -M 128G -R "rusage[mem=128G]" -o $outputfile -e $errorfile mpiexec -np 3 /home/meerhofj/.conda/envs/fedxgboost_mpi/bin/python /home/meerhofj/Documents/Federated_XGBoost_Python/tests/main.py $experiment
