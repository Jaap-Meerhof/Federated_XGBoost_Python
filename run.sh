#!/bin/bash
name="$1"
outputfile="output_${name}.txt"
errorfile="error_${name}.txt"

rm $outputfile $errorfile & pip install . && bsub -J $name -q bio -n 6 -M 64G -R "rusage[mem=64G]" -o $outputfile -e $errorfile mpiexec -np 3 /home/meerhofj/.conda/envs/fedxgboost_mpi/bin/python /home/meerhofj/Documents/Federated_XGBoost_Python/tests/main.py
