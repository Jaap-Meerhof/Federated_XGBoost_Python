#!/bin/bash
testnum="$1"
name="$2"
outputfile="output${testnum}.txt"
errorfile="error${testnum}.txt"

rm $outputfile $errorfile & pip install . && bsub -J $name -q bio -n 6 -M 64G -R "rusage[mem=64G]" -o $outputfile -e $errorfile mpiexec -np 3 /home/meerhofj/.conda/envs/fedxgboost_mpi/bin/python /home/meerhofj/Documents/Federated_XGBoost_Python/tests/main.py
