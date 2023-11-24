#!/bin/bash
name="$1"
experiment="$2"
outputfile="output_${name}.txt"
errorfile="error_${name}.txt"

case "$experiment" in
    "1")
        rm $outputfile $errorfile & pip install . && python tests/main.py $experiment
        ;;
    "2" | "3")
        rm $outputfile $errorfile & pip install . && mpiexec -np 3 python tests/main.py $experiment
        ;;
    *)  
        echo "only experiment 1, 2, 3 exist!"
        ;;
esac