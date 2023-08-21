#!/bin/bash
param1=""
param2=""

while getopts "h1:2:" opt; do
   case $opt in
      h)
        show_help=1
        ;;
      1)
        param1="$OPTARG"
        ;;
      \?)
        echo "invalid option: -$OPTARG" >&2
        exit 1
        ;;
   esac
done

if [ $show_help -eq 1 ]; then
  echo "Usage $0 -h [-1 value] [-2 value]"
  exit 0
fi

echo "Parameter 1: $param1"
echo "parameter 2: $param2"
