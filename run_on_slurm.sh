#!/bin/bash
#
#SBATCH --job-name=training
#SBATCH --partition=All

python3 -m pip install -r ./requirements.txt
python3 src/main.py
