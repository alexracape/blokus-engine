#!/bin/bash
#SBATCH --mail-type=BEGIN,END,FAIL

cd /mnt/hpc/tmp/aracape
cp /home/aracape/Blokus-Engine/training_requirements.txt .
python -m venv venv
source venv/bin/activate
pip install --upgrade --no-cache-dir blokus-engine
pip install -r training_requirements.txt
cd /home/aracape/Blokus-Engine

python model/training.py --cpus $SLURM_NPROCS --load /home/aracape/Blokus-Engine/weights/new_loss.pt --save less_exp.pt
