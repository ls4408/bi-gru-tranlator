#!/bin/bash
#
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=DecompAtt
#SBATCH --mail-type=END
#SBATCH --mail-user=ls4408@nyu.edu
#SBATCH --output=slurm_%j.out
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

module load pytorch/python3.5/0.2.0_3

module load cuda/8.0.44
module load cudnn/8.0v5.1

time python3 train_model_cloud2.py