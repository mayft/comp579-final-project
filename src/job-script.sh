#!/bin/bash
#SBATCH --job-name=eswm_open_arena 
#SBATCH --output=logs/out_%j.txt 
#SBATCH --error=logs/err_%j.txt  
#SBATCH --time=2:00:00
#SBATCH --partition=all
#SBATCH --account=winter2026-comp579
#SBATCH --qos=comp579-1gpu-12h
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --propagate=NONE
#SBATCH -c 8

module load slurm
module load miniconda/miniconda-winter2025

python eswm.py
