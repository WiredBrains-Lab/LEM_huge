#!/bin/bash
#SBATCH --job-name=LEM_huge
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=750gb
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --account=bgross
#SBATCH --partition=pi_bgross
#SBATCH --output=%x-%j.out

cd $HOME/git/LEM_huge

module load code-server/4.92.2 git/2.30.0
module load cuda/12.3 cudnn/8.9.7-12.x

module add pytorch

deepspeed run_test.py 
