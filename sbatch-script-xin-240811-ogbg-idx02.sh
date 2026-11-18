#!/bin/bash
#SBATCH --job-name=1108-ft-ogbg-molbace
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4500MB
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a100_1g.10gb
##Submitted batch job 5448

cd /export/home/s2997786/workspace/GDC-Eval-hcm/reGen/

module load slurm/23.11.6
module load anaconda3/2024.02
source activate autoeval

for ly_w in 1 0.5 0.1
do
  for lfgw_w in 1 0.5 0.1 0.05 0.01
  do
    for lstrcut_w in 0.5 0.1 0.05 0.01
    do
      CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps=30 --batch_size 64 \
      --ly_w=${ly_w} --lfgw_w=${lfgw_w} --lstrut_w=${lstrcut_w} \
      --p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
      |tee 2>&1 log_main-test-aug-ogb_molbace_1108ft_lyw_${ly_w}_lfgw_${lfgw_w}_lstrut_${lstrcut_w}.txt
    done
  done
done

