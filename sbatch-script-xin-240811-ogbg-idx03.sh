#!/bin/bash
#SBATCH --job-name=1108-smallft-ogbg-molbace
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4500MB
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a100_1g.10gb
##Submitted batch job 5449

cd /export/home/s2997786/workspace/GDC-Eval-hcm/reGen/

module load slurm/23.11.6
module load anaconda3/2024.02
source activate autoeval


CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps=40 --batch_size 64 \
--ly_w 1 --lfgw_w 1 --lstrut_w 0.1 \
--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
|tee 2>&1 log_main-test-aug-ogb_molbace_1108_idx01.txt


CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps=40 --batch_size 64 \
--ly_w 1 --lfgw_w 0.1 --lstrut_w 0.1 \
--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
|tee 2>&1 log_main-test-aug-ogb_molbace_1108_idx02.txt

CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps=40 --batch_size 64 \
--ly_w 1 --lfgw_w 1 --lstrut_w 0.05 \
--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
|tee 2>&1 log_main-test-aug-ogb_molbace_1108_idx03.txt


CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps=40 --batch_size 64 \
--ly_w 1 --lfgw_w 0.1 --lstrut_w 0.05 \
--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
|tee 2>&1 log_main-test-aug-ogb_molbace_1108_idx04.txt

for snr in 0.1 0.3 0.5 0.6 0.7 0.8 0.9
do
CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps=40 --batch_size 64 \
--ly_w 1 --lfgw_w 0.1 --lstrut_w 0.05 --snr=${snr} \
--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
|tee 2>&1 log_main-test-aug-ogb_molbace_1108_idx05_snr_${snr}.txt
done

not good!

