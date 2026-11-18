#!/bin/bash
#SBATCH --job-name=1108-v2-ogbg-molbace
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4500MB
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a100_1g.10gb
##Submitted batch job 5447

cd /export/home/s2997786/workspace/GDC-Eval-hcm/reGen/

module load slurm/23.11.6
module load anaconda3/2024.02
source activate autoeval

for sample_steps in 10 20 30
do
CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps=${sample_steps} --batch_size 64 \
--ly_w 1 --lfgw_w 1 --lstrut_w 1 \
--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
|tee 2>&1 log_main-test-aug-ogb_molbace_ly_w_1_STEP_${sample_steps}_1108.txt
done


for snr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps 20 --batch_size 64 \
--ly_w 0.1 --lfgw_w 1 --lstrut_w 1 --snr=${snr} \
--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
|tee 2>&1 log_main-test-aug-ogb_molbace_ly_w_01_SNR_${snr}_1108.txt
done

for sample_steps in 10 20 30
do
CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps=${sample_steps} --batch_size 32 \
--ly_w 1 --lfgw_w 1 --lstrut_w 1 \
--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
|tee 2>&1 log_main-test-aug-ogb_molbace_BS32_ly_w_1_STEP_${sample_steps}_1108.txt
done

for snr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps 20 --batch_size 64 \
--ly_w 1 --lfgw_w 1 --lstrut_w 1 --snr=${snr} \
--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
|tee 2>&1 log_main-test-aug-ogb_molbace_ly_w_1_SNR_${snr}_1108.txt
done

08/11 11:29:08 PM Perf AUC = {'auc_aug': 0.9695705094766126, 'auc_aug_avg': 0.822813423752391, 'auc_ori': 0.8109893931490175}
08/11 11:29:08 PM Finish! Log_dir final is: diff_results_xz/diff-ogbg-molbace-gin-20240811-224941-727382

08/11 11:55:56 AM Perf AUC = {'auc_aug': 0.9695705094766127, 'auc_aug_avg': 0.822813423752391, 'auc_ori': 0.8109893931490175}
08/11 11:55:56 AM Finish! Log_dir final is: diff_results_xz/diff-ogbg-molbace-gin-20240811-111714-470974


CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps 20 --batch_size 64 \
--ly_w 1 --lfgw_w 1 --lstrut_w 1 --snr=0.2 --pretrain_model_path='pretrain_models/ogbg-molbace-gin-20240814-185025-054548/gin_ogbg-molbace_model.pt' \
--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
|tee 2>&1 log_main-test-aug-ogb_molbace_1408.txt