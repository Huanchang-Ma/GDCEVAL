#!/bin/bash
##6782.cadmin
#PBS -m abe
#PBS -N Molbace-ogb-0814
##PBS -q gpuq2
#PBS -q workq
#PBS -M s2997786@griffithuni.edu.au
###PBS -W group_list=aspen -A aspen

###PBS -W group_list=deeplearning -A deeplearning
###Other options group_list=aspen -A aspen

#PBS -q gpuq2
#PBS -l select=1:ncpus=8:ngpus=1:mem=10gb,walltime=100:00:00

### Add current shell environment to job (comment out if not needed)
# The job's working directory
#cd /export/home/s2997786/workspace/GDC-Evaluator-hcm/reGen/
#
#source /usr/local/bin/s3proxy.sh
#module load anaconda3/2022.10
#source activate autoeval2
#
#for snr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#do
#CUDA_VISIBLE_DEVICES=0 python main_test_aug_ogb_v2.py --sample_steps 20 --batch_size 64 \
#--ly_w 1 --lfgw_w 1 --lstrut_w 1 --snr=${snr} \
#--pretrain_model_path='pretrain_models/ogbg-molbace-gin-20240814-193055-567011/gin_ogbg-molbace_model.pt' \
#--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
#|tee 2>&1 log_main-test-aug-ogb_molbace_snr_${snr}_1408.txt
#done

for snr in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
CUDA_VISIBLE_DEVICES=0 python3 main_test_aug_ogb.py --sample_steps 20 --batch_size 64 \
--ly_w 1 --lfgw_w 1 --lstrut_w 1 --snr=${snr} \
#--pretrain_model_path='pretrain_models/ogbg-molbace-gin-20240814-193055-567011/gin_ogbg-molbace_model.pt' \
--pretrain_model_path='pretrain_models/ogbg-molbbbp-gin-20240829-161648-933582/gin_ogbg-molbbbp_model.pt' \
--p_ratio_lists 0.01 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
|tee 2>&1 log_main-test-aug-ogb_molbace_snr_${snr}_1408.txt
done
