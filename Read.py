import csv

# 假设文本文件的名称是 'data.txt'
filename = 'output_samples.txt'

# 初始化数据字典
data = {'avg_mmd_samples_value', 'max validity w/o correction', 'Average validity w/o correction', 'mse_ori_samples', 'mse_aug_samples', 'mse_ori_aug_samples', 'rmse_ori_samples', 'rmse_aug_samples', 'rmse_ori_aug_samples', 'mae_ori_samples', 'mae_aug_samples', 'mae_ori_aug_samples', 'r2_ori_samples', 'r2_aug_samples', 'r2_ori_aug_samples', 'l2_ori_samples', 'l2_aug_samples', 'l2_ori_aug_samples', 'normalized_MSE_samples_ori_aug', 'normalized_RMSE_samples_ori_aug', 'normalized_L2_ori_aug_samples', 'auc_ori_samples', 'auc_aug_samples', 'acc_ground_ori', 'acc_ground_aug', 'acc_ori_aug_samples'}

# 打开文件并读取数据
with open(filename, 'r') as file:
    reader = csv.reader(file, delimiter='：')  # 假设每个数据项由 '：' 分隔
    
    # 读取每一行数据
    for row in reader:
        # 跳过空行
        if not row:
            continue
        
        key = row[0]
        value = row[1]
        
        # 跳过特定类型的数据
        if key.startswith('mmd_samples_value_set'):
            continue
        
        # 将数据添加到字典中
        if key in data:
            data[key].append(value)
        else:
            data[key] = [value]

# 准备写入到 CSV 文件
with open('output.csv', 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=',')
    
    # 写入不同类型的数据
    for key in data:
        values = data[key]
        if values:  # 确保列表不为空
            # 写入标题行
            writer.writerow([key])
            # 写入数据行
            writer.writerows(values)
