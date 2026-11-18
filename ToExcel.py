import pandas as pd
import re


def extract_data_from_txt(file_path):
    data = []
    with open(file_path, 'r') as file:
        content = file.read()

        # 分割内容为不同的部分（每个步骤的数据）
        steps = content.split(
            '-----------------------------------------------------------------------------------------------------------------')
        for step in steps:
            if not step.strip():
                continue

            # 使用正则表达式提取数值
            values = {}
            for line in step.splitlines():
                line = line.strip()
                if '：' in line:  # 处理包含中文冒号的行
                    key_value = line.split('：', 1)
                elif '=' in line:  # 处理包含等号的行
                    key_value = line.split('=', 1)
                else:
                    continue

                if len(key_value) == 2:
                    key = key_value[0].strip()
                    value = key_value[1].strip()
                    values[key] = value

            if values:
                data.append(values)

    return data


def save_to_excel(data, output_file):
    if not data:  # 检查是否有数据
        print("No data to save.")
        return

    # 创建DataFrame并转置
    df = pd.DataFrame(data).T
    df.to_excel(output_file, index=True, header=False)


if __name__ == "__main__":
    txt_file_path = 'output_samples250410molclintox-gin(sigma0.1)20.txt'  # 替换为你的TXT文件路径
    excel_file_path = 'output_samples250410.xlsx'  # 输出的Excel文件名（确保是 .xlsx 扩展名）

    # 提取数据并保存
    extracted_data = extract_data_from_txt(txt_file_path)
    save_to_excel(extracted_data, excel_file_path)

    print(f"Data has been successfully extracted to {excel_file_path}.")