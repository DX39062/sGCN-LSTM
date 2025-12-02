import os
import glob
import scipy.io
import numpy as np

def calculate_average_matrix():
    """
    在当前路径下查找所有 .mat 文件，
    加载其中 116x116 的数据矩阵，
    检测并排除包含 NaN 的异常矩阵，
    计算剩余有效矩阵的逐元素平均值，
    并将结果保存为 'FC.csv'。
    """
    
    # 定义预期的数据维度
    expected_shape = (116, 116)
    output_filename = "FC.csv"
    
    # 1. 查找当前路径下所有的 .mat 文件
    mat_files = glob.glob('*.mat')
    
    if not mat_files:
        print("在当前路径下未找到 .mat 文件。")
        return

    print(f"找到了 {len(mat_files)} 个 .mat 文件，开始处理...")
    
    # 2. 初始化变量
    total_matrix_sum = np.zeros(expected_shape, dtype=np.float64)
    valid_file_count = 0
    
    # 新增：记录包含 NaN 的文件
    nan_file_count = 0
    nan_files_list = []
    
    # 3. 循环处理每一个 .mat 文件
    for mat_filename in mat_files:
        # print(f"--- 正在处理: {mat_filename} ---") # 可以注释掉以减少刷屏
        
        try:
            # 4. 加载 .mat 文件
            data_dict = scipy.io.loadmat(mat_filename)
            
            # 5. 提取数据矩阵
            data_matrix = None
            variable_name = None
            
            for key, value in data_dict.items():
                if key.startswith('__'):
                    continue
                
                # 检查数据是否符合 116x116 的要求
                if isinstance(value, np.ndarray) and value.shape == expected_shape:
                    data_matrix = value
                    variable_name = key
                    break
            
            # 6. 验证、检测 NaN 并累加
            if data_matrix is not None:
                # --- 新增修改：检测 NaN ---
                if np.isnan(data_matrix).any():
                    print(f"!!! 警告: 文件 {mat_filename} 包含 NaN 值，已排除 !!!")
                    nan_file_count += 1
                    nan_files_list.append(mat_filename)
                else:
                    # 没有 NaN，视为有效数据
                    total_matrix_sum += data_matrix
                    valid_file_count += 1
                    # print(f"已累加: {mat_filename}") 
            else:
                print(f"忽略: {mat_filename} (未找到 {expected_shape} 矩阵)")

        except Exception as e:
            print(f"错误: 处理 {mat_filename} 时发生异常: {e}")

    # 7. 汇总报告
    print("\n" + "="*40)
    print(f"处理汇总报告:")
    print(f"1. 找到文件总数: {len(mat_files)}")
    print(f"2. 有效文件数量: {valid_file_count}")
    print(f"3. 包含 NaN 被排除: {nan_file_count}")
    
    if nan_file_count > 0:
        print("\n以下文件因包含 NaN 被排除:")
        for name in nan_files_list:
            print(f" - {name}")
    print("="*40 + "\n")

    # 8. 检查是否有有效文件进行计算
    if valid_file_count == 0:
        print(f"错误: 没有有效的矩阵用于计算 (所有文件都包含 NaN 或格式不对)。")
        print(f"未生成 {output_filename}。")
        return

    # 9. 计算平均值
    print(f"正在计算 {valid_file_count} 个文件的平均值...")
    
    # 逐元素相除
    average_matrix = total_matrix_sum / valid_file_count

    # 10. 保存为 .csv 文件
    # 注意：Fisher-Z 变换矩阵对角线通常为 Inf，保存到 CSV 会显示为 inf
    print(f"正在保存结果到 {output_filename}...")
    np.savetxt(output_filename, average_matrix, delimiter=',')
    
    print(f"完成！已生成 {output_filename}。")

# --- 运行脚本 ---
if __name__ == "__main__":
    calculate_average_matrix()