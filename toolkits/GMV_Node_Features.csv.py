import nibabel as nib
import numpy as np
import os
import pandas as pd
from nilearn.image import resample_to_img

def batch_extract_gmv(data_dir, atlas_path, output_csv="GMV_Node_Features.csv"):
    # 1. 检查模板是否存在
    if not os.path.exists(atlas_path):
        print(f"【错误】找不到模板文件: {atlas_path}")
        return

    print(f"正在加载 AAL 模板: {os.path.basename(atlas_path)}")
    atlas_img = nib.load(atlas_path)
    
    # 2. 扫描目录下所有 mwc1 开头的 .nii 文件 (灰质体积文件)
    # 注意：这里我们过滤掉以 's' 开头的 (smwc1)，因为那是平滑过的，通常用于 VBM 统计分析
    # 对于 GNN 特征提取，通常使用 mwc1 即可；如果想要平滑后的特征，也可以改用 smwc1
    files = [f for f in os.listdir(data_dir) if f.startswith('mwc1') and f.endswith('.nii')]
    files.sort()
    
    if len(files) == 0:
        print("【错误】未找到 mwc1 开头的文件！请确认路径是否正确。")
        return

    print(f"找到 {len(files)} 个受试者的灰质体积图 (mwc1)，开始处理...")
    
    all_features = []
    subject_ids = []

    # 3. 循环处理每个受试者
    for idx, file_name in enumerate(files):
        # 提取文件名作为 ID (去掉 mwc1 前缀可能更好看，这里暂时保留原名)
        subject_id = file_name.split('.')[0] 
        
        img_path = os.path.join(data_dir, file_name)
        print(f"[{idx+1}/{len(files)}] 处理中: {file_name} ...", end="\r")
        
        try:
            img = nib.load(img_path)
            
            # 自动重采样 (解决分辨率不一致问题)
            # 使用 nearest 插值保持模板标签的整数性质
            if img.shape != atlas_img.shape:
                resampled_atlas = resample_to_img(atlas_img, img, interpolation='nearest')
                atlas_data = resampled_atlas.get_fdata()
            else:
                atlas_data = atlas_img.get_fdata()
            
            img_data = img.get_fdata()
            
            # 4. 提取 116 个脑区的平均灰质体积
            roi_features = []
            # AAL 模板通常是从 1 到 116
            for roi_idx in range(1, 117): 
                mask = (atlas_data == roi_idx)
                voxels = img_data[mask]
                
                # 计算均值：代表该脑区的平均灰质体积特征
                if len(voxels) > 0:
                    mean_val = np.mean(voxels)
                else:
                    mean_val = 0.0
                
                roi_features.append(mean_val)
            
            all_features.append(roi_features)
            subject_ids.append(subject_id)
            
        except Exception as e:
            print(f"\n【错误】处理 {file_name} 失败: {e}")

    # 5. 保存结果到 CSV
    print(f"\n处理完成！正在保存结果到 {output_csv}...")
    
    column_names = [f"ROI_{i}" for i in range(1, 117)]
    df = pd.DataFrame(all_features, columns=column_names)
    df.insert(0, "Subject_ID", subject_ids)
    
    df.to_csv(output_csv, index=False)
    print("✅ GMV 特征提取成功！你可以直接用 Excel 打开查看。")

# ================= 配置路径 =================
# 你的分割结果文件夹 (根据你之前的截图)
data_directory = r"C:\Raw_Data\T1ImgNewSegment"

# 你的 AAL 模板路径 (请使用之前测试成功的那个 aal.nii)
atlas_file_path = r"D:\Software\DPABI_V7.0_230110\Templates\aal.nii"

# 运行提取
batch_extract_gmv(data_directory, atlas_file_path)