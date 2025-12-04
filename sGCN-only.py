import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import time

# ==========================================
# 0. 日志配置 (Setup)
# ==========================================

def setup_logger():
    # 确保日志目录存在
    os.makedirs("./logging", exist_ok=True)
    
    # 创建带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"./logging/training_log_gcnonly_{timestamp}.txt"
    
    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_filename), # 输出到文件
            logging.StreamHandler()            # 输出到控制台
        ]
    )
    return log_filename

# ==========================================
# 第一部分: 数据加载
# ==========================================

def load_adjacency_matrix(adj_file, n_nodes=116):
    logging.info(f"--- 正在加载邻接矩阵: {adj_file} ---")
    try:
        adj_np = pd.read_csv(adj_file, header=None).values.astype(np.float32)
        if adj_np.shape != (n_nodes, n_nodes):
            raise ValueError(f"邻接矩阵形状错误: 期望 ({n_nodes}, {n_nodes}), 实际 {adj_np.shape}")
        
        np.fill_diagonal(adj_np, 0)
        adj_torch = torch.tensor(adj_np)
        
        A_tilde = adj_torch + torch.eye(n_nodes)
        degrees = torch.sum(A_tilde, dim=1)
        D_inv_sqrt = torch.pow(degrees, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        D_mat = torch.diag(D_inv_sqrt)
        adj_normalized = D_mat @ A_tilde @ D_mat
        
        logging.info(" 邻接矩阵加载并归一化完成。")
        return adj_normalized
    except Exception as e:
        logging.error(f"加载邻接矩阵时出错: {e}")
        raise e

class MultimodalDataset(Dataset):
    def __init__(self, fmri_dir, smri_file, label_file, n_time_steps=140, n_nodes=116):
        self.fmri_dir = fmri_dir
        self.n_time_steps = n_time_steps
        self.n_nodes = n_nodes
        
        # 1. 解析标签
        self.label_map = {}
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if ':' in line:
                        parts = line.strip().split(':')
                        clean_id = parts[0].strip().replace("'", "").replace('"', "")
                        clean_label = int(parts[1].strip().replace(",", ""))
                        self.label_map[clean_id] = clean_label
        except Exception as e:
            logging.error(f"解析标签文件失败: {e}")
            raise e
            
        # 2. 加载 sMRI 并清洗
        if not os.path.exists(smri_file):
            raise FileNotFoundError(f"找不到 sMRI 文件: {smri_file}")
            
        # 读取原始数据
        logging.info(f"--- 正在加载并清洗 sMRI 数据: {smri_file} ---")
        self.smri_df = pd.read_csv(smri_file)
        
        # 设置索引 (Subject_ID)
        if 'Subject_ID' in self.smri_df.columns:
            self.smri_df = self.smri_df.set_index('Subject_ID')
        else:
            self.smri_df = self.smri_df.set_index(self.smri_df.columns[0])

        # ==========================================
        # 数据清洗逻辑: 剔除含 0 的样本
        # ==========================================
        initial_count = len(self.smri_df)
        
        # 检查每一行，如果该行所有列(脑区)都不为0，则保留
        valid_mask = (self.smri_df != 0).all(axis=1)
        self.smri_df = self.smri_df[valid_mask]
        
        dropped_count = initial_count - len(self.smri_df)
        if dropped_count > 0:
            logging.warning(f" 警告: 已剔除 {dropped_count} 个含有 0 值(异常脑区)的样本。")
            logging.info(f"   剩余有效 sMRI 样本数: {len(self.smri_df)}")
        else:
            logging.info(" sMRI 数据质量良好，未发现含 0 值的样本。")

        # 3. 匹配数据 (fMRI + sMRI + Label)
        self.data_list = [] 
        self.labels_list = [] 
        
        if not os.path.exists(fmri_dir):
            raise FileNotFoundError(f"找不到 fMRI 文件夹: {fmri_dir}")
            
        fmri_files = sorted([f for f in os.listdir(fmri_dir) if f.endswith('.csv')])
        
        match_count = 0
        for f_file in fmri_files:
            long_id = os.path.splitext(f_file)[0]
            parts = long_id.split('_')
            if len(parts) >= 3:
                short_id = "_".join(parts[:3])
            else:
                short_id = long_id
            
            # 只有当 ID 同时存在于 清洗后的 sMRI表 和 标签表 中才匹配
            if short_id in self.label_map and long_id in self.smri_df.index:
                label = self.label_map[short_id]
                f_path = os.path.join(fmri_dir, f_file)
                self.data_list.append((f_path, long_id, label))
                self.labels_list.append(label)
                match_count += 1
                
        logging.info(f" 最终匹配完成: 共有 {match_count} 个完整且有效的样本参与训练。")
        if match_count == 0:
            raise RuntimeError("数据匹配失败，请检查是否所有样本都被清洗掉了？")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        f_path, subject_id, label = self.data_list[idx]
        
        try:
            # 读取 fMRI
            fmri_data = pd.read_csv(f_path, header=None).values.astype(np.float32)
            if fmri_data.shape != (self.n_time_steps, self.n_nodes):
                if fmri_data.shape == (self.n_nodes, self.n_time_steps):
                    fmri_data = fmri_data.T
                elif fmri_data.shape[0] > self.n_time_steps:
                    fmri_data = fmri_data[:self.n_time_steps, :]
                else:
                    padding = np.zeros((self.n_time_steps - fmri_data.shape[0], self.n_nodes))
                    fmri_data = np.vstack([fmri_data, padding])

            scaler = StandardScaler()
            fmri_data = scaler.fit_transform(fmri_data) 

            # 读取 sMRI
            smri_row = self.smri_df.loc[subject_id].values.astype(np.float32)
            if smri_row.shape[0] != self.n_nodes:
                 smri_row = smri_row[:self.n_nodes]
            
            # 归一化
            if np.std(smri_row) > 0:
                smri_row = (smri_row - np.mean(smri_row)) / np.std(smri_row)
            else:
                smri_row = smri_row - np.mean(smri_row)

            return (
                torch.tensor(fmri_data, dtype=torch.float32), 
                torch.tensor(smri_row, dtype=torch.float32), 
                torch.tensor(label, dtype=torch.long)
            )
        except Exception as e:
            logging.error(f"读取数据出错 (ID: {subject_id}): {e}")
            return (torch.zeros(self.n_time_steps, self.n_nodes), 
                    torch.zeros(self.n_nodes), 
                    torch.tensor(0, dtype=torch.long))

# ==========================================
# 第二部分: 仅包含 GCN 的模型
# ==========================================

class StructureGatedGCN(nn.Module):
    """
    残差 + BN + 结构门控
    """
    def __init__(self, n_nodes=116, feature_dim=64):
        super(StructureGatedGCN, self).__init__()
        self.n_nodes = n_nodes
        
        self.struct_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.fmri_linear = nn.Linear(1, feature_dim)
        self.residual_proj = nn.Linear(1, feature_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, fmri, smri_gmv, adj_static):
        # 结构门控
        node_integrity = self.struct_gate(smri_gmv.unsqueeze(-1)) 
        struct_mask = (node_integrity @ node_integrity.transpose(1, 2)) + 0.1
        adj_dynamic = adj_static.unsqueeze(0) * struct_mask
        
        # GCN
        fmri_feat = self.fmri_linear(fmri.unsqueeze(-1)) 
        fmri_feat = F.relu(fmri_feat)
        
        out = torch.einsum('bmn, btnf -> btmf', adj_dynamic, fmri_feat)
        
        # 残差连接
        residual = self.residual_proj(fmri.unsqueeze(-1))
        out = out + residual 
        
        out = F.relu(out)
        out = self.dropout(out)
        
        # 节点池化: (Batch, Time, Nodes, Feat) -> (Batch, Time, Feat)
        out = torch.mean(out, dim=2) 
        
        return out 

class GCN_Only_Model(nn.Module):
    """
    无 LSTM 版本：直接对时间维度取平均
    """
    def __init__(self, n_nodes=116, n_classes=2):
        super(GCN_Only_Model, self).__init__()
        self.hidden_dim = 64  
        
        # GCN 模块
        self.struct_gcn = StructureGatedGCN(n_nodes=n_nodes, feature_dim=self.hidden_dim)
        
        # Batch Normalization (直接对特征维度归一化)
        self.bn_final = nn.BatchNorm1d(self.hidden_dim)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fmri, smri, adj_static):
        # 1. GCN 提取特征
        # 输出形状: (Batch, Time, Hidden_Dim)
        gcn_out = self.struct_gcn(fmri, smri, adj_static)
        
        # 2. 全局时间平均池化 (Global Average Pooling over Time)
        # 将时序维度压缩：(Batch, 140, 64) -> (Batch, 64)
        feat = torch.mean(gcn_out, dim=1)
        
        # 3. BN 和 分类
        feat = self.bn_final(feat)
        logits = self.classifier(feat)
        
        return logits

# ==========================================
# 第三部分: 训练流程 (五折 CV)
# ==========================================

def train_k_fold():
    # --- 初始化日志 ---
    log_file = setup_logger()
    
    # 记录开始时间
    start_time = datetime.now()
    logging.info(f"==========================================")
    logging.info(f" 训练任务开始 (GCN Only)")
    logging.info(f" 开始时间: {start_time}")
    logging.info(f" 日志文件: {log_file}")
    logging.info(f"==========================================")

    # --- 配置 ---
    base_path = "./"
    fmri_dir = os.path.join(base_path, "datasets", "fMRI")
    smri_file = os.path.join(base_path, "datasets", "GMV_Node_Features.csv")
    label_file = os.path.join(base_path, "datasets", "labels.csv")
    adj_file = os.path.join(base_path, "datasets", "FC.csv")
    
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001 
    NUM_EPOCHS = 80
    K_FOLDS = 5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f" Device: {device} (Mode: GCN Only - No Early Stopping)")

    # 1. 数据准备
    if not os.path.exists(adj_file):
        logging.error(f"Missing: {adj_file}")
        return
    adj_static = load_adjacency_matrix(adj_file).to(device)
    
    full_dataset = MultimodalDataset(fmri_dir, smri_file, label_file)
    all_labels = np.array(full_dataset.labels_list)
    all_indices = np.arange(len(full_dataset))

    # 2. 交叉验证
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    fold_results = []
    
    logging.info(f" 开始 {K_FOLDS} 折交叉验证 (GCN Only, Full Epochs)...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels)):
        logging.info(f"\n=== Fold {fold+1}/{K_FOLDS} 开始 ===")
        fold_start_time = time.time()
        
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        # 加权采样
        y_train = all_labels[train_idx]
        class_counts = np.bincount(y_train)
        class_weights = np.nan_to_num(1. / class_counts, posinf=0.0)
        sample_weights = class_weights[y_train]
        
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).float(),
            num_samples=len(train_idx),
            replacement=True
        )
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 初始化模型 (使用 GCN_Only_Model)
        model = GCN_Only_Model(n_nodes=116, n_classes=2).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
        
        best_fold_bacc = 0.0
        best_fold_acc = 0.0
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            
            for fmri, smri, labels in train_loader:
                fmri, smri, labels = fmri.to(device), smri.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(fmri, smri, adj_static)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                running_loss += loss.item()

            # 验证
            model.eval()
            preds_list = []
            targets_list = []
            
            with torch.no_grad():
                for fmri, smri, labels in val_loader:
                    fmri, smri, labels = fmri.to(device), smri.to(device), labels.to(device)
                    outputs = model(fmri, smri, adj_static)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    preds_list.extend(predicted.cpu().numpy())
                    targets_list.extend(labels.cpu().numpy())
            
            epoch_acc = np.mean(np.array(preds_list) == np.array(targets_list)) * 100
            epoch_bacc = balanced_accuracy_score(targets_list, preds_list) * 100
            
            # 仅记录最佳
            if epoch_bacc > best_fold_bacc:
                best_fold_bacc = epoch_bacc
                best_fold_acc = epoch_acc
                logging.info(f"  [Fold {fold+1}] Epoch {epoch+1}: B-Acc {epoch_bacc:.2f}% (New Best)")
                # 保存最佳权重
                #torch.save(model.state_dict(), f"./save/best_gcn_fold_{fold+1}.pth")
            
            # 每10轮打印一次日志
            if (epoch+1) % 10 == 0:
                logging.info(f"  [Fold {fold+1}] Epoch {epoch+1}: Loss {running_loss/len(train_loader):.4f} | Val B-Acc: {epoch_bacc:.2f}%")

        fold_duration = time.time() - fold_start_time
        logging.info(f" Fold {fold+1} 完成. 耗时: {fold_duration/60:.2f}分. Best Balanced Acc: {best_fold_bacc:.2f}% (Acc: {best_fold_acc:.2f}%)")
        fold_results.append({'fold': fold+1, 'bacc': best_fold_bacc, 'acc': best_fold_acc})

    # 汇总
    logging.info("\n" + "="*35)
    logging.info("   GCN-Only (No LSTM) Results   ")
    logging.info("="*35)
    avg_bacc = sum([r['bacc'] for r in fold_results]) / K_FOLDS
    avg_acc = sum([r['acc'] for r in fold_results]) / K_FOLDS
    
    for res in fold_results:
        logging.info(f"Fold {res['fold']}: Balanced Acc = {res['bacc']:.2f}% | Acc = {res['acc']:.2f}%")
        
    logging.info("-" * 35)
    logging.info(f"Avg Balanced Acc: {avg_bacc:.2f}%")
    logging.info(f"Avg Accuracy    : {avg_acc:.2f}%")
    logging.info("="*35)

    # 记录结束时间
    end_time = datetime.now()
    duration = end_time - start_time
    logging.info(f"==========================================")
    logging.info(f" 训练任务结束")
    logging.info(f" 结束时间: {end_time}")
    logging.info(f" 总耗时: {duration}")
    logging.info(f"==========================================")
    print(f"训练日志已保存至: {log_file}")

if __name__ == "__main__":
    train_k_fold()