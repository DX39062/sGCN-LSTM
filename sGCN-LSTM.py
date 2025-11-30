import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 第一部分: 结构门控 GCN (经典版本)
# ==========================================

class StructureGatedGCN(nn.Module):
    def __init__(self, n_nodes=116, feature_dim=8):
        super(StructureGatedGCN, self).__init__()
        self.n_nodes = n_nodes
        self.feature_dim = feature_dim
        
        # 1. 结构门控网络 (sMRI -> Gate)
        # 学习 "灰质体积(GMV) -> 连接可靠性权重" 的非线性映射
        self.struct_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 输出 0~1，表示完整度
        )
        
        # 2. fMRI 特征变换 (Feature Mapping)
        # 将 fMRI 标量信号升维
        self.fmri_linear = nn.Linear(1, feature_dim)
        
        # Dropout 防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, fmri, smri_gmv, adj_static):
        """
        参数:
        fmri: (Batch, Time, Nodes) - fMRI 时序数据
        smri_gmv: (Batch, Nodes) - sMRI 灰质体积数据
        adj_static: (Nodes, Nodes) - 静态平均功能连接矩阵
        """
        B, T, N = fmri.shape
        
        # === A. 结构门控 (Structure Gating) ===
        # 1. 计算每个节点的完整度权重
        # smri_gmv.unsqueeze(-1) -> (B, N, 1)
        node_integrity = self.struct_gate(smri_gmv.unsqueeze(-1)) 
        
        # 2. 生成动态掩码 (Dynamic Mask)
        # 两个脑区的连接权重 = ROI_i完整度 * ROI_j完整度
        struct_mask = node_integrity @ node_integrity.transpose(1, 2)
        
        # 3. 融合生成个性化动态图
        # 广播 adj_static: (1, N, N) * (B, N, N)
        adj_dynamic = adj_static.unsqueeze(0) * struct_mask
        
        # === B. 批处理动态图卷积 (Batch GCN) ===
        # 1. fMRI 特征升维
        # (B, T, N) -> (B, T, N, 1) -> (B, T, N, K)
        fmri_feat = self.fmri_linear(fmri.unsqueeze(-1))
        fmri_feat = F.relu(fmri_feat)
        
        # 2. 执行图卷积
        # 物理含义: 聚合邻居信息，且受 sMRI 结构约束
        out = torch.einsum('bmn, btnf -> btmf', adj_dynamic, fmri_feat)
        out = F.relu(out)
        out = self.dropout(out)
        
        # 3. 图池化 (Readout)
        # 将 116 个节点的信息聚合为 feature_dim 维的全脑状态向量
        # (B, T, N, K) -> (B, T, K)
        out = torch.mean(out, dim=2)
        
        # [修改] 移除了原有的 torch.atan(out)。
        # 原代码使用 atan 是为了将数值限制在 (-pi/2, pi/2) 以适应量子角度编码。
        # 在经典网络中，ReLU 输出已经是特征，直接传递即可 (或可视情况加 Tanh)。
        
        return out 

# ==========================================
# 第二部分: 经典 LSTM 单元
# ==========================================
# 原代码中的 QLSTM 是单层的循环结构 (尽管 VQC 有 n_layers=2 的深度，但 LSTM 只有一层)。
# 这里我们直接使用 PyTorch 的 nn.LSTM 替代。

# ==========================================
# 第三部分: Fused MGRN 主模型 (经典版)
# ==========================================

class Fused_MGRN_Classic(nn.Module):
    def __init__(self, n_nodes=116, n_time_steps=140, n_classes=2):
        super(Fused_MGRN_Classic, self).__init__()
        
        # 超参数
        # [修改] 将 n_qubits 改为 hidden_dim。
        # 注意：为了与量子版本进行公平对比，这里默认仍设为 8。
        # 但在纯经典任务中，增加此维度 (如 32, 64) 通常能获得更好的性能。
        self.hidden_dim = 8 
        self.pool_kernel = 4 # 时间池化窗口大小 (140 -> 35)
        
        # 模块 1: 结构门控 GCN (融合引擎)
        self.struct_gcn = StructureGatedGCN(n_nodes=n_nodes, feature_dim=self.hidden_dim)
        
        # 模块 2: 经典 LSTM (时序引擎)
        # 替代了原有的 QLSTM
        self.lstm = nn.LSTM(
            input_size=self.hidden_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=1, # 原 QLSTM 逻辑上只有一层时间循环
            batch_first=True
        )
        
        # 模块 3: 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, fmri, smri, adj_static):
        """
        完整的前向传播路径
        """
        # 1. 结构门控与动态 GCN
        # 输出: (B, 140, hidden_dim)
        gcn_out = self.struct_gcn(fmri, smri, adj_static)
        
        # 2. 时间池化 (Temporal Pooling) 
        # (B, 140, 8) -> (B, 35, 8)
        gcn_out = gcn_out.permute(0, 2, 1) 
        gcn_out = F.avg_pool1d(gcn_out, kernel_size=self.pool_kernel)
        lstm_in = gcn_out.permute(0, 2, 1)
        
        # 3. 经典 LSTM 处理
        # 输入: (B, 35, 8)
        # output: (B, 35, 8) - 所有时间步的输出
        # (h_n, c_n): (num_layers, B, 8) - 最后一个时间步的隐状态
        self.lstm.flatten_parameters() # 优化显存
        _, (h_n, _) = self.lstm(lstm_in)
        
        # 取最后一层的最后一个时间步状态
        # h_n 形状为 (1, B, 8)，我们需要 (B, 8)
        final_feat = h_n[-1]
        
        # 4. 最终分类
        logits = self.classifier(final_feat)
        
        return logits

# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    # 模拟 Batch=2 的数据
    B = 2
    dummy_fmri = torch.randn(B, 140, 116)
    dummy_smri = torch.randn(B, 116) # 模拟归一化后的 GMV
    dummy_adj = torch.randn(116, 116)
    dummy_adj = (dummy_adj + dummy_adj.T) / 2 # 对称化
    
    # 实例化模型
    model = Fused_MGRN_Classic()
    print(model)
    
    # 前向传播
    output = model(dummy_fmri, dummy_smri, dummy_adj)
    
    print(f"模型输出形状: {output.shape}") # 应为 (2, 2)
    print("Fused MGRN (Classic) 构建成功。")