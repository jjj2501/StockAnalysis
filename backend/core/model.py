import torch
import torch.nn as nn

class HybridModel(nn.Module):
    """
    Transformers + LSTM 混合模型
    结合 Transformer 获取全局注意力特征和 LSTM 获取局部时序特征
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        super(HybridModel, self).__init__()
        
        self.d_model = hidden_dim
        
        # 特征映射层: 将输入特征维度映射到隐藏层维度
        self.embedding = nn.Linear(input_dim, self.d_model)
        
        # Transformer Encoder: 用于捕捉长距离依赖
        # d_model: 特征维度
        # nhead: 多头注意力头数 (假设为4)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # LSTM Layer: 用于捕捉时序动态
        self.lstm = nn.LSTM(self.d_model, hidden_dim, num_layers=1, batch_first=True)
        
        # 全连接输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        
        # 1. 嵌入/投影
        x = self.embedding(x) # -> (batch_size, seq_len, hidden_dim)
        x = self.dropout(x)
        
        # 2. Transformer 编码
        x = self.transformer_encoder(x) # -> (batch_size, seq_len, hidden_dim)
        
        # 3. LSTM 处理
        # LSTM output: (output, (h_n, c_n))
        # output shape: (batch_size, seq_len, hidden_dim)
        x, _ = self.lstm(x)
        
        # 4. 取最后一个时间步的输出进行预测
        out = x[:, -1, :] # -> (batch_size, hidden_dim)
        
        # 5. 输出层
        prediction = self.fc(out) # -> (batch_size, output_dim)
        
        return prediction
