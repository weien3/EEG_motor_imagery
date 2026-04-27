import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    """
    EEGNet for Motor Imagery Classification
    """
    
    def __init__(self, n_channels=64, n_classes=3, n_timepoints=512, 
                 F1=8, D=2, F2=16, dropout_rate=0.25):
        super(EEGNet, self).__init__()
        
        # Block 1: 时间卷积 + 深度卷积
        # 输入: (batch, 1, n_channels, n_timepoints)
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding='same')
        self.bn1 = nn.BatchNorm2d(F1)
        
        # 深度卷积 (空间卷积)
        self.depthwise_conv = nn.Conv2d(F1, F1 * D, (n_channels, 1), 
                                        groups=F1, padding='valid')
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.avgpool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Block 2: 可分离卷积
        self.sep_conv1 = nn.Conv2d(F1 * D, F1 * D, (1, 16), 
                                   groups=F1 * D, padding='same')
        self.sep_conv2 = nn.Conv2d(F1 * D, F2, 1, padding='same')
        self.bn3 = nn.BatchNorm2d(F2)
        self.avgpool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 使用全局平均池化，避免计算维度
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(F2, n_classes)  # 全局平均池化后特征维度是 F2
        )
        
    def forward(self, x):
        """
        前向传播
        """
        # 确保输入有4个维度
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, channels, time) -> (batch, 1, channels, time)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.avgpool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.sep_conv1(x)
        x = self.sep_conv2(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.avgpool2(x)
        x = self.dropout2(x)
        
        # 全局平均池化
        x = self.global_avg_pool(x)
        
        # 分类
        x = self.classifier(x)
        
        return x