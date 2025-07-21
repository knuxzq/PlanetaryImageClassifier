import os
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# from torchsummary import summary  # 暂时注释掉，使用其他方式打印模型信息


class CelestialModel(nn.Module):
    def __init__(self, num_classes, model_type="efficientnet", dropout_rate=0.5):
        """
        构建用于天体分类的模型，支持多种基础模型

        参数:
            num_classes: 分类类别数量
            model_type: 模型类型 ("resnet50", "efficientnet", "ensemble")
            dropout_rate: Dropout率，用于正则化
        """
        super(CelestialModel, self).__init__()
        self.model_type = model_type
        
        if model_type == "resnet50":
            # 使用预训练的ResNet50
            base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            num_features = base_model.fc.in_features
            
            # 移除最后的全连接层
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            
            # 添加分类头
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
            
            # 冻结特征提取器参数
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
        elif model_type == "efficientnet":
            # 使用预训练的EfficientNet-B3
            base_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            num_features = base_model.classifier[1].in_features
            
            # 移除最后的分类层
            self.feature_extractor = base_model.features
            
            # 添加分类头 (不使用BatchNorm1d以避免小批量大小问题)
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(num_features, 1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),  # 第二层使用较小的dropout
                nn.Linear(512, num_classes)
            )
            
            # 冻结特征提取器参数
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                
        elif model_type == "ensemble":
            # 使用两个不同的基础模型
            resnet_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            efficient_model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
            
            # 获取特征提取器
            self.resnet_features = nn.Sequential(*list(resnet_model.children())[:-1])
            self.efficient_features = efficient_model.features
            
            # 获取特征维度
            resnet_features_dim = resnet_model.fc.in_features
            efficient_features_dim = efficient_model.classifier[1].in_features
            combined_features_dim = resnet_features_dim + efficient_features_dim
            
            # 添加分类头 (移除BatchNorm1d)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(combined_features_dim, 1024),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(512, num_classes)
            )
            
            # 冻结特征提取器参数
            for param in self.resnet_features.parameters():
                param.requires_grad = False
            for param in self.efficient_features.parameters():
                param.requires_grad = False
                
            # 添加池化层
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
            
    def forward(self, x):
        if self.model_type == "resnet50" or self.model_type == "efficientnet":
            features = self.feature_extractor(x)
            return self.classifier(features)
        elif self.model_type == "ensemble":
            # 从每个模型提取特征
            resnet_out = self.resnet_features(x)
            efficient_out = self.efficient_features(x)
            
            # 应用池化
            resnet_out = self.avg_pool(resnet_out)
            efficient_out = self.avg_pool(efficient_out)
            
            # 展平并拼接特征
            resnet_out = torch.flatten(resnet_out, 1)
            efficient_out = torch.flatten(efficient_out, 1)
            combined = torch.cat([resnet_out, efficient_out], dim=1)
            
            # 分类
            return self.classifier(combined)


def compile_model(model, learning_rate=0.001, device='cuda'):
    """
    设置模型的优化器和损失函数

    参数:
        model: 要编译的模型
        learning_rate: 学习率
        device: 运行设备 ('cuda' 或 'cpu')

    返回:
        model: 模型
        optimizer: 优化器
        criterion: 损失函数
    """
    # 移动模型到指定设备
    model = model.to(device)
    
    # 配置优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-07,
        weight_decay=1e-5
    )
    
    # 配置损失函数
    criterion = nn.CrossEntropyLoss()
    
    return model, optimizer, criterion


def print_model_info(model):
    """简单打印模型信息，无需依赖外部库"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型信息:")
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"冻结参数量: {total_params - trainable_params:,}")
    
    # 打印模型结构
    print("\n模型结构:")
    print(model)


def test_model(model, device, batch_size=4):
    """测试模型的简单前向传播"""
    print("\n测试模型前向传播:")
    
    # 创建随机测试数据
    test_input = torch.randn(batch_size, 3, 224, 224).to(device)
    print(f"测试输入形状: {test_input.shape}")
    
    # 切换到评估模式
    model.eval()
    
    # 前向传播
    with torch.no_grad():
        outputs = model(test_input)
    
    print(f"输出形状: {outputs.shape}")
    
    # 获取预测类别
    _, predicted = torch.max(outputs, 1)
    print(f"预测类别: {predicted}")
    
    return outputs


# 示例用法
if __name__ == "__main__":
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 设备数量: {torch.cuda.device_count()}")
        print(f"当前 GPU 设备名称: {torch.cuda.get_device_name(0)}")
    
    print("\n构建模型中...")
    # 构建模型示例
    num_classes = 7
    model = CelestialModel(num_classes, model_type="efficientnet", dropout_rate=0.5)
    print(f"模型类型: {model.model_type}")
    
    print("\n配置优化器和损失函数...")
    # 编译模型
    model, optimizer, criterion = compile_model(model, learning_rate=0.001, device=device)
    
    # 打印模型信息
    print_model_info(model)
    
    # 测试模型
    test_model(model, device, batch_size=4)
    
    print("\n脚本执行完成!")