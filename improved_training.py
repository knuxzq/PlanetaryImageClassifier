import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from sklearn.utils import class_weight
import json
import random
from pathlib import Path
import time
from improved_model_architecture import CelestialModel, compile_model
from torch.utils.tensorboard import SummaryWriter


class CelestialDataset(Dataset):
    """自定义行星图像数据集"""
    def __init__(self, root_dir, split='train', img_size=(224, 224), transform=None):
        """
        参数:
            root_dir: 数据集根目录
            split: 'train', 'val', 或 'test'
            img_size: 图像大小
            transform: 图像转换
        """
        self.root_dir = Path(root_dir) / split
        self.img_size = img_size
        self.transform = transform
        self.samples = []
        self.class_indices = {}
        self.class_counts = {}
        
        # 获取所有类别
        self.classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(self.root_dir / d)])
        
        # 创建类别索引映射
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 收集所有样本
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            class_idx = self.class_to_idx[class_name]
            self.class_indices[class_name] = class_idx
            
            # 获取类别下的所有图片
            img_files = [f for f in os.listdir(class_dir) 
                         if os.path.isfile(class_dir / f) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # 记录每个类别的样本数
            self.class_counts[class_idx] = len(img_files)
            
            # 将样本添加到列表
            for img_file in img_files:
                self.samples.append((str(class_dir / img_file), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(img_size=(224, 224), split='train'):
    """
    根据不同数据集创建对应的数据增强
    
    参数:
        img_size: 目标图像大小
        split: 数据集类型，'train', 'val', 或 'test'
    
    返回:
        对应的转换
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    if split == 'train':
        # 训练集使用强数据增强
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomRotation(40),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    elif split == 'val':
        # 验证集使用轻度数据增强
        return transforms.Compose([
            transforms.Resize((int(img_size[0]*1.1), int(img_size[1]*1.1))),
            transforms.CenterCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        # 测试集只进行标准化
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            normalize,
        ])


def prepare_dataloaders(data_dir, img_size=(224, 224), batch_size=16, num_workers=4):
    """
    准备数据加载器
    
    参数:
        data_dir: 数据根目录
        img_size: 图像大小
        batch_size: 批次大小
        num_workers: 数据加载线程数
    
    返回:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
        class_indices: 类别索引映射
        class_weights: 类别权重
    """
    print(f"\n准备数据加载器...")
    print(f"数据目录: {data_dir}")
    print(f"图像尺寸: {img_size}")
    print(f"批次大小: {batch_size}")
    print(f"数据加载线程数: {num_workers}\n")
    
    # 创建数据集
    print("正在创建训练数据集...")
    train_dataset = CelestialDataset(
        root_dir=data_dir,
        split='train',
        img_size=img_size,
        transform=get_transforms(img_size, 'train')
    )
    
    print("正在创建验证数据集...")
    val_dataset = CelestialDataset(
        root_dir=data_dir,
        split='val',
        img_size=img_size,
        transform=get_transforms(img_size, 'val')
    )
    
    print("正在创建测试数据集...")
    test_dataset = CelestialDataset(
        root_dir=data_dir,
        split='test',
        img_size=img_size,
        transform=get_transforms(img_size, 'test')
    )
    
    # 创建数据加载器
    print("正在创建数据加载器...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # 获取类别索引映射
    class_indices = train_dataset.class_indices
    
    # 计算类别权重
    print("正在计算类别权重...")
    class_counts = np.array([train_dataset.class_counts.get(i, 0) for i in range(len(class_indices))])
    if len(class_counts) > 0 and np.min(class_counts) > 0:
        # sklearn方式计算权重
        classes = np.array(list(range(len(class_indices))))
        labels = []
        for _, label in train_dataset.samples:
            labels.append(label)
        
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        
        # 转换为PyTorch张量
        class_weights = torch.FloatTensor(class_weights)
    else:
        class_weights = torch.ones(len(class_indices))
    
    # 打印数据信息
    print(f"\n找到 {len(train_dataset)} 张训练图像，{len(class_indices)} 个类别")
    print(f"找到 {len(val_dataset)} 张验证图像")
    print(f"找到 {len(test_dataset)} 张测试图像")
    print(f"类别映射: {class_indices}")
    print(f"类别权重: {class_weights.tolist()}")
    
    # 保存类别映射到文件
    print("正在保存类别映射...")
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)
    
    # 可视化一些增强后的图像
    print("正在生成数据增强示例图像...")
    visualize_augmentations(train_loader, "augmentation_examples.png")
    
    return train_loader, val_loader, test_loader, class_indices, class_weights


def visualize_augmentations(data_loader, save_path=None, num_samples=5):
    """可视化数据增强效果"""
    print(f"正在准备 {num_samples} 个数据增强示例...")
    
    try:
        # 获取一批数据
        x_batch, _ = next(iter(data_loader))
        
        plt.figure(figsize=(15, 3 * num_samples))
        for i in range(min(num_samples, len(x_batch))):
            # 转换回PIL图像以便显示
            img = TF.to_pil_image(x_batch[i])
            
            plt.subplot(num_samples, 3, i * 3 + 1)
            plt.imshow(img)
            plt.title(f"样本 {i + 1}")
            plt.axis('off')
            
            # 再次获取增强后的图像
            x_batch2, _ = next(iter(data_loader))
            img2 = TF.to_pil_image(x_batch2[i % len(x_batch2)])
            
            plt.subplot(num_samples, 3, i * 3 + 2)
            plt.imshow(img2)
            plt.title(f"增强 1")
            plt.axis('off')
            
            # 第三次获取增强图像
            x_batch3, _ = next(iter(data_loader))
            img3 = TF.to_pil_image(x_batch3[i % len(x_batch3)])
            
            plt.subplot(num_samples, 3, i * 3 + 3)
            plt.imshow(img3)
            plt.title(f"增强 2")
            plt.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"增强示例已保存至 '{save_path}'")
        
        # 不自动显示图像，避免阻塞程序执行
        plt.close()
    except Exception as e:
        print(f"生成数据增强示例时出错: {e}")
        print("继续执行程序...")


def cosine_annealing_schedule(initial_lr, epoch, epochs, warmup_epochs=5):
    """余弦退火学习率计划，带预热"""
    if epoch < warmup_epochs:
        # 预热阶段：线性增加学习率
        return initial_lr * ((epoch + 1) / warmup_epochs)
    else:
        # 余弦退火阶段
        return initial_lr * 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))


class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """计算topk准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, class_weights=None):
    """训练一个周期"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    
    # 设置进度打印的频率
    print_freq = max(len(train_loader) // 20, 1)  # 更频繁地打印进度
    
    # 如果使用类权重，将其移到设备上
    if class_weights is not None:
        class_weights = class_weights.to(device)
        weighted_criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        weighted_criterion = criterion
    
    start_time = time.time()
    print(f"[训练进度]: ", end="", flush=True)
    
    for i, (images, target) in enumerate(train_loader):
        # 移动数据到设备
        images, target = images.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        output = model(images)
        loss = weighted_criterion(output, target)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        acc1, acc2 = accuracy(output, target, topk=(1, 2))
        
        # 更新统计
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top2.update(acc2.item(), images.size(0))
        
        # 打印进度
        if i % print_freq == 0:
            elapsed = time.time() - start_time
            progress = (i / len(train_loader)) * 100
            
            # 创建进度条
            progress_chars = int(progress / 5)
            progress_bar = "[" + "#" * progress_chars + "-" * (20 - progress_chars) + "]"
            
            print(f"\r[训练进度]: {progress_bar} {progress:.1f}% ({i}/{len(train_loader)}) - "
                  f"Time: {elapsed:.1f}s - "
                  f"Loss: {losses.val:.4f} ({losses.avg:.4f}) - "
                  f"Acc@1: {top1.val:.2f}% ({top1.avg:.2f}%) - "
                  f"Acc@2: {top2.val:.2f}% ({top2.avg:.2f}%)", end="", flush=True)
    
    # 结束一个周期后打印最终统计
    total_time = time.time() - start_time
    print(f"\r[完成]: 轮次 {epoch+1} 训练完成 - "
          f"耗时: {total_time:.1f}s - "
          f"平均损失: {losses.avg:.4f} - "
          f"平均Acc@1: {top1.avg:.2f}% - "
          f"平均Acc@2: {top2.avg:.2f}%")
    
    return {'loss': losses.avg, 'top1': top1.avg, 'top2': top2.avg}


def validate(model, val_loader, criterion, device):
    """在验证集上评估模型"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    
    print(f"[验证]: 在 {len(val_loader)} 批次的验证集上评估模型...", end="", flush=True)
    start_time = time.time()
    
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            images, target = images.to(device), target.to(device)
            
            # 前向传播
            output = model(images)
            loss = criterion(output, target)
            
            # 计算准确率
            acc1, acc2 = accuracy(output, target, topk=(1, 2))
            
            # 更新统计
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top2.update(acc2.item(), images.size(0))
            
            # 打印简单进度
            if i % 5 == 0:
                progress = (i / len(val_loader)) * 100
                progress_bar = "." * (i // 2)
                print(f"\r[验证]: {progress_bar} {progress:.1f}%", end="", flush=True)
    
    total_time = time.time() - start_time
    print(f"\r[验证完成]: 耗时: {total_time:.1f}s - Validation Acc@1: {top1.avg:.2f}% - Acc@2: {top2.avg:.2f}% - Loss: {losses.avg:.4f}")
    
    return {'loss': losses.avg, 'top1': top1.avg, 'top2': top2.avg}


def train_model(model, train_loader, val_loader, device, 
               epochs=100, initial_epochs=15, fine_tune_at=100,
               model_dir="model_checkpoints", class_weights=None,
               initial_learning_rate=0.001):
    """
    两阶段训练模型：先训练顶层，然后微调
    
    参数:
        model: PyTorch模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 运行设备
        epochs: 总训练轮数
        initial_epochs: 第一阶段训练轮数
        fine_tune_at: 微调时解冻的层数
        model_dir: 保存模型的目录
        class_weights: 类别权重
        initial_learning_rate: 初始学习率
    
    返回:
        model: 训练后的模型
        history: 合并的训练历史
    """
    print("\n" + "="*80)
    print(f"开始训练模型")
    print(f"总轮数: {epochs}，第一阶段轮数: {initial_epochs}")
    print(f"初始学习率: {initial_learning_rate}")
    print(f"模型保存目录: {model_dir}")
    print("="*80 + "\n")
    
    # 创建模型保存目录
    print(f"创建模型保存目录: {model_dir}")
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建TensorBoard日志
    log_dir = os.path.join(model_dir, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"创建TensorBoard日志目录: {log_dir}")
    writer = SummaryWriter(log_dir)
    
    # 初始化损失函数和优化器
    print("初始化损失函数和优化器")
    criterion = nn.CrossEntropyLoss()
    
    # 第一阶段：只训练分类层
    print("\n" + "-"*80)
    print("第1阶段: 训练顶层分类器...")
    print("-"*80)
    
    # 冻结特征提取器参数
    print("冻结特征提取器参数...")
    for param in model.feature_extractor.parameters():
        param.requires_grad = False
    
    # 只优化分类器参数
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_learning_rate,
        weight_decay=1e-5
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_annealing_schedule(
            initial_learning_rate, epoch, initial_epochs, warmup_epochs=3
        ) / initial_learning_rate
    )
    
    # 存储训练历史
    history = {
        'loss': [], 'top1': [], 'top2': [],
        'val_loss': [], 'val_top1': [], 'val_top2': [],
        'lr': []
    }
    
    best_acc = 0.0
    # 第一阶段训练
    for epoch in range(initial_epochs):
        # 训练一个轮次
        print(f"\nEpoch {epoch+1}/{initial_epochs} (Stage 1)")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 记录学习率
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 训练和验证
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, class_weights)
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        for k in train_metrics:
            history[k].append(train_metrics[k])
            history[f'val_{k}'].append(val_metrics[k])
            writer.add_scalar(f'Stage1/{k}', train_metrics[k], epoch)
            writer.add_scalar(f'Stage1/val_{k}', val_metrics[k], epoch)
        
        # 保存最佳模型
        if val_metrics['top1'] > best_acc:
            best_acc = val_metrics['top1']
            torch.save(model.state_dict(), 
                      os.path.join(model_dir, f'stage1_best_model.pth'))
            print(f"保存最佳模型，验证准确率: {best_acc:.2f}%")
    
    # 第二阶段：微调部分特征提取器
    print("\n" + "-"*80)
    print("第2阶段: 微调模型...")
    print("-"*80)
    
    # 解冻部分特征提取层
    # 对于efficientnet模型，解冻最后几个块
    if model.model_type == "efficientnet":
        print(f"解冻EfficientNet模型的最后 {fine_tune_at} 个块...")
        # 解冻最后几个块
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
            
        # 计算要解冻的层数
        feature_blocks = list(model.feature_extractor.children())
        num_blocks = len(feature_blocks)
        unfreeze_blocks = min(fine_tune_at, num_blocks)
        
        # 解冻最后几个块
        for i in range(num_blocks - unfreeze_blocks, num_blocks):
            for param in feature_blocks[i].parameters():
                param.requires_grad = True
    
    # 输出可训练的参数
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(f"总参数: {total_count:,}, 可训练参数: {trainable_count:,}, 比例: {trainable_count/total_count:.2%}")
    
    # 使用较小的学习率
    print(f"创建第2阶段优化器，学习率: {initial_learning_rate / 10:.6f}")
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=initial_learning_rate / 10,
        weight_decay=1e-5
    )
    
    # 新的学习率调度器
    print("创建第2阶段学习率调度器...")
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: cosine_annealing_schedule(
            initial_learning_rate / 10, epoch, epochs - initial_epochs, warmup_epochs=5
        ) / (initial_learning_rate / 10)
    )
    
    # 第二阶段训练
    best_acc = 0.0
    for epoch in range(epochs - initial_epochs):
        actual_epoch = epoch + initial_epochs
        print(f"\nEpoch {actual_epoch+1}/{epochs} (Stage 2)")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 记录学习率
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # 训练和验证
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, actual_epoch, class_weights)
        val_metrics = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        for k in train_metrics:
            history[k].append(train_metrics[k])
            history[f'val_{k}'].append(val_metrics[k])
            writer.add_scalar(f'Stage2/{k}', train_metrics[k], actual_epoch)
            writer.add_scalar(f'Stage2/val_{k}', val_metrics[k], actual_epoch)
        
        # 保存最佳模型
        if val_metrics['top1'] > best_acc:
            best_acc = val_metrics['top1']
            torch.save(model.state_dict(), 
                      os.path.join(model_dir, f'stage2_best_model.pth'))
            print(f"保存最佳模型，验证准确率: {best_acc:.2f}%")
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'celestial_classification_final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存至 {final_model_path}")
    
    return model, history


def plot_training_history(history, save_path='training_history.png'):
    """
    绘制训练历史图表
    
    参数:
        history: 训练历史字典
        save_path: 保存图表的路径
    """
    print("正在准备绘制训练历史图表...")
    
    # 创建3x2布局的图表
    plt.figure(figsize=(18, 12))
    
    # 重置字体设置，使用默认字体
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})
    
    # 绘制准确率
    plt.subplot(2, 3, 1)
    plt.plot(history['top1'], label='Train Accuracy')
    plt.plot(history['val_top1'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # 绘制损失
    plt.subplot(2, 3, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # 绘制Top-2准确率
    plt.subplot(2, 3, 3)
    plt.plot(history['top2'], label='Train Top-2 Accuracy')
    plt.plot(history['val_top2'], label='Validation Top-2 Accuracy')
    plt.title('Top-2 Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # 绘制学习率
    plt.subplot(2, 3, 4)
    plt.semilogy(history['lr'])  # 使用对数刻度
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate (log scale)')
    plt.grid(True)
    
    # 添加训练阶段标记
    stage1_epochs = len(history['lr']) // 3  # 估计第一阶段的轮数
    for i in range(1, 5):
        ax = plt.subplot(2, 3, i)
        ax.axvline(x=stage1_epochs, color='r', linestyle='--')
        ax.text(stage1_epochs + 0.5, ax.get_ylim()[1] * 0.9, 'Stage 2', color='r')
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"训练历史图表已保存至 '{save_path}'")
    plt.close()


def check_data_directory(data_dir):
    """检查数据目录是否存在和包含所需文件结构"""
    print(f"\n正在检查数据目录: {data_dir}")
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"错误: 数据目录 '{data_dir}' 不存在!")
        print("请提供一个有效的数据目录，或者创建一个示例数据集用于测试")
        return False
    
    print("√ 数据目录存在")
    
    # 检查训练、验证和测试子目录
    required_dirs = ['train', 'val', 'test']
    print("正在检查必需的子目录...")
    
    for subdir in required_dirs:
        if not (data_path / subdir).exists():
            print(f"错误: '{subdir}' 子目录不存在!")
            return False
        print(f"√ 找到 '{subdir}' 子目录")
    
    # 检查是否有类别子目录
    train_dir = data_path / 'train'
    class_dirs = [d for d in train_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"错误: 训练目录中没有找到类别子目录!")
        return False
    
    # 计算总图像数
    total_images = 0
    print(f"\n正在扫描类别子目录和图像...")
    
    # 输出找到的类别
    print(f"找到 {len(class_dirs)} 个类别:")
    for cls_dir in class_dirs:
        jpg_images = list(cls_dir.glob('*.jpg'))
        png_images = list(cls_dir.glob('*.png'))
        num_samples = len(jpg_images) + len(png_images)
        total_images += num_samples
        print(f"  - {cls_dir.name}: {num_samples} 张图像")
    
    print(f"√ 数据目录结构有效")
    print(f"√ 共找到 {total_images} 张图像，分布在 {len(class_dirs)} 个类别中")
    
    return True


# 主程序
if __name__ == "__main__":
    print("\n" + "="*80)
    print("行星图像分类训练程序")
    print("="*80)
    
    # 设置随机种子
    seed = 42
    print(f"\n正在设置随机种子: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 检查CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n检查设备信息:")
    print(f"使用设备: {device}")
    print(f"PyTorch 版本: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 设备数量: {torch.cuda.device_count()}")
        print(f"当前 GPU 设备名称: {torch.cuda.get_device_name(0)}")
        # 显示可用GPU内存
        try:
            free_memory, total_memory = torch.cuda.mem_get_info()
            print(f"GPU内存: 可用 {free_memory/1024**3:.1f}GB / 总共 {total_memory/1024**3:.1f}GB")
        except:
            print("无法获取GPU内存信息")
    
    # 数据目录
    data_dir = "./data/celestial_dataset"  # 修改为您实际的数据目录
    
    # 检查数据目录
    if not check_data_directory(data_dir):
        # 如果目录不存在或结构不正确，退出程序
        print("程序终止: 请修正数据目录问题后再运行。")
        exit(1)
    
    # 设置较小的批次大小以避免内存问题，并使用更大的图像尺寸
    img_size = (299, 299)  # 更大的图像尺寸可以提供更多细节
    batch_size = 16  # 较小的批次大小
    
    print("\n" + "="*80)
    print("训练配置:")
    print(f"图像尺寸: {img_size}")
    print(f"批次大小: {batch_size}")
    print(f"总训练轮数: 100 (第一阶段: 15, 第二阶段: 85)")
    print(f"初始学习率: 0.0005")
    print("模型类型: EfficientNet")
    print("="*80 + "\n")
    
    # 准备数据加载器
    train_loader, val_loader, test_loader, class_indices, class_weights = prepare_dataloaders(
        data_dir, img_size=img_size, batch_size=batch_size
    )
    
    # 构建EfficientNet模型
    num_classes = len(class_indices)
    print(f"\n正在构建EfficientNet模型，类别数: {num_classes}...")
    model = CelestialModel(
        num_classes,
        model_type="efficientnet",  # 使用EfficientNet
        dropout_rate=0.5  # 增加dropout以减少过拟合
    )
    
    # 将模型移到设备上
    print(f"将模型移至设备: {device}")
    model = model.to(device)
    
    # 训练模型，使用余弦退火学习率和更长的训练周期
    initial_lr = 0.0005  # 使用较小的初始学习率
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=100,  # 增加总轮数
        initial_epochs=15,  # 第一阶段轮数
        fine_tune_at=5,  # 微调更多层
        class_weights=class_weights,  # 使用类别权重
        initial_learning_rate=initial_lr
    )
    
    # 绘制训练历史
    print("\n正在生成训练历史图表...")
    plot_training_history(history)
    
    print("\n" + "="*80)
    print("Model training completed! / 训练完成！")
    print("="*80 + "\n")