"""
宇宙天体分类模型 - 评估与可视化脚本 (PyTorch版本)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import json
import argparse
import seaborn as sns
import pandas as pd
from pathlib import Path
import cv2
from improved_model_architecture import CelestialModel

# 设置全局字体为Arial，避免中文字体问题
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial']})


class CelestialDataset(Dataset):
    """自定义行星图像数据集"""
    def __init__(self, root_dir, split='test', img_size=(224, 224), transform=None):
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
            
            # 将样本添加到列表
            for img_file in img_files:
                self.samples.append((str(class_dir / img_file), class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        # 调整原始图像大小
        orig_image = image.resize(self.img_size)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, np.array(orig_image) / 255.0


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='行星分类模型评估脚本 (PyTorch版本)')

    parser.add_argument('--model_path', type=str, 
                        default="model_checkpoints/celestial_classification_final_model.pth",
                        help='模型文件路径 (.pth)')
    parser.add_argument('--data_dir', type=str, default="./data/celestial_dataset",
                        help='数据集目录路径')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='评估结果输出目录')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--img_size', type=int, default=299,
                        help='图像尺寸')
    parser.add_argument('--model_type', type=str, default='efficientnet',
                        help='模型类型 (efficientnet, resnet)')

    return parser.parse_args()


def prepare_test_data(data_dir, img_size=(299, 299), batch_size=16):
    """准备测试数据"""
    # 标准化变换
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建测试数据集
    test_dataset = CelestialDataset(
        root_dir=data_dir,
        split='test',
        img_size=img_size,
        transform=test_transform
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 加载类别索引
    class_indices = test_dataset.class_indices
    
    # 保存类别映射到文件
    with open('class_indices.json', 'w') as f:
        json.dump(class_indices, f)
        
    return test_loader, class_indices


def evaluate_model(model, test_loader, class_names, device, output_dir='evaluation_results'):
    """评估模型性能"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置模型为评估模式
    model.eval()
    
    # 收集预测和真实标签
    y_true = []
    y_pred = []
    y_pred_probs = []
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    
    print("Evaluating model...")
    
    # 关闭梯度计算以加速评估
    with torch.no_grad():
        for inputs, targets, _ in test_loader:
            # 将数据移动到指定设备
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # 收集真实标签和预测
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_pred_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
    
    # 计算平均损失和准确率
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    
    print("=== Model Evaluation Results ===")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # 转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_probs = np.array(y_pred_probs)
    
    # 分类报告
    class_report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(class_report).transpose()
    
    print("\nClassification Report:")
    print(report_df)
    
    # 保存分类报告
    report_path = os.path.join(output_dir, 'classification_report.csv')
    report_df.to_csv(report_path)
    print(f"Classification report saved to {report_path}")
    
    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    # 保存混淆矩阵
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")
    plt.close()
    
    # 绘制归一化混淆矩阵
    plt.figure(figsize=(12, 10))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format='.2f')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    
    # 保存归一化混淆矩阵
    cm_norm_path = os.path.join(output_dir, 'normalized_confusion_matrix.png')
    plt.savefig(cm_norm_path)
    print(f"Normalized confusion matrix saved to {cm_norm_path}")
    plt.close()
    
    # 计算每个类别的准确率
    class_correct = {}
    for i, class_name in enumerate(class_names):
        class_indices = np.where(y_true == i)[0]
        if len(class_indices) > 0:
            correct = np.sum(y_pred[class_indices] == i)
            accuracy = correct / len(class_indices)
            class_correct[class_name] = accuracy
    
    # 绘制每个类别的准确率
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_correct.keys()), y=list(class_correct.values()))
    plt.title('Per-Class Accuracy')
    plt.ylim([0, 1])
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存类别准确率图
    class_acc_path = os.path.join(output_dir, 'class_accuracy.png')
    plt.savefig(class_acc_path)
    print(f"Per-class accuracy plot saved to {class_acc_path}")
    plt.close()
    
    return y_true, y_pred, y_pred_probs


def visualize_model_predictions(model, test_loader, class_names, device, num_samples=10, output_dir='evaluation_results'):
    """可视化模型预测结果"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置模型为评估模式
    model.eval()
    
    # 收集图像、真实标签和预测ll
    images = []
    true_labels = []
    pred_labels = []
    probabilities = []
    
    print("\nGenerating prediction visualizations...")
    
    # 关闭梯度计算
    with torch.no_grad():
        for inputs, targets, orig_images in test_loader:
            # 将输入移动到设备
            inputs = inputs.to(device)
            
            # 前向传播
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            # 收集数据
            for i in range(inputs.size(0)):
                images.append(orig_images[i])
                true_labels.append(targets[i].item())
                pred_labels.append(preds[i].cpu().item())
                probabilities.append(probs[i].cpu().numpy())
                
                if len(images) >= num_samples:
                    break
                    
            if len(images) >= num_samples:
                break
    
    # 确保我们有足够的样本
    images = images[:num_samples]
    true_labels = true_labels[:num_samples]
    pred_labels = pred_labels[:num_samples]
    probabilities = probabilities[:num_samples]
    
    print(f"Collected {len(images)} samples for prediction visualization")
    print(f"Classes in visualization: {[class_names[label] for label in true_labels]}")
    
    # 可视化
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(images):
            # 显示图像
            ax.imshow(images[i])
            
            # 获取真实和预测标签
            true_label = class_names[true_labels[i]]
            pred_label = class_names[pred_labels[i]]
            prob = probabilities[i][pred_labels[i]]
            
            # 设置标题颜色 (绿色表示正确，红色表示错误)
            title_color = 'green' if true_labels[i] == pred_labels[i] else 'red'
            
            # 设置标题
            ax.set_title(f"True: {true_label}\nPred: {pred_label} ({prob:.2f})",
                        color=title_color)
            ax.axis('off')
    
    plt.tight_layout()
    
    # 保存可视化结果
    vis_path = os.path.join(output_dir, 'prediction_visualization.png')
    plt.savefig(vis_path)
    print(f"Prediction visualization saved to {vis_path}")
    plt.close()
    
    # 为错误分类的样本创建一个单独的可视化
    misclassified = []
    misclassified_true = []
    misclassified_pred = []
    
    for i in range(len(images)):
        if true_labels[i] != pred_labels[i]:
            misclassified.append(images[i])
            misclassified_true.append(true_labels[i])
            misclassified_pred.append(pred_labels[i])
    
    if misclassified:
        # 确定绘图尺寸
        n_cols = min(5, len(misclassified))
        n_rows = (len(misclassified) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
        if n_rows * n_cols > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, ax in enumerate(axes):
            if i < len(misclassified):
                # 显示图像
                ax.imshow(misclassified[i])
                
                # 获取真实和预测标签
                true_label = class_names[misclassified_true[i]]
                pred_label = class_names[misclassified_pred[i]]
                
                # 设置标题
                ax.set_title(f"True: {true_label}\nMisclassified as: {pred_label}")
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        
        # 保存错误分类的可视化
        misc_path = os.path.join(output_dir, 'misclassified_samples.png')
        plt.savefig(misc_path)
        print(f"Misclassified samples visualization saved to {misc_path}")
        plt.close()
    else:
        print("No misclassified samples found - all predictions are correct!")


def visualize_gradcam(model, test_loader, class_names, device, num_samples=5, output_dir='evaluation_results'):
    """
    生成简化版的特征可视化，只基于激活图，不依赖梯度，更加简单稳定
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        print("Generating feature activation visualizations for all classes...")
        
        # 设置模型为评估模式
        model.eval()
        
        # 收集不同类别的样本
        class_samples = {}  # 每个类别的样本
        test_inputs = []    # 所有输入样本
        test_images = []    # 所有原始图像
        test_labels = []    # 所有标签
        
        print("Collecting samples from each class...")
        with torch.no_grad():
            for inputs, targets, orig_images in test_loader:
                for i in range(len(inputs)):
                    label = targets[i].item()
                    cls_name = class_names[label]
                    
                    # 每个类别最多收集2个样本
                    if cls_name not in class_samples:
                        class_samples[cls_name] = 0
                    
                    if class_samples[cls_name] < 2:
                        test_inputs.append(inputs[i:i+1])
                        test_images.append(orig_images[i])
                        test_labels.append(label)
                        class_samples[cls_name] += 1
                
                # 一旦收集了足够的样本，退出循环
                if len(test_labels) >= num_samples or len(class_samples) * 2 <= len(test_labels):
                    break
        
        # 保留前num_samples个样本
        test_inputs = test_inputs[:num_samples]
        test_images = test_images[:num_samples]
        test_labels = test_labels[:num_samples]
        
        print(f"Collected samples for visualization: {[class_names[l] for l in test_labels]}")
        
        # 找到目标层 - 使用倒数第二层特征
        target_layer = None
        
        # 对于EfficientNet模型，尝试找到合适的卷积层
        if hasattr(model, 'feature_extractor'):
            # 记录模型中所有卷积层
            conv_layers = []
            
            # 遍历特征提取器的所有模块
            for name, module in model.feature_extractor.named_modules():
                if isinstance(module, nn.Conv2d):
                    conv_layers.append((name, module))
            
            # 使用倒数第二个卷积层作为目标
            if len(conv_layers) >= 2:
                target_name, target_layer = conv_layers[-2]
                print(f"Using layer {target_name} for feature visualization")
            elif len(conv_layers) > 0:
                target_name, target_layer = conv_layers[-1]
                print(f"Using last conv layer {target_name} for feature visualization")
            else:
                print("No convolutional layers found in feature extractor")
                return
        
        if target_layer is None:
            print("Could not find a suitable target layer for visualization")
            return
        
        # 创建图表
        plt.figure(figsize=(15, 4 * min(num_samples, len(test_labels))))
        
        # 为每个样本生成特征可视化
        for idx, (input_tensor, img, label) in enumerate(zip(test_inputs, test_images, test_labels)):
            # 存储该样本的激活图
            activations = []
            
            # 钩子函数
            def hook_fn(module, input, output):
                activations.append(output.detach().cpu())
            
            # 注册钩子
            hook = target_layer.register_forward_hook(hook_fn)
            
            # 前向传播
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                model(input_tensor)
            
            # 删除钩子
            hook.remove()
            
            # 检查是否获取到激活
            if not activations:
                print(f"No activation captured for sample {idx} ({class_names[label]})")
                continue
            
            # 提取特征图
            feature_map = activations[0][0]  # 第一个batch的第一个样本
            
            # 对通道维度求平均得到热力图
            activation_map = torch.mean(feature_map, dim=0).numpy()
            
            # 应用ReLU以突出显示正值
            activation_map = np.maximum(activation_map, 0)
            
            # 归一化到[0,1]
            if np.max(activation_map) > 0:
                activation_map = activation_map / np.max(activation_map)
            
            # 调整大小以匹配原始图像
            heatmap_resized = cv2.resize(activation_map, (img.shape[1], img.shape[0]))
            
            # 生成彩色热力图
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # 生成叠加图像
            superimposed = np.uint8(0.6 * img + 0.4 * heatmap_colored)
            
            # 显示原始图像
            plt.subplot(min(num_samples, len(test_labels)), 3, idx*3+1)
            plt.imshow(img)
            true_label = class_names[label]
            plt.title(f"Original: {true_label}")
            plt.axis('off')
            
            # 显示热力图
            plt.subplot(min(num_samples, len(test_labels)), 3, idx*3+2)
            plt.imshow(heatmap_resized, cmap='jet')
            plt.title(f"Feature Map: {true_label}")
            plt.axis('off')
            
            # 显示叠加图像
            plt.subplot(min(num_samples, len(test_labels)), 3, idx*3+3)
            plt.imshow(superimposed)
            plt.title(f"Overlay: {true_label}")
            plt.axis('off')
        
        # 保存图表
        plt.tight_layout()
        cam_path = os.path.join(output_dir, 'feature_activation_visualization.png')
        plt.savefig(cam_path)
        print(f"Feature activation visualizations saved to {cam_path}")
        plt.close()
        
    except Exception as e:
        import traceback
        print(f"Error generating feature visualizations: {e}")
        print(traceback.format_exc())


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置图像尺寸
    img_size = (args.img_size, args.img_size)
    
    print("=== Celestial Objects Classification Model Evaluation (PyTorch) ===")
    print(f"Model path: {args.model_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Image size: {img_size}")
    
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 准备测试数据
    test_loader, class_indices = prepare_test_data(
        args.data_dir,
        img_size=img_size,
        batch_size=args.batch_size
    )
    
    # 类别名称列表
    class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    
    # 初始化模型
    model = CelestialModel(
        num_classes=num_classes,
        model_type=args.model_type
    )
    
    # 加载模型权重
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # 移动模型到设备
    model = model.to(device)
    
    # 评估模型
    y_true, y_pred, y_pred_probs = evaluate_model(
        model,
        test_loader,
        class_names,
        device,
        output_dir=args.output_dir
    )
    
    # 可视化预测结果
    visualize_model_predictions(
        model,
        test_loader,
        class_names,
        device,
        num_samples=10,
        output_dir=args.output_dir
    )
    
    # 生成Grad-CAM可视化
    visualize_gradcam(
        model,
        test_loader,
        class_names,
        device,
        num_samples=5,
        output_dir=args.output_dir
    )
    
    print(f"\nEvaluation complete! All results saved to {args.output_dir} directory")


if __name__ == "__main__":
    main() 