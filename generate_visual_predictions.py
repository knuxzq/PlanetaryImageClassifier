"""
生成单张预测可视化图片的独立脚本
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
import random
from improved_model_architecture import CelestialModel

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def generate_visual_predictions(model_path, mapping_file, num_samples=30):
    """
    生成可视化预测图片

    参数:
        model_path: 模型文件路径
        mapping_file: 盲测映射文件路径
        num_samples: 生成的样本数量
    """
    print("=" * 60)
    print("生成可视化预测图片")
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载映射文件
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        image_mapping = data['mapping']
        class_names = data['class_names']

    print(f"找到 {len(image_mapping)} 张图片")
    print(f"类别: {class_names}")

    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = CelestialModel(num_classes=len(class_names), model_type='efficientnet')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 创建输出目录
    output_dir = Path('visual_predictions')
    output_dir.mkdir(exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    # 中文类别名映射
    chinese_names = {
        "earth": "地球",
        "jupiter": "木星",
        "mars": "火星",
        "moon": "月球",
        "neptune": "海王星",
        "saturn": "土星",
        "uranus": "天王星"
    }

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 随机选择样本
    samples = random.sample(image_mapping, min(num_samples, len(image_mapping)))

    print(f"\n生成 {len(samples)} 张可视化图片...")

    # 统计正确和错误的预测
    correct_count = 0
    wrong_count = 0

    for idx, sample in enumerate(samples):
        try:
            # 加载图像
            img_path = sample['new_path']
            original_img = Image.open(img_path).convert('RGB')
            true_class = sample['true_class']

            # 预测
            img_tensor = transform(original_img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            pred_class = class_names[predicted.item()]
            pred_confidence = confidence.item()

            # 判断是否正确
            is_correct = (pred_class == true_class)
            if is_correct:
                correct_count += 1
            else:
                wrong_count += 1

            # 创建可视化
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))

            # 显示图像
            ax.imshow(original_img)

            # 设置标题
            title_color = 'green' if is_correct else 'red'
            result_symbol = "✓" if is_correct else "✗"

            # 获取中文名称
            true_chinese = chinese_names.get(true_class, true_class)
            pred_chinese = chinese_names.get(pred_class, pred_class)

            # 创建标题文本
            if is_correct:
                title_text = f"{result_symbol} 预测: {pred_chinese} ({pred_confidence * 100:.1f}%)"
            else:
                title_text = f"{result_symbol} 预测: {pred_chinese} ({pred_confidence * 100:.1f}%)\n实际: {true_chinese}"

            ax.set_title(title_text, fontsize=20, fontweight='bold',
                         color=title_color, pad=20)
            ax.axis('off')

            # 添加彩色边框
            rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                 linewidth=5, edgecolor=title_color, facecolor='none')
            ax.add_patch(rect)

            # 保存图片
            if is_correct:
                filename = f"correct_{idx + 1:03d}_{true_class}_{pred_confidence:.3f}.png"
            else:
                filename = f"wrong_{idx + 1:03d}_{true_class}_as_{pred_class}_{pred_confidence:.3f}.png"

            save_path = output_dir / filename
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"  [{idx + 1}/{len(samples)}] 已生成: {filename}")

        except Exception as e:
            print(f"  处理第 {idx + 1} 张图片时出错: {e}")
            plt.close()
            continue

    print(f"\n生成完成!")
    print(f"正确预测: {correct_count} 张")
    print(f"错误预测: {wrong_count} 张")
    print(f"可视化图片保存在: {output_dir}")

    # 创建一个展示汇总图
    create_showcase_image(output_dir, correct_count, wrong_count)


def create_showcase_image(output_dir, correct_count, wrong_count):
    """创建一个展示汇总图，显示一些预测示例"""

    print("\n创建展示汇总图...")

    # 获取生成的图片
    all_images = list(output_dir.glob("*.png"))
    correct_images = [img for img in all_images if img.name.startswith("correct_")]
    wrong_images = [img for img in all_images if img.name.startswith("wrong_")]

    # 创建展示图
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(f'预测结果展示 (正确: {correct_count}, 错误: {wrong_count})',
                 fontsize=24, fontweight='bold')

    # 显示最多6个正确和6个错误的例子
    num_correct_show = min(6, len(correct_images))
    num_wrong_show = min(6, len(wrong_images))

    # 显示正确的预测
    for i in range(num_correct_show):
        ax = plt.subplot(3, 4, i + 1)
        img = Image.open(correct_images[i])
        ax.imshow(img)
        ax.set_title("正确预测", color='green', fontsize=16)
        ax.axis('off')

    # 显示错误的预测
    for i in range(num_wrong_show):
        ax = plt.subplot(3, 4, i + 7)
        img = Image.open(wrong_images[i])
        ax.imshow(img)
        ax.set_title("错误预测", color='red', fontsize=16)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'showcase_summary.png', dpi=200, bbox_inches='tight')
    plt.close()

    print("展示汇总图已创建")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='生成可视化预测图片')
    parser.add_argument('--model_path', type=str,
                        default='model_checkpoints/celestial_classification_final_model.pth',
                        help='模型文件路径')
    parser.add_argument('--mapping_file', type=str,
                        help='盲测映射文件路径（如不指定，自动寻找最新的）')
    parser.add_argument('--num_samples', type=int, default=30,
                        help='生成的样本数量')

    args = parser.parse_args()

    # 如果没有指定映射文件，自动寻找最新的
    if not args.mapping_file:
        mapping_files = list(Path('.').glob('blind_test_mapping_*.json'))
        if mapping_files:
            args.mapping_file = str(max(mapping_files, key=lambda x: x.stat().st_mtime))
            print(f"使用最新的映射文件: {args.mapping_file}")
        else:
            print("错误: 找不到映射文件，请先运行盲测")
            exit(1)

    # 生成可视化
    generate_visual_predictions(args.model_path, args.mapping_file, args.num_samples)