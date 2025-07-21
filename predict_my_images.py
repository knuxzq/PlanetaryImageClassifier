"""
简单的批量预测脚本 - 预测完自动退出
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from torchvision import transforms
import json
import pandas as pd
from improved_model_architecture import CelestialModel

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def batch_predict(folder_path='my_images'):
    """批量预测文件夹中的所有图片"""

    print("=" * 60)
    print("批量天体图片预测")
    print("=" * 60)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载类别信息
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    except:
        class_names = ["earth", "jupiter", "mars", "moon", "neptune", "saturn", "uranus"]

    # 中文名称映射
    chinese_names = {
        "earth": "地球",
        "jupiter": "木星",
        "mars": "火星",
        "moon": "月球",
        "neptune": "海王星",
        "saturn": "土星",
        "uranus": "天王星"
    }

    # 加载模型
    model_path = 'model_checkpoints/celestial_classification_final_model.pth'
    print(f"加载模型: {model_path}")

    model = CelestialModel(num_classes=len(class_names), model_type='efficientnet')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # 检查文件夹
    folder = Path(folder_path)
    if not folder.exists():
        print(f"\n错误: 文件夹 '{folder}' 不存在!")
        print(f"请创建文件夹并放入图片后再运行。")
        return

    # 查找所有图片
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(folder.glob(f'*{ext}'))
        image_files.extend(folder.glob(f'*{ext.upper()}'))

    if not image_files:
        print(f"\n在 '{folder}' 中没有找到图片文件!")
        return

    print(f"\n找到 {len(image_files)} 张图片")
    print("=" * 60)

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 创建输出目录
    output_dir = Path('my_predictions')
    output_dir.mkdir(exist_ok=True)

    # 预测所有图片
    results = []

    for idx, img_path in enumerate(image_files, 1):
        try:
            print(f"\n[{idx}/{len(image_files)}] 预测: {img_path.name}")

            # 加载图片
            original_img = Image.open(img_path).convert('RGB')
            img_tensor = transform(original_img).unsqueeze(0).to(device)

            # 预测
            with torch.no_grad():
                outputs = model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)

            # 获取结果
            probs = probabilities[0].cpu().numpy()
            pred_idx = np.argmax(probs)
            pred_class = class_names[pred_idx]
            pred_prob = probs[pred_idx]

            # 打印结果
            pred_chinese = chinese_names.get(pred_class, pred_class)
            print(f"  → 预测结果: {pred_chinese} ({pred_class})")
            print(f"  → 置信度: {pred_prob * 100:.2f}%")

            # 保存可视化
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.imshow(original_img)
            ax.set_title(f"预测: {pred_chinese} ({pred_prob * 100:.1f}%)",
                         fontsize=18, fontweight='bold', color='darkblue')
            ax.axis('off')

            # 添加边框
            rect = plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                 linewidth=3, edgecolor='darkblue', facecolor='none')
            ax.add_patch(rect)

            # 保存
            save_name = f"pred_{idx:03d}_{pred_class}_{img_path.stem}.png"
            plt.savefig(output_dir / save_name, dpi=150, bbox_inches='tight')
            plt.close()

            # 记录结果
            results.append({
                'filename': img_path.name,
                'prediction': pred_class,
                'prediction_chinese': pred_chinese,
                'confidence': f"{pred_prob * 100:.2f}%"
            })

        except Exception as e:
            print(f"  → 错误: {e}")
            results.append({
                'filename': img_path.name,
                'prediction': 'error',
                'prediction_chinese': '错误',
                'confidence': '0%'
            })

    # 保存汇总结果
    print("\n" + "=" * 60)
    if results:
        df = pd.DataFrame(results)
        summary_path = output_dir / 'batch_prediction_summary.csv'
        df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        print(f"汇总结果已保存到: {summary_path}")

        # 打印汇总
        print("\n预测汇总:")
        print("-" * 40)
        for result in results:
            print(f"{result['filename']:<30} → {result['prediction_chinese']:<10} ({result['confidence']})")

    print("\n" + "=" * 60)
    print(f"批量预测完成！")
    print(f"- 处理图片数: {len(image_files)}")
    print(f"- 结果保存在: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import sys

    # 检查是否指定了文件夹
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        folder = 'my_images'

    # 执行批量预测
    batch_predict(folder)