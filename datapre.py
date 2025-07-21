import os
import shutil
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def organize_dataset(raw_data_dir, output_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    将原始数据集组织成训练、验证和测试目录

    参数:
        raw_data_dir: 包含天体图像文件夹的目录
        output_dir: 存储组织后数据的基础目录
        split_ratio: (训练集、验证集、测试集)比例的元组
    """
    print("开始组织数据集...")

    # 创建必要的目录
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 获取所有天体类别
    celestial_classes = [d for d in os.listdir(raw_data_dir) 
                          if os.path.isdir(os.path.join(raw_data_dir, d))]

    for celestial_class in celestial_classes:
        # 为每个类别创建目录
        os.makedirs(os.path.join(train_dir, celestial_class), exist_ok=True)
        os.makedirs(os.path.join(val_dir, celestial_class), exist_ok=True)
        os.makedirs(os.path.join(test_dir, celestial_class), exist_ok=True)

        # 获取该类别的所有图像
        class_dir = os.path.join(raw_data_dir, celestial_class)
        images = [img for img in os.listdir(class_dir) 
                 if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if len(images) == 0:
            print(f"警告: {celestial_class} 类别没有图像")
            continue

        # 拆分为训练集、验证集和测试集
        train_val_images, test_images = train_test_split(
            images, 
            test_size=split_ratio[2],
            random_state=42
        )

        train_images, val_images = train_test_split(
            train_val_images,
            test_size=split_ratio[1]/(split_ratio[0]+split_ratio[1]),
            random_state=42
        )

        # 复制文件到相应的目录
        print(f"处理 {celestial_class} 类...")
        for img_list, target_dir in [(train_images, train_dir), 
                                     (val_images, val_dir), 
                                     (test_images, test_dir)]:
            for img in img_list:
                src = os.path.join(class_dir, img)
                dst = os.path.join(target_dir, celestial_class, img)
                shutil.copy(src, dst)

    # 打印数据集统计信息
    print("\n数据集组织完成!")
    print(f"总类别数: {len(celestial_classes)}")

    for split, directory in [("训练集", train_dir), 
                             ("验证集", val_dir), 
                             ("测试集", test_dir)]:
        total_images = 0
        for cls in celestial_classes:
            class_path = os.path.join(directory, cls)
            if os.path.exists(class_path):
                class_count = len(os.listdir(class_path))
                total_images += class_count
                print(f"  {split} - {cls}: {class_count} 张图像")
        print(f"  {split} 总计: {total_images} 张图像")

def preprocess_images(data_dir, img_size=(224, 224), sample_visualization=True):
    """
    检查图像并预处理它们(调整大小，标准化等)
    为了验证，这里只对几张图像进行可视化

    参数:
        data_dir: 包含train, val, test子目录的数据目录
        img_size: 调整图像大小的目标尺寸
        sample_visualization: 是否可视化样本图像
    """
    print("\n检查和预处理图像...")

    # 示例可视化
    if sample_visualization:
        classes = os.listdir(os.path.join(data_dir, 'train'))
        plt.figure(figsize=(15, 10))

        for i, cls in enumerate(classes[:min(5, len(classes))]):
            class_dir = os.path.join(data_dir, 'train', cls)
            images = os.listdir(class_dir)

            for j, img_name in enumerate(images[:min(4, len(images))]):
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path)

                # 调整图像大小并转换为数组
                img = img.resize(img_size)
                img_array = np.array(img)

                # 显示图像
                plt.subplot(min(5, len(classes)), 4, i*4+j+1)
                plt.imshow(img_array)
                plt.title(f"{cls}")
                plt.axis('off')

        plt.tight_layout()
        plt.savefig('sample_images.png')
        print("样本图像已保存至 'sample_images.png'")

    # 检查所有图像是否可以正确加载和调整大小
    splits = ['train', 'val', 'test']
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"警告: {split_dir} 不存在")
            continue

        classes = os.listdir(split_dir)

        for cls in classes:
            class_dir = os.path.join(split_dir, cls)
            images = os.listdir(class_dir)

            print(f"检查 {split}/{cls} 中的 {len(images)} 张图像...")

            for img_name in tqdm(images):
                try:
                    img_path = os.path.join(class_dir, img_name)
                    img = Image.open(img_path)

                    # 检查通道数
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                        img.save(img_path)  # 保存转换后的图像

                    # 检查图像是否可以调整大小
                    img = img.resize(img_size)
                except Exception as e:
                    print(f"  处理 {img_path} 时出错: {e}")

    print("预处理完成!")

# 示例用法
if __name__ == "__main__":
    # 首先组织数据集
    organize_dataset(r"C:\Users\USER\Desktop\PlanetaryImageClassifier\data\input", r"C:\Users\USER\Desktop\PlanetaryImageClassifier\data\celestial_dataset")

    # 然后预处理图像
    preprocess_images(r"C:\Users\USER\Desktop\PlanetaryImageClassifier\data\celestial_dataset")