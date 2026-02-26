import os
import shutil
from pathlib import Path

# ===================== 配置 =====================
# 原始验证集图片路径
VAL_IMG_DIR = r"D:\AI_Traffic_Detection\data\processed\val\images"
# 筛选后施工样本保存路径
CONSTRUCTION_SAMPLE_DIR = r"D:\AI_Traffic_Detection\data\supplement\construction_samples"
# 关键词（筛选包含施工标识的图片，可根据实际命名调整）
CONSTRUCTION_KEYWORDS = ["construction", "施工", "警示牌", "10189", "roadwork"]


# ===================== 核心功能 =====================
def filter_construction_samples():
    """筛选验证集中包含施工标识的图片"""
    os.makedirs(CONSTRUCTION_SAMPLE_DIR, exist_ok=True)
    img_files = [f for f in os.listdir(VAL_IMG_DIR) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    filtered_count = 0
    for img_file in img_files:
        # 按文件名关键词筛选
        if any(key in img_file.lower() for key in CONSTRUCTION_KEYWORDS):
            src_path = os.path.join(VAL_IMG_DIR, img_file)
            dst_path = os.path.join(CONSTRUCTION_SAMPLE_DIR, img_file)
            shutil.copy(src_path, dst_path)
            filtered_count += 1
            print(f"✅ 筛选施工样本：{img_file}")

    print(f"\n📊 共筛选出 {filtered_count} 张施工样本，保存至：{CONSTRUCTION_SAMPLE_DIR}")
    print("💡 下一步：用LabelImg标注这些图片，标注类名：")
    print("   - 施工标识：construction（ID=0）")
    print("   - 行人：illegal_crossing（ID=1）")


def move_to_train_set():
    """将标注好的施工样本移动到训练集"""
    # 标注后的标签文件路径（需手动替换为你的标注路径）
    LABEL_DIR = r"D:\AI_Traffic_Detection\data\supplement\construction_samples_labels"

    # 移动图片
    for img_file in os.listdir(CONSTRUCTION_SAMPLE_DIR):
        if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            img_src = os.path.join(CONSTRUCTION_SAMPLE_DIR, img_file)
            img_dst = os.path.join(r"D:\AI_Traffic_Detection\data\processed\train\images", img_file)
            shutil.move(img_src, img_dst)

            # 移动对应标签
            label_file = Path(img_file).stem + ".txt"
            label_src = os.path.join(LABEL_DIR, label_file)
            if os.path.exists(label_src):
                label_dst = os.path.join(r"D:\AI_Traffic_Detection\data\processed\train\labels", label_file)
                shutil.move(label_src, label_dst)
                print(f"✅ 移动样本：{img_file} + {label_file} 到训练集")

    print("\n📊 施工样本已全部补充到训练集，可开始反混淆训练")


if __name__ == "__main__":
    # 第一步：筛选施工样本
    filter_construction_samples()

    # 第二步：标注完成后，取消注释执行
    # move_to_train_set()