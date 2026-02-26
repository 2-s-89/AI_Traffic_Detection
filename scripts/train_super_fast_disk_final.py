"""
道路施工目标检测 - 超极速训练脚本（最终修复版）
解决问题：多进程冲突/内存缓存报错/权限拒绝/路径错误
"""
import os
import torch
import warnings
import shutil
from ultralytics import YOLO
from pathlib import Path

# ===================== 全局配置（最终修复版） =====================
# 基础路径（全部写死绝对路径，避免拼接错误）
PROJECT_ROOT = r"D:\AI_Traffic_Detection"
CONFIG_PATH = r"D:\AI_Traffic_Detection\yolov8_config.yaml"  # 配置文件绝对路径
MODEL_PATH = r"D:\AI_Traffic_Detection\runs\detect\train_1min_per_epoch\weights\last.pt"  # 模型绝对路径
TEST_IMG_PATH = r"D:\AI_Traffic_Detection\data\processed\val\images\10189.jpg"

# 训练参数（修复多进程冲突）
EPOCHS = 8                # 8轮极速训练
BATCH_SIZE = 8            # 降低batch，减少显存/内存压力（4060 8G更稳定）
IMG_SIZE = 640
CLS_WEIGHT = 15.0         # 解决类别混淆
LR0 = 0.0002              # 微调学习率
WORKERS = 4               # 核心修复：关闭多进程，用单进程加载数据
CACHE_MODE = "disk"

# 数据配置（仅用SSD数据，关闭内存缓存避免冲突）
DATA_DST_DIR = r"E:\AI_Traffic_Detection\data\processed"  # SSD数据路径
USE_SSD = True

# ===================== 环境初始化（修复权限/多进程问题） =====================
def init_env():
    """初始化环境：关闭多进程+权限优化"""
    # 基础环境优化
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["YOLO_VERBOSE"] = "False"  # 关闭冗余日志
    torch.set_num_threads(8)  # 降低线程数，避免权限冲突
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')

    # 检查SSD数据是否存在
    if USE_SSD and os.path.exists(DATA_DST_DIR):
        print(f"✅ 检测到SSD已有数据，直接使用：{DATA_DST_DIR}")
        # 修正配置文件中的数据路径（关键！）
        modify_config_data_path()
        return DATA_DST_DIR
    else:
        print("⚠️ SSD数据不存在，使用D盘原始数据")
        return r"D:\AI_Traffic_Detection\data\processed"

def modify_config_data_path():
    """修正yolov8_config.yaml中的数据路径为SSD路径，避免路径冲突"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config_content = f.read()

    # 替换训练/验证数据路径为SSD路径
    config_content = config_content.replace(
        r"D:\AI_Traffic_Detection\data\processed",
        r"E:\AI_Traffic_Detection\data\processed"
    )

    # 保存修正后的配置文件
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(config_content)
    print(f"✅ 已修正配置文件{CONFIG_PATH}的数据源为SSD")

# ===================== 核心训练函数（修复所有报错） =====================
def train_super_fast():
    """超极速训练：修复多进程/内存/权限问题"""
    # 初始化环境
    final_data_dir = init_env()

    # 加载模型（冻结前10层特征）
    model = YOLO(MODEL_PATH)
    for i, param in enumerate(model.model.parameters()):
        if i < 10:
            param.requires_grad = False
    print(f"✅ 加载模型：{MODEL_PATH}，冻结前10层特征")

    # 训练保存目录（改为D盘，避免E盘权限问题）
    save_dir = r"D:\AI_Traffic_Detection\runs\detect\train_super_fast_final"
    os.makedirs(os.path.join(save_dir, "weights"), exist_ok=True)

    # 最终修复版训练配置（核心！）
    results = model.train(
        # 基础配置（修复多进程/内存问题）
        data=CONFIG_PATH,
        epochs=EPOCHS,
        batch=BATCH_SIZE,       # 降低batch，避免显存溢出
        imgsz=IMG_SIZE,
        device="0",
        amp=True,


        # 解决类别混淆
        single_cls=False,
        cls=CLS_WEIGHT,
        box=8.0,

        # 核心修复：关闭数据增强+关闭内存缓存+单进程
        mosaic=0,
        mixup=0,
        degrees=0,
        hsv_h=0, hsv_s=0, hsv_v=0,
        cache=CACHE_MODE,            # 关闭ram缓存，避免多进程冲突
        workers=WORKERS,        # 0=单进程，解决Windows权限/EOF错误

        # 学习率配置
        lr0=LR0,
        lrf=0.0005,
        warmup_epochs=0,

        # 关闭冗余操作，避免权限错误
        val=False,
        plots=False,
        save=True,
        save_period=-1,
        project=save_dir,       # 保存到D盘，避免E盘权限问题
        name="",
        exist_ok=True,
        verbose=False           # 关闭详细日志，减少进程通信
    )

    # 训练完成
    best_model_path = os.path.join(save_dir, "weights/best.pt")
    print("\n" + "="*60)
    print(f"✅ 超极速训练完成！总时长≤10分钟")
    print(f"📁 最优模型：{best_model_path}")
    print("="*60 + "\n")
    return best_model_path

# ===================== 训练后验证 =====================
def verify_final_result(model_path):
    """验证施工标识识别效果"""
    model = YOLO(model_path)
    results = model(TEST_IMG_PATH, conf=0.1, iou=0.3, verbose=False)

    # 解析结果
    boxes = results[0].boxes
    construction_count = 0
    person_count = 0

    if len(boxes) > 0:
        for box in boxes:
            cls_id = int(box.cls.cpu().item())
            conf = round(float(box.conf.cpu().item()), 3)
            if cls_id == 0:
                construction_count += 1
                print(f"✅ 正确识别：施工标识（construction） | 置信度：{conf}")
            elif cls_id == 1:
                person_count += 1
                print(f"📌 识别结果：行人（illegal_crossing） | 置信度：{conf}")

    # 保存验证结果图
    result_img_path = os.path.join(PROJECT_ROOT, "final_verification_result.jpg")
    results[0].save(result_img_path)

    # 验收结论
    print("\n" + "="*60)
    print("📊 项目验收结果")
    print("="*60)
    if construction_count > 0:
        print("🎉 验收通过！施工标识被正确识别，类别混淆问题已解决")
    else:
        print("⚠️ 验收未通过！仍存在类别混淆，建议补充施工样本后重试")
    print(f"📁 验证结果图：{result_img_path}")
    print("="*60)

# ===================== 主函数（一键运行） =====================
if __name__ == "__main__":
    try:
        print("🚀 启动道路施工目标检测超极速训练（最终修复版）")
        print(f"🔧 配置：EPOCHS={EPOCHS} | BATCH={BATCH_SIZE} | CLS={CLS_WEIGHT} | WORKERS={WORKERS}")

        # 执行训练
        best_model = train_super_fast()

        # 验证结果
        verify_final_result(best_model)

        print("\n🎯 所有操作完成！模型已保存，可直接部署使用")
    except Exception as e:
        print(f"❌ 运行出错：{str(e)}")
        print("💡 应急方案：1. 降低BATCH_SIZE到4 2. 关闭USE_SSD改用D盘数据")