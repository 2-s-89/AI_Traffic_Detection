import os
import json
import torch
from ultralytics import YOLO
import glob

# ===================== 配置 =====================
BEST_MODEL_PATH = r"D:\AI_Traffic_Detection\runs\detect\train_deconfusion_enhanced\weights\best.pt"
VAL_IMG_DIR = r"D:\AI_Traffic_Detection\data\processed\val\images"
ACCEPTANCE_REPORT = "project_acceptance_report.json"
# 验收标准
ACCEPTANCE_CRITERIA = {
    "construction_detection_rate": 0.8,  # 施工标识检测率≥80%
    "person_detection_rate": 0.8,  # 行人检测率≥80%
    "min_confidence": 0.2,  # 最小置信度≥0.2
    "no_class_confusion": True  # 无类别混淆
}


# ===================== 验收逻辑 =====================
def final_acceptance():
    """全维度验收项目"""
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"🚫 核心模型不存在：{BEST_MODEL_PATH}")
        return

    model = YOLO(BEST_MODEL_PATH)
    val_img_paths = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg"))[:50]  # 抽样50张验证
    report = {
        "total_test_images": len(val_img_paths),
        "construction": {"detected": 0, "total": 0, "confusion": 0},
        "person": {"detected": 0, "total": 0, "confusion": 0},
        "pass_criteria": {},
        "final_result": "FAIL"
    }

    print("✅ 开始项目最终验收，抽样测试50张验证集图片...")
    for img_path in val_img_paths:
        # 标记图片是否包含施工/行人（可根据标注文件优化）
        has_construction = any(key in img_path.lower() for key in ["construction", "施工", "10189"])
        has_person = any(key in img_path.lower() for key in ["person", "行人", "crossing"])

        if has_construction:
            report["construction"]["total"] += 1
        if has_person:
            report["person"]["total"] += 1

        # 推理
        results = model(img_path, conf=0.1, iou=0.3)
        boxes = results[0].boxes

        # 统计检测结果
        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls.cpu().item())
                conf = box.conf.cpu().item()

                # 施工标识检测
                if cls_id == 0 and has_construction and conf >= ACCEPTANCE_CRITERIA["min_confidence"]:
                    report["construction"]["detected"] += 1
                # 行人检测
                elif cls_id == 1 and has_person and conf >= ACCEPTANCE_CRITERIA["min_confidence"]:
                    report["person"]["detected"] += 1
                # 类别混淆
                elif (cls_id == 1 and has_construction) or (cls_id == 0 and has_person):
                    if cls_id == 1 and has_construction:
                        report["construction"]["confusion"] += 1
                    else:
                        report["person"]["confusion"] += 1

    # 计算检测率
    if report["construction"]["total"] > 0:
        construction_rate = report["construction"]["detected"] / report["construction"]["total"]
    else:
        construction_rate = 1.0

    if report["person"]["total"] > 0:
        person_rate = report["person"]["detected"] / report["person"]["total"]
    else:
        person_rate = 1.0

    # 验证验收标准
    report["pass_criteria"] = {
        "construction_detection_rate": construction_rate >= ACCEPTANCE_CRITERIA["construction_detection_rate"],
        "person_detection_rate": person_rate >= ACCEPTANCE_CRITERIA["person_detection_rate"],
        "min_confidence": True,  # 已过滤低置信度
        "no_class_confusion": (report["construction"]["confusion"] + report["person"]["confusion"]) == 0
    }

    # 最终结论
    all_pass = all(report["pass_criteria"].values())
    report["final_result"] = "PASS" if all_pass else "FAIL"

    # 保存验收报告
    with open(ACCEPTANCE_REPORT, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    # 输出验收结果
    print("\n" + "=" * 60)
    print("📊 项目验收报告")
    print("=" * 60)
    print(f"施工标识检测率：{construction_rate:.2%} (要求≥{ACCEPTANCE_CRITERIA['construction_detection_rate']:.0%})")
    print(f"行人检测率：{person_rate:.2%} (要求≥{ACCEPTANCE_CRITERIA['person_detection_rate']:.0%})")
    print(f"类别混淆数：{report['construction']['confusion'] + report['person']['confusion']} (要求=0)")
    print(f"最终结论：{'✅ 项目验收通过' if all_pass else '❌ 项目验收未通过'}")
    print(f"验收报告保存：{ACCEPTANCE_REPORT}")
    print("=" * 60)


if __name__ == "__main__":
    final_acceptance()