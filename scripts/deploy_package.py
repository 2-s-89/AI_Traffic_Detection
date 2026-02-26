import os
import torch
from ultralytics import YOLO
from dataclasses import dataclass


# ===================== 部署配置 =====================
@dataclass
class DetectorConfig:
    model_path: str = r"D:\AI_Traffic_Detection\runs\detect\train_deconfusion_enhanced\weights\best.pt"
    conf_threshold: float = 0.2
    iou_threshold: float = 0.3
    max_detections: int = 20
    device: str = "0" if torch.cuda.is_available() else "cpu"


# ===================== 检测器封装类 =====================
class TrafficDetector:
    """交通目标检测器（施工标识+行人）"""

    def __init__(self, config: DetectorConfig = DetectorConfig()):
        self.config = config
        self.model = YOLO(config.model_path)
        self.class_names = {0: "construction", 1: "illegal_crossing"}
        print(f"✅ 加载检测器完成：{config.model_path}")
        print(f"🔧 推理配置：conf={config.conf_threshold}, iou={config.iou_threshold}")

    def detect(self, img_path: str, save_result: bool = True, save_dir: str = "deploy_results"):
        """
        检测图片中的施工标识和行人
        :param img_path: 图片路径
        :param save_result: 是否保存检测结果图
        :param save_dir: 结果保存目录
        :return: 结构化检测结果
        """
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"图片不存在：{img_path}")

        # 推理
        results = self.model(
            img_path,
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            max_det=self.config.max_detections,
            device=self.config.device,
            verbose=False
        )

        # 解析结果
        detections = []
        boxes = results[0].boxes
        if len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls.cpu().item())
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.class_names[cls_id],
                    "confidence": round(float(box.conf.cpu().item()), 3),
                    "bbox": [round(float(coord), 2) for coord in box.xyxy.cpu().numpy()[0]],
                    "bbox_format": "x1,y1,x2,y2"
                })

        # 保存结果图
        if save_result:
            os.makedirs(save_dir, exist_ok=True)
            img_name = os.path.basename(img_path)
            save_path = os.path.join(save_dir, f"detected_{img_name}")
            results[0].save(save_path)
            detections.append({"result_image_path": save_path})

        return detections


# ===================== 调用示例 =====================
if __name__ == "__main__":
    # 初始化检测器
    detector = TrafficDetector()

    # 单图检测
    test_img = r"D:\AI_Traffic_Detection\data\processed\val\images\10189.jpg"
    try:
        results = detector.detect(test_img)
        print("\n📊 检测结果：")
        for res in results:
            if "class_name" in res:
                print(f"- {res['class_name']} | 置信度：{res['confidence']} | 坐标：{res['bbox']}")
            else:
                print(f"- 结果图保存：{res['result_image_path']}")
    except Exception as e:
        print(f"❌ 检测失败：{str(e)}")