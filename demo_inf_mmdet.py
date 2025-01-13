from mmdeploy_runtime import Detector
import cv2

# 读取图片
img = cv2.imread('mmpose-dev-1.x/2.jpg')

# 创建检测器
detector = Detector(model_path='work_dir/yolox_det', device_name='cpu', device_id=0)
# 执行推理
bboxes, labels, _ = detector(img)
# 使用阈值过滤推理结果，并绘制到原图中
indices = [i for i in range(len(bboxes))]
for index, bbox, label_id in zip(indices, bboxes, labels):
  [left, top, right, bottom], score = bbox[0:4].astype(int),  bbox[4]
  if score < 0.3:
      continue
  cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0))

cv2.imwrite('output_detection.png', img)