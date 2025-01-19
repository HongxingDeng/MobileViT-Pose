# MobileViT-Pose

**Abstract:** Accurate measurement of cattle body size is crucial for assessing growth status and making breeding decisions. Traditional manual measurement techniques are inefficient, unsafe, and prone to inaccuracies. Existing automated methods either lack precision or suffer from long processing times. In this study, a rapid and non-contact cattle body size measurement method based on stereo vision was carried out. Lateral images of cattle were initially captured using a stereo camera, and depth information was derived from these images using the CREStereo algorithm. The MobileViT-Pose algorithm was then applied to predict body size keypoints, including the head, body, front limbs, and hind limbs. The final body size measurements were obtained by integrating depth data with these keypoints. To minimize measurement errors, the Isolation Forest algorithm was used to detect and remove outliers, with the final measurement computed as the average of multiple results. Compared to traditional stereo matching algorithms, CREStereo provided more detailed disparity information and demonstrated greater robustness across varying resolutions. Pose estimation accuracy of the MobileViT-Pose algorithm reached 92.4%, while improving efficiency and reducing both the number of parameters and FLOPs. Additionally, a lightweight version, LiteMobileViT-Pose, was introduced, featuring only 1.735 M parameters and 0.272 G FLOPs. In practical evaluations, the maximum measurement deviations for body length, body height, hip height, and rump length were 4.55%, 4.87%, 4.99%, and 6.76%, respectively, when compared to manual measurements. Additionally, the MobileViT model was deployed, achieving an average body size measurement error of only 2.85% and a measurement speed of 18.9 fps. The proposed method provides a practical solution for the rapid and accurate measurement of cattle body size.
Keywords: Cattle; Body Size; Automatic measurement; Stereo vision; Pose estimation

![image](https://github.com/user-attachments/assets/70ada5c9-d181-4167-a658-4c8769c8f1e7)

Visual workflow of cattle body size measurement. ①: dairy cattle; ②: beef cattle; (a): left image; (b): right image; (c): cattle detection, classification and selection of optimal measurement cattle based on object detection scores; (d): individual pose estimation of cattle and location of measurement keypoints; (e): disparity map; (f): depth map; (g): body size measurement.
## 💡 Highlights

    🔥Stereo vision and pose estimation were fused for accurate cow’s body size measurement
    🔥CREStereo was robust and with high-precision for agricultural context description
    🔥MobileViT-Pose model was rapid and accurate for point localization measurement
    🔥The proposed method was deployable for rapidly cow’s body size measurement

## 📜 Todo
- [√] Source code and models for the open-source MobileViT-Pose and MobileViT-Pose。
- [√] Public pose estimation datasets.
- [√] Training code for MobileViT-Pose and MobileViT-Pose.
- [√] Evaluation code for MobileViT-Pose and MobileViT-Pose.
- [ ] Usage example notebook of MobileViT-Pose and MobileViT-Pose.
## 👨‍💻 Data
Pose estimation dataset was developed for cattle object detection and measurement point localization, consist of left-eye images captured using a ZED Stereo camera, data from the [NWAFU-Cattle dataset](https://github.com/MicaleLee/Database/blob/master/NWAFU-CattleDataset), [the Hereford cattle dataset](https://github.com/ruchaya/CowDatabase), and [the Angus cattle dataset](https://github.com/ruchaya/CowDatabase2).
## 🛠️ Usage
### Installation
#### Prerequisites
    Python 3.7+、CUDA 9.2+ 和 PyTorch 1.8+
```shell
# create python env
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
# install pytorch
conda install pytorch torchvision -c pytorch
# install mmcv
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.1"
mim install "mmdet>=3.1.0"
# install mmpose
cd mmpose-dev-1.x
pip install -r requirements.txt
pip install -v -e .
# install mmcls
cd mmpretrain
pip install -r requirements.txt
pip install -v -e .
```
### Training
```shell
# MobileViT-Pose
python tools/train.py configs/body_2d_keypoint/rtmpose/coco/rtmpose-Mobilevit-Pose.py
# LiteMobilevit-Pose
python tools/train.py configs/body_2d_keypoint/rtmpose/coco/rtmpose-LiteMobilevit-Pose.py
```
## ❤️ Acknowledgments
Our code is built based on [mmpose](https://github.com/open-mmlab/mmpose.git), [mmpretrain](https://github.com/open-mmlab/mmpretrain), and [CREStereo](https://github.com/megvii-research/CREStereo.git). Thanks for their wonderful work.
