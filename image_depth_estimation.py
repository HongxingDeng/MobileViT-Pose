import os
import time

import cv2
import numpy as np
from imread_from_url import imread_from_url
import camera_config_2i as camera_configs
from crestereo import CREStereo


def DepthMap(imgs, KL, KR, baseline):
    """
        This function is used to computing the depth map.
        Params Description:
            - 'imgs' is the list of the image data.
            - 'K' is the intrinsic matrix of the input images.
            - 'baseline' is the baseline of the stereo camera.
    """
    # Getting the focal length.
    fx = KL[0][0]
    fy = KL[1][1]
    xl = KL[0][2]
    xr = KR[0][2]

    # Initializing the depth map list.
    depthMaps = []
    # Computing the depth maps.
    for i in range(len(imgs)):
        # Getting the image size.
        h = imgs[i].shape[0]
        # Initializing the depth map.
        depthMap = np.zeros(h)
        # Creating the depth map.
        for k in range(0, h):
            b = (imgs[i][k] + abs(xl - xr))
            # f = math.sqrt((fx * fx + fy * fx)/2)
            # depthMap[k] = f * baseline / b
            # depthMap[k] = fx * baseline / ((imgs[i][k]))
            depthMap[k] = fx * baseline / b
        # np.set_printoptions(threshold = np.inf)
        # print(depthMap)
        # Storing the depth map.
        depthMaps.append(depthMap)
    # Returning the depth map.
    return depthMaps


# Model Selection options (not all options supported together)
iters = 20  # Lower iterations are faster, but will lower detail.
# Options: 2, 5, 10, 20

shape = (120, 160)  # Input resolution.
# Options: (120,160), (160,240) , (240,320), (360,640), (480,640), (720, 1280)

version = "init"  # The combined version does 2 passes, one to get an initial estimation and a second one to refine it.
# Options: "init", "combined"

# Initialize model
model_path = f'models/crestereo_{version}_iter{iters}_{shape[0]}x{shape[1]}.onnx'
depth_estimator = CREStereo(model_path)
path = '/home/xing/ONNX-CREStereo-Depth-Estimation-main/datasets/00/0/'
filelist = os.listdir(path + 'left')
for file in filelist:
    if file.split('.')[-1] == 'png':
        pass
    else:
        continue
    left_path = path + 'left/' + file
    right_path = path + 'right/right' + file.strip('left')
    csv_path = path + 'result/' + version + str(iters) + '_' + str(shape[0]) + 'x' + str(shape[1]) + '_' + file.split('.')[0] + '.csv'
    out_path = '/home/xing/ONNX-CREStereo-Depth-Estimation-main/result/depth' + file
    # csv_path =
    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    left_img = cv2.resize(left_img, dsize=(shape[1], shape[0]), fx=1, fy=1)
    right_img = cv2.resize(right_img, dsize=(shape[1], shape[0]), fx=1, fy=1)
    # Estimate the depth
    disparity_map = depth_estimator(left_img, right_img)
    t = float(2208) / float(shape[1])
    disparity = cv2.resize(disparity_map, dsize=(2208, 1242), fx=1, fy=1, interpolation=cv2.INTER_LINEAR) * t
    depthMaps = DepthMap(disparity, camera_configs.left_camera_matrix, camera_configs.right_camera_matrix,
                         camera_configs.t)
    depth = np.array(depthMaps, dtype=float)
    np.savetxt(csv_path, depth, delimiter=",")
    # depth_img = cv2.normalize(depth, None, 0, 255)
    # cv2.imwrite("out.jpg", depth_img)


    # color_disparity, disparity_map = depth_estimator.draw_disparity()
