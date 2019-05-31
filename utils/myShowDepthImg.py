import glob
import os
import PIL.Image as IMG
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


rgb_dir = '/home/bird/data2/dataset/kitti_depth_completion/test_depth_completion_anonymous/image'
raw_dir = '/home/bird/data2/dataset/kitti_depth_completion/test_depth_completion_anonymous/velodyne_raw'
pred_dir = '/home/bird/code/depth_completion/test_output_epoch_39/'

img_files = glob.glob(os.path.join(rgb_dir, '*.png'))

for i in range(0, len(img_files)):
    base_dir, fname = os.path.split(img_files[i])
    rgb = np.asarray(IMG.open(os.path.join(rgb_dir, fname)))
    raw = np.asarray(IMG.open(os.path.join(raw_dir, fname)))
    pred = np.asarray(IMG.open(os.path.join(pred_dir, fname)))

    max1 = np.max(raw)
    max2 = np.max(pred)
    max = np.maximum(max1, max2)

    norm = colors.Normalize(vmin=0, vmax=max)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)

    ax1.imshow(rgb)
    ax2.imshow(raw, cmap='jet', norm=norm)
    ax3.imshow(pred, cmap='jet', norm=norm)
    plt.show()
