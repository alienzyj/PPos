import os
import time

import numpy as np
import torch

from models import Classifier
from models import UNetClassifier
from models import UNet
from my_utils import post_process, my_iou, hausdorff_distance, dice_coef, unet_predict, fine_segmentation, pixel_acc, pixel_precision_recall_f1

t = 19
u = 52
c = 3

print(time.strftime('%Y-%m-%d %H:%M:%S'))

unet_classifier = UNetClassifier((5, 4, 4, 4), (3, 3, 4, 4), (2, 2, 2, 2), 8, (5, 9, 17), (3, 5, 9), 0.5)
unet_classifier.load_state_dict(torch.load(f'weights/unet_classifier/unet_classifier{t}.pth', "cpu")["model"])

unet = UNet((3, 4, 6, 3), (2, 3, 4, 3, 2), (5, 9, 17))
unet.load_state_dict(torch.load(f'weights/unet/unet{u}.pth', 'cpu')['model'])

threshold_coarse = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# threshold_coarse = (0.6,)
threshold_fine = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
# threshold_fine = (0.7,)
print(threshold_coarse)
print(threshold_fine)

classifier = Classifier((3, 4, 23, 3))
classifier.load_state_dict(torch.load(f'weights/classifier/classifier{c}.pth', 'cpu')['model'])

patient_path = f'test/samples'
patients = os.listdir(patient_path)
patients.sort()

mask_path = f'test/masks'

files = f'{t}-{u}-{c}'
dir = f'metrics/unet/{files}'
print(dir)

iou = my_iou()

test_iou = np.zeros((len(threshold_coarse), len(threshold_fine), len(patients)))
hausdorff = np.zeros((len(threshold_coarse), len(threshold_fine), len(patients)))
test_dice = np.zeros((len(threshold_coarse), len(threshold_fine), len(patients)))
test_precision = np.zeros((len(threshold_coarse), len(threshold_fine), len(patients)))
test_recall = np.zeros((len(threshold_coarse), len(threshold_fine), len(patients)))
test_acc = np.zeros((len(threshold_coarse), len(threshold_fine), len(patients)))
test_f1 = np.zeros((len(threshold_coarse), len(threshold_fine), len(patients)))
for i, patient in enumerate(patients):
    print(patient)
    sample = np.load(os.path.join(patient_path, patient))
    pred = unet_predict(sample, (256, 512), unet)
    mask = np.load(os.path.join(mask_path, patient))
    for j, unet_threshold_coarse in enumerate(threshold_coarse):
        output_coarse = post_process(sample, pred, classifier, unet_threshold_coarse, 1.5, (48, 96, 96))
        for k, unet_threshold_fine in enumerate(threshold_fine):
            output_fine = fine_segmentation(sample, output_coarse, unet_classifier, unet_threshold_fine, (64, 128, 128))
            if np.sum(output_fine) > 0 and np.sum(mask) > 0:
                hausdorff[j][k][i] = hausdorff_distance(output_fine, mask)
            else:
                hausdorff[j][k][i] = -1
            test_iou[j][k][i] = iou(torch.from_numpy(output_fine), torch.from_numpy(mask))
            test_dice[j][k][i] = dice_coef(torch.from_numpy(output_fine.astype(np.float32)),
                                           torch.from_numpy(mask.astype(np.float32))).item()
            test_acc[j][k][i] = pixel_acc(output_fine, mask)
            test_precision[j][k][i], test_recall[j][k][i], test_f1[j][k][i] = pixel_precision_recall_f1(output_fine, mask)

os.makedirs(dir)

np.save(f'{dir}/iou.npy', test_iou)
np.save(f'{dir}/hd.npy', hausdorff)
np.save(f'{dir}/dice.npy', test_dice)
np.save(f'{dir}/acc.npy', test_acc)
np.save(f'{dir}/precision.npy', test_precision)
np.save(f'{dir}/recall.npy', test_recall)
np.save(f'{dir}/f1.npy', test_f1)

print('finish')
print(time.strftime('%Y-%m-%d %H:%M:%S'))
