# -*- coding: utf-8 -*-
# Author: Xiangde Luo (https://luoxd1996.github.io).
# Date:   16 Dec. 2021
# Implementation for computing the DSC and HD95 in the WORD dataset.
# # Reference:
#   @article{luo2021word,
#   title={{WORD}: A large scale dataset, benchmark and clinical applicable study for abdominal organ segmentation from CT image},
#   author={Xiangde Luo, Wenjun Liao, Jianghong Xiao, Jieneng Chen, Tao Song, Xiaofan Zhang, Kang Li, Dimitris N. Metaxas, Guotai Wang, Shaoting Zhang},
#   journal={arXiv preprint arXiv:2111.02403},
#   year={2021}
# }

import os
import numpy as np
import SimpleITK as sitk
from medpy import metric

def cal_metric(gt, pred, voxel_spacing):
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt, voxelspacing=voxel_spacing)
        return np.array([dice, hd95])
    else:
        return np.array([0.0, 50])

def each_cases_metric(gt, pred, voxel_spacing):
    classes_num = gt.max() + 1
    class_wise_metric = np.zeros((classes_num-1, 2))
    for cls in range(1, classes_num):
        class_wise_metric[cls-1, ...] = cal_metric(pred==cls, gt==cls, voxel_spacing)
    print(class_wise_metric)
    return class_wise_metric

all_results = np.zeros((30,16,2))
for ind, case in enumerate(os.listdir("..WORD/labelsTs/")):
    gt_itk = sitk.ReadImage("../WORD/labelsTs/"+case)
    voxel_spacing = (gt_itk.GetSpacing()[2], gt_itk.GetSpacing()[0], gt_itk.GetSpacing()[1])
    gt_array = sitk.GetArrayFromImage(gt_itk)
    pred_itk = sitk.ReadImage("../WORD/predictionsTs/"+case)
    pred_array = sitk.GetArrayFromImage(pred_itk)
    print(case)
    all_results[ind, ...] = each_cases_metric(gt_array, pred_array, voxel_spacing)
    
np.save("../WORD/predictionsTs_DSC_HD95.npy", all_results)
