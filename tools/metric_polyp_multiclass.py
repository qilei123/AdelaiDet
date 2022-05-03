import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
# from scipy.interpolate import spline
# from scipy.interpolate import make_interp_spline, BSpline
from matplotlib import colors
# from utils.colormap import colormap
# from utils import cv2_util
from collections import defaultdict
from itertools import chain
import json


# import json
class Metric(object):
    def __init__(self, mode='center', iou_thresh=0, visualize=False, visualization_root='demo/',
                 image_classification=False, classes=('Adenomatous', 'non-Adenomatous')):

        self.classes = classes
        self.TPs = defaultdict(list)
        self.FNs = defaultdict(list)
        self.FPs = defaultdict(list)
        self.TNs = defaultdict(list)

        self.TPs_binary = []
        self.FNs_binary = []
        self.FPs_binary = []
        self.TNs_binary = []

        self.adenomatous_to_non_adenomatous_size = []
        self.non_adenomatous_to_adenomatous_size = []
        self.adenomatous_size = []
        self.non_adenomatous_size = []
        """
                GT
                 1        |        2
        ------------------------------------
    Pred     |   
          1  |
          -  |
          2  |
             |   
        """
        self.confusion_matrix = defaultdict(lambda: defaultdict(int))
        assert mode == 'center' or mode == 'iou', f'({mode}) mode is not supported'
        self.mode = mode
        self.iou_thresh = iou_thresh
        self.image_classification = image_classification
        self.mask_iou_sum = 0
        self.mask_iou_count = 0
        self.dice_sum = 0
        # in BGR order
        self.FP_color = (255, 0, 0)
        self.Detection_color = (0, 255, 0)
        self.GT_color = (0, 0, 255)
        self.visualize = visualize
        self.total_gt = 0.0

        if visualize:
            #  create image folder for saving detection result
            self.detection_folder = visualization_root + 'ALL/'
            self.false_positive_folder = visualization_root + 'FP/'
            self.false_negative_folder = visualization_root + 'FN/'
            os.makedirs(self.detection_folder, exist_ok=True)
            os.makedirs(self.false_positive_folder, exist_ok=True)
            os.makedirs(self.false_negative_folder, exist_ok=True)
            os.popen('rm -r ' + self.detection_folder + '*')
            os.popen('rm -r ' + self.false_positive_folder + '*')
            os.popen('rm -r ' + self.false_negative_folder + '*')

    def eval_add_result(self,
                        ground_truth: list,
                        pred_points: list,
                        image: np.ndarray = None,
                        image_name=None,
                        masks=None,
                        mask_target=None
                        ):
        '''

        Args:
            ground_truth: Nx5 array in format of [x,y,x,y,class]
            pred_points: Nx5 array in format of [x,y,x,y,class]
            image:
            image_name:
            masks:
            mask_target:

        Returns:

        '''
        self.eval_add_result_binary(ground_truth, pred_points)
        if self.visualize:
            FPimage = image.copy()
            FNimage = image.copy()
            Detectionimage = image.copy()

            for idx, pt in enumerate(pred_points):
                pt1 = tuple([int(pt[0]), int(pt[1])])
                pt2 = tuple([int(pt[2]), int(pt[3])])
                cv2.rectangle(Detectionimage, pt1, pt2, self.Detection_color, 2)
                cv2.putText(Detectionimage,
                            str(pt[4]),
                            (int(pt[0]), int(pt[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .5, self.Detection_color, 1
                            )

            if masks is not None:
                Detectionimage = self.overlay_mask(Detectionimage, masks)

        missing = False
        self.total_gt += len(ground_truth)
        for index_gt_box, gt_box in enumerate(ground_truth):
            hasTP = False
            gt = gt_box
            if gt[4] == 1:
                self.adenomatous_size.append(
                    (gt[2] - gt[0]) * (gt[3] - gt[1]) * 512 * 512 / (image.shape[0] * image.shape[1]))
            else:
                self.non_adenomatous_size.append(
                    (gt[2] - gt[0]) * (gt[3] - gt[1]) * 512 * 512 / (image.shape[0] * image.shape[1]))
            not_matched = []
            for index_pred, j in enumerate(pred_points):
                if self.mode == 'center':
                    ctx = j[0] + (j[2] - j[0]) * 0.5
                    cty = j[1] + (j[3] - j[1]) * 0.5
                    bbox_matched = gt[0] < ctx < gt[2] and gt[1] < cty < gt[3]

                elif self.mode == 'iou':
                    query_area = (j[2] - j[0]) * (j[3] - j[1])
                    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                    iw = (min(j[2], gt[2]) - max(j[0], gt[0]))
                    ih = (min(j[3], gt[3]) - max(j[1], gt[1]))
                    iw = max(0, iw)
                    ih = max(0, ih)
                    ua = query_area + gt_area - (iw * ih)
                    overlaps = (iw * ih) / float(ua)
                    bbox_matched = overlaps > self.iou_thresh
                # 如果打中GT框
                if bbox_matched:
                    # 如果GT框没有被match 过， 并且pred和GT 的class一样 判定为TP +1
                    if not hasTP and gt[4] == j[4]:
                        self.TPs[int(j[4])].append(j)
                        hasTP = True
                    # 如果match到GT 但是 class不一样 添加到候选框，等待下一轮GT match
                    elif gt[4] != j[4]:
                        not_matched.append(j)
                        gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1]) * 512 * 512 / (image.shape[0] * image.shape[1])
                        if gt[4] == 1:
                            self.adenomatous_to_non_adenomatous_size.append(gt_area)
                        else:
                            self.non_adenomatous_to_adenomatous_size.append(gt_area)
                    # 不管gt match的唯一性， 所有的match的prediction都要添加到confusion matrix
                    gt_cls = int(j[4])
                    pred_cls = int(gt[4])
                    self.confusion_matrix[pred_cls][gt_cls] += 1
                    if masks is not None:
                        self.mask_iou_sum += self._mask_iou(masks[index_pred].squeeze(), mask_target[index_gt_box])
                        self.dice_sum += self._dice(masks[index_pred].squeeze(), mask_target[index_gt_box])
                        self.mask_iou_count += 1
                else:
                    not_matched.append(j)
            pred_points = not_matched
            # 如果GT 没有被match过 FN+1
            if not hasTP:
                self.FNs[int(gt[4])].append(gt)

                if self.visualize:
                    # Draw False negative rect
                    missing = True
                    pt1 = tuple([int(gt[0]), int(gt[1])])
                    pt2 = tuple([int(gt[2]), int(gt[3])])
                    cv2.rectangle(FNimage, pt1, pt2, self.GT_color, 2)
                    cv2.putText(FNimage,
                                self.classes[int(gt[4] - 1)],
                                pt1,
                                cv2.FONT_HERSHEY_SIMPLEX,
                                .5, self.GT_color, 1
                                )

            if self.visualize:
                # Draw groundturth on detection and FP images
                pt1 = tuple([int(gt[0]), int(gt[1])])
                pt2 = tuple([int(gt[2]), int(gt[3])])
                cv2.rectangle(Detectionimage, pt1, pt2, self.GT_color, 2)
                cv2.rectangle(FPimage, pt1, pt2, self.GT_color, 2)
                cv2.putText(Detectionimage,
                            self.classes[int(gt[4] - 1)],
                            pt1,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .5, self.GT_color, 1
                            )
                cv2.putText(FPimage,
                            self.classes[int(gt[4] - 1)],
                            pt1,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .5, self.GT_color, 1
                            )

        if self.visualize:
            if missing:
                cv2.imwrite(self.false_negative_folder + str(image_name) + '.jpg', FNimage)
            cv2.imwrite(self.detection_folder + str(image_name), Detectionimage)

        if len(pred_points) > 0 and self.visualize:
            # Draw false positive rect
            for fp in pred_points:
                pt1 = tuple([int(fp[0]), int(fp[1])])
                pt2 = tuple([int(fp[2]), int(fp[3])])
                cv2.rectangle(FPimage, pt1, pt2, self.FP_color, 2)
                cv2.putText(FPimage,
                            self.classes[int(fp[4] - 1)],
                            pt1,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            .5, self.FP_color, 1
                            )

            cv2.imwrite(self.false_positive_folder + str(image_name) + '.jpg', FPimage)
        # 剩下的predict框都是FP
        for p in pred_points:
            self.FPs[int(p[4])].append(p)

    def eval_add_result_binary(self, ground_truth: list,
                               pred_points: list,
                               ):

        for index_gt_box, gt_box in enumerate(ground_truth):
            hasTP = False
            gt = gt_box
            not_matched = []
            for index_pred, j in enumerate(pred_points):
                if self.mode == 'center':
                    ctx = j[0] + (j[2] - j[0]) * 0.5
                    cty = j[1] + (j[3] - j[1]) * 0.5
                    bbox_matched = gt[0] < ctx < gt[2] and gt[1] < cty < gt[3]

                elif self.mode == 'iou':
                    query_area = (j[2] - j[0]) * (j[3] - j[1])
                    gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
                    iw = (min(j[2], gt[2]) - max(j[0], gt[0]))
                    ih = (min(j[3], gt[3]) - max(j[1], gt[1]))
                    iw = max(0, iw)
                    ih = max(0, ih)
                    ua = query_area + gt_area - (iw * ih)
                    overlaps = (iw * ih) / float(ua)
                    bbox_matched = overlaps > self.iou_thresh
                if bbox_matched:
                    if not hasTP:
                        self.TPs_binary.append(j)
                        hasTP = True
                else:
                    not_matched.append(j)
            pred_points = not_matched
            if not hasTP:
                self.FNs_binary.append(gt)
        #  add FP here
        self.FPs_binary += pred_points

    def get_precision_recall(self):
        TP = len(self.TPs) * 1.0
        FP = len(self.FPs) * 1.0
        FN = len(self.FNs) * 1.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        return precision, recall

    def get_miou_mdice(self):
        mask_iou_count = self.mask_iou_count + 1e-5
        return self.mask_iou_sum / mask_iou_count, self.dice_sum / mask_iou_count

    def get_result(self):
        '''

        Returns: a dict of following format:
        {
            1:                       <<--------- '1' (int) is class number
                {
                    'F1': ,
                    'F2': ,
                    'precision': ,
                    'recall' : ,
                    'TP': ,             <<--------- number of TPs
                    'TN': ,             <<--------- number of TNs
                    'FP': ,             <<--------- number of FPs
                    'FN': ,             <<--------- number of FNs
                },
            ...
            'overall':
                {
                    'F1': ,
                    'F2': ,
                    'precision': ,
                    'recall' : ,
                    'TP': ,             <<--------- number of TPs
                    'TN': ,             <<--------- number of TNs
                    'FP': ,             <<--------- number of FPs
                    'FN': ,             <<--------- number of FNs
                },
            'confusion_matrix':
                {
                    1:{ 1: ,
                        ...
                        3:
                        },
                    ...
                    3:{
                        1: ,
                        ...
                        3:
                    }

                }
        }

        '''
        unique_cat = sorted(set([k for k in chain.from_iterable([self.TPs.keys(),
                                                                 self.FPs.keys(),
                                                                 self.TNs.keys(),
                                                                 self.FNs.keys()]
                                                                )]))
        result = dict()
        result['overall'] = defaultdict(int)
        # calculate performance metric for each category
        for cat in unique_cat:
            TP = len(self.TPs[cat]) * 1.0
            FP = len(self.FPs[cat]) * 1.0
            TN = len(self.TNs[cat]) * 1.0
            FN = len(self.FNs[cat]) * 1.0
            result['overall']['TP'] += TP
            result['overall']['FP'] += FP
            result['overall']['TN'] += TN
            result['overall']['FN'] += FN
            result[cat] = self.__get_performance(FN, FP, TN, TP)
        # calculate performance metric for overall
        result['overall'] = self.__get_performance(
            result['overall']['FN'],
            result['overall']['FP'],
            result['overall']['TN'],
            result['overall']['TP']
        )
        # add binary performance matrix here
        result['binary'] = dict()
        result['binary']['FN'] = len(self.FNs_binary) * 1.0
        result['binary']['FP'] = len(self.FPs_binary) * 1.0
        result['binary']['TN'] = len(self.TNs_binary) * 1.0
        result['binary']['TP'] = len(self.TPs_binary) * 1.0
        result['binary'] = self.__get_performance(
            result['binary']['FN'],
            result['binary']['FP'],
            result['binary']['TN'],
            result['binary']['TP']
        )
        # fill in missing cat and value if possible form confusion_matrix
        for c in unique_cat:
            self.confusion_matrix[cat][c] = self.confusion_matrix[cat][c]
        result['confusion_matrix'] = self.confusion_matrix

        return result

    def __get_performance(self, FN, FP, TN, TP):
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * (precision * recall) / max((precision + recall), 1e-5)
        F2 = 5 * (precision * recall) / max((4 * precision + recall), 1e-5)
        return {
            'F1': F1,
            'F2': F2,
            'precision': precision,
            'recall': recall,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN,
        }

    def reset(self):
        self.TPs = defaultdict(list)
        self.FNs = defaultdict(list)
        self.FPs = defaultdict(list)
        self.TNs = defaultdict(list)

        self.TPs_binary = []
        self.FNs_binary = []
        self.FPs_binary = []
        self.TNs_binary = []
        self.total_gt = 0.0

    # def overlay_mask(self, image, masks):
    #     import torch
    #     if isinstance(masks, torch.Tensor):
    #         masks = masks.numpy()
    #
    #     # original
    #     # colors = self.compute_colors_for_labels(labels).tolist()
    #
    #     # Detectron.pytorch for matplotlib colors
    #     colors = colormap(rgb=True).tolist()
    #     # colors = self.compute_colors_for_labels_yolact(labels)
    #
    #     mask_img = np.copy(image)
    #
    #     for mask, color in zip(masks, colors):
    #         # color_mask = color_list[color_id % len(color_list)]
    #         # color_id += 1
    #
    #         thresh = mask[0, :, :, None].astype(np.uint8)
    #         contours, hierarchy = cv2_util.findContours(
    #             thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    #         )
    #         mask_img = cv2.drawContours(mask_img, contours, -1, color, 2)
    #
    #     # composite = image
    #     alpha = 0.6
    #     composite = cv2.addWeighted(image, 1.0 - alpha, mask_img, alpha, 0)
    #     # composite = cv2.addWeighted(image, 1.0 - alpha, mask, alpha, 0)
    #
    #     return composite

    def _mask_iou(self, mask1, mask2):
        intersection = np.sum(mask1 * mask2)
        area1 = np.sum(mask1)
        area2 = np.sum(mask2)
        union = (area1 + area2) - intersection
        ret = intersection / union
        return ret

    def _dice(self, mask1, mask2):
        intersection = np.sum(mask1 * mask2)
        area1 = np.sum(mask1)
        area2 = np.sum(mask2)
        ret = 2 * intersection / (area1 + area2)
        return ret


if __name__ == '__main__':
    import pprint

    m = Metric()
    gt_box = [
        [0, 0, 100, 100, 1],
        [120, 120, 200, 200, 2],
    ]
    pred_box = [
        [0, 0, 50, 50, 1],
        [25, 25, 50, 50, 1],
        [25, 25, 30, 30, 2],
    ]
    m.eval_add_result(gt_box, pred_box)
    pprint.pprint(m.get_result())
