import numpy as np
import cv2
from numpy import random
import torch
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform
from albumentations.augmentations.bbox_utils import convert_bboxes_from_albumentations, convert_bboxes_to_albumentations

PASCAL_VOC = 'pascal_voc'


class Expand(DualTransform):
    def __init__(self, mean, always_apply=False, p=0.5, ):
        super(Expand, self).__init__(always_apply, p)
        self.mean = mean

    @property
    def targets(self):
        return {
            'image': self.apply,
            'mask': self.apply_to_mask,
            "masks": self.apply_to_masks,
            'bboxes': self.apply_to_bboxes
        }

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width, depth = img.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width * ratio - width)
        top = random.uniform(0, height * ratio - height)
        return {
            'ratio': ratio,
            'left': left,
            'top': top
        }

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["image"]

    def apply_to_bboxes(self, bboxes, ratio=None, left=None, top=None, **params):
        if len(bboxes) > 0:
            bboxes = convert_bboxes_from_albumentations(bboxes, PASCAL_VOC, params['rows'], params['cols'])
            bboxes = np.array(bboxes)
            bboxes[:, :2] += (int(left), int(top))
            bboxes[:, 2:4] += (int(left), int(top))
            new_image_rows, new_image_cols = int(params['rows'] * ratio), int(params['cols'] * ratio)
            bboxes = convert_bboxes_to_albumentations(bboxes, PASCAL_VOC, new_image_rows, new_image_cols)
        return bboxes

    def apply_to_mask(self, img, ratio=None, left=None, top=None, **params):
        return self.apply(img, ratio, left, top, is_mask=True, **params)

    def apply(self, img, ratio=None, left=None, top=None, is_mask=False, **params):
        height, width, depth = img.shape
        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=img.dtype)
        expand_image[:, :, :] = self.mean if not is_mask else 0
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = img
        image = expand_image
        return image

    def get_transform_init_args_names(self):
        return "mean"


class MinIoURandomCrop(DualTransform):
    def __init__(self, always_apply=False, p=0.5, ):
        super(MinIoURandomCrop, self).__init__(always_apply, p)
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    @property
    def targets(self):
        return {
            'image': self.apply,
            'mask': self.apply,
            "masks": self.apply_to_masks,
            'bboxes': self.apply_to_bboxes,
        }

    def apply(self, img, crop_rect=None, **params):
        if crop_rect is None:
            return img
        current_image = img[crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2], :]
        return current_image

    def apply_to_masks(self, masks, box_select_mask=None, **params):
        if box_select_mask is None:
            return masks
        masks = np.stack(masks,axis=0)
        masks = masks[box_select_mask]
        return [self.apply(mask, **params) for mask in masks]

    def apply_to_bboxes(self, bboxes, crop_rect=None, box_select_mask=None, **params):

        if box_select_mask is None or crop_rect is None or len(bboxes) == 0:
            return bboxes

        bboxes = np.array(convert_bboxes_from_albumentations(bboxes, PASCAL_VOC, params['rows'], params['cols']))

        # take only matching gt boxes
        current_boxes = bboxes[box_select_mask, :]
        # should we use the box left and top corner or the crop's
        current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                          crop_rect[:2])
        # adjust to crop (by substracting crop's left,top)
        current_boxes[:, :2] -= crop_rect[:2]

        current_boxes[:, 2:4] = np.minimum(current_boxes[:, 2:4],
                                           crop_rect[2:])
        # adjust to crop (by substracting crop's left,top)
        current_boxes[:, 2:4] -= crop_rect[:2]
        current_boxes = convert_bboxes_to_albumentations(current_boxes, PASCAL_VOC, crop_rect[3] - crop_rect[1],
                                                         crop_rect[2] - crop_rect[0])
        return current_boxes

    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        rows, cols = image.shape[:2]
        albu_bboxes = params["bboxes"]
        boxes = np.array(convert_bboxes_from_albumentations(albu_bboxes, PASCAL_VOC, rows, cols))
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:
                return {}

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                if len(boxes) == 0:
                    # current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                    #                 :]
                    return {
                        'crop_rect': rect,

                    }

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = self.jaccard_numpy(boxes[:, :4], rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                # current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                #                               :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                box_select_mask = m1 * m2

                # have any valid boxes? try again if not
                if not box_select_mask.any():
                    continue

                # # take only matching gt boxes
                # current_boxes = boxes[mask, :].copy()
                #
                # # take only matching gt labels
                # current_labels = labels[mask]
                #
                # # should we use the box left and top corner or the crop's
                # current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                #                                   rect[:2])
                # # adjust to crop (by substracting crop's left,top)
                # current_boxes[:, :2] -= rect[:2]
                #
                # current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                #                                   rect[2:])
                # # adjust to crop (by substracting crop's left,top)
                # current_boxes[:, 2:] -= rect[:2]

                return {
                    'crop_rect': rect,
                    'box_select_mask': box_select_mask
                }

    def jaccard_numpy(self, box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: Multiple bounding boxes, Shape: [num_boxes,4]
            box_b: Single bounding box, Shape: [4]
        Return:
            jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
        """
        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))  # [A,B]
        area_b = ((box_b[2] - box_b[0]) *
                  (box_b[3] - box_b[1]))  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]

    def intersect(self, box_a, box_b):
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]

    def get_params(self):
        return {}

    @property
    def targets_as_params(self):
        return ["image", "bboxes"]

    def get_transform_init_args_names(self):
        return "mean"


class PhotometricDistort(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(PhotometricDistort, self).__init__(always_apply, p)
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))
        self.pd = [
            self.randomContrast,
            self.convertColorBGR2HSV,
            self.randomSaturation,
            self.randomHue,
            self.convertColorHSV2BGR,
            self.randomContrast
        ]

    def apply(self, img, **params):
        im = img.astype(np.float32)
        # RGB to BGR
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = self.randomBrightness(im)
        if random.randint(2):
            for function in self.pd[:-1]:
                im = function(im)
        else:
            for function in self.pd[1:]:
                im = function(im)
        im = self.randomLightingNoise(im)
        #  BGR To RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im

    def randomSaturation(self, image, lower=0.5, upper=1.5):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(lower, upper)
        return image

    def randomHue(self, image, delta=18.0):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-delta, delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image

    def randomContrast(self, image, lower=0.5, upper=1.5):
        if random.randint(2):
            alpha = random.uniform(lower, upper)
            image *= alpha
        return image

    def randomBrightness(self, image, delta=32):
        if random.randint(2):
            delta = random.uniform(-delta, delta)
            image += delta
        return image

    def convertColorBGR2HSV(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        return image

    def convertColorHSV2BGR(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        return image

    def randomLightingNoise(self, img):
        channels = self.perms[random.randint(len(self.perms))]
        img = F.channel_shuffle(img, channels)
        return img

    @property
    def targets(self):
        return {
            'image': self.apply
        }


class SubtractMeans(ImageOnlyTransform):
    def __init__(self, mean, always_apply=False, p=1.0):
        super(SubtractMeans, self).__init__(always_apply, p)
        self.mean = mean

    def apply(self, img, **params):
        image = img.astype(np.float32)
        image -= self.mean
        return image


class Normalize(ImageOnlyTransform):
    def __init__(self, mean, std, to_bgr=False, to_float=False, always_apply=False, p=1.0):
        super(Normalize, self).__init__(always_apply, p)
        self.mean = mean
        self.std = std
        self.to_bgr = to_bgr
        self.to_float =to_float

    def apply(self, img, **params):
        image = img.astype(np.float32)
        if self.to_bgr:
            image = image[:,:,(2,1,0)]
        image = (image - self.mean) / self.std
        if self.to_float:
            image = image/255
        return image
