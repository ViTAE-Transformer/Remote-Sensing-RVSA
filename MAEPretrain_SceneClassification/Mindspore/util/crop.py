import math

import mindspore as ms
import mindspore.dataset.vision as vision_transforms
import mindspore.numpy as mnp
from PIL import Image

class RandomResizedCrop:
    """
    RandomResizedCrop for matching TF/TPU implementation: no for-loop is used.
    This may lead to results different with torchvision's version.
    Following BYOL's TF code:
    https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L206
    """
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = img.size  # 获取图像尺寸
        area = height * width

        target_area = area * ms.Tensor(1).uniform(scale[0], scale[1]).asnumpy().item()
        log_ratio = mnp.log(ms.Tensor(ratio))
        aspect_ratio = mnp.exp(ms.Tensor(1).uniform(log_ratio[0], log_ratio[1])).asnumpy().item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        w = min(w, width)
        h = min(h, height)

        i = ms.Tensor(1).uniform(0, height - h + 1).astype(ms.int32).asnumpy().item()
        j = ms.Tensor(1).uniform(0, width - w + 1).astype(ms.int32).asnumpy().item()

        return i, j, h, w
