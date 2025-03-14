import math
import torch
import numpy as np
import torchvision.transforms as transforms
from timm.data.mixup import Mixup
from torch import Tensor
from typing import List, Tuple, Optional, Dict
from torchvision.transforms import functional as F, InterpolationMode
import random

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def get_train_aug(args):
    if args.DS == 'dvs_cifar10':
        return transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    elif args.DS == 'dvs_gesture':
        return transforms.Compose([
            transforms.Resize(size=(32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

def get_test_aug(args):
    if args.DS == 'dvs_cifar10':
        return transforms.Compose([
            transforms.Resize(size=(64, 64)),
        ])
    elif args.DS == 'dvs_gesture':
        return transforms.Compose([
            transforms.Resize(size=(32, 32)),
        ])

def get_trival_aug():
    return SNNAugmentWide()

def get_mixup_fn(args, num_classes: int):
    mixup_args = dict(
        mixup_alpha=args.mixup, prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, 
        mode=args.mixup_mode, label_smoothing=args.smoothing, num_classes=num_classes)
    return Mixup(**mixup_args)

class SNNAugmentWide(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, num_magnitude_bins: int = 31, interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.cutout = transforms.RandomErasing(p=1, scale=(0.001, 0.11), ratio=(1, 1))


    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        return {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(-0.3, 0.3, num_bins), True),
            "TranslateX": (torch.linspace(-5.0, 5.0, num_bins), True),
            "TranslateY": (torch.linspace(-5.0, 5.0, num_bins), True),
            "Rotate": (torch.linspace(-30.0, 30.0, num_bins), True),
            "Cutout": (torch.linspace(1.0, 30.0, num_bins), True)
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) \
            if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0
        if op_name == "Cutout":
            return self.cutout(img)
        else:
            return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)

def _apply_op(img: Tensor, op_name: str, magnitude: float,
              interpolation: InterpolationMode, fill: Optional[List[float]]):
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                       interpolation=interpolation, fill=fill)
    elif op_name == "ShearY":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                       interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)

    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img

class DVSTransform:
    def __init__(self, c, d, e, n):
        self.c = c
        self.d = d
        self.e = e
        self.n = n

    def __call__(self, img):
        # T C H W
        w = img.shape[-1]
        img = torch.Tensor(img.astype(np.float32))

        if random.random() > 0.5:
            img = F.hflip(img)

        # 1
        a = int(random.uniform(-self.c, self.c))
        b = int(random.uniform(-self.c, self.c))
        img = torch.roll(img, shifts=(a, b), dims=(1, 2))

        # 2
        mask = 0
        length = random.uniform(1, self.e)
        height = random.uniform(1, self.e)
        center_x = random.uniform(0, w)
        center_y = random.uniform(0, w)

        small_y = int(center_y - height / 2)
        big_y = int(center_y + height / 2)
        small_x = int(center_x - length / 2)
        big_x = int(center_x + length / 2)

        if small_y < 0:
            small_y = 0
        if small_x < 0:
            small_x = 0
        if big_y > w:
            big_y = w
        if big_x > w:
            big_x = w

        img[:, :, small_y:big_y, small_x:big_x] = mask

        return img


class Cutout:
    """Randomly mask out one or more patches from an image.
    Altered from https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    Args:
    n_holes (int): Number of patches to cut out of each image.
    length (int): The length (in pixels) of each square patch.
    """

    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
        img (Tensor): Tensor image of size (C, H, W).
        Returns:
        Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(-2)
        w = img.size(-1)

        mask = torch.ones((h, w)).type_as(img)

        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.0

            mask = mask.expand_as(img)
            img = img * mask

        return img