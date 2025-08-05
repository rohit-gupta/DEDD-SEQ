import torch
import torchvision.transforms as T


def convert_image_to_rgb(image):
    return image.convert("RGB")


def get_resnet_transforms():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = T.Compose(
        [
            T.RandomCrop(224),
            convert_image_to_rgb,
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Normalize(mean, std),
        ]
    )

    val_transform = T.Compose(
        [
            T.CenterCrop(224),
            convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    return {"train": train_transform, "val": val_transform}


def get_convnext_transforms():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = T.Compose(
        [
            T.RandomCrop(224),
            convert_image_to_rgb,
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Normalize(mean, std),
        ]
    )

    val_transform = T.Compose(
        [
            T.CenterCrop(224),
            convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    return {"train": train_transform, "val": val_transform}


def get_swin_transforms():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = T.Compose(
        [
            T.RandomCrop(224),
            convert_image_to_rgb,
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Normalize(mean, std),
        ]
    )

    val_transform = T.Compose(
        [
            T.CenterCrop(224),
            convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    test_transform = T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            T.CenterCrop(224),
            convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    return {"train": train_transform, "val": val_transform, "test": test_transform}


# def get_clip_transforms():
#     mean = (0.48145466, 0.4578275, 0.40821073)
#     std = (0.26862954, 0.26130258, 0.27577711)

#     train_transform = T.Compose([
#         T.RandomCrop(224),
#         convert_image_to_rgb,
#         T.ToTensor(),
#         T.RandomHorizontalFlip(),
#         T.Normalize(mean, std),
#     ])

#     val_transform = T.Compose([
#         T.CenterCrop(224),
#         convert_image_to_rgb,
#         T.ToTensor(),
#         T.Normalize(mean, std),
#     ])

#     test_transform = T.Compose([
#         T.CenterCrop(224),
#         convert_image_to_rgb,
#         T.ToTensor(),
#         T.Normalize(mean, std),
#     ])

#     return {'train': train_transform, 'val': val_transform, 'test': test_transform}


def get_clip_transforms():
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    train_transform = T.Compose(
        [
            T.RandomResizedCrop(
                size=224, scale=(0.6, 1.0), ratio=(0.999, 1.001), antialias=True
            ),
            convert_image_to_rgb,
            T.ToTensor(),
            T.RandomHorizontalFlip(),
            T.Normalize(mean, std),
        ]
    )

    eval_transform = T.Compose(
        [
            T.Resize(256, antialias=True),
            T.CenterCrop(224),
            convert_image_to_rgb,
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    return {"train": train_transform, "val": eval_transform, "test": eval_transform}


def random_rotation(image, p=0.3):
    random_num = torch.rand(1).item()
    if random_num <= p / 3:
        image = T.functional.rotate(image, 90)
    elif p / 3 < random_num <= 2 * p / 3:
        image = T.functional.rotate(image, 180)
    elif 2 * p / 3 < random_num <= p:
        image = T.functional.rotate(image, 270)
    return image


import torchvision.transforms.functional as F
from typing import Optional
import copy


class MultiImageRandomResizedCrop(T.RandomResizedCrop):
    """
    Adapted from TORCHVISION.TRANSFORMS.TRANSFORMS source code.
    https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomResizedCrop
    """

    def __init__(
        self,
        size,
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        interpolation=F.InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ):
        super(MultiImageRandomResizedCrop, self).__init__(
            size, scale, ratio, interpolation, antialias
        )

    def forward(self, images):
        i, j, h, w = self.get_params(images[0], self.scale, self.ratio)
        return [
            F.resized_crop(
                image,
                i,
                j,
                h,
                w,
                self.size,
                self.interpolation,
                antialias=self.antialias,
            )
            for image in images
        ]


class MultiImageCenterCrop(T.CenterCrop):
    """
    Adapted from TORCHVISION.TRANSFORMS.TRANSFORMS source code.
    https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#CenterCrop
    """

    def __init__(self, size):
        super(MultiImageCenterCrop, self).__init__(size)

    def forward(self, images):
        return [F.center_crop(image, self.size) for image in images]


class MultiImageRandomHorizontalFlip(T.RandomHorizontalFlip):
    """
    Adapted from TORCHVISION.TRANSFORMS.TRANSFORMS source code.
    https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomHorizontalFlip
    """

    def __init__(self, p=0.5):
        super(MultiImageRandomHorizontalFlip, self).__init__(p)

    def forward(self, images):
        if torch.rand(1) < self.p:
            return [F.hflip(image) for image in images]
        return images


class MultiImageNormalize(T.Normalize):
    """
    Adapted from TORCHVISION.TRANSFORMS.TRANSFORMS source code.
    https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#Normalize
    """

    def __init__(self, mean, std, inplace=False):
        super(MultiImageNormalize, self).__init__(mean, std, inplace)

    def forward(self, images):
        return [
            F.normalize(image, self.mean, self.std, self.inplace) for image in images
        ]


class MultiImageToTensor(T.ToTensor):
    """
    Adapted from TORCHVISION.TRANSFORMS.TRANSFORMS source code.
    https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#ToTensor
    """

    def __init__(self):
        super(MultiImageToTensor, self).__init__()

    def __call__(self, pics):
        return [F.to_tensor(pic) for pic in pics]


def get_feature_diff_transforms():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_transform = T.Compose(
        [
            MultiImageRandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.90, 1.11)),
            MultiImageToTensor(),
            MultiImageRandomHorizontalFlip(),
            MultiImageNormalize(mean, std),
        ]
    )

    val_transform = T.Compose(
        [
            MultiImageCenterCrop(size=224),
            MultiImageToTensor(),
            MultiImageNormalize(mean, std),
        ]
    )

    test_transform = copy.deepcopy(val_transform)

    return {"train": train_transform, "val": val_transform, "test": test_transform}
