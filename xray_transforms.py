"""Use full prepared square images consistently, with no random or center crops."""

from torchvision import transforms as T
from ultralytics.data.dataset import ClassificationDataset
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator


class FullImageDataset(ClassificationDataset):
    def __init__(self, root, args, augment=False, prefix="", names=None):
        super().__init__(root, args, augment=augment, prefix=prefix, names=names)
        # Inputs are already letterboxed by prepare. The resize is primarily for
        # the 64-pixel smoke test; all stages preserve the entire square image.
        self.torch_transforms = T.Compose([
            T.Resize((args.imgsz, args.imgsz)),
            T.ToTensor(),
        ])


class FullImageTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        return FullImageDataset(img_path, self.args, augment=mode == "train",
                                prefix=mode, names=self.data["names"])


class FullImageValidator(ClassificationValidator):
    def build_dataset(self, img_path):
        return FullImageDataset(img_path, self.args, prefix=self.args.split, names=self.names)
