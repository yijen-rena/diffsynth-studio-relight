import torch, os, torchvision
from torchvision import transforms
import pandas as pd
from PIL import Image

# from data_utils import read_hdr, env_map_to_cam_to_world_by_convention, reinhard_tonemap

class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, steps_per_epoch=10000, height=1024, width=1024, center_crop=True, random_flip=False):
        self.steps_per_epoch = steps_per_epoch
        # metadata = pd.read_csv(os.path.join(dataset_path, "train/metadata.csv"))
        metadata = pd.read_csv("/ocean/projects/cis250002p/rju/datasets/metadata.csv")
        # self.path = [os.path.join(dataset_path, "train", file_name) for file_name in metadata["file_name"]]
        # self.text = metadata["text"].to_list()
        self.path = metadata["file_name"].to_list()
        self.text = [""] * len(self.path)
        self.height = height
        self.width = width
        self.image_processor = transforms.Compose(
            [
                transforms.CenterCrop((height, width)) if center_crop else transforms.RandomCrop((height, width)),
                transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        text = self.text[data_id]
        image = Image.open(self.path[data_id]).convert("RGB")
        target_height, target_width = self.height, self.width
        width, height = image.size
        scale = max(target_width / width, target_height / height)
        shape = [round(height*scale),round(width*scale)]
        image = torchvision.transforms.functional.resize(image,shape,interpolation=transforms.InterpolationMode.BILINEAR)
        image = self.image_processor(image)
        return {"text": text, "image": image}


    def __len__(self):
        return self.steps_per_epoch
