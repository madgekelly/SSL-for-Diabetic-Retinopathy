from PIL import Image
import pandas as pd
from torch.utils.data import Dataset


class DataSetFromFolder(Dataset):
    def __init__(self, image_dir, csv, transform, label=True, index=True, mode='train'):
        self.label = label
        self.index = index
        self.image_dir = image_dir
        df = pd.read_csv(csv)
        images = df[df[mode]]['image'].tolist()
        self.total_imgs = [image_dir + i for i in images]
        self.labels = df[df[mode]]['level'].tolist()
        self.indices = df[df[mode]].index.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        image = Image.open(self.total_imgs[idx]).convert("RGB")
        transform_image = self.transform(image)
        y = self.labels[idx]
        if self.index:
            if self.label:
                data = self.indices[idx], transform_image, self.labels[idx]
            else:
                data = self.indices[idx], transform_image
        else:
            if self.label:
                data = transform_image, self.labels[idx]
            else:
                data = transform_image
        return data
