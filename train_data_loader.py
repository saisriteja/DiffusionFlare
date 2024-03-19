from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pyheif

latent_size = 256

class IITM_Dataset(Dataset):
    def __init__(self, data_dir = 'iitm_data', patch_size = 512):
        self.data_dir = data_dir
        self.patch_size = patch_size
        self.image_filenames = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.heic')]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        # get me a 512,512 image
        img_path = self.image_filenames[index]
        img = None

        if img_path.endswith('.heic'):
            heif_file = pyheif.read(img_path)
            img = Image.frombytes(heif_file.mode, 
                heif_file.size, 
                heif_file.data,
                "raw",
                heif_file.mode,
                heif_file.stride,
            )
        else:
            img = Image.open(img_path)

        if img is None:
            print(f"Failed to load {img_path}")
            return None

        width, height = img.size
        left = np.random.randint(0, width - self.patch_size)
        top = np.random.randint(0, height - self.patch_size)
        right = left + self.patch_size
        bottom = top + self.patch_size

        patch = img.crop((left, top, right, bottom))

        # Apply transformations if needed (e.g., normalization)
        transform = transforms.Compose([
            transforms.Resize((latent_size, latent_size)),
            transforms.ToTensor(),
            # Add more transformations if needed
        ])
        patch = transform(patch)

        return patch