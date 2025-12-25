from torch.utils.data import Dataset, DataLoader

import os
from PIL import Image


def find_file_with_keyword(folder, keyword):
    for fname in os.listdir(folder):
        if keyword in fname.lower():
            return os.path.join(folder, fname)
    return None


def read_split_file(split_path):
    with open(split_path, 'r') as f:
        return [line.strip() for line in f.readlines()]

class MRISliceDatasetSubset(Dataset):
    def __init__(self, sample_dirs, transform=None):
        self.samples = sample_dirs
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = self.samples[idx]

        # case_path = os.path.join(sample_dir)

        img_path = find_file_with_keyword(sample_dir, 'img')
        mask_path = find_file_with_keyword(sample_dir, 'mask')

        img = Image.open(img_path).convert('L')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        return img, mask


def get_train_val_loaders_from_txt(root_dir, transform=None, batch_size=2,
                                   train_txt='dataset/train.txt', val_txt='dataset/val.txt', test_txt='dataset/test.txt'):
    # Read folder names from text files
    train_ids = read_split_file(train_txt)
    val_ids = read_split_file(val_txt)
    test_ids = read_split_file(test_txt)

    # Create full paths
    train_samples = [os.path.join(root_dir, folder) for folder in train_ids]
    val_samples = [os.path.join(root_dir, folder) for folder in val_ids]
    test_samples = [os.path.join(root_dir, folder) for folder in test_ids]

    # Initialize datasets
    train_dataset = MRISliceDatasetSubset(train_samples, transform)
    val_dataset = MRISliceDatasetSubset(val_samples, transform)
    test_dataset = MRISliceDatasetSubset(test_samples, transform)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

