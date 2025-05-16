import os, glob, torch, numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

def augment_eeg(data, noise_std=0.01,shift_max=10, scale_range=(0.90, 1.10)):
    noise = torch.randn_like(data) * noise_std
    data = data + noise

    scale = torch.empty(1).uniform_(*scale_range).item()  # ±10% amplitude jitter
    data = data * scale

    shift = np.random.randint(-shift_max, shift_max + 1)
    data = torch.roll(data, shifts=shift, dims=-1)

    return data

class CaueegDataset(Dataset):
    def __init__(self, data_dir, augment=False, noise_std=0.01, shift_max=10, scale_range=(0.95, 1.05)):
        self.augment = augment
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.scale_range = scale_range

        all_files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.files = []
        self.label_counts = {}

        for f in all_files:
            parts = os.path.basename(f).replace(".npy", "").split("_")
            raw_label = int(parts[2])
            self.files.append(f)
            self.label_counts[raw_label] = self.label_counts.get(raw_label, 0) + 1

        print(f"Loaded {len(self.files)} EEG files from {data_dir}")
        print("Label distribution →", self.label_counts)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        parts = os.path.basename(filepath).replace(".npy", "").split("_")
        age = int(parts[1])
        raw_label = int(parts[2])  # No remapping

        data = np.load(filepath)
        data = data[np.newaxis, :, :, :]  # shape: (1, C, H, W)
        data = torch.tensor(data, dtype=torch.float32)

        # Only augment dementia (raw_label == 2)
        if self.augment and raw_label == 2:
            data = augment_eeg(data, self.noise_std, self.shift_max, self.scale_range)

        return data, raw_label, age

def get_dataloaders(train_dir, val_dir, test_dir, batch_size=4):
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    return (
        DataLoader(CaueegDataset(train_dir, augment=True), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(CaueegDataset(val_dir, augment=False), batch_size=batch_size, num_workers=num_workers),
        DataLoader(CaueegDataset(test_dir, augment=False), batch_size=batch_size, num_workers=num_workers)
    )
"""
class CaueegDataset(Dataset):
    def __init__(self, data_dir, augment=False, noise_std=0.01, shift_max=10, scale_range=(0.95, 1.05)):
        self.augment = augment
        self.noise_std = noise_std
        self.shift_max = shift_max
        self.scale_range = scale_range

        all_files = glob.glob(os.path.join(data_dir, "*.npy"))
        self.files = []
        self.label_counts = {}

        for f in all_files:
            parts = os.path.basename(f).replace(".npy", "").split("_")
            raw_label = int(parts[2])
            if raw_label in [1, 2]:  # Only keep MCI and Dementia
                self.files.append(f)
                mapped_label = 0 if raw_label == 1 else 1  # MCI → 0, Dementia → 1
                self.label_counts[mapped_label] = self.label_counts.get(mapped_label, 0) + 1

        print(f"Loaded {len(self.files)} MCI/Dementia files from {data_dir}")
        print("Label distribution →", self.label_counts)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        parts = os.path.basename(filepath).replace(".npy", "").split("_")
        age = int(parts[1])
        raw_label = int(parts[2])
        mapped_label = 0 if raw_label == 1 else 1  # MCI → 0, Dementia → 1

        data = np.load(filepath)
        data = data[np.newaxis, :, :, :]  # shape: (1, C, H, W)
        data = torch.tensor(data, dtype=torch.float32)

        if self.augment and mapped_label == 1:
            data = augment_eeg(data, self.noise_std, self.scale_range)

        return data, mapped_label, age
"""

"""
# ========== Custom Transform Wrappers ==========

class ForceTuple(nn.Module):
    def __init__(self, transform):
        super().__init__()
        self.transform = transform

    def forward(self, x, y):
        x = self.transform(x)
        return x, y

class SafeCompose(Compose):
    def forward(self, x, y=None):
        for transform in self.transforms:
            out = transform(x, y)
            if isinstance(out, tuple) and len(out) == 2:
                x, y = out
            else:
                x = out
        return x, y

class GaussianNoise(nn.Module):
    def __init__(self, probability=1.0, std=0.01):
        super().__init__()
        self.probability = probability
        self.std = std

    def forward(self, x, y=None):
        if torch.rand(1).item() < self.probability:
            x = x + torch.randn_like(x) * self.std
        return x, y

class AmplitudeScaling(nn.Module):
    def __init__(self, probability=1.0, scale_range=(0.95, 1.05)):
        super().__init__()
        self.probability = probability
        self.scale_range = scale_range

    def forward(self, x, y=None):
        if torch.rand(1).item() < self.probability:
            scale = torch.empty(1).uniform_(*self.scale_range).item()
            x = x * scale
        return x, y

class TimeShift(nn.Module):
    def __init__(self, probability=1.0, shift_max=10):
        super().__init__()
        self.probability = probability
        self.shift_max = shift_max

    def forward(self, x, y=None):
        if torch.rand(1).item() < self.probability:
            shift = np.random.randint(-self.shift_max, self.shift_max + 1)
            x = torch.roll(x, shifts=shift, dims=-1)
        return x, y

# ========== Augmentation Pipeline ==========
""" """
def get_bd_transform(prob_dict=None):
    prob_dict = prob_dict or {}

    return SafeCompose([
        GaussianNoise(probability=prob_dict.get("noise", 1.0), std=0.01),
        AmplitudeScaling(probability=prob_dict.get("scale", 1.0), scale_range=(0.95, 1.05)),
        TimeShift(probability=prob_dict.get("shift", 1.0), shift_max=10),
        TimeReverse(probability=prob_dict.get("reverse", 0.5)),
        FTSurrogate(probability=prob_dict.get("ftsurrogate", 0.5)),
        ChannelsDropout(probability=prob_dict.get("dropout", 0.5)),  # ✅ only one argument!
    ])
"""
"""
def get_bd_transform(prob_dict=None):
    prob_dict = prob_dict or {}

    return SafeCompose([
        FTSurrogate(probability=prob_dict.get("ftsurrogate", 0.5)),
        ChannelsDropout(probability=prob_dict.get("dropout", 0.5))
    ])



class CaueegDataset(Dataset):
    def __init__(self, data_dir, augment=False, transform=None, class_pair=(1, 2)):
        self.augment = augment
        self.transform = transform
        self.class_pair = class_pair
        self.files = []
        self.label_counts = {0: 0, 1: 0}

        for f in glob.glob(os.path.join(data_dir, "*.npy")):
            filename = os.path.basename(f)
            parts = filename.replace(".npy", "").split("_")
            if len(parts) == 3:
                label = int(parts[2])
                if label in class_pair:
                    # Remap labels: class_pair[0] → 0, class_pair[1] → 1
                    mapped_label = 0 if label == class_pair[0] else 1
                    self.files.append((f, mapped_label))
                    self.label_counts[mapped_label] += 1

        print(f"Loaded {len(self.files)} files from {data_dir}")
        print("Filtered and remapped classes:", class_pair, "→ [0, 1]")
        print("Label distribution →", self.label_counts)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath, label = self.files[idx]
        parts = os.path.basename(filepath).replace(".npy", "").split("_")
        age = int(parts[1])

        data = np.load(filepath)
        data = torch.tensor(data, dtype=torch.float32)

        if self.augment and self.transform is not None:
            data = data.permute(1, 0, 2).contiguous()   # (19, 238, 400)
            data = data.view(19, -1)                    # (19, 95200)
            data, _ = self.transform(data, None)
            data = data.view(19, 238, 400)
            data = data.permute(1, 0, 2).contiguous()   # (238, 19, 400)

        return data.unsqueeze(0), label, age





def get_dataloaders(train_dir, val_dir, test_dir, batch_size=4, class_pair=(1, 2)):
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    train_transform = get_bd_transform(prob_dict={
        "noise": 1.0,
        "scale": 1.0,
        "shift": 1.0,
        "reverse": 0.5,
        "ftsurrogate": 0.5,
        "dropout": 0.5
    })

    return (
        DataLoader(CaueegDataset(train_dir, augment=True, transform=train_transform, class_pair=class_pair),
                    batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(CaueegDataset(val_dir, augment=False, class_pair=class_pair),
                    batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(CaueegDataset(test_dir, augment=False, class_pair=class_pair),
                    batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )




bd_transform = SafeCompose([
    GaussianNoise(1.0, 0.01),      # Might return just x → now safe
    ChannelsDropout(1.0, 0.2),     # Might return just x → now safe
    TimeReverse(0.5)               # Returns (x, y) → safe
])

def get_classwise_transforms():
    mci_transform = SafeCompose([
        GaussianNoise(probability=0.2, std=0.01),             # More noise
        AmplitudeScaling(probability=0.2, scale_range=(0.95, 1.05)),
        ChannelsDropout(probability=0.2)                      # More aggressive
    ])

    dementia_transform = SafeCompose([
        GaussianNoise(probability=0.1, std=0.005),
        AmplitudeScaling(probability=0.1, scale_range=(0.98, 1.02)),
        ChannelsDropout(probability=0.1)
    ])

    return {1: mci_transform, 2: dementia_transform}

class CaueegDataset(Dataset):
    def __init__(self, data_dir, augment=False, transform_dict=None):
        self.augment = augment
        self.transform_dict = transform_dict or {}
        self.files = []
        self.label_counts = {}
 

        for f in glob.glob(os.path.join(data_dir, "*.npy")):
            filename = os.path.basename(f)
            parts = filename.replace(".npy", "").split("_")
            if len(parts) == 3:
                label = int(parts[2])  # Use all labels (e.g., 0, 1, 2)
                self.files.append(f)
                self.label_counts[label] = self.label_counts.get(label, 0) + 1

        print(f"Loaded {len(self.files)} files from {data_dir}")
        print("Label distribution →", self.label_counts)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = self.files[idx]
        parts = os.path.basename(filepath).replace(".npy", "").split("_")
        age = int(parts[1])
        label = int(parts[2])

        data = np.load(filepath)
        data = torch.tensor(data, dtype=torch.float32)

        if self.augment and label in self.transform_dict:
            transform = self.transform_dict[label]
            reshaped = data.permute(1, 0, 2).contiguous()
            reshaped = reshaped.view(-1, 400)
            reshaped, _ = transform(reshaped, None)
            reshaped = reshaped.view(19, 238, 400)
            data = reshaped.permute(1, 0, 2).contiguous()

        return data.unsqueeze(0), label, age

       
        if self.augment and self.transform is not None:
            reshaped = data.permute(1, 0, 2).contiguous()
            reshaped = reshaped.view(-1, 400)
            reshaped, _ = self.transform(reshaped, None)
            reshaped = reshaped.view(19, 238, 400)
            data = reshaped.permute(1, 0, 2).contiguous()
       
        return data.unsqueeze(0), label, age

def get_dataloaders(train_dir, val_dir, test_dir, batch_size=4):
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    classwise_transforms = get_classwise_transforms()

    return (
        DataLoader(CaueegDataset(train_dir, augment=True, transform_dict=classwise_transforms),
                   batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(CaueegDataset(val_dir, augment=False),
                   batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(CaueegDataset(test_dir, augment=False),
                   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )

"""

