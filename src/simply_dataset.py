import os, glob, torch, numpy as np
from torch.utils.data import Dataset, DataLoader

class CaueegDataset(Dataset):
    def __init__(self, data_dir):
        self.files = []
        self.label_counts = {0: 0, 1: 0}  # 0 = original class 1, 1 = original class 2
        self.original_labels = []

        all_files = glob.glob(os.path.join(data_dir, "*.npy"))

        print(f"[INFO] 🔍 Scanning directory: {data_dir}")
        for file in all_files:
            filename = os.path.basename(file)
            parts = filename.replace(".npy", "").split("_")
            if len(parts) == 3:
                raw_label = int(parts[2])
                if raw_label == 1:
                    mapped_label = 0
                elif raw_label == 2:
                    mapped_label = 1
                else:
                    continue  # Ignore other classes

                self.files.append((file, mapped_label))
                self.label_counts[mapped_label] += 1
                self.original_labels.append(raw_label)

        print(f"[INFO] ✅ Loaded {len(self.files)} files")
        print(f"[INFO] 🧾 Class Mapping: original 1 → 0 | original 2 → 1")
        print(f"[INFO] 📊 Final Label Distribution → {self.label_counts}")
        print(f"[INFO] 🧠 Classes used → {sorted(set(self.original_labels))}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath, label = self.files[idx]
        parts = os.path.basename(filepath).replace(".npy", "").split("_")
        age = int(parts[1])

        data = np.load(filepath)
        data = data[np.newaxis, :, :, :]  # Shape: (1, 160, 19, 500)
        data = torch.tensor(data, dtype=torch.float32)

        return data, label, age


def get_dataloaders(train_dir, val_dir, test_dir, batch_size=4):
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    print("[INFO] 🧪 Initializing dataloaders...")

    return (
        DataLoader(CaueegDataset(train_dir), batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(CaueegDataset(val_dir), batch_size=batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(CaueegDataset(test_dir), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    )
