import os
import re
import mne
import pandas as pd
import numpy as np

def normalize_z(x):
    channel_mean = np.mean(x, axis=1)
    channel_std = np.std(x, axis=1)
    norm = (x - channel_mean[:, np.newaxis]) / channel_std[:, np.newaxis]
    return norm

def filter(raw):
    raw.set_eeg_reference(projection=True).apply_proj()
    raw.filter(1., 40., fir_design='firwin', n_jobs=1)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    raw_csd = mne.preprocessing.compute_current_source_density(raw)
    return raw_csd

def drop_channels(signal):
    return np.delete(signal, [19, 20], axis=0)

def custom_crop(raw, tmin=0.0, tmax=None, include_tmax=True):
    tmax = min((raw.n_times - 1) / raw.info['sfreq'], tmax)
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)
    return raw

def multichannel_sliding_window(X, size, step):
    shape = (X.shape[0] - X.shape[0] + 1, (X.shape[1] - size + 1) //
             step, X.shape[0], size)
    strides = (X.strides[0], X.strides[1] * step, X.strides[0], X.strides[1])
    return np.lib.stride_tricks.as_strided(X, shape, strides)[0]

# ----- Main Execution -----
# Define paths
root_dir = '/home/shubham/D1/dataset/ds004504/derivatives'
tsv_file = '/home/shubham/D1/dataset/ds004504/participants.tsv'
output_csv = '/home/shubham/D1/dataset/paths.csv'
dest_dir = '/home/shubham/D1/vtf_images/alz_88'

# Load metadata
df = pd.read_csv(tsv_file, sep='\t')
os.makedirs(dest_dir, exist_ok=True)

for id, each_row in enumerate(zip(df["participant_id"], df["Gender"], df["Group"], df["Age"], df["MMSE"])):
    subject = each_row[0]
    eeg_file = f"{root_dir}/{subject}/eeg/{subject}_task-eyesclosed_eeg.set"

    if os.path.exists(eeg_file):
        print(f"📥 Reading EEG: {eeg_file}")
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True)
        raw.filter(1., 40., fir_design='firwin', n_jobs=1)
        raw.resample(sfreq=100)
        raw = custom_crop(raw, tmin=60, tmax=360)
        normalized = normalize_z(raw._data)
        split_timestamps = multichannel_sliding_window(raw._data, 500, 250)
        print(f"✅ Shape: {split_timestamps.shape}")

        if split_timestamps.shape == (118, 19, 500):
            filename = f'{subject}{each_row[1]}{each_row[2]}{each_row[3]}{each_row[4]}_{id}.npy'
            np.save(os.path.join(dest_dir, filename), split_timestamps)
    else:
        print(f"❌ Missing EEG file: {eeg_file}")
