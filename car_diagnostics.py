import soundfile as sf
import tempfile, uuid
import random
import numpy as np
import pandas as pd
import librosa
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)


# Audio config
SR         = 22050   # Sample rate to resample to
DURATION   = None    # None = use full file; set e.g. 3.0 to truncate/pad
N_MELS     = 128     # Mel bands for spectrograms
N_MFCC     = 40      # MFCC coefficients
HOP_LENGTH = 512
N_FFT      = 2048
FMAX       = 8000


# State colors
STATE_COLORS = {
    'braking_state':  '#E07B54',
    'idle_state':     '#5B8DB8',
    'startup_state':  '#6BAE75',
}


def discover_files(root: Path) -> pd.DataFrame:
    """Walk the dataset directory and return a DataFrame of (path, label, state)."""
    records = []
    for state_dir in sorted(root.iterdir()):
        if not state_dir.is_dir():
            continue
        for class_dir in sorted(state_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            label = class_dir.name
            for wav_file in class_dir.glob('*.wav'):
                records.append({
                    'path':  str(wav_file),
                    'label': label,
                    'state': state_dir.name,
                })
    return pd.DataFrame(records)


def load_audio(path, sr=SR, duration=None):
    y, _ = librosa.load(path, sr=sr, duration=duration)
    return y


def get_mel_spectrogram(y, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX):
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length, fmax=fmax
    )
    return librosa.power_to_db(mel, ref=np.max)


def plot_distribution(df, state_colors=STATE_COLORS):
    # Class distribution bar chart
    counts = df.groupby(['state', 'label']).size().reset_index(name='count')
    color_list = [state_colors[s] for s in counts['state']]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(counts['label'], counts['count'], color=color_list, edgecolor='white', linewidth=0.8)
    ax.set_title('Class Distribution', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Class Label')
    ax.set_ylabel('Number of Samples')
    plt.xticks(rotation=35, ha='right')
    
    # Add value labels
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(int(bar.get_height())), ha='center', va='bottom', fontsize=9)
    
    # Legend for states
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=s) for s, c in state_colors.items()]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()


def plot_waveforms(samples, sampling_rate=SR, state_colors=STATE_COLORS):
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    fig.suptitle('Waveforms — One Sample per Class', fontsize=15, fontweight='bold', y=1.01)
    
    for ax, (_, row) in zip(axes.flat, samples.iterrows()):
        y = load_audio(row['path'], sampling_rate)
        times = np.linspace(0, len(y)/sampling_rate, len(y))
        color = state_colors.get(row['state'], '#888888')
        ax.plot(times, y, color=color, linewidth=0.6, alpha=0.85)
        ax.set_title(row['label'].replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.set_xlabel('Time (s)', fontsize=8)
        ax.set_ylabel('Amplitude', fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_facecolor('#f8f8f8')
        # Add state badge
        ax.text(0.98, 0.95, row['state'].replace('_', ' '),
                transform=ax.transAxes, ha='right', va='top',
                fontsize=6.5, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8))
    
    plt.tight_layout()
    plt.show()


def plot_mel_spectograms(samples, sampling_rate=SR, n_mels=N_MELS, n_fft=N_FFT,
                         hop_length=HOP_LENGTH, fmax=FMAX, state_colors=STATE_COLORS):
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    fig.suptitle('Mel Spectrograms — One Sample per Class', fontsize=15, fontweight='bold', y=0.95)
    
    for ax, (_, row) in zip(axes.flat, samples.iterrows()):
        y = load_audio(row['path'], sampling_rate)
        mel_db = get_mel_spectrogram(y, sampling_rate, n_mels, n_fft, hop_length, fmax)
        img = librosa.display.specshow(
            mel_db, sr=sampling_rate, hop_length=hop_length,
            x_axis='time', y_axis='mel', fmax=fmax, ax=ax, cmap='magma'
        )
        ax.set_title(row['label'].replace('_', ' ').title(), fontsize=10, fontweight='bold')
        ax.tick_params(labelsize=7)
        color = state_colors.get(row['state'], '#888888')
        ax.text(0.98, 0.95, row['state'].replace('_', ' '),
                transform=ax.transAxes, ha='right', va='top',
                fontsize=6.5, color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85))
    
    fig.subplots_adjust(right=0.88, hspace=0.45)  # increase hspace to taste
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    fig.colorbar(img, cax=cbar_ax, format='%+2.0f dB')
    plt.show()


def plot_comparisons(compare_rows, sampling_rate=SR, n_mels=N_MELS, n_fft=N_FFT,
                     hop_length=HOP_LENGTH, fmax=FMAX, n_mfcc=N_MFCC,
                     state_colors=STATE_COLORS):
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    fig.suptitle('Deep Dive: dead_battery vs normal_engine_startup', fontsize=13, fontweight='bold')
    
    for col, (label, row) in enumerate(compare_rows.iterrows()):
        y = load_audio(row['path'], sampling_rate)
        mel_db = get_mel_spectrogram(y, sampling_rate, n_mels, n_fft, hop_length, fmax)
        mfccs  = librosa.feature.mfcc(y=y, sr=sampling_rate, n_mfcc=n_mfcc)
        color  = state_colors.get(row['state'], '#888888')
        times  = np.linspace(0, len(y)/sampling_rate, len(y))
    
        # Waveform
        axes[col][0].plot(times, y, color=color, linewidth=0.7)
        axes[col][0].set_title(f'{label} — Waveform', fontsize=10)
        axes[col][0].set_xlabel('Time (s)')
    
        # Mel Spectrogram
        librosa.display.specshow(mel_db, sr=sampling_rate, hop_length=hop_length,
                                 x_axis='time', y_axis='mel', ax=axes[col][1], cmap='magma')
        axes[col][1].set_title(f'{label} — Mel Spectrogram', fontsize=10)
    
        # MFCCs
        librosa.display.specshow(mfccs, sr=sampling_rate, hop_length=hop_length,
                                 x_axis='time', ax=axes[col][2], cmap='coolwarm')
        axes[col][2].set_title(f'{label} — MFCCs', fontsize=10)
        axes[col][2].set_ylabel('MFCC Coefficient')
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history, title, ax_acc, ax_loss):
    epochs = range(1, len(history.history['accuracy']) + 1)
    ax_acc.plot(epochs, history.history['accuracy'],     label='Train', linewidth=2)
    ax_acc.plot(epochs, history.history['val_accuracy'], label='Val',   linewidth=2, linestyle='--')
    ax_acc.set_title(f'{title} — Accuracy', fontweight='bold')
    ax_acc.set_ylabel('Accuracy'); ax_acc.legend()

    ax_loss.plot(epochs, history.history['loss'],     label='Train', linewidth=2)
    ax_loss.plot(epochs, history.history['val_loss'], label='Val',   linewidth=2, linestyle='--')
    ax_loss.set_title(f'{title} — Loss', fontweight='bold')
    ax_loss.set_ylabel('Loss'); ax_loss.set_xlabel('Epoch'); ax_loss.legend()


def plot_confusion_matrix(y_true, y_pred, classes, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor='white', vmin=0, vmax=1)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', rotation=0, labelsize=8)


# Feature Extraction ##########################################################

def extract_mfcc_features(path, sr=SR, n_mfcc=N_MFCC, duration=DURATION):
    """Return a flat feature vector: mean + std of MFCCs, delta-MFCCs, and chroma."""
    y = load_audio(path, sr=sr, duration=duration)
    mfccs       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc  = librosa.feature.delta(mfccs)
    delta2_mfcc = librosa.feature.delta(mfccs, order=2)
    chroma      = librosa.feature.chroma_stft(y=y, sr=sr)
    zcr         = librosa.feature.zero_crossing_rate(y)
    rms         = librosa.feature.rms(y=y)

    features = np.hstack([
        mfccs.mean(axis=1),       mfccs.std(axis=1),
        delta_mfcc.mean(axis=1),  delta_mfcc.std(axis=1),
        delta2_mfcc.mean(axis=1), delta2_mfcc.std(axis=1),
        chroma.mean(axis=1),      chroma.std(axis=1),
        zcr.mean(axis=1),         zcr.std(axis=1),
        rms.mean(axis=1),         rms.std(axis=1),
    ])
    return features


def extract_mel_spectrogram_array(path, sr=SR, n_mels=N_MELS,
                                   target_length=128, duration=DURATION):
    """Return a fixed-size (n_mels, target_length) mel spectrogram array."""
    y = load_audio(path, sr=sr, duration=duration)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=FMAX
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Normalise to [0, 1]
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    # Pad or trim to target_length
    if mel_db.shape[1] < target_length:
        mel_db = np.pad(mel_db, ((0,0),(0, target_length - mel_db.shape[1])), mode='constant')
    else:
        mel_db = mel_db[:, :target_length]
    return mel_db


def extract_yamnet_waveform(path, sr=SR, target_sr=16000, duration=DURATION):
    """YAMNet expects 16 kHz mono waveform."""
    y = load_audio(path, sr=sr, duration=duration)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y_resampled.astype(np.float32)


# Waveform Augmentation Functions #############################################

def aug_time_stretch(y, sr, rate=None):
    """Stretch or compress time by a random rate in [0.85, 1.15]."""
    rate = rate or np.random.uniform(0.85, 1.15)
    return librosa.effects.time_stretch(y, rate=rate)


def aug_pitch_shift(y, sr, n_steps=None):
    """Shift pitch by a random number of semitones in [-2, 2]."""
    n_steps = n_steps if n_steps is not None else np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def aug_add_noise(y, sr, snr_db=None):
    """Add Gaussian white noise at a random SNR between 20 and 40 dB."""
    snr_db  = snr_db or np.random.uniform(20, 40)
    sig_pow = np.mean(y ** 2)
    noise_pow = sig_pow / (10 ** (snr_db / 10))
    noise   = np.random.randn(len(y)) * np.sqrt(noise_pow)
    return y + noise


def aug_time_shift(y, sr, max_shift_frac=0.1):
    """Shift the waveform forward or backward by up to max_shift_frac of its length."""
    shift = int(np.random.uniform(-max_shift_frac, max_shift_frac) * len(y))
    return np.roll(y, shift)


def aug_volume_scale(y, sr, low=0.7, high=1.3):
    """Randomly scale amplitude."""
    scale = np.random.uniform(low, high)
    return y * scale


def augment_waveform(y, sr, n_transforms=2):
    """
    Apply n_transforms randomly chosen augmentations in sequence.
    Using 2 transforms per sample gives more variety than 1.
    """
    AUG_TRANSFORMS = [
        aug_time_stretch,
        aug_pitch_shift,
        aug_add_noise,
        # aug_time_shift,
        aug_volume_scale,
    ]

    transforms = random.sample(AUG_TRANSFORMS, k=n_transforms)
    for t in transforms:
        y = t(y, sr)
    # Clip to [-1, 1] to avoid float overflow artifacts
    y = np.clip(y, -1.0, 1.0)
    return y
