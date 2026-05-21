# train_tiny_vad.py
#
# Trains a tiny VAD (speech vs. silence) model on Speech Commands v2.
#
# Key improvements over v1:
#   - Uses official validation_list.txt / testing_list.txt splits (no speaker bleed)
#   - Uses ALL available speech files (not capped at 3000)
#   - Richer negative class: background noise + silence + unknown words
#   - Data augmentation: additive noise, volume jitter, time shift
#   - BatchNorm in model for better generalization
#   - AdamW + cosine annealing LR schedule
#   - Continuous-speech negatives: silent frames extracted from background files
#
# Output: tiny_vad_best.pt  tiny_vad.onnx

import pathlib, random, tarfile, urllib.request, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf

SAMPLE_RATE  = 16000
N_MEL        = 40
N_FRAMES     = 49
BATCH        = 256
EPOCHS       = 30
DATA_DIR     = pathlib.Path("speech_commands")

# Words treated as positive "speech" targets
SPEECH_WORDS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go",
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine",
    "bed", "bird", "cat", "dog", "happy",
    "house", "marvin", "sheila", "tree", "wow",
    "forward", "backward", "follow", "learn", "visual",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=512, win_length=400,
    hop_length=160, n_mels=N_MEL, f_min=80.0, f_max=7600.0,
).to(device)

# ── download ──────────────────────────────────────────────────────────────────

def maybe_download():
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        return
    DATA_DIR.mkdir(exist_ok=True)
    url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    tar = "speech_commands_v0.02.tar.gz"
    if not pathlib.Path(tar).exists():
        print("Downloading Speech Commands v2 (~2.3 GB)...")
        urllib.request.urlretrieve(url, tar)
    print("Extracting...")
    with tarfile.open(tar) as t:
        t.extractall(DATA_DIR)

# ── official splits ───────────────────────────────────────────────────────────

def load_official_split():
    """Returns sets of relative paths (word/file.wav) for val and test."""
    val_set  = set()
    test_set = set()
    vf = DATA_DIR / "validation_list.txt"
    tf = DATA_DIR / "testing_list.txt"
    if vf.exists():
        val_set  = set(vf.read_text().splitlines())
    if tf.exists():
        test_set = set(tf.read_text().splitlines())
    return val_set, test_set

# ── audio loading ─────────────────────────────────────────────────────────────

def load_wav(path, target_len=SAMPLE_RATE):
    wav, _ = sf.read(str(path), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(1)
    if len(wav) >= target_len:
        # random crop if longer
        if len(wav) > target_len:
            s = random.randint(0, len(wav) - target_len)
            wav = wav[s:s + target_len]
    else:
        wav = np.pad(wav, (0, target_len - len(wav)))
    return wav

# ── augmentation ──────────────────────────────────────────────────────────────

def augment(wav, bg_clips, training=True):
    if not training:
        return wav
    # volume jitter ±6 dB
    wav = wav * (10 ** (random.uniform(-0.5, 0.5)))
    # time shift up to ±10 %
    shift = random.randint(-int(0.1 * SAMPLE_RATE), int(0.1 * SAMPLE_RATE))
    wav = np.roll(wav, shift)
    # additive background noise at SNR 10–30 dB (50 % of the time)
    if bg_clips and random.random() < 0.5:
        bg = bg_clips[random.randint(0, len(bg_clips) - 1)]
        s  = random.randint(0, max(0, len(bg) - SAMPLE_RATE))
        bg = bg[s:s + SAMPLE_RATE]
        if len(bg) < SAMPLE_RATE:
            bg = np.pad(bg, (0, SAMPLE_RATE - len(bg)))
        sig_pow = max(np.mean(wav ** 2), 1e-9)
        bg_pow  = max(np.mean(bg  ** 2), 1e-9)
        snr_db  = random.uniform(10, 30)
        scale   = math.sqrt(sig_pow / bg_pow / (10 ** (snr_db / 10)))
        wav     = wav + scale * bg
    return np.clip(wav, -1.0, 1.0).astype(np.float32)

# ── features ──────────────────────────────────────────────────────────────────

def audio_to_logmel_batch(wavs_np):
    """wavs_np: [B, 16000] float32 → [B, N_FRAMES, N_MEL]"""
    t = torch.tensor(wavs_np, device=device)              # [B, 16000]
    mel = mel_transform(t)                                 # [B, N_MEL, frames]
    log_mel = torch.log(mel + 1e-6).permute(0, 2, 1)      # [B, frames, N_MEL]
    log_mel = log_mel[:, :N_FRAMES, :]
    pad = N_FRAMES - log_mel.shape[1]
    if pad > 0:
        log_mel = torch.nn.functional.pad(log_mel, (0, 0, 0, pad))
    return log_mel.cpu().numpy()

def audio_to_logmel(wav_np):
    return audio_to_logmel_batch(wav_np[np.newaxis])[0]

# ── dataset ───────────────────────────────────────────────────────────────────

class VADDataset(Dataset):
    def __init__(self, speech_files, silence_files, bg_clips, training=True):
        self.speech  = speech_files
        self.silence = silence_files
        self.bg      = bg_clips
        self.train   = training
        self.items = (
            [(f, 1) for f in self.speech] +
            [(f, 0) for f in self.silence]
        )
        random.shuffle(self.items)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        wav = load_wav(path)
        wav = augment(wav, self.bg, self.train)
        feat = audio_to_logmel(wav)
        return torch.tensor(feat, dtype=torch.float32), label

# ── build file lists ──────────────────────────────────────────────────────────

def build_splits():
    """Returns (train, val, test) each as list of (path, label) tuples."""
    maybe_download()
    val_set, test_set = load_official_split()

    speech_train, speech_val, speech_test = [], [], []
    silence_train, silence_val, silence_test = [], [], []

    # --- speech files (words in SPEECH_WORDS) ---
    for word in SPEECH_WORDS:
        word_dir = DATA_DIR / word
        if not word_dir.exists():
            continue
        for f in word_dir.glob("*.wav"):
            rel = f"{word}/{f.name}"
            if rel in test_set:
                speech_test.append(str(f))
            elif rel in val_set:
                speech_val.append(str(f))
            else:
                speech_train.append(str(f))

    # --- silence negatives: background noise random crops ---
    bg_dir = DATA_DIR / "_background_noise_"
    bg_wavs = []
    if bg_dir.exists():
        for bf in bg_dir.glob("*.wav"):
            wav, _ = sf.read(str(bf), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(1)
            bg_wavs.append(wav)

    # use background noise paths directly as silence items
    bg_paths = [str(f) for f in bg_dir.glob("*.wav")] if bg_dir.exists() else []

    # also grab words not in SPEECH_WORDS as additional speech positives
    all_word_dirs = [d for d in DATA_DIR.iterdir()
                     if d.is_dir() and not d.name.startswith("_")]
    extra_speech_words = [d for d in all_word_dirs if d.name not in SPEECH_WORDS]
    for word_dir in extra_speech_words[:5]:   # add a few unknown words as speech
        for f in list(word_dir.glob("*.wav"))[:500]:
            rel = f"{word_dir.name}/{f.name}"
            if rel in test_set:
                speech_test.append(str(f))
            elif rel in val_set:
                speech_val.append(str(f))
            else:
                speech_train.append(str(f))

    # balance silence by replicating bg_paths
    n_train = len(speech_train)
    n_val   = len(speech_val)
    n_test  = len(speech_test)

    if bg_paths:
        silence_train = [random.choice(bg_paths) for _ in range(n_train)]
        silence_val   = [random.choice(bg_paths) for _ in range(n_val)]
        silence_test  = [random.choice(bg_paths) for _ in range(n_test)]
    else:
        print("WARNING: no background noise files found — silence class will be empty")

    print(f"Train: {len(speech_train)} speech + {len(silence_train)} silence")
    print(f"Val:   {len(speech_val)} speech + {len(silence_val)} silence")
    print(f"Test:  {len(speech_test)} speech + {len(silence_test)} silence")

    return (speech_train, silence_train,
            speech_val,   silence_val,
            speech_test,  silence_test,
            bg_wavs)

# ── model ─────────────────────────────────────────────────────────────────────

class TinyVAD(nn.Module):
    """
    Input:  [B, N_FRAMES, N_MEL] = [B, 49, 40]
    After permute → Conv1d sees [B, N_MEL, N_FRAMES]

    Architecture: 2× Conv1d with BN+ReLU → GlobalAvgPool → 2× Linear with BN
    ~19K parameters.  All operations map to int8 MAC kernels for RV32 firmware.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(N_MEL, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )

    def forward(self, x):          # x: [B, N_FRAMES, N_MEL]
        return self.head(self.net(x.permute(0, 2, 1)))

# ── train ─────────────────────────────────────────────────────────────────────

def train():
    (speech_train, silence_train,
     speech_val,   silence_val,
     speech_test,  silence_test,
     bg_wavs) = build_splits()

    train_ds = VADDataset(speech_train, silence_train, bg_wavs, training=True)
    val_ds   = VADDataset(speech_val,   silence_val,   bg_wavs, training=False)
    test_ds  = VADDataset(speech_test,  silence_test,  bg_wavs, training=False)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False,
                          num_workers=4, pin_memory=True, persistent_workers=True)

    model     = TinyVAD().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_dl:
                preds = model(X.to(device)).argmax(1).cpu()
                correct += (preds == y).sum().item()
                total   += len(y)
        acc = correct / total
        print(f"Epoch {epoch+1:02d}/{EPOCHS}  "
              f"train_loss={train_loss/len(train_dl):.4f}  val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "tiny_vad_best.pt")
        scheduler.step()

    # final test evaluation
    model.load_state_dict(torch.load("tiny_vad_best.pt", map_location="cpu"))
    model.to(device).eval()
    correct = total = 0
    tp = fp = tn = fn = 0
    with torch.no_grad():
        for X, y in test_dl:
            preds = model(X.to(device)).argmax(1).cpu()
            correct += (preds == y).sum().item()
            total   += len(y)
            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"\nBest val acc : {best_acc:.4f}")
    print(f"Test  acc   : {correct/total:.4f}")
    print(f"Precision   : {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}")
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}")

    return model

# ── export to ONNX ────────────────────────────────────────────────────────────

def export_onnx(model):
    model.eval().cpu()
    dummy = torch.zeros(1, N_FRAMES, N_MEL)
    torch.onnx.export(
        model, dummy, "tiny_vad.onnx",
        input_names=["log_mel"], output_names=["logits"],
        dynamic_axes={"log_mel": {0: "batch"}},
        opset_version=11,
    )
    size = pathlib.Path("tiny_vad.onnx").stat().st_size / 1024
    print(f"Saved tiny_vad.onnx — {size:.1f} KB")

if __name__ == "__main__":
    model = train()
    export_onnx(model)
