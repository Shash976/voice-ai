# Run on laptop with RTX 3050 — outputs tiny_vad.onnx for Pi deployment
import pathlib, random, tarfile, urllib.request
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf

SAMPLE_RATE  = 16000
N_MEL        = 40
N_FRAMES     = 49
BATCH        = 128
EPOCHS       = 15
DATA_DIR     = pathlib.Path("speech_commands")
SPEECH_WORDS = ["yes","no","up","down","left","right","on","off","stop","go"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=512, win_length=400,
    hop_length=160, n_mels=N_MEL, f_min=80.0, f_max=7600.0,
)

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

# ── features ──────────────────────────────────────────────────────────────────
def audio_to_logmel(wav_np):
    wav_t = torch.tensor(wav_np).float().unsqueeze(0)      # [1, 16000]
    mel   = mel_transform(wav_t)                            # [1, N_MEL, frames]
    log_mel = torch.log(mel + 1e-6).squeeze(0).T           # [frames, N_MEL]
    log_mel = log_mel[:N_FRAMES]
    pad = N_FRAMES - log_mel.shape[0]
    if pad > 0:
        log_mel = torch.nn.functional.pad(log_mel, (0, 0, 0, pad))
    return log_mel.numpy()

def load_wav_clip(path):
    wav, _ = sf.read(path, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    return wav[:SAMPLE_RATE] if len(wav) >= SAMPLE_RATE else np.pad(wav, (0, SAMPLE_RATE - len(wav)))

def load_bg_clip(path):
    wav, _ = sf.read(path, dtype="float32")
    if wav.ndim > 1: wav = wav.mean(1)
    if len(wav) > SAMPLE_RATE:
        s = random.randint(0, len(wav) - SAMPLE_RATE)
        wav = wav[s:s + SAMPLE_RATE]
    return np.pad(wav, (0, max(0, SAMPLE_RATE - len(wav))))

# ── dataset ───────────────────────────────────────────────────────────────────
class SpeechDataset(Dataset):
    def __init__(self, speech_files, bg_files, n_silence):
        self.speech = speech_files
        self.bg     = bg_files
        self.n_sil  = n_silence

    def __len__(self):
        return len(self.speech) + self.n_sil

    def __getitem__(self, idx):
        if idx < len(self.speech):
            wav, label = load_wav_clip(self.speech[idx]), 1
        else:
            wav, label = load_bg_clip(self.bg[idx % len(self.bg)]), 0
        feat = audio_to_logmel(wav)
        return torch.tensor(feat, dtype=torch.float32), label

def get_files(n=3000):
    speech = []
    for word in SPEECH_WORDS:
        files = list((DATA_DIR / word).glob("*.wav"))
        speech += random.sample(files, min(n // len(SPEECH_WORDS), len(files)))
    bg = list((DATA_DIR / "_background_noise_").glob("*.wav"))
    return [str(f) for f in speech], [str(f) for f in bg]

# ── model ─────────────────────────────────────────────────────────────────────
class TinyVAD(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # input: [batch, N_MEL, N_FRAMES] after permute
            nn.Conv1d(N_MEL, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(16,    32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):              # x: [B, N_FRAMES, N_MEL]
        return self.head(self.net(x.permute(0, 2, 1)))

# ── train ─────────────────────────────────────────────────────────────────────
def train():
    maybe_download()
    speech_files, bg_files = get_files()
    random.shuffle(speech_files)
    split = int(0.8 * len(speech_files))

    train_ds = SpeechDataset(speech_files[:split],  bg_files, split)
    val_ds   = SpeechDataset(speech_files[split:],  bg_files, len(speech_files) - split)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model     = TinyVAD().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0
    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_dl:
                preds = model(X.to(device)).argmax(1).cpu()
                correct += (preds == y).sum().item()
                total   += len(y)
        acc = correct / total
        print(f"Epoch {epoch+1:02d}/{EPOCHS}  val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "tiny_vad_best.pt")
        scheduler.step()

    print(f"Best val accuracy: {best_acc:.3f}")
    model.load_state_dict(torch.load("tiny_vad_best.pt"))
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
