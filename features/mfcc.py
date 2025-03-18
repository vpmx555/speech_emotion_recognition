import torch
import torchaudio
import torchaudio.transforms as T

# --- 1. Hàm Trích Xuất MFCC bằng torchaudio ---
def extract_mfcc(y, sr=16000, n_mfcc=13, device="cpu"):
    """ Trích xuất MFCC + Delta + Delta-Delta bằng torchaudio (tất cả trên device). """

    # Chuyển đổi sang tensor trên device20
    if not isinstance(y, torch.Tensor):
        waveform = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        waveform = y.clone().detach().to(torch.float32).to(device).unsqueeze(0)

    # Bộ trích xuất MFCC
    mfcc_transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": int(0.04 * sr), "hop_length": int(0.02 * sr)},
    ).to(device)

    # Tính MFCC
    mfcc = mfcc_transform(waveform)  # (1, n_mfcc, time_steps)

    # Tính delta và delta-delta
    delta_mfcc = torchaudio.functional.compute_deltas(mfcc)
    delta2_mfcc = torchaudio.functional.compute_deltas(delta_mfcc)

    # Tính năng lượng
    energy = (waveform**2).mean(dim=1, keepdim=True)  # (1, 1)
    energy = energy.expand(-1, mfcc.shape[2])  # (1, time_steps)
    energy = energy.unsqueeze(1)  # (1, 1, time_steps)

    # Kết hợp các đặc trưng lại thành tensor (1, 40, time_steps)
    features = torch.cat([mfcc, delta_mfcc, delta2_mfcc, energy], dim=1)

    return features.squeeze(0).permute(1, 0)  # (time_steps, 40)

