import torch
import torch.nn.functional as F

def collate_fn(batch):
    mfccs, prosodies, huberts, labels = zip(*batch)  # Tách từng loại feature

    # 1️⃣ **Tìm max_length để padding**
    max_hubert_len = max(feat.shape[1] for feat in huberts)  # Tìm max time steps của HuBERT

    # 2️⃣ **Padding cho MFCC**
    padded_mfccs = [F.pad(feat, (0, 0, 0, max_hubert_len - feat.shape[0])) for feat in mfccs]
    padded_mfccs = torch.stack(padded_mfccs)  # (batch_size, max_mfcc_len, 40)

    # 3️⃣ **Padding cho HuBERT**
    padded_huberts = [F.pad(feat, (0, 0, 0, max_hubert_len - feat.shape[1])) for feat in huberts]
    padded_huberts = torch.stack(padded_huberts).squeeze(1)

    # 4️⃣ **Chuyển Prosody thành tensor**
    prosodies = torch.stack([feat.clone().detach().float() for feat in prosodies])

    # 5️⃣ **Chuyển Labels thành tensor**
    labels = torch.stack(labels)
    return padded_mfccs, prosodies, padded_huberts, labels