import os
import glob
import torch
import torchaudio
from torch.utils.data import Dataset
from features import prosody, mfcc, HuBERT
import torch.nn.functional as F

# 🔹 Mapping cảm xúc từ ID sang nhãn
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# 🔹 1️⃣ Load file âm thanh và nhãn cảm xúc
def load_audio_files(folder_path, device="cuda"):
    audio_files_names = glob.glob(os.path.join(folder_path, "*.wav"))
    audio_list = []
    labels = []

    for file in audio_files_names:
        filename = os.path.basename(file)  # Chỉ lấy tên file
        emotion_id = filename.split("-")[2]  # Lấy Emotion ID
        emotion_label = EMOTION_MAP.get(emotion_id, "unknown")  # Chuyển ID thành nhãn

        waveform, sr = torchaudio.load(file)
        waveform = waveform.mean(dim=0)  # Chuyển về mono nếu cần
        audio_list.append(waveform.to(device))
        labels.append(emotion_label)

    assert len(audio_list) != 0, "❌ Không có file nào được load!"
    return audio_files_names, audio_list, labels

# 🔹 2️⃣ Dataset Class để sử dụng DataLoader
class RAVDESSDataset(Dataset):
    def __init__(self, folder_path, device="cuda"):

        self.folder_path = folder_path
        self.audio_files_names, self.audio_list, self.labels = load_audio_files(folder_path, device)
        self.device = device

        self.feature_extractor, self.hubert_model = HuBERT.load_hubert_model(device)

        self.mfcc_features = self.get_list_mfcc_feature()
        self.prosody_features = self.get_list_prosody_feature()
        self.hubert_features = self.get_list_hubert_feature()

        assert len(self.mfcc_features) == len(self.audio_list), "MFCC feature extraction failed!"
        assert len(self.prosody_features) == len(self.audio_list), "Prosody feature extraction failed!"
        assert len(self.hubert_features) == len(self.audio_list), "HuBERT feature extraction failed!"

    def __len__(self):
        return len(self.audio_list)

    def __getitem__(self, idx):
        mfcc = self.mfcc_features[idx]
        prosody = self.prosody_features[idx]
        hubert = self.hubert_features[idx]

        # Chuyển nhãn từ string sang số (0-7)
        label_index = list(EMOTION_MAP.values()).index(self.labels[idx])
        label_one_hot = F.one_hot(torch.tensor(label_index, dtype=torch.long), num_classes=8).float()

        return mfcc, prosody, hubert, label_one_hot

    def get_list_mfcc_feature(self):
        """Trích xuất đặc trưng MFCC cho toàn bộ dataset."""
        return [mfcc.extract_mfcc(audio, 16000, 13, self.device) for audio in self.audio_list]

    def get_list_prosody_feature(self):
        """Trích xuất đặc trưng Prosody cho toàn bộ dataset."""
        full_audio_paths = [os.path.join(self.folder_path, filename) for filename in self.audio_files_names]

        prosody_list = prosody.extract_prosody_batch(full_audio_paths, self.device)  # (num_samples, 103)
        for i in range(len(prosody_list)):
            if torch.isnan(prosody_list[i]).any():
                prosody_list[i] = torch.nan_to_num(prosody_list[i], nan=0.0)
        return prosody_list

    def get_list_hubert_feature(self):
        """Trích xuất đặc trưng HuBERT cho toàn bộ dataset."""
        return [HuBERT.extract_hubert_features(audio, 16000, self.feature_extractor, self.hubert_model, self.device)
                for audio in self.audio_list]