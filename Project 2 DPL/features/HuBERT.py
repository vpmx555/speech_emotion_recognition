from transformers import Wav2Vec2FeatureExtractor, HubertModel
import torch

def load_hubert_model(device):
    """ Load HuBERT model và chuyển lên GPU nếu có. """
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    hubert_model = HubertModel.from_pretrained("facebook/hubert-base-ls960").to(device)  # Đưa model lên GPU
    return feature_extractor, hubert_model

def extract_hubert_features(y, sr, feature_extractor, hubert_model, device):
    """ Trích xuất đặc trưng 768-D từ HuBERT trên GPU. """
    inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt").to(device)  # Đưa input lên GPU
    with torch.no_grad():
        features = hubert_model(**inputs).last_hidden_state  # Trích xuất đặc trưng
    return features.to(device)  # Mean và đảm bảo nằm trên GPU
