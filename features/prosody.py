from disvoice.prosody.prosody import Prosody
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def extract_prosody_batch(audio_list_name, device="cuda"):
    prosodyf = Prosody()
    prosody_list = []

    for file_audio in audio_list_name:
        features = prosodyf.extract_static_features(file_audio, plots=False, fmt="torch")
        prosody_list.append(features.to(device))

    #prosody_batch = torch.stack(prosody_list) #(files_audio, 103)
    return prosody_list

