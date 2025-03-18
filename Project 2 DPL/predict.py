import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from model.main_model import HuMP_CAT  # Import model
from dataset import EMOTION_MAP
from utils import collate_fn
from dataset import RAVDESSDataset
from torch.utils.data import DataLoader

# ✅ Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 8  # Số lớp cảm xúc
model = HuMP_CAT(embed_dim=256, num_heads=4, embed_dim_x1=256, embed_dim_x2=256, num_classes=num_classes).to(device)
model.load_state_dict(torch.load("model/hump_cat.pth", weights_only=True))  # Load trọng số
model.eval()  # Chuyển về chế độ evaluation
print("✅ Model loaded successfully!")

# ✅ Load dataset test
test_folder = r"D:\Study\Programming\DPL\Project_2\RAVDESS_dataset\Actor_02"
test_data = RAVDESSDataset(test_folder, device)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate_fn)

print(f"✅ Dữ liệu test có {len(test_data)} samples.")

# 🔹 Khởi tạo danh sách lưu nhãn dự đoán và nhãn thực tế
all_preds = []
all_labels = []

# 🔹 Dự đoán trên tập test
correct, total = 0, 0

for i, (mfcc, prosody, hubert, labels) in enumerate(test_loader):
    mfcc, prosody, hubert, labels = mfcc.to(device), prosody.to(device), hubert.to(device), labels.to(device)

    with torch.no_grad():
        logits = model(hubert, mfcc, prosody, device)
        probs = F.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        true_label = torch.argmax(labels, dim=1).item()

    all_preds.append(pred_label)
    all_labels.append(true_label)

    is_correct = pred_label == true_label
    correct += is_correct
    total += 1

    print(f"🎯 Sample {i+1}: Dự đoán -> {list(EMOTION_MAP.values())[pred_label]} | Thực tế -> {list(EMOTION_MAP.values())[true_label]} {'✅ Đúng' if is_correct else '❌ Sai'}")

# 🔹 Tính độ chính xác
accuracy = correct / total * 100
print(f"\n📊 Độ chính xác trên tập test: {accuracy:.2f}%")

# 🔹 Tạo Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

# 🔹 Vẽ Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=list(EMOTION_MAP.values()), yticklabels=list(EMOTION_MAP.values()))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Heatmap")
plt.show()
