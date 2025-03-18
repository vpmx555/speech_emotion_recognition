import torch
import torch.nn as nn
import torch.optim as optim
from dataset import RAVDESSDataset
from torch.utils.data import DataLoader
from model.main_model import HuMP_CAT
from utils import collate_fn


folder_path = r"D:\Study\Programming\DPL\Project_2\RAVDESS_dataset\test_v2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = RAVDESSDataset(folder_path, device)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_fn)


num_classes = 8  # Số nhãn trong dataset
model = HuMP_CAT(embed_dim=256, num_heads=4, embed_dim_x1=256, embed_dim_x2=256, num_classes=num_classes).to(device)

# 🔹 Loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

# 🔹 Train loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for mfcc, prosody, hubert, labels in train_loader:
        hubert, mfcc, prosody, labels = hubert.to(device), mfcc.to(device), prosody.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(hubert, mfcc, prosody, device)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        labels = labels.argmax(dim=1)  # Chuyển từ one-hot về label index
        correct += (logits.argmax(dim=1) == labels).sum().item()

        total += labels.size(0)

    acc = correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}, Acc: {acc:.4f}")

# 🔹 Lưu mô hình
torch.save(model.state_dict(), "model/hump_cat.pth")
print("✅ Mô hình đã được lưu!")

