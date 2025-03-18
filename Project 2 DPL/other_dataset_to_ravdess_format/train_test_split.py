import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Đường dẫn thư mục chứa tất cả file .wav
full_dataset_dir = r"D:\Study\Programming\DPL\Project_2\RAVDESS_dataset\full_dataset"

# Đường dẫn thư mục mới cho train và test
train_dir = os.path.join(os.path.dirname(full_dataset_dir), "train")
test_dir = os.path.join(os.path.dirname(full_dataset_dir), "test")

# Tạo thư mục nếu chưa tồn tại
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Lấy danh sách tất cả file .wav
wav_files = [f for f in os.listdir(full_dataset_dir) if f.endswith(".wav")]

# Chia thành 80% train, 20% test
train_files, test_files = train_test_split(wav_files, test_size=0.2, random_state=42)

# Hàm copy file vào thư mục tương ứng
def copy_files(file_list, source_folder, dest_folder):
    for file in file_list:
        shutil.copy2(os.path.join(source_folder, file), os.path.join(dest_folder, file))

# Copy file vào thư mục train và test
copy_files(train_files, full_dataset_dir, train_dir)
copy_files(test_files, full_dataset_dir, test_dir)

print(f"Train: {len(train_files)} files, Test: {len(test_files)} files")
