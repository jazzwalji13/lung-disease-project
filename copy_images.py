import os
import random
import shutil

# Copy some test images
folders = ['chest_xray/train/NORMAL', 'chest_xray/train/PNEUMONIA']
for folder in folders:
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith(('.png','.jpg','.jpeg'))]
        for f in random.sample(files, min(3, len(files))):
            shutil.copy2(os.path.join(folder, f), 'static/uploads/')
        print(f'Copied from {folder}')
print('✅ Test images ready!')