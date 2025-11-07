# check_dataset.py
import os

def check_dataset_structure():
    print("🔍 Checking dataset structure...")
    
    folders_to_check = [
        'chest_xray/train/NORMAL',
        'chest_xray/train/PNEUMONIA',
        'chest_xray/test/NORMAL', 
        'chest_xray/test/PNEUMONIA',
        'chest_xray/val/NORMAL',
        'chest_xray/val/PNEUMONIA'
    ]
    
    for folder in folders_to_check:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if not f.startswith('._') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"📁 {folder}: {len(files)} images")
            if files:
                print(f"   Sample files: {files[:3]}")  # Show first 3 files
        else:
            print(f"❌ {folder}: DOES NOT EXIST")

if __name__ == '__main__':
    check_dataset_structure()