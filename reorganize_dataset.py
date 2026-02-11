import os
import shutil
import random

base_dir = "Rice_Leaf_AUG"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

classes = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf scald",
    "Sheath Blight"
]

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for class_name in classes:
    class_path = os.path.join(base_dir, class_name)
    if not os.path.exists(class_path):
        continue
        
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    
    files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
    if not files:
        print(f"No files in {class_name}, maybe already moved.")
        continue
        
    random.shuffle(files)
    
    split_idx = int(len(files) * 0.8)
    train_files = files[:split_idx]
    val_files = files[split_idx:]
    
    print(f"Moving {len(train_files)} files to train and {len(val_files)} to validation for {class_name}")
    
    for f in train_files:
        try:
            shutil.move(os.path.join(class_path, f), os.path.join(train_class_dir, f))
        except Exception as e:
            print(f"Error moving {f}: {e}")
        
    for f in val_files:
        try:
            shutil.move(os.path.join(class_path, f), os.path.join(val_class_dir, f))
        except Exception as e:
            print(f"Error moving {f}: {e}")
        
    # Try to remove original folder but don't fail if it doesn't work
    try:
        shutil.rmtree(class_path)
        print(f"Removed {class_name}")
    except Exception as e:
        print(f"Could not remove {class_name}: {e}")

print("Dataset organization attempt complete.")
