import os
import shutil

base_dir = "Rice_Leaf_AUG"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

for d in [train_dir, val_dir]:
    if os.path.exists(d):
        for class_name in os.listdir(d):
            src = os.path.join(d, class_name)
            dst = os.path.join(base_dir, class_name)
            if os.path.isdir(src):
                if not os.path.exists(dst):
                    os.makedirs(dst)
                for f in os.listdir(src):
                    shutil.move(os.path.join(src, f), os.path.join(dst, f))
                os.rmdir(src)
        # Try to remove the train/val folder if empty
        try:
            os.rmdir(d)
        except OSError:
            pass

print("Dataset reset complete.")
