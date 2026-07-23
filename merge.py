import os
import shutil

# --- CONFIGURATION ---
RAW_DATA_DIR = "Raw_Datasets"
MASTER_DATA_DIR = "Master_Dataset"

# Master IDs: 0:Elephant, 1:Monkey, 2:Wild_Boar, 3:Cow, 4:Deer, 5:Leopard
# Mapping format: {Original_ID : Master_ID}
MAPPING = {
    "Dataset_1": {1: 0, 2: 5},                                               # Elephant(1)->0, Leopard(2)->5
    "Dataset_2": {0: 0, 6: 5, 9: 1},                                         # Elephant(0)->0, Leopard(6)->5, Monkey(9)->1
    "Dataset_3": {0: 3, 1: 3},                                               # Cow sitted(0)->3, Cow standing(1)->3 (Merge)
    "Dataset_4": {0: 4, 1: 4},                                               # buck(0)->4, doe(1)->4 (Merge, ignored 'face')
    "Dataset_5": {1: 2, 4: 1},                                               # boar(1)->2, monkey(4)->1
    "Dataset_6": {1: 2, 4: 3, 6: 0, 11: 1},                                  # boar(1)->2, cow(4)->3, elephant(6)->0, monkey(11)->1
    "Dataset_7": {0: 1}                                                      # monkey(0)->1
}

splits = ['train', 'valid', 'test']

# Create Master Dataset folders
for split in splits:
    os.makedirs(os.path.join(MASTER_DATA_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(MASTER_DATA_DIR, split, 'labels'), exist_ok=True)

image_counter = 0

print("🚀 Starting Data Merge\n")

for folder, id_map in MAPPING.items():
    folder_path = os.path.join(RAW_DATA_DIR, folder)
    if not os.path.exists(folder_path):
        print(f"⚠️ Missing folder: {folder}. Skipping...")
        continue
        
    for split in splits:
        img_dir = os.path.join(folder_path, split, 'images')
        lbl_dir = os.path.join(folder_path, split, 'labels')
        
        # Some datasets use 'val' instead of 'valid'
        if split == 'valid' and not os.path.exists(img_dir):
            img_dir = os.path.join(folder_path, 'val', 'images')
            lbl_dir = os.path.join(folder_path, 'val', 'labels')

        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue

        for label_file in os.listdir(lbl_dir):
            if not label_file.endswith('.txt'): continue
            
            with open(os.path.join(lbl_dir, label_file), 'r') as f:
                lines = f.readlines()
                
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts: continue
                
                orig_id = int(parts[0])
                
                if orig_id in id_map:
                    new_id = id_map[orig_id]
                    new_lines.append(f"{new_id} " + " ".join(parts[1:]) + "\n")
            if new_lines:
                image_counter += 1
                base_name = os.path.splitext(label_file)[0]
                
                # FIND Image file(jpg, png, jpeg)
                img_ext = '.jpg'
                for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
                    if os.path.exists(os.path.join(img_dir, base_name + ext)):
                        img_ext = ext
                        break
                
                new_file_name = f"chaukidar_img_{image_counter}"
                
                # Copy Image
                shutil.copy(os.path.join(img_dir, base_name + img_ext), 
                            os.path.join(MASTER_DATA_DIR, split, 'images', new_file_name + img_ext))
                
                # Save New Label
                with open(os.path.join(MASTER_DATA_DIR, split, 'labels', new_file_name + '.txt'), 'w') as f:
                    f.writelines(new_lines)

# Generate final data.yaml
yaml_content = f"""train: ../train/images
val: ../valid/images
test: ../test/images

nc: 6
names: ['Elephant', 'Monkey', 'Wild_Boar', 'Cow', 'Deer', 'Leopard']
"""
with open(os.path.join(MASTER_DATA_DIR, 'data.yaml'), 'w') as f:
    f.write(yaml_content)

print(f"\n✅ BOOM! Master Dataset is Ready! Total Images Processed: {image_counter}")