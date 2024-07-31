import os

image_dir_path = 'data/images'
image_dirs = os.listdir(image_dir_path)
image_ids = {image_id[:-4] for image_id in os.listdir(image_dir_path + '/' + image_dirs[0]) + os.listdir(image_dir_path + '/' + image_dirs[1])}
print(f'Found {len(image_ids)} images.')

label_dir_path = 'data/labels'
label_dirs = os.listdir(label_dir_path)
label_id_paths = [label_dirs[0] + '/' + label for label in os.listdir(label_dir_path + '/' + label_dirs[0])] 
label_id_paths += [label_dirs[1] + '/' + label for label in os.listdir(label_dir_path + '/' + label_dirs[1])]
print(f'Found {len(label_id_paths)} labels.')

print("Cleaning to ensure all labels map to an image. . .")

for label_path in label_id_paths:
    if label_path.split('/')[1][:-4] not in image_ids:
        os.remove(label_dir_path + '/' + label_path)

print('Cleaned.')

