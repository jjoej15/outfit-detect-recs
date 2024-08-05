import json
import os
from PIL import Image


val_data = {image[:-4] for image in os.listdir('data/images/validation')}
cats = {}


# YOLO format: {class} {x_center} {y_center} {width} {height}
for file in os.listdir('annos'):
    with open(f'annos/{file}') as fh:
        image_id = file[:-5]
        data = json.load(fh)
        dataset = 'validation' if image_id in val_data else 'train'
        img = Image.open(f'data/images/{dataset}/{image_id}.jpg') 
        width, height = img.size

        label_string = ''
        for item in data:
            if item[:4] == 'item':
                cat_id = data[item]['category_id']

                if cat_id not in cats:
                    cats[cat_id] = data[item]['category_name']

                x_c = ((data[item]['bounding_box'][0] + data[item]['bounding_box'][2]) / 2) / width
                y_c = ((data[item]['bounding_box'][1] + data[item]['bounding_box'][3]) / 2) / height
                width_box = (data[item]['bounding_box'][2] - data[item]['bounding_box'][0]) / width
                height_box = (data[item]['bounding_box'][3] - data[item]['bounding_box'][1]) / height

                label_string += f'{cat_id} {x_c} {y_c} {width_box} {height_box}\n'

        

        with open(f'data/labels/{dataset}/{image_id}.txt', 'w') as label_file:
            label_file.write(label_string)


# with open('categories.txt', 'w') as cat_file:
#     cat_file.write('names:\n')

#     cat_list = [int(cat) for cat in cats]
#     cat_list.sort()
#     for i in cat_list:
#         cat_file.write(f'\t{i}: {cats[i]}\n')


# from PIL import Image
  
# # get image 
# filepath = "data/images/train/000002.jpg"
# img = Image.open(filepath) 
  
# # get width and height 
# width,height = img.size 
# print(width, height)