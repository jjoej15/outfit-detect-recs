import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager


label_item_map = {
    # Key represents LabelName in bbox csv files, first item in tuple value is type of item, second item represents class ID
    "/m/01n4qj": ("shirt", 0),
    "/m/0fly7": ("jeans", 1),
    "/m/01d40f": ("dress", 2),
    "/m/02wv6h6": ("skirt", 3),
    "/m/01bfm9": ("shorts", 4),
    "/m/02dl1y": ("hat", 5),
    "/m/01b638": ("boots", 6),
    "/m/0176mf": ("belt", 7),
    "/m/09j5n": ("footwear", 8)
}

# used_ids = {}


def create_id_list(item_name, label_name, df, n, used_ids):
    open(f'id-lists/{item_name}_ID_LIST.txt', 'w').close()

    ids_under_label = df[df['LabelName'] == label_name]['ImageID']

    num_used = 0

    with open(f'id-lists/{item_name}_ID_LIST.txt', 'a') as list_file:
        for id in ids_under_label:
            if num_used == n:
                break

            elif id not in used_ids:
                used_ids[id] = True
                list_file.write(f'train/{id}\n')

                # Creating annotation for image if it doesn't already exist
                file_path = f'data/labels/validation/{id}.txt' if num_used % 10 == 0 else f'data/labels/train/{id}.txt'
                with open(file_path, 'w') as annotation_file:
                    pieces_in_img = df[(df['ImageID'] == id) & (df['LabelName'].isin(label_item_map.keys()))]
                    
                    data = {
                        "LabelNames": list(pieces_in_img['LabelName'].items()), 
                        "XMins": list(pieces_in_img['XMin'].items()), 
                        "XMaxes": list(pieces_in_img['XMax'].items()),
                        "YMins": list(pieces_in_img['YMin'].items()),
                        "YMaxes": list(pieces_in_img['YMax'].items()),  
                    }

                    annotations_str = ""
                    for i in range(len(data["LabelNames"])):
                        # YOLO format: {class} {x_center} {y_center} {width} {height}
                        class_id = label_item_map[data["LabelNames"][i][1]][1]
                        x_center = (data["XMins"][i][1] + data["XMaxes"][i][1]) / 2
                        y_center = (data["YMins"][i][1] + data["YMaxes"][i][1]) / 2
                        width = -data["XMins"][i][1] + data["XMaxes"][i][1]
                        height = -data["YMins"][i][1] + data["YMaxes"][i][1]

                        annotations_str += f'{class_id} {x_center} {y_center} {width} {height}\n'

                    annotation_file.write(annotations_str)

                    num_used += 1


def download_images(item_name):
    os.system(f'python downloader.py id-lists/{item_name}_ID_LIST.txt --num_processes=5')


def main():
    if not os.path.exists('data'):
        os.makedirs('data')
        os.makedirs('data/labels')
        os.makedirs('data/labels/train')
        os.makedirs('data/labels/validation')
    if not os.path.exists('id-lists'):
        os.makedirs('id-lists')

    df = pd.read_csv('oidv6-train-annotations-bbox.csv', usecols=['ImageID', 'LabelName', "XMin", "XMax", "YMin", "YMax"])


    with Manager() as manager:
        used_ids = manager.dict()

        with ProcessPoolExecutor() as executor:
            for key in label_item_map:
                print(f'Creating {label_item_map[key][0]} id list and annotations')
                executor.submit(create_id_list, label_item_map[key][0], key, df, 400, used_ids)

    for key in label_item_map:
        print(f'Downloading {label_item_map[key][0]} images')
        download_images(label_item_map[key][0])


if __name__ == '__main__':
    main()