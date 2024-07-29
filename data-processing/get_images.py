import pandas as pd
import os


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

used_ids = {}


def create_id_list(item_name, label_name, df, n):
    open(f'id-lists/{item_name}_ID_LIST.txt', 'w').close()

    ids_under_label = df[df['LabelName'] == label_name]['ImageID']

    num_used = 0

    with open(f'id-lists/{item_name}_ID_LIST.txt', 'a') as list_file:
        for id in ids_under_label:
            if num_used == n:
                break

            elif id not in used_ids:
                num_used += 1
                used_ids[id] = True
                list_file.write(f'train/{id}\n')

                # Creating annotation for image if it doesn't already exist
                if not os.path.exists(f'labels/{id}.txt'):
                    with open(f'labels/{id}.txt', 'w') as annotation_file:
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
                            annotations_str += f'{label_item_map[data["LabelNames"][i][1]][1]} {(data["XMins"][i][1] + data["XMaxes"][i][1])/2} {(data["YMins"][i][1] + data["YMaxes"][i][1])/2} {-data["XMins"][i][1] + data["XMaxes"][i][1]} {-data["YMins"][i][1] + data["YMaxes"][i][1]}\n'

                        annotation_file.write(annotations_str)


def download_images(item_name):
    os.system(f'python downloader.py id-lists/{item_name}_ID_LIST.txt --download_folder=images/{item_name} --num_processes=5')


def main():
    df = pd.read_csv('oidv6-train-annotations-bbox.csv')

    for key in label_item_map:
        print(f'Creating {label_item_map[key][0]} id list and annotations')
        create_id_list(label_item_map[key][0], key, df, 400)

    for key in label_item_map:
        print(f'Downloading {label_item_map[key][0]} images')
        download_images(label_item_map[key][0])


if __name__ == '__main__':
    main()