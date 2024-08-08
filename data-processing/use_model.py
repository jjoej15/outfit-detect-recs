from ultralytics import YOLO
import cv2
import supervision as sv
import time
from PIL import Image
import numpy as np
import scipy
# import scipy.misc
import scipy.cluster


# Code for this function was written by Peter Hansen at https://stackoverflow.com/a/3244061 but slightly modified for my use case
def get_object_color(frame: np.ndarray):
    # Reading image
    img = Image.fromarray(frame)
    img = img.resize((150, 150)) # Resizing to reduce time
    arr = np.asarray(img)
    shape = arr.shape
    # Reshaping to 2D array where row represents pixel and col represents r/g/b value
    arr = arr.reshape(shape[0] * shape[1], 3).astype(float) 

    codes, _ = scipy.cluster.vq.kmeans(arr, 5) # Finding most dominant colors
    vecs, _ = scipy.cluster.vq.vq(arr, codes) # Assigning each pixel to one of the dominant colors
    counts, _ = np.histogram(vecs, len(codes)) # Counting occurrences

    index_max = np.argmax(counts) # Find most frequent
    peak = codes[index_max] # Getting RGB value of most frequent
    return [int(val) for val in peak]


def get_isolated_object(bbox, frame):
    x1, y1, x2, y2 = bbox
    isolated_object = frame[int(y1):int(y2), int(x1):int(x2)]
    isolated_object = cv2.cvtColor(isolated_object, cv2.COLOR_BGR2RGB)

    return isolated_object


def save_img(class_name: str, img: np.ndarray):
    img = Image.fromarray(img)
    img.save(f"{class_name}.jpg")


def use_camera():
    model = YOLO("train/weights/best.pt")

    box_annotator = sv.BoundingBoxAnnotator(
        thickness=2
    )
    label_annotator = sv.LabelAnnotator()

    vid = cv2.VideoCapture(1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    frame_count = 0
    # t0 = time.time()
    detections_dict = {}

    while True: 
        _, frame = vid.read() 
        frame_count += 1
        # if frame_count == 125:
        #     break
        # hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        result = model(frame, agnostic_nms=True)[0]

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence >= .4]

        labels = []
        for bbox, _, confidence, _, _, class_dict in detections:
            class_name = class_dict['class_name']
            label = f'{class_name} {confidence:0.2f}'
            labels.append(label)
            
            if class_name not in detections_dict:
                isolated_object = get_isolated_object(bbox, frame)
                # img = Image.fromarray(isolated_object)
                # detections_dict[class_name] = [confidence, img.copy(), 0]
                detections_dict[class_name] = {
                    'conf': confidence,
                    'img': isolated_object,
                    'detection count': 0
                }

            elif confidence > detections_dict[class_name]['conf']:
                isolated_object = get_isolated_object(bbox, frame)
                # img = Image.fromarray(isolated_object)
                # detections_dict[class_name] = [confidence, isolated_object, detections_dict[class_name]]
                detections_dict[class_name]['conf'] = confidence
                detections_dict[class_name]['img'] = isolated_object

            detections_dict[class_name]['detection count'] += 1
        
        # labels = [f'{class_dict['class_name']} {confidence:0.2f}' for _, _, confidence, _, _, class_dict in detections]

        # bboxes = [f'{bbox}:{class_dict['class_name']}' for bbox, _, _, _, _, class_dict in detections]
        # print(f'BBOXS: {bboxes}')
        # bboxs: [x1 y1 x2 y2]
# Detections(xyxy=array([[     205.26,      372.57,      1237.2,
# 720]], dtype=float32), mask=None, confidence=array([    0.73218], dtype=float32), class_id=array([1]), tracker_id=None, data={'class_name': array(['short sleeve top'], dtype='<U16')})

# [(array([     218.03,       308.2,      1236.6,         720], dtype=float32), None, 0.831849, 1, None, {'class_name': 'short sleeve top'})]       

        try:
            frame = box_annotator.annotate(
                scene=frame, 
                detections=detections
            )
            frame = label_annotator.annotate(
                scene=frame,
                detections=detections,
                labels=labels
            )

            cv2.imshow('frame', frame) 

        except:
            break

        # finally: 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    vid.release() 

    cv2.destroyAllWindows()     

    objects_detected = sorted(
        [(
            class_name, 
            detections_dict[class_name]['conf'], 
            detections_dict[class_name]['img'], 
            detections_dict[class_name]['detection count']
        ) 
        for class_name in detections_dict], 
        key=lambda c : c[3], 
        reverse=True
    )

    for obj_name, conf, img, count in objects_detected[:3]:
        print(obj_name, conf, count)

        color = get_object_color(img)

        print(f'Color: {color}')
        # save_img(obj_name, img)


if __name__ == '__main__':
    use_camera()

