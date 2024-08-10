import cv2
import supervision as sv

from ultralytics import YOLO

from PIL import Image
import numpy as np
import scipy
import scipy.cluster

from openai import OpenAI
from config import OPENAI_API_KEY


clothing_groups = {
    "short sleeve top": "top", 
    "long sleeve top": "top", 
    "short sleeve outwear": "top", 
    "long sleeve outwear": "top", 
    "vest": "top",
    "shorts": "bottom",
    "trousers": "bottom",
    "skirt": "bottom",
    "short sleeve dress": "dress",
    "long sleeve dress": "dress",
    "vest dress": "dress",
    "sling dress": "dress",
    "sling": "other"
}


def get_detections(arr: np.ndarray, model: YOLO):
    result = model(arr, agnostic_nms=True)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence >= .4]

    return detections


# Code for this function was written by Peter Hansen at https://stackoverflow.com/a/3244061 but slightly modified for my use case
def get_object_color(frame: np.ndarray):
    # Reading image
    img = Image.fromarray(frame)
    img = img.resize((150, 150)) # Resizing to reduce time
    arr = np.asarray(img)
    # Reshaping to 2D array where row represents pixel and col represents r/g/b value
    arr = arr.reshape(arr.shape[0] * arr.shape[1], 3).astype(float) 

    codes, _ = scipy.cluster.vq.kmeans(arr, 5) # Finding most dominant colors
    vecs, _ = scipy.cluster.vq.vq(arr, codes) # Assigning each pixel to one of the dominant colors
    counts, _ = np.histogram(vecs, len(codes)) # Counting occurrences

    index_max = np.argmax(counts) # Find most frequent
    peak = codes[index_max] # Getting RGB value of most frequent
    return f'({int(peak[0])}, {int(peak[1])}, {int(peak[2])})'


def get_isolated_object(bbox, frame):
    x1, y1, x2, y2 = bbox
    isolated_object = frame[int(y1):int(y2), int(x1):int(x2)]
    isolated_object = cv2.cvtColor(isolated_object, cv2.COLOR_BGR2RGB)

    return isolated_object


def detect_outfit_camera():
    model = YOLO("train/weights/best.pt")

    box_annotator = sv.BoundingBoxAnnotator(
        thickness=2
    )
    label_annotator = sv.LabelAnnotator()

    vid = cv2.VideoCapture(1)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    detections_dict = {}
    while True: 
        _, frame = vid.read() 
        
        detections = get_detections(frame, model)

        labels = []
        for bbox, _, confidence, _, _, class_dict in detections:
            class_name = class_dict['class_name']
            label = f'{class_name} {confidence:0.2f}'
            labels.append(label)
            
            if class_name not in detections_dict:
                isolated_object = get_isolated_object(bbox, frame)
                detections_dict[class_name] = {
                    'conf': confidence,
                    'img': isolated_object,
                    'detection count': 0
                }

            elif confidence > detections_dict[class_name]['conf']:
                isolated_object = get_isolated_object(bbox, frame)
                detections_dict[class_name]['conf'] = confidence
                detections_dict[class_name]['img'] = isolated_object

            detections_dict[class_name]['detection count'] += 1
            if detections_dict[class_name]['detection count'] == 500 : break

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

            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

        except:
            break
        
    vid.release() 

    cv2.destroyAllWindows()     

    objects_detected = [
        (
            class_name, 
            detections_dict[class_name]['conf'], 
            detections_dict[class_name]['img'], 
            detections_dict[class_name]['detection count']
        ) 
        for class_name in detections_dict
    ]

    objects_detected = sorted(
        objects_detected, 
        key=lambda o : o[3], # Sorting by number of frames objects were detected
        reverse=True # Sorting from highest to lowest
    )

    user_is_wearing = []
    detected_groups = {}
    for obj_name, _, img, _ in objects_detected:
        if clothing_groups[obj_name] not in detected_groups:
            color = get_object_color(img)
            detected_groups[clothing_groups[obj_name]] = True
            user_is_wearing.append({'class name': obj_name, 'color': color})

    return user_is_wearing


def detect_outfit_pic(file_path):
    try:
        model = YOLO("train/weights/best.pt")
        img = Image.open(file_path)
        arr = np.asarray(img)

        detections = get_detections(arr, model)

        user_is_wearing = []
        for bbox, _, _, _, _, class_dict in detections:
            isolated_object = get_isolated_object(bbox, arr)

            color = get_object_color(isolated_object)
            user_is_wearing.append({'class name': class_dict['class_name'], 'color': color})

        return user_is_wearing
    
    except FileNotFoundError:
        print("File path not found.")

    

def get_recs(outfit):
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = "You are an expert in fashion and recommend styling tips to others. " \
                    "When I ask you for recommendations for the outfit that I'm wearing, follow these guidelines:\n\n" \
                    "- Style: Bullet points. The header of the bullet point should be a 1-3 word summary of the information" \
                    "in the rest of the bullet point. Min. of 3 points but Max. of 5 points.\n" \
                    "- Tone: Professional.\n" \
                    "- Consider both the pieces and their corresponding colors in you answer. " \
                    "Try to give specific advice regarding the outfit given, instead of general styling tips.\n" \
                    "- Do not mention any specific RGB values in your response\n" \
                    "- When given an RGB value, don't assume that it's the exact color of the clothing piece. " \
                    "Instead, assume it's a color somewhat similar to the given RGB value.\n" \
                    "- Don't assume that the color of the clothing is monotone. Instead, only assume that the given color " \
                    "is the dominant color of the piece.\n" \
                    "- Don't assume any specific style of given outfit pieces nor any specific fit." \
                    "The only information that's safe to assume is the information given to you."
    
    user_prompt = f"I am wearing {outfit[0]['class name']} in the color of RGB value {outfit[0]['color']}"
    for i in range(1, len(outfit)):
        user_prompt += f" and a {outfit[i]['class name']} in the color of RGB value {outfit[i]['color']}"
    user_prompt += ". Can you provide some recommendations for my outfit?"
    
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=256,
        temperature=0.6
    )

    print(user_prompt, '\n')

    return completion


if __name__ == '__main__':
    outfit = detect_outfit_camera()
    # outfit = detect_outfit_pic('nettspend.jpg')
    
    if len(outfit) > 0:
        recs = get_recs(outfit)
        print(recs.choices[0].message.content)
    else:
        print("No clothing detected.")


    

    


