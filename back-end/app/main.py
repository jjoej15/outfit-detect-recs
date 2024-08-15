import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
import uvicorn

import numpy as np
from PIL import Image
import cv2
import scipy
import scipy.cluster

from ultralytics import YOLO
import supervision as sv

from openai import OpenAI

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model = YOLO("best.pt")
    dummy_frame = np.zeros((640, 480, 3), dtype=np.uint8)
    get_detections(dummy_frame)
    
    yield
    del model

app = FastAPI(lifespan=lifespan)
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


origins = [
    "http://localhost:5173/",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# fastapi run main.py --port 8080

# class ImageBytes(BaseModel):
#     bytes: bytes


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


def get_detections(arr: np.ndarray):
    result = model(arr, agnostic_nms=True, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections = detections[detections.confidence >= .4]

    return detections


def get_isolated_object(bbox, frame):
    x1, y1, x2, y2 = bbox
    isolated_object = frame[int(y1):int(y2), int(x1):int(x2)]
    # isolated_object = cv2.cvtColor(isolated_object, cv2.COLOR_BGR2RGB)

    return isolated_object


async def use_model_webcam(websocket: WebSocket, queue: asyncio.Queue, detections_dict: dict):
    # model = YOLO("best.pt")

    box_annotator = sv.BoundingBoxAnnotator(
        thickness=2
    )
    label_annotator = sv.LabelAnnotator()

    while True:
        bytes = await queue.get()
        arr = np.frombuffer(bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, 1)

        detections = get_detections(frame)

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
            if detections_dict[class_name]['detection count'] == 25:
                await websocket.close()
    
            
            # print(class_name, detections_dict[class_name]['detection count'])
        
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections
        )
        frame = label_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
        )

        encoded_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        await websocket.send_bytes(encoded_bytes)


async def receive(websocket: WebSocket, queue: asyncio.Queue):
    bytes = await websocket.receive_bytes()
    try:
        queue.put_nowait(bytes)
    except asyncio.QueueFull:
        pass


@app.websocket("/webcam/")
async def use_camera_detection(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue(maxsize=10)
    detections_dict = {}
    detect_task = asyncio.create_task(use_model_webcam(websocket, queue, detections_dict))

    try:
        while True:
            await receive(websocket, queue)
            
    except WebSocketDisconnect:
        detect_task.cancel()
        try:
            await websocket.close()
        except RuntimeError as e:
            if str(e) == "Unexpected ASGI message 'websocket.close', after sending 'websocket.close' or response already completed.":
                pass
            else:
                print(e)

    # finally:
        # clothing_groups = {
        #     "short sleeve top": "top", 
        #     "long sleeve top": "top", 
        #     "short sleeve outwear": "top", 
        #     "long sleeve outwear": "top", 
        #     "vest": "top",
        #     "shorts": "bottom",
        #     "trousers": "bottom",
        #     "skirt": "bottom",
        #     "short sleeve dress": "dress",
        #     "long sleeve dress": "dress",
        #     "vest dress": "dress",
        #     "sling dress": "dress",
        #     "sling": "other"
        # }

    #     objects_detected = [
    #         (
    #             class_name, 
    #             detections_dict[class_name]['conf'], 
    #             detections_dict[class_name]['img'], 
    #             detections_dict[class_name]['detection count']
    #         ) 
    #         for class_name in detections_dict
    #     ]

    #     objects_detected = sorted(
    #         objects_detected, 
    #         key=lambda o : o[3], # Sorting by number of frames objects were detected
    #         reverse=True # Sorting from highest to lowest
    #     )

    #     user_is_wearing = []
    #     detected_groups = {}
    #     for obj_name, _, img, _ in objects_detected:
    #         if clothing_groups[obj_name] not in detected_groups:
    #             color = get_object_color(img)
    #             detected_groups[clothing_groups[obj_name]] = True
    #             user_is_wearing.append({'class name': obj_name, 'color': color})

    #     return user_is_wearing

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')