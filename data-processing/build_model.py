from ultralytics import YOLO

def build_model():
    # Building new model from scratch
    model = YOLO("yolov8n.yaml") 

    # Training model
    model.train(data="clothing_data.yaml", epochs=200, batch=.75, patience=50)


if __name__ == '__main__':
    build_model()

