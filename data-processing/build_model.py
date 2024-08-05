from ultralytics import YOLO
# from ultralytics.utils.benchmarks import benchmark


def build_model():
    # Building new model from scratch
    model = YOLO("yolov8n.yaml") 

    # Train model
    model.train(data="clothing_data.yaml", epochs=200, batch=.75, patience=50)


if __name__ == '__main__':
    build_model()

    # YOLO("train/weights/best.pt").export(format="onnx")
    # benchmark(model="train/weights/best.pt", data="clothing_data.yaml", imgsz=640, half=False, device=0)
