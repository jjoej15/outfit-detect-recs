from ultralytics import YOLO

if __name__ == '__main__':
    # Load model
    model = YOLO("yolov8n.yaml")  # build a new model from scratch

    # Use model
    model.train(data="clothing_data.yaml", epochs=1, batch=4)  # train the model
    metrics = model.val()  # evaluate model performance on the validation set
    # results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
    # path = model.export(format="onnx")  # export the model to ONNX format