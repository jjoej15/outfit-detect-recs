from ultralytics import YOLO
import cv2
import supervision as sv


def use_camera():
    model = YOLO("train/weights/best.pt")

    box_annotator = sv.BoundingBoxAnnotator(
        thickness=2
    )
    label_annotator = sv.LabelAnnotator()

    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
  
    while True: 
        _, frame = vid.read() 
    
        # hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        result = model(frame, agnostic_nms=True)[0]

        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence >= .4]
        labels = [f'{class_dict['class_name']} {confidence:0.2f}' for _, _, confidence, _, _, class_dict in detections]

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

        except ValueError:
            pass

        finally: 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break

    vid.release() 

    cv2.destroyAllWindows()     


if __name__ == '__main__':
    use_camera()

