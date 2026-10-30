import cv2
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

source = 0
# source = "rtsp://admin:password@192.168.1.100:554/stream1"

cap = cv2.VideoCapture(source)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model(frame, conf=0.5, verbose=False)
    annotated_frame = results[0].plot()

    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()