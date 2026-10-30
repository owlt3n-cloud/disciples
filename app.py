import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolo11n.pt")

source = 0
# source = "rtsp://admin:password@192.168.1.100:554/stream1"


def generate_frames():
    cap = cv2.VideoCapture(source)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, conf=0.5, verbose=False)
            annotated_frame = results[0].plot()

            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Live Stream</title>
        <style>
            body { background: #111; color: #fff; font-family: sans-serif; text-align: center; margin-top: 40px; }
            img { border: 2px solid #333; border-radius: 8px; max-width: 90%; height: auto; }
        </style>
    </head>
    <body>
        <h2>Видеомониторинг YOLO в реальном времени</h2>
        <img src="/video_feed" alt="Video Stream" />
    </body>
    </html>
    """