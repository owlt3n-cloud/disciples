import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from ultralytics import YOLO

app = FastAPI()
model = YOLO("yolo11n-pose.pt")

source = 0
hood_alert = False


def generate_frames():
    global hood_alert
    cap = cv2.VideoCapture(source)
    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model(frame, conf=0.5, verbose=False)
            annotated_frame = frame.copy()
            current_hood_detected = False

            for r in results:
                if r.keypoints is not None and len(r.keypoints.data) > 0:
                    for idx, kp in enumerate(r.keypoints.data):
                        # Индексы точек в YOLO Pose: 0 - нос, 1,2 - глаза, 3,4 - уши, 5,6 - плечи
                        nose_conf = kp[0][2].item()
                        left_eye_conf = kp[1][2].item()
                        right_eye_conf = kp[2][2].item()
                        left_ear_conf = kp[3][2].item()
                        right_ear_conf = kp[4][2].item()

                        face_visible = nose_conf > 0.6 and (left_eye_conf > 0.6 or right_eye_conf > 0.6)
                        ears_hidden = left_ear_conf < 0.45 and right_ear_conf < 0.45

                        # Дополнительная проверка: если видна коробка человека, смотрим верхнюю часть головы
                        if r.boxes is not None and len(r.boxes.xyxy) > idx:
                            x1, y1, x2, y2 = map(int, r.boxes.xyxy[idx])
                            
                            if face_visible and ears_hidden:
                                current_hood_detected = True
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(
                                    annotated_frame,
                                    "HOOD DETECTED / VIOLATION",
                                    (x1, max(30, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 0, 255),
                                    2,
                                )
                            else:
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(
                                    annotated_frame,
                                    "OK",
                                    (x1, max(30, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    (0, 255, 0),
                                    2,
                                )

            hood_alert = current_hood_detected

            ret, buffer = cv2.imencode(".jpg", annotated_frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        cap.release()


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/status")
def get_status():
    return JSONResponse({"hood_alert": hood_alert})


@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Anti-Fraud & Liveness Check</title>
        <style>
            body { background: #0f172a; color: #f8fafc; font-family: system-ui, sans-serif; text-align: center; margin: 0; padding: 30px; }
            .container { max-width: 800px; margin: 0 auto; }
            #alert-box {
                display: none;
                background: #ef4444;
                color: white;
                padding: 16px;
                border-radius: 8px;
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 20px;
                box-shadow: 0 4px 15px rgba(239, 68, 68, 0.5);
                animation: pulse 1s infinite alternate;
            }
            @keyframes pulse { from { opacity: 1; } to { opacity: 0.75; } }
            img { border: 3px solid #334155; border-radius: 12px; max-width: 100%; height: auto; }
            .alert-border { border-color: #ef4444 !important; }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Система контроля биометрии и безопасности</h2>
            <div id="alert-box">
                ОШИБКА ИДЕНТИФИКАЦИИ: Обнаружен капюшон или головной убор! Пожалуйста, откройте голову полностью.
            </div>
            <img id="stream" src="/video_feed" alt="Video Stream" />
        </div>

        <script>
            const alertBox = document.getElementById('alert-box');
            const streamImg = document.getElementById('stream');

            setInterval(async () => {
                try {
                    const res = await fetch('/status');
                    const data = await res.json();
                    if (data.hood_alert) {
                        alertBox.style.display = 'block';
                        streamImg.classList.add('alert-border');
                    } else {
                        alertBox.style.display = 'none';
                        streamImg.classList.remove('alert-border');
                    }
                } catch (e) {}
            }, 300);
        </script>
    </body>
    </html>
    """