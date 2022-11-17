from fastapi import FastAPI
from fastapi.responses import StreamingResponse

import cv2
import numpy

app = FastAPI()

camera = cv2.VideoCapture(0)


def gen_frames():
    # generate frame by frame from camera
    # check if camera is opened and open it if not

    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.get("/")
def root():
    # check if camera is opened and close it if it is
    if camera.isOpened():
        camera.release()
    return {"message": "Hello World"}


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")
