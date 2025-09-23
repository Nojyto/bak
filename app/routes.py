from flask import Blueprint, Response, render_template
from app.camera import Camera

main = Blueprint("main", __name__)

camera = Camera()

def generate_frames():
    if not camera.is_running:
        camera.start()

    while True:
        frame = camera.get_jpeg_frame()
        if frame is None:
            print("Camera feed stopped. Stopping...")
            break
        yield frame


@main.route("/")
def index():
    return render_template("index.html")


@main.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")
