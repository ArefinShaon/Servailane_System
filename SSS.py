import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import pyaudio
import wave
import threading
from ultralytics.utils.plotting import Annotator
from email.message import EmailMessage
import ssl
import smtplib
from datetime import datetime, timedelta
import io
import face_recognition
import numpy as np
import os

# =========================
# Load Owner's Face Encoding
# =========================
owner_image = face_recognition.load_image_file("owner.jpg")
owner_encoding = face_recognition.face_encodings(owner_image)[0]

def is_owner(frame):
    """Check if the detected face matches the owner's face."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for encoding in face_encodings:
        matches = face_recognition.compare_faces([owner_encoding], encoding, tolerance=0.5)
        if True in matches:
            return True
    return False


# =========================
# Behavior Detection Helper
# =========================
def detect_suspicious_pose(keypoints_list):
    """
    Rule-based detection:
    - Both wrists above head = suspicious/fight
    """
    if keypoints_list is None or len(keypoints_list) == 0:
        return False

    for person in keypoints_list:
        person = np.array(person)  # shape: [num_joints, 2]

        if person.shape[0] < 11:  # need at least head + wrists
            continue

        left_wrist = person[9]    # [x, y]
        right_wrist = person[10]  # [x, y]
        head_y = person[0][1]     # y of head

        # Convert to scalar floats
        if float(left_wrist[1]) < float(head_y) and float(right_wrist[1]) < float(head_y):
            return True

    return False


class ObjectDetectionApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.configure(bg="#f0f0f0")

        self.audio_playing = False
        self.running = True

        # Initialize PyAudio
        self.p = pyaudio.PyAudio()

        # Load YOLO models
        self.object_model = YOLO('yolov8l.pt')        # weapons detection
        self.pose_model = YOLO('yolov8n-pose.pt')     # human pose detection

        # Open the camera
        self.cap = cv2.VideoCapture(0)

        # Cooldown period for sending email (in seconds)
        self.email_cooldown = 60
        self.last_email_time = None

        # Create GUI elements
        self.canvas = tk.Canvas(window, bg="#f0f0f0", highlightthickness=0, width=640, height=480)
        self.canvas.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        self.btn_quit = tk.Button(window, text="Quit", command=self.close_app, bg="#ff4d4d", fg="white",
                                  font=("Helvetica", 14, "bold"), padx=20, pady=10, bd=0)
        self.btn_quit.pack(side=tk.BOTTOM, padx=(0, 10), pady=10)

        # Fullscreen toggle
        self.window.bind("f", self.toggle_fullscreen)
        self.window.attributes("-fullscreen", False)
        self.fullscreen = False

        # Start detection loop
        self.detect_objects()

    def detect_objects(self):
        if not self.running:
            return

        ret, img = self.cap.read()
        if not ret:
            self.window.after(10, self.detect_objects)
            return

        # 1️⃣ Weapon detection
        results = self.object_model.predict(img, verbose=False)
        annotator = Annotator(img)
        alert_triggered = False
        alert_label = ""

        for r in results:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                c = box.cls
                detected_object = self.object_model.names[int(c)]

                if detected_object in ['knife', 'scissors'] and not is_owner(img):
                    alert_triggered = True
                    alert_label = detected_object
                    annotator.box_label(b, detected_object)

        # 2️⃣ Pose detection for suspicious behavior
        pose_results = self.pose_model.predict(img, verbose=False)
        keypoints_list = []
        for r in pose_results:
            if r.keypoints is not None:
                keypoints_list.extend(r.keypoints.xy.cpu().numpy())  # each person shape: (17, 2)

        if detect_suspicious_pose(keypoints_list) and not is_owner(img):
            alert_triggered = True
            alert_label = "Suspicious Behavior"

        # Annotate pose skeletons
        for r in pose_results:
            img = r.plot()

        # Trigger audio/email if needed
        if alert_triggered:
            self.play_audio()
            self.send_email(alert_label, img)

            # 🔴 Overlay ALERT text on screen
            cv2.putText(img, f"ALERT: {alert_label}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Display in Tkinter
        img_resized = cv2.resize(img, (640, 480))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.img = img_tk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        self.window.after(10, self.detect_objects)

    def play_audio(self):
        if not self.audio_playing:
            threading.Thread(target=self._play_audio).start()

    def _play_audio(self):
        chunk = 1024
        self.audio_playing = True
        f = wave.open("repeating-alarm-tone-metal-detector.wav", "rb")
        f.rewind()
        data = f.readframes(chunk)
        stream = self.p.open(format=self.p.get_format_from_width(f.getsampwidth()),
                             channels=f.getnchannels(),
                             rate=f.getframerate(),
                             output=True)
        while data:
            stream.write(data)
            data = f.readframes(chunk)
        stream.stop_stream()
        stream.close()
        self.audio_playing = False

    def send_email(self, detected_object, img):
        if self.last_email_time is None or (datetime.now() - self.last_email_time) >= timedelta(seconds=self.email_cooldown):
            threading.Thread(target=self._send_email, args=(detected_object, img)).start()
            self.last_email_time = datetime.now()

    @staticmethod
    def _send_email(detected_object, img):
        current_datetime = datetime.now()
        email_sender = 'arefinshaon99@gmail.com'
        email_pass = 'dnru fenp vatz fytp'  # ⚠️ Recommend storing in env variable
        email_receiver = 'akib3008@gmail.com'

        subject = f"Security Alert - {detected_object} Detected"
        body = f"""Dear User,
We have detected a {detected_object} in your premises at {current_datetime.strftime("%Y-%m-%d %H:%M:%S")}.
For your safety, please take necessary actions.

Best regards,
S3 - Smart Security System

Note: This is an automated alert from your home security system.
"""

        em = EmailMessage()
        em["From"] = email_sender
        em["To"] = email_receiver
        em["Subject"] = subject
        em.set_content(body)

        # attach screenshot
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_io = io.BytesIO()
        pil_img.save(img_io, 'JPEG')
        img_io.seek(0)
        em.add_attachment(img_io.getvalue(), maintype='image', subtype='jpeg', filename='screenshot.jpg')

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as smtp:
            smtp.login(email_sender, email_pass)
            smtp.sendmail(email_sender, email_receiver, em.as_string())

    def close_app(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.p.terminate()
        self.window.destroy()

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.window.attributes("-fullscreen", self.fullscreen)


if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root, "S3 - Security System")
    root.mainloop()
