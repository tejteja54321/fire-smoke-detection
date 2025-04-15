from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory,Response
from ultralytics import YOLO
import os
import subprocess
from werkzeug.utils import secure_filename
import simpleaudio as sa
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# Initialize Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'


# Folder for uploaded images and results
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER


users = {}

#model = YOLO(r'F:/TEJA/NLP/PROJ14/runs/detect/train3/weights/best.pt')
model=YOLO(r'../runs/detect/train/weights/best.pt')


# Home Page
@app.route('/')
def index():
    return render_template('index.html')

# Home Page After Login
@app.route('/about')
def about():
    if 'user' in session:
        return render_template('about.html', user=session['user'])
    return redirect(url_for('login'))

# Login Page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['pswd']
        if email in users and users[email] == password:
            session['user'] = email
            flash('Login successful!', 'success')
            return redirect(url_for('about'))
        else:
            flash('Invalid email or password!', 'danger')
    
    return render_template('login.html')

# Register Page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('pswd')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return render_template('register.html')  # Stay on register page
        elif email in users:
            flash('Email is already registered!', 'warning')
        else:
            users[email] = password
            flash("Registration Successful! Redirecting to login...", "success")
            return render_template('register.html')  # Stay on register page first     
    return render_template('register.html')  # Show register form


    return render_template('register.html')






@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("Image1")
        return redirect(url_for('about'))

    file = request.files['file']
    if file.filename == '':
        print("Image2")
        return redirect(url_for('about'))

    if file:
        print("Image3")
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Video formats
            result_video_path = process_video(file_path, filename)
            return render_template('about.html', 
                                   original_file=filename, 
                                   result_video=result_video_path)

        else:
            results = model(file_path)
            boxes = results[0].boxes
            frame = cv2.imread(file_path)

            custom_labels = {0: "Fire", 1: "Fire/Smoke", 2: "Smoke"}
            class_counts = {}
            trigger_classes = [0, 2]  # Only Fire and Smoke

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in trigger_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = custom_labels.get(cls_id, "Unknown")

                    # Count per class
                    class_counts[label] = class_counts.get(label, 0) + 1

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw count summary for class 0 and 2
            y_offset = 50
            font_scale = 1.5
            font_thickness = 3
            total_count = sum(class_counts.values())

            for class_name, count in class_counts.items():
                text = f"{class_name}: {count}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x, text_y = 10, y_offset
                cv2.rectangle(frame, 
                            (text_x, text_y - text_size[1] - 10), 
                            (text_x + text_size[0] + 10, text_y + 10), 
                            (0, 0, 0), -1)
                cv2.putText(frame, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                y_offset += text_size[1] + 20

            cv2.putText(frame, f"Total: {total_count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)

            result_img_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")
            cv2.imwrite(result_img_path, frame)

            return render_template('about.html', 
                                original_file=filename, 
                                result_file=f"result_{filename}")
        



@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename)

@app.route('/results/videos/<filename>')
def result_video(filename):
    return send_from_directory(app.config['RESULTS_FOLDER'], filename, mimetype='video/mp4')

def process_video(input_path, filename):
    cap = cv2.VideoCapture(input_path)
    output_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    temp_output = os.path.join(app.config['RESULTS_FOLDER'], "temp_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, frame_size)

    custom_labels = {0: "Fire", 1: "Fire/Smoke", 2: "Smoke"}
    trigger_classes = [0, 2]  # Only draw for Fire and Smoke

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes
        annotated_frame = frame.copy()

        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id in trigger_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = custom_labels.get(cls_id, "Unknown")

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(annotated_frame)

    cap.release()
    out.release()

    # Re-encode with FFmpeg
    ffmpeg_command = [
        "ffmpeg", "-y",
        "-i", temp_output,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]

    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        raise

    os.remove(temp_output)

    return f"result_{filename}"





# Live Video Page After Login
@app.route('/live_video')
def live_video():
    if 'user' in session:
        return render_template('live_video.html', user=session['user'])
    return redirect(url_for('login'))

# Global variable for camera
camera = None

def play_alarm():
    """Plays an alarm sound when an object is detected."""
    alarm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "static/alarm.wav"))
    
    if not os.path.exists(alarm_path):
        print("Error: Alarm sound file not found!")
        return
    print(f"Trying to play alarm from: {alarm_path}")  # Debugging
    wave_obj = sa.WaveObject.from_wave_file(alarm_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()



def generate_frames():
    """Captures frames from the camera and performs object detection for Fire and Smoke only (skips 'Fire/Smoke' or others)."""
    global camera
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open the camera.")
        return

    custom_labels = {0: "Fire", 1: "Fire/Smoke", 2: "Smoke"}
    trigger_classes = [0, 2]  # Only fire and smoke

    while True:
        success, frame = camera.read()
        if not success:
            break

        results = model(frame)
        boxes = results[0].boxes
        annotated_frame = frame.copy()

        alarm_triggered = False

        for box in boxes:
            cls_id = int(box.cls[0])

            if cls_id in trigger_classes:
                label = custom_labels.get(cls_id, "Unknown")

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                alarm_triggered = True  # Only if fire or smoke detected

        if alarm_triggered:
            play_alarm()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')




@app.route('/video_feed')
def video_feed():
    """Stream the camera feed to the browser."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    

@app.route('/close_camera', methods=['POST'])
def close_camera():
    global camera
    if camera:
        camera.release()
        camera = None
    return '', 204


# Global variables
video_capture = None
stop_detection_flag = False
video_path = None


@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_path, stop_detection_flag

    file = request.files['file']
    if file:
        upload_folder = os.path.join('static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)  # ✅ Creates the folder if it doesn't exist

        filename = file.filename
        video_path = os.path.join(upload_folder, filename)
        file.save(video_path)

        stop_detection_flag = False  # Reset for fresh detection
        return redirect(url_for('video_analysis'))

    return 'No file uploaded', 400

@app.route('/video_analysis')
def video_analysis():
    return render_template('live_video.html') 

@app.route('/video_feed2')
def video_feed2():
    return Response(generate_video_detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video_detection():
    global video_capture, stop_detection_flag, video_path

    video_capture = cv2.VideoCapture(video_path)
    custom_labels = {0: "Fire", 1: "Fire/Smoke", 2: "Smoke"}
    trigger_classes = [0, 2]  # Only trigger for Fire and Smoke

    while video_capture.isOpened():
        if stop_detection_flag:
            break
        success, frame = video_capture.read()
        if not success:
            break

        frame = cv2.resize(frame, (640, 480))
        results = model(frame)
        detected = False

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()

            for box, conf, class_id in zip(boxes, confs, cls):
                if conf > 0.5 and int(class_id) in trigger_classes:
                    detected = True
                    label = custom_labels[int(class_id)]

                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if detected:
            play_alarm()

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    video_capture.release()


    
def generate_video_frames(video_path):
    if not video_path or not os.path.exists(video_path):
        return  

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # Resize frame to 640x480 for faster processing
        frame = cv2.resize(frame, (640, 480))

        # Run inference on frame
        results = model(frame)

        detected = False  # Track if fire/smoke is detected

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
            confs = result.boxes.conf.cpu().numpy()  # Get confidence scores
            cls = result.boxes.cls.cpu().numpy()     # Get class IDs

            print(f"Detected Objects: {len(cls)}")  # Debugging

            for box, conf, class_id in zip(boxes, confs, cls):
                if conf > 0.5:  # Confidence threshold
                    detected = True
                    x1, y1, x2, y2 = map(int, box)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Fire/Smoke: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if detected:
            play_alarm()  # Trigger alarm in a separate thread

        # Convert frame for streaming
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/stop_video', methods=['POST'])
def stop_video():
    global stop_detection_flag, video_capture
    stop_detection_flag = True
    if video_capture:
        video_capture.release()
    return '', 204







# Performance Page After Login
@app.route('/performance')
def performance():
    if 'user' in session:
        return render_template('performance.html', user=session['user'])
    return redirect(url_for('login'))

# Charts Page After Login
@app.route('/charts')
def charts():
    if 'user' in session:
        return render_template('charts.html', user=session['user'])
    return redirect(url_for('login'))


# Logout Route
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out!', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
