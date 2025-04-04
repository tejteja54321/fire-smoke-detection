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
model=YOLO(r'F:/TEJA/NLP/PROJ14/runs/detect/train/weights/best.pt')


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

        else:  # Image file
            results = model(file_path)
            result_img_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")
            annotated_frame = results[0].plot()
            
            # Object counting
            class_counts = results[0].boxes.cls.cpu().numpy()
            class_counts = {model.names[int(cls_id)]: list(class_counts).count(cls_id) for cls_id in set(class_counts)}
            total_count = sum(class_counts.values())

            # Annotate frame with counts
            y_offset = 50
            font_scale = 1.5
            font_thickness = 3
            for class_name, count in class_counts.items():
                text = f"{class_name}: {count}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                text_x, text_y = 10, y_offset
                cv2.rectangle(annotated_frame, 
                              (text_x, text_y - text_size[1] - 10), 
                              (text_x + text_size[0] + 10, text_y + 10), 
                              (0, 0, 0), -1)  # Black background
                cv2.putText(annotated_frame, text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
                y_offset += text_size[1] + 20
            cv2.putText(annotated_frame, f"Total: {total_count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
            
            cv2.imwrite(result_img_path, annotated_frame)
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
    # OpenCV Video Capture
    cap = cv2.VideoCapture(input_path)
    output_path = os.path.join(app.config['RESULTS_FOLDER'], f"result_{filename}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    # Temporary file for processed frames
    temp_output = os.path.join(app.config['RESULTS_FOLDER'], "temp_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Temporary OpenCV video
    out = cv2.VideoWriter(temp_output, fourcc, fps, frame_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # YOLO Processing: Annotate the frame with detections
        results = model(frame)
        annotated_frame = results[0].plot()
        
        # Object counting
        class_counts = results[0].boxes.cls.cpu().numpy()
        frame_class_counts = {model.names[int(cls_id)]: list(class_counts).count(cls_id) for cls_id in set(class_counts)}
        frame_total_count = sum(frame_class_counts.values())
        
        # Annotate frame with counts
        y_offset = 50
        font_scale = 1.5
        font_thickness = 3
        for class_name, count in frame_class_counts.items():
            text = f"{class_name}: {count}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            text_x, text_y = 10, y_offset
            cv2.rectangle(annotated_frame, 
                          (text_x, text_y - text_size[1] - 10), 
                          (text_x + text_size[0] + 10, text_y + 10), 
                          (0, 0, 0), -1)  # Black background
            cv2.putText(annotated_frame, text, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
            y_offset += text_size[1] + 20
        cv2.putText(annotated_frame, f"Total: {frame_total_count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), font_thickness)
        
        out.write(annotated_frame)

    cap.release()
    out.release()

    # FFmpeg Encoding for Final Output
    ffmpeg_command = [
        "ffmpeg", "-y",  # Overwrite output if exists
        "-i", temp_output,   # Input temporary video
        "-c:v", "libx264",   # H.264 codec for encoding
        "-preset", "medium", # Encoding speed
        "-crf", "23",        # Quality (Lower is better)
        "-c:a", "aac",       # Audio codec
        "-strict", "experimental",
        output_path          # Final output video
    ]

    try:
        # Run FFmpeg to re-encode
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        raise

    # Remove temporary file
    os.remove(temp_output)

    return f"result_{filename}"  # Return relative path



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

    wave_obj = sa.WaveObject.from_wave_file(alarm_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()

def generate_frames():
    """Captures frames from the camera and performs object detection."""
    global camera
    camera = cv2.VideoCapture(0)  # Open the default camera

    if not camera.isOpened():
        print("Error: Could not open the camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Perform YOLO detection
        results = model.predict(frame, verbose=False)

        detected = False  # Track if any object is detected

        # Annotate detections
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = result
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))  # Bounding box coordinates
            label = f"{model.names[int(cls)]} {conf:.2f}"  # Label with confidence

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detected = True  # Set flag if object detected

        # Play alarm if an object is detected
        if detected:
            play_alarm()

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame to browser
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


# Upload Video for Detection
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        print("live1")
        flash('No file selected!', 'danger')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        print("live1")
        flash('No file selected!', 'danger')
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    session['uploaded_video'] = file_path
    print("live1")
    return redirect(url_for('live_video'))  # Reload the same page after uploading

def play_alarm():
    alarm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "static/alarm.wav"))
    print(f"Trying to play alarm from: {alarm_path}")  # Debugging
    wave_obj = sa.WaveObject.from_wave_file(alarm_path)
    play_obj = wave_obj.play()
    play_obj.wait_done()




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


@app.route('/video_feed2')
def video_feed2():
    video_path = session.get('uploaded_video', '')  # Fetch uploaded video path
    return Response(generate_video_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')








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
