from flask import Flask, render_template, Response, request
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np

app = Flask(__name__)

face_classifier = cv2.CascadeClassifier("C:\\Users\\user\\Desktop\\pythonProject1\\haarcascade_frontalface_default.xml")
classifier = load_model("C:\\Users\\user\\Desktop\\pythonProject1\\model.h5")

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

def detect_emotion(frame):
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    bar_height = 20
    bar_width = 300
    bar_padding = 10

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            predictions = classifier.predict(roi)[0]
            max_index = np.argmax(predictions)
            label = emotion_labels[max_index]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Draw bar chart
            for i, prediction in enumerate(predictions):
                bar_x = 10
                bar_y = frame.shape[0] - (i + 1) * (bar_height + bar_padding)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(prediction * bar_width), bar_y + bar_height), (0,255,0), -1)
                percentage_text = f"{emotion_labels[i]}: {prediction*100:.2f}%"
                cv2.putText(frame, percentage_text, (bar_x + bar_width + bar_padding, bar_y + bar_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        else:
            cv2.putText(frame, 'No Faces', (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    return frame

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame = detect_emotion(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emotion_detection')
def emotion_detection():
    return render_template('hii.html')

@app.route('/analysis')
def analysis():
    # Perform dataset analysis here
    # For example, count the occurrences of each emotion in your dataset
    # Replace the dummy data in the table with the actual analysis results
    return render_template('dataset.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/us')
def us():
    return render_template('us.html')


@app.route('/upload_image')
def upload_image():
    return render_template('upload_image.html')


# Add this route for handling image upload and prediction
@app.route('/predict_image', methods=['POST'])
def predict_image():
    # Get the uploaded image file
    file = request.files['image']

    # Save the file to a temporary location
    file.save('temp.jpg')

    # Load the saved image using OpenCV
    img = cv2.imread('temp.jpg')

    # Perform emotion detection on the image
    result_img = detect_emotion(img)

    # Convert the result image to bytes for rendering
    _, buffer = cv2.imencode('.jpg', result_img)
    result_img_bytes = buffer.tobytes()

    # Render the result image in the template
    return render_template('result_image.html', result_img=result_img_bytes)

if __name__ == '__main__':
    app.run(debug=True)
