import cv2
import glob
from flask import Flask, send_from_directory, request
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet_v2 import preprocess_input
import numpy as np
import threading
from time import sleep
import uuid
from twilio.rest import Client
import datetime

# Your Twilio Account SID and Auth Token
account_sid = 'ACcc10781e79695c4f86d966784018e431'
auth_token = 'e01d910031224ede4c37859bcc4409b1'

app = Flask(__name__, static_url_path='/static', static_folder='tmp')
video_captures = []
CORS(app)
model = load_model('./model.h5')
IMAGE_THRESHHOLD = 100

# Initialize Twilio client
client = Client(account_sid, auth_token)


def send_video(status, frames):
    fileName = str(uuid.uuid4()) + '.mp4'
    frame_height, frame_width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("./tmp/" + fileName, fourcc,
                          10, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    out.release()
    print("Video created successfully!")

    message = client.messages.create(
        body=status + "\n" + "Video: " +
        "https://01c8-49-37-243-18.ngrok-free.app/static/" + fileName,
        from_="whatsapp:+14155238886",
        to="whatsapp:+919972649867",
    )
    print("Message sent successfully!")


def video_capture(index):
    global video_captures
    capture = cv2.VideoCapture(video_captures[index]['url'])
    if not capture.isOpened():
        print("Error: Unable to open webcam.")
        return

    while True:
        ret, frame = capture.read()
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # convert to numpy array
        video_captures[index]['video'].append(frame)
        frm = cv2.resize(frame, (224, 224))
        img_array = np.expand_dims(frm, axis=0)
        img_array = preprocess_input(img_array)

        predictions = model.predict(img_array)

        class_labels = ['default', 'fire', 'smoke']
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]

        if predicted_class == 'fire' and predictions[0][1] > 0.8:
            video_captures[index]['status'] = 'Fire Detected'
        elif predicted_class == 'smoke' and predictions[0][2] > 0.8:
            video_captures[index]['status'] = 'Smoke Detected'
        else:
            video_captures[index]['status'] = 'Running'
        video_captures[index]['count'].append(predicted_class)

        if len(video_captures[index]['count']) > IMAGE_THRESHHOLD:
            video_captures[index]['count'].pop(0)
            video_captures[index]['video'].pop(0)

        if len(video_captures[index]['count']) == IMAGE_THRESHHOLD:
            if video_captures[index]['count'].count('fire') > IMAGE_THRESHHOLD * 0.5:
                send_video('Fire Detected', video_captures[index]['video'])
            elif video_captures[index]['count'].count('smoke') > IMAGE_THRESHHOLD * 0.8:
                send_video('Smoke Detected', video_captures[index]['video'])

        cv2.imshow('Video Capture', frame)
        sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


def connect(url):
    global video_captures

    video_captures.append({
        'url': url,
        'video': [],
        'count': [],
    })
    video_thread = threading.Thread(
        target=video_capture, args=(len(video_captures) - 1,))
    video_thread.daemon = True
    video_thread.start()
    video_captures[-1]['thread'] = video_thread


@app.route('/', methods=['GET'])
def main():
    return send_from_directory('static', 'index.html')


@app.route('/videos', methods=['GET'])
def videos():
    return send_from_directory('static', 'videos.html')


@app.route('/video', methods=['GET'])
def video():
    return send_from_directory('static', 'video.html')


@app.route('/upload', methods=['GET'])
def upload():
    return send_from_directory('static', 'upload.html')


@app.route('/screenshot', methods=['GET'])
def screenshot():
    return send_from_directory('static', 'screenshot.html')


@app.route('/checkPhoto', methods=['POST'])
def photoCheckPhoto():
    image = request.files['fileUpload']

    if image is None:
        return {
            'status': 404,
            'message': 'Image not found.'
        }

    image.save('tmp/image.jpg')

    frame = cv2.imread('tmp/image.jpg')
    frame = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(frame, axis=0)
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    class_labels = ['default', 'fire', 'smoke']
    return {
        'status': 200,
        'message': class_labels[np.argmax(predictions)]
    }


@app.route('/getScreenshot', methods=['GET'])
def getAllScreenshots():
    screenshots = glob.glob('tmp/*.jpg')
    return {
        'status': 200,
        'screenshots': screenshots
    }


@app.route('/getVideos', methods=['GET'])
def getAllVideos():
    global video_captures
    videos = []
    for i in range(len(video_captures)):
        videos.append({
            'index': i,
            'url': video_captures[i]['url'],
            'status': video_captures[i]['status']
        })

    return {
        'status': 200,
        'videos': videos
    }


@app.route('/screenshot/<int:index>', methods=['GET'])
def getScreenshot(index):
    global video_captures
    if index < 0 or index >= len(video_captures):
        return {
            'status': 404,
            'message': 'Video capture not found.'
        }

    today = datetime.datetime.now()
    date_time = today.strftime("%m-%d-%Y %H-%M-%S")
    print('screenshot/' + date_time + '---' +
          video_captures[index]['status'] + '.jpg')
    cv2.imwrite(f'tmp/' + date_time + '---' +
                video_captures[index]['status'] + '.jpg', video_captures[index]['video'][-1])
    return {
        'status': 200,
        'message': video_captures[index]['status']
    }


@app.route('/status/<int:index>', methods=['GET'])
def getStatus(index):
    global video_captures
    if index < 0 or index >= len(video_captures):
        return {
            'status': 404,
            'message': 'Video capture not found.'
        }

    return {
        'status': 200,
        'message': video_captures[index]['status']
    }


if __name__ == '__main__':
    connect('http://172.20.10.2:8080/video')
    app.run(host='0.0.0.0', port=5001, debug=True)
