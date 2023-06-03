from flask import Flask,render_template,Response
import cv2
import time
from flask import Flask, render_template, Response, request,redirect,url_for
import cv2
import os
import datetime
from flask import Markup
import numpy as np
import pandas as pd
from utilss.disease import disease_dic
from utilss.fertilizer import fertilizer_dic
import requests
import requests
import config
import pickle
import io
import os
import torch
from torchvision import transforms
from PIL import Image
from utilss.model import ResNet9
import cv2
import torch
import cv2
import time
import numpy as np
try:
    os.mkdir('./capture')
except OSError as error:
    pass
now = datetime.datetime.now()
p = os.path.sep.join(['capture', "shot_{}.png".format(str(now).replace(":",''))])
import subprocess
disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']



capture_index=0#"https://25.174.224.29:8080/video"      #"http://25.14.39.28:8080/video"
cap=cv2.VideoCapture(capture_index)
model = torch.hub.load('ultralytics/yolov5', 'custom',path="yolo.pt",force_reload=True)
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\n\nDevice Used:",device)
print("-------------------------------------------")


def score_frame(frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        model.to(device)
        frame = [frame]
        results = model(frame)
     
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # print("------------------------------------------------------------")
        # print(labels)
        # print("------------------------------------------------------------")
        return labels, cord
#result={                                                }

def class_to_label(x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label

    """
    names= ["Pepper__bell___Bacterial_spot","Pepper__bell___healthy","Potato___Early_blight","Potato___Late_blight","Tomato__Target_Spot","Tomato__Tomato_YellowLeaf__Curl_Virus","Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy","Tomato_Late_blight","Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot","Tomato_Spider_mites_Two_spotted_spider_mite"]
    # names= ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    #     'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    #     'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    #     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    #     'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    #     'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    #     'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    #     'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    #     'hair drier', 'toothbrush']
    name=names[int(x)]
    return name


def plot_boxes(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results
    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            j=0
            j=j+1
            tet=class_to_label(labels[i])+str(j)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, tet, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
           

    return frame

def generate_frames_yolo():
    while True:
        start_time = time.perf_counter()
        start_frame_number = 50
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)    
        ## read the camera frame
        success,frame=cap.read()
        # print("-@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@----------------------------------------------")
        # print(frame.shape)
        # print("-----------------------------------------------")
        if not success:
            break
        else:
            results = score_frame(frame)
            
            frame = plot_boxes(results, frame)
            end_time = time.perf_counter()
            fps = 1 / np.round(end_time - start_time, 3)
            cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
           

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=cap.read()
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
app = Flask(__name__, static_url_path='/static')


@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

#
# as manny files are dependent and to move from any page to main page i have added index.html so this function is created
# use if you want else make use of route('/')
@app.route("/index.html", methods=["GET"])
def index_back():
    return render_template('index.html')


@app.route("/about.html")
def about():
    return render_template('about.html')

@app.route("/products.html")
def product():
    return render_template('products.html')

@app.route("/sign-in.html")
def signin():
    return render_template('sign-in.html')


@app.route("/sign-up.html")
def signup():
    return render_template('sign-up.html')

@app.route("/product-detail.html")
def product_detect():
    return render_template('product-detail.html')

@app.route("/faq.html")
def faq():
    return render_template('faq.html')

@app.route("/contact.html")
def contact():
    return render_template('contact.html')

@app.route("/scan-mode.html")
def scanmode():
    return render_template('scan-mode.html')


# @app.route("/result.html")
# def result():
#     return render_template('result.html')


@app.route('/video')
def video():
    # detection = WebcamDetection(capture_index="https:25.138.100.155:8080/video",model_name="best.pt")
    return Response(generate_frames_yolo(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/videonormal')
def videonormal():
    # detection = WebcamDetection(capture_index="https:25.138.100.155:8080/video",model_name="best.pt")
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/takeimage', methods = ["GET","POST"])
# def takeimage():
#     # name = request.form['name']
#     # print(name)
#     _, frame = cap.read()
#     p = os.path.sep.join(['capture', "shot_{}.png".format(str(now).replace(":",''))])
#     cv2.imwrite(p, frame)
#     print(Response(status = 200))
#     disease_model_path = 'plant_disease_model.pth'
#     disease_model = ResNet9(3, len(disease_classes))
#     disease_model.load_state_dict(torch.load(
#         disease_model_path, map_location=torch.device('cpu')))
#     disease_model.eval()
#     def predict_image(img, model=disease_model):
#         """
#         Transforms image to tensor and predicts disease label
#         :params: image
#         :return: prediction (string)
#         """
#         transform = transforms.Compose([
#             transforms.Resize(256),
#             transforms.ToTensor(),
#         ])
#         image = Image.open(io.BytesIO(img))
#         img_t = transform(image)
#         img_u = torch.unsqueeze(img_t, 0)

#         # Get predictions from model
#         yb = model(img_u)
#         # Pick index with highest probability
#         _, preds = torch.max(yb, dim=1)
#         prediction = disease_classes[preds[0].item()]
#         # Retrieve the class label
#         return prediction
    
#     r=os.listdir("E:\A_Vihave-Yolo-Dashboard\capture")

#     print(r[-1])
#     d="E:\A_Vihave-Yolo-Dashboard\capture\{fname}".format(fname=r[-1])
#     with open(d, "rb") as image:
#         f = image.read()
#         b = bytearray(f)
#     prediction = predict_image(b)
#     prediction = Markup(str(disease_dic[prediction]))
#     print("#################################################")
#     print(prediction)
#     return render_template('scan-mode.html', prediction_text='Your disease is: {}'.format(prediction))

if __name__=="__main__":
    app.run(debug=True,port=8050)
#  <div class="col-lg-6 col-12 mt-4 mt-lg-0">
#                                     <button type="submit" class="btn custom-btn cart-btn"
#                                             data-bs-toggle="modal" data-bs-target="#cart-modal"
#                                     onclick="location.href = 'takeimage'">Scan</button>
#                                 </div>