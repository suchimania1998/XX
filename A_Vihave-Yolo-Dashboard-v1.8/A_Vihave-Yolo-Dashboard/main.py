from flask import Flask,render_template,Response,request, Markup
import cv2
import time
from flask import Flask, render_template, Response, request,redirect
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
import what3words
from torchvision import transforms
from PIL import Image
from utilss.model import ResNet9
import cv2
import torch
import time
import requests
import config
import pickle
import io
import torch
import pyttsx3
from utilss.yolo_disease import medicine
try:
    os.mkdir('./capture')
except OSError as error:
    pass
now = datetime.datetime.now()
p = os.path.sep.join(['capture', "shot_{}.png".format(str(now).replace(":",''))])

# ==============================================================================================

# -------------------------LOADING THE TRAINED MODELS -----------------------------------------------

# Loading plant disease classification model

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

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(
    disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()


# Loading crop recommendation model

crop_recommendation_model_path = 'models/RandomForest.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))


# =========================================================================================

# Custom functions for calculations


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = "9d7cde1f6d07ec55650544be1631307e"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    """
    Transforms image to tensor and predicts disease label
    :params: image
    :return: prediction (string)
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)

    # Get predictions from model
    yb = model(img_u)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    # Retrieve the class label
    return prediction

# ======================================================================================================================================
# ------------------------------------ FLASK APP ----------------------------------------------------------------------------------------

# ===============================================================================================

# RENDER PREDICTION PAGES



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

    name=names[int(x)]
    return name

def name_img(results, frame):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    :param results: contains labels and coordinates predicted by model on the given frame.
    :param frame: Frame which has been scored.
    :return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results
    n = len(labels)
    frame=cv2.imread(frame)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            name=class_to_label(labels[i])
            print(x1, y1, x2, y2)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
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
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            n=class_to_label(labels[i])
            cv2.putText(frame, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
           

    return frame,n

def generate_frames_yolo():
    while True:
            
        ## read the camera frame
        start_time = time.perf_counter()
        success,frame=cap.read()
        # print("-@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@----------------------------------------------")
        # print(frame.shape)
        # print("-----------------------------------------------")
        if not success:
            break
        else:
            results = score_frame(frame)
            
            frame,lebel1 = plot_boxes(results, frame)
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
def name_img(results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        frame=cv2.imread(frame)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                dis=class_to_label(labels[i])
               
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        return dis
image1=0
pr="No disease detected"
def plot_boxes_obj(results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        frame=cv2.imread(frame)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.2:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                
                image=cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                global image1
                global pr
                image1=cv2.putText(image, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                pr=class_to_label(labels[i])
                print(x1, y1, x2, y2)
                
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        return image1,pr
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
    pyttsx3.speak("You are Opening Crop disease detection and entering into step 1 verfication mode")
    return render_template('product-detail.html')

@app.route("/faq.html")
def faq():
    return render_template('faq.html')

@app.route("/contact.html")
def contact():
    return render_template('contact.html')

@app.route("/scan-mode.html")
def scanmode():
    pyttsx3.speak("You are entering into Step-2 verfication mode")
    return render_template('scan-mode.html')


@app.route('/fertilizer.html')
def fertilizer_recommendation():
    title = 'Fertilizer Suggestion'
    pyttsx3.speak("You are entering into fertlizer recommendation system")

    return render_template('fertilizer.html', title=title)


@app.route('/fertilizer-predict', methods=['POST'])
def fert_recommend():
    title = ' Harvestify-Fertilizer Suggestion '
    crop_name = str(request.form['cropname'])
    N = int(request.form['nitrogen'])
    P = int(request.form['phosphorous'])
    K = int(request.form['pottasium'])
    # ph = float(request.form['ph'])

    df = pd.read_csv('E:\A_Vihave-Yolo-Dashboard\fertilizer.csv')

    nr = df[df['Crop'] == crop_name]['N'].iloc[0]
    pr = df[df['Crop'] == crop_name]['P'].iloc[0]
    kr = df[df['Crop'] == crop_name]['K'].iloc[0]
    n = nr - N
    p = pr - P
    k = kr - K
    temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
    max_value = temp[max(temp.keys())]
    if max_value == "N":
        if n < 0:
            key = 'NHigh'
        else:
            key = "Nlow"
    elif max_value == "P":
        if p < 0:
            key = 'PHigh'
        else:
            key = "Plow"
    else:
        if k < 0:
            key = 'KHigh'
        else:
            key = "Klow"

    response = Markup(str(fertilizer_dic[key]))

    return render_template('fertilizer-result.html', recommendation=response, title=title)


# render crop recommendation result page
@app.route("/crop.html")
def crop():
    pyttsx3.speak("You are entering into Crop recommendation system")
    return render_template('crop.html')


@ app.route('/crop-predict', methods=['POST'])
def crop_prediction():
    title = 'Harvestify - Crop Recommendation'

    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('crop-result.html', prediction=final_prediction, title=title)

        else:

            return render_template('try_again.html', title=title)


# # render disease prediction result page
# @app.route("/disease.html")
# def disease():
#     return render_template('disease.html')



@app.route('/disease-predict', methods=["GET",'POST'])
def disease_prediction():
    title = 'Harvestify - Disease Detection'
    pyttsx3.speak("You are entering into mobile crop disease detection system")

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('disease.html', title=title)
        try:
            img = file.read()

            prediction = predict_image(img)

            prediction = Markup(str(disease_dic[prediction]))
            return render_template('disease-result.html', prediction=prediction, title=title)
        except:
            pass
    return render_template('disease.html', title=title)


# ===============================================================================================

# @app.route("/result.html")
# def result():
#     return render_template('result.html')
@app.route('/image')
def image():
    r=os.listdir("E:\A_Vihave-Yolo-Dashboard\static\yolo")

    imager="E:\A_Vihave-Yolo-Dashboard\static\yolo\{fname}".format(fname=r[-1])
   
    return imager 

@app.route('/video')
def video():
    # detection = WebcamDetection(capture_index="https:25.138.100.155:8080/video",model_name="best.pt")
    return Response(generate_frames_yolo(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/videonormal')
def videonormal():
    # detection = WebcamDetection(capture_index="https:25.138.100.155:8080/video",model_name="best.pt")
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/takeimage', methods = ["GET","POST"])
def takeimage():
    _, frame = cap.read()
    directory ='E:\A_Vihave-Yolo-Dashboard\capture'
    os.chdir(directory)
    now = datetime.datetime.now()
    p = os.path.sep.join(['E:\A_Vihave-Yolo-Dashboard\capture', "shot_{}.png".format(str(now).replace(":",''))])
    cv2.imwrite(p, frame)
    print(Response(status = 200))
    print("#######################################")
    print("Image captured stored into capture folder")

    print("##########################################")

    print("******************************")
   
    r=os.listdir("E:\A_Vihave-Yolo-Dashboard\capture")

    frame="E:\A_Vihave-Yolo-Dashboard\capture\{fname}".format(fname=r[-1])
    print("*******************")

    print(frame)
    results = model(frame)

    print("******************************")

    #results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
    #classes = model.names
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("\n\nDevice Used:",device)
    print("-------------------------------------------")
    results=score_frame(frame)
    #prediction=name_img(results,frame)
    image1,pr = plot_boxes_obj(results,frame)
    print("+++++++++++++++++++++++++++++++++++")
    print(image1)
    print("+++++++++++++++++++++++++++++++++")
    prediction=pr
    med=medicine[prediction]
    dirt ='E:\A_Vihave-Yolo-Dashboard\static\yolo'
    os.chdir(dirt)
    p = os.path.sep.join(['E:\A_Vihave-Yolo-Dashboard\static\yolo', "shot_{}.png".format(str(now).replace(":",''))])
    d = "".join(['yolo/',"shot_{}.png".format(str(now).replace(":",''))])
    cv2.imwrite(p,image1)
    print("+++++++++++++++++++++++++++++++++++")
    print("Added to yolo image folder")
    print(p)
    print("+++++++++++++++++++++++++++++++++")
    geocoder = what3words.Geocoder("HQZK687Y")
    res = geocoder.convert_to_coordinates('enact.accomplishment.deeply')
    lng=res['coordinates']['lng']
    lat=res['coordinates']['lat']
    map=res["map"]
    engine = pyttsx3.init()
    engine.say("Your result is displaying on your screen")
    engine.runAndWait()


    return render_template('scan-mode.html', prediction_text='Your disease is: {}'.format(prediction),path=d,Medi="Your medication is: {}".format(med),lng='Found at lng: {}'.format(lng),lat='Found at lat: {}'.format(lat),maptext='click the below link to see the location',map=map)

if __name__=="__main__":
    app.run(debug=True,port=8050)
#  <div class="col-lg-6 col-12 mt-4 mt-lg-0">
#                                     <button type="submit" class="btn custom-btn cart-btn"
#                                             data-bs-toggle="modal" data-bs-target="#cart-modal"
#                                     onclick="location.href = 'takeimage'">Scan</button>
#                                 </div>