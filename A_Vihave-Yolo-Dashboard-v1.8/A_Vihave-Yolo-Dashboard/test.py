import torch
import cv2
import time
import datetime
import requests
# # Model
import os
model = torch.hub.load('ultralytics/yolov5', 'custom',path="yolo.pt",force_reload=True)
r=os.listdir("E:\A_Vihave-Yolo-Dashboard\capture")
frame=cv2.VideoCapture(0)
print("*******************")

print(frame)
results = model(frame)

print("******************************")

results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
classes = model.names
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("\n\nDevice Used:",device)
print("-------------------------------------------")

def class_to_label(x):
    names= ["Pepper__bell___Bacterial_spot","Pepper__bell___healthy","Potato___Early_blight","Potato___Late_blight","Tomato__Target_Spot","Tomato__Tomato_YellowLeaf__Curl_Virus","Tomato_Bacterial_spot","Tomato_Early_blight","Tomato_healthy","Tomato_Late_blight","Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot","Tomato_Spider_mites_Two_spotted_spider_mite"]
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
    frame=cv2.imread(frame)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
            bgr = (0, 255, 0)
            
            image=cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            image=cv2.putText(image, class_to_label(labels[i]), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(class_to_label(labels[i]))
            print(x1, y1, x2, y2)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    return image    

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
        print("------------------------------------------------------------")
        print(labels)
        print(cord)
        print("------------------------------------------------------------")
        return labels, cord
now = datetime.datetime.now()
results=score_frame(frame)
image = plot_boxes(results,frame)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
print(image)
print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# p = os.path.sep.join(['yoloimg', "shot_dis{}.png".format(str(now).replace(":",''))])
# cv2.imwrite(p,image)


# cv2.imwrite(filename,image)
window_name = 'image'
cv2.imshow(window_name,image)

cv2.waitKey(0)


# # directory ='C:/Users/soumy/Downloads/A_Vihave-Yolo-Dashboard/yoloimg'
# # os.chdir(directory)
 
# # print("Before saving image:")  
# # print(os.listdir(directory))  

# # filename = 'savedImage.jpg'
# # cv2.imwrite(filename, frame12)
# # print("After saving image:")  
# # print(os.listdir(directory))
print(results)
