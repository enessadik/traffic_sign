import cv2
import numpy as np
from tensorflow.python.keras.models import load_model


#Camera Resolution
frameWidth=640
frameHeight=480
brightness=180
threshold=0.70
font=cv2.FONT_HERSHEY_SIMPLEX

#Setup the video camera
cap=cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,frameHeight)
cap.set(10,brightness)


#Loaded Model
model_out=load_model("model_trained.h5")


def grayScale(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img=cv2.equalizeHist(img)
    return img


#preproccess
def preProcess(img):
    img=grayScale(img)
    img=equalize(img)
    img=img / 255    
    return img




def getClassName(classNo):
    if   classNo == 0: return "Hiz Limiti 20 km/h"
    elif classNo == 1: return "Hiz limiti 30 km/h"
    elif classNo == 2: return "Hiz limiti 50 km/h"
    elif classNo == 3: return "Hiz limiti 60 km/h"
    elif classNo == 4: return "Hiz limiti 70 km/h"
    elif classNo == 5: return "Hiz limiti 80 km/h"
    elif classNo == 6: return "Son Hiz  80 km/h"
    elif classNo == 7: return "Hiz limiti 100 km/h"
    elif classNo == 8: return "Hiz limiti 120 km/h"
    elif classNo == 9: return "Gecis yok"
    


while True:
    #read image
    success,imgOrignal=cap.read()
    
    #proccess image
    img=np.asarray(imgOrignal)
    img=cv2.resize(img,(32,32))
    img=preProcess(img)
    cv2.imshow("Islenmis Resim",img)
    
    img=img.reshape(1,32,32,1)
    cv2.putText(imgOrignal, "Class: ", (20,35),font,0.75,(0,0,255),2,cv2.LINE_AA)
    cv2.putText(imgOrignal, "Probability: ",(20,75),font,0.75,(255,0,0),2,cv2.LINE_AA)
    
    #predict image
    predictions=model_out.predict(img)
    classIndex=model_out.predict_classes(img)
    probVal=np.amax(predictions)
    
    if probVal > threshold:
        cv2.putText(imgOrignal,str(classIndex)+"  "+str(getClassName(classIndex)),(120,35),font,0.75,(0,0,255),2,cv2.LINE_AA)
        cv2.putText(imgOrignal,str(round(probVal*100,2) )+"%",(180,75),font,0.75,(255,0,0),2,cv2.LINE_AA)             
    
    cv2.imshow("result", imgOrignal)
   


    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
               


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    