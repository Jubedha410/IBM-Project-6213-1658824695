import requests
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session
from twilio.rest import Client
from playsound import playsound
import cv2

#load the forest fire model from IBM Watson Studio.
model = load_model("Forest_fire.h5")

#home page
app= Flask(__name__, template_folder='template')
@app.route("/")
def index():
    return render_template('index.html')
app.run(host='0.0.0.0' , port=5001) 
@app.route('/')
def prediction():
    video=cv2.VideoCapture(r'C:\Users\admin\Downloads\ai\forest_fire_1.mp4')
    while (1):
        success, frame= video.read()
        cv2.imwrite('forest.jpg', frame)
        img= image.load_img('forest.jpg', target_size=(64,64,3))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        pred=model.predict(x)
        p= int(pred[0])
        print (p)
        name=['forest without fire','forest with fire']
        cv2.putText(frame, 'predicted class ='+str(name[p]), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),1)
        pred= model.predict(x)
        if pred[0]==1:
            account_sid = 'AC8d900acfe2f3400cbed2b4a9abbcda90'
            auth_token= '6d169a03f7ef5659901e40f2b5403931'
            client = Client (account_sid, auth_token)
            message = client.messages      .create(
                body='Forest Fire is detected, stay alert',
                from_= '+18176702460',
                to='+919962278677')
            print(message.sid)
            print ('Fire detected')
            print ('Message Sent')
            cv2.imshow('image',frame)
        else:
            print ('No Danger')
            cv2.imshow('image',frame)
            if cv2.waitKey(1) & 0xFF == ord('a'):
                break
                
    video.release() 
    cv2.destroyAllWindows()  
if __name__ == "__main__":
    app.run(debug=True)