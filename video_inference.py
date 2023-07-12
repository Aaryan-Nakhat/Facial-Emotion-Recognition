# Imports
import numpy as np
import cv2
from PIL import Image
from time import sleep
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, pipeline
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from typing import Optional
import argparse

# Defining the Command Line Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_to_use", type=int, help="which model to use (pass an integer among [0, 1, 2, 3, 4]",  required=True)
parser.add_argument("--enable_real_time_video_inferencing", type=bool, help="whether to enable real-time video inferencing (type: bool)", required=True)
parser.add_argument("--prerecorded_video_path", help="path to the pre-recorded video")

# Helper Variables
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
available_models = ['./models/Emotion_Recognition_CNN.h5', 
                    './models/Emotion_Recognition_CNN_with_Data_Aug.h5', 
                    'Aaryan333/vit-base-finetuned-fer2013',
                    'Aaryan333/convnext-tiny-finetuned-fer2013',
                    'Aaryan333/convnextv2-tiny-384-finetuned-fer2013'
                   ]

# Video FER Inferencing Function
def do_video_inferencing(model_to_use: int, enable_real_time_video_inferencing: Optional[bool] = False, 
               prerecorded_video_path: Optional[str] = None):
        
    if model_to_use in range(5):
        face_detector = cv2.CascadeClassifier("haarcascades_models/haarcascade_frontalface_default.xml")
        if model_to_use < 2:
            emotion_recognition_model = load_model(available_models[model_to_use])   
        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained(available_models[model_to_use])
            model = AutoModelForImageClassification.from_pretrained(available_models[model_to_use])
            pipe = pipeline("image-classification", model=model, feature_extractor=feature_extractor)   
            
        if enable_real_time_video_inferencing:
            cap = cv2.VideoCapture(0)
        else:
            if prerecorded_video_path == None:
                raise ValueError("Please provide a pre-recorded video path!")  
            video_path = fr"{prerecorded_video_path}"
            cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read() # ret is a boolean variable that returns true if the frame is available. frame is an image array vector captured based on the default frames per second    
            if not ret:
                break
            gray_scale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(image=gray_scale_frame, scaleFactor=1.3, minNeighbors=5)

            for (x,y,w,h) in faces:
                cv2.rectangle(img=frame, pt1=(x,y), pt2=(x+w,y+h), color=(255,0,0), thickness=2)
                roi_gray_scale_frame = gray_scale_frame[y:y+h,x:x+w]
                roi_gray_scale_frame = cv2.resize(roi_gray_scale_frame, (48,48), interpolation=cv2.INTER_AREA)

                if model_to_use < 2:
                    roi = roi_gray_scale_frame.astype('float')/255.0  # Doing the preprocessing as per the trained emotion recog. model
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)  # Expand dims to get it ready for prediction (1, 48, 48, 1)
                    prediction = emotion_recognition_model.predict(roi, verbose=None)[0]
                    label = class_labels[np.argmax(prediction, axis=-1)]

                else:
                    roi = Image.fromarray(roi_gray_scale_frame)
                    label = pipe(roi)[0]['label']

                label_position = (x,y)
                cv2.putText(img=frame, text=label, org=label_position, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), 
                    thickness=2)

            cv2.imshow('Real-time Emotion Recognizer', frame)

            k = cv2.waitKey(30) & 0xff
            if k == 27:      #Press Esc to stop the video
                break
        cap.release()
        cv2.destroyAllWindows()               
    else:
        raise ValueError("model_to_use must be an integer among [0, 1, 2, 3, 4]")


# Getting the Command Line Arguments
args = parser.parse_args().__dict__ 
a = args['model_to_use']
boolean_checker = isinstance(args['enable_real_time_video_inferencing'], bool)
if boolean_checker:
	b = args['enable_real_time_video_inferencing']
else:
	raise ValueError("enable_real_time_video_inferencing should be a boolean [True, False]")
string_checker = isinstance(args['prerecorded_video_path'], str)
if string_checker:
	c = args['prerecorded_video_path']
else:
	c = None

# Calling the function
if __name__ == "__main__":
	do_video_inferencing(model_to_use=a, enable_real_time_video_inferencing=b, prerecorded_video_path=c)
