from ultralytics import YOLO
import cv2
from gtts import gTTS
import pygame
import os
import pyttsx3
from playsound import playsound


engine = pyttsx3.init()
# Function that converts TTS and says the message
# @param: The text that you want to convert to audio
def say(mytext):
    engine.say(mytext)
    engine.runAndWait()

#load yolov8 model
model = YOLO('yolov8n.pt')

video_path = './manhattan.mp4'
vid = cv2.VideoCapture(video_path)

prevSize=0
ret = True
#read each frame of video
while ret:
    ret, frame = vid.read()

    # detect and track objects using yolov8
    results = model.track(frame, persist=True,classes=0, conf=.5) # only track people, (class id is 0 in yolo default model)

    #draw bounding boxes around the results we got, this is our 'new frame' to be displayed
    new_frame_ = results[0].plot()
    size = len(results[0]) # get the count of all the people that are being tracked (in context, this every person being detected that's being captured in the video)
    l = f"Number of people present: {size}"
    font = cv2.FONT_HERSHEY_SIMPLEX 
    cv2.putText(new_frame_,l, (50, 50),font, 1,(0, 255, 255),2, cv2.LINE_4) 
    cv2.imshow('frame',new_frame_) # show each frame of the video with the new drawn on boxes
    if size>prevSize:
        say('Person entered')
    if size<prevSize:
        say('Person exited')
    prevSize = size
    if cv2.waitKey(25) & 0xFF == ord('q'): # if we press q, then quit
        break