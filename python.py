import tkinter as tk
import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import math
import requests
import pandas as pd
from pandastable import Table, TableModel
import threading
import numpy as np

root=tk.Tk()
root.geometry('1600x1400')
root.title('Tkinter Hub')

w = 600
h = 600
color = "#581845"
frame_1 = Frame(root, width=1600, height=70, bg="black").place(x=0, y=0)
#bg="gray16"



cap = cv2.VideoCapture(0)




i = tk.PhotoImage(file="C:/Users/Admin/OneDrive/Desktop/cricketsort/log/Screenshot 2023-03-25 202032.png")
# Create a label widget with the image as its background
img = tk.Label(frame_1, image=i,width=1600,height=730,bg="blue")
img.place(x=0,y=59)



stop = False
labels = []

label1=None
label2=None
label3=None
label4=None
label5=None
label6=None
current_thread = None



a=None
label6 = tk.Label(frame_1,width=300,height=1)
label6.place(x=2, y=60)
label6.configure(text='FPS: {}'.format(a))

###########################################################################################################################
#2
#image clasification
def select_file1():
    global label3
    global current_thread
    
    # stop the previous thread if it is currently running
    if current_thread and current_thread.is_alive():
        current_thread.close()
    file_path1 = filedialog.askopenfilename()
    select_img(file_path1)
    current_thread = threading.Thread(target=([select_img]))
    current_thread.start()
    
    
select_button1 = Button(frame_1, text='image', command=select_file1).place(x=400, y=15)

def select_img(file_path1):
  global label3
  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

  def calculateAngle(a, b, c):
    # Calculate the angle between three points
    radians = math.atan2(c.y - b.y, c.x - b.x) - math.atan2(a.y - b.y, a.x - b.x)
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

  # Load the input image
  

  image = cv2.imread(file_path1)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # Get the pose landmarks using Mediapipe Pose
  results = pose.process(image)

  if results.pose_landmarks:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    landmarks = results.pose_landmarks.landmark

    # Get the angle between the left knee, hip and ankle points
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])

    # Get the angle between the right knee, hip and ankle points
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])

    # Get the angle between the left hip, shoulder and elbow points
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value])

    # Get the angle between the right hip, shoulder and elbow points
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    # Get the angle between the left ankle, knee, and hip points. 
    left_ankle_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                  landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    # Get the angle between the right ankle, knee, and hip points. 
    right_ankle_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])

    # Get the angle between the left and right hips
    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value])
    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                  landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])

    # Check if the pose matches a cricket sweep shot
    if  (left_knee_angle > 80 and left_knee_angle < 170 and right_knee_angle > 80 and right_knee_angle < 170)and(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].visibility > landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].visibility )and(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].visibility > landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].visibility )and(left_hip_angle > 45 and left_hip_angle< 72 and right_hip_angle >= 0 and right_hip_angle < 21)and(left_shoulder_angle > 20 and left_shoulder_angle < 60 and right_shoulder_angle > 0 and right_shoulder_angle < 60) :
        
                           fps="sweep-shot "
                           cv2.putText(image, '{}'.format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    elif  (left_knee_angle > 165 and left_knee_angle < 180 and right_knee_angle >135 and right_knee_angle < 180)and(left_shoulder_angle > 6 and left_shoulder_angle < 155 and right_shoulder_angle > 25 and right_shoulder_angle < 65):
        
             fps=" Straight-drive shot "
             cv2.putText(image, '{}'.format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    elif   (left_hip_angle > 5 and left_hip_angle< 16)and( right_knee_angle >100 and right_knee_angle < 180)and(right_shoulder_angle > 19 and right_shoulder_angle < 80):
            fps="Pull-shot "
            cv2.putText(image, '{}'.format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    elif(left_knee_angle > 75 and left_knee_angle < 150 and right_knee_angle >125 and right_knee_angle < 155)and(left_hip_angle > 35 and left_hip_angle< 70)and(left_shoulder_angle > 65 and left_shoulder_angle < 135 and right_shoulder_angle > 50 and right_shoulder_angle < 65):
        fps=" forwerd-defence shot "
        cv2.putText(image, '{}'.format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    else:
        fps="Unknow-shot"
        cv2.putText(image, '{}'.format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
  # Show the image
  #cv2.imshow("Image", image)
  label3 = tk.Label(root, width=1400, height=900)
  label3.place(x=100, y=70)
  image = Image.fromarray(image)
  
  iago = ImageTk.PhotoImage(image)
  label3.configure(image=iago)
  label3.image = iago
  
  
  #cv2.waitKey(0)
  #cv2.destroyAllWindows()

#############################################################################################################################
#3
#developed

def select_file2():
    
    global current_thread
    
    # stop the previous thread if it is currently running
    if current_thread and current_thread.is_alive():
        current_thread.close()
    deve( w, h)
    deve1( w, h)
    current_thread = threading.Thread(target=([deve,deve1]))
    current_thread.start()
   
select_button2 = Button(frame_1, text='Research', command=select_file2).place(x=750, y=31)


def deve( w, h):
  global label2
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_selfie_segmentation = mp.solutions.selfie_segmentation
  mp_pose = mp.solutions.pose

  # Load the input image
  input_image = cv2.imread("C:/Users/Admin/OneDrive/Pictures/Screenshot 2023-03-13 110302.jpg")
  

  # Initialize the MediaPipe Selfie Segmentation and Pose models
  with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation, \
     mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
  
    # Convert the BGR image to RGB before processing
    image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Process the image and get the segmentation mask
    results = selfie_segmentation.process(image)

    # Draw the segmentation on the image
    condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    bg_image = np.zeros(image.shape, dtype=np.uint8)
    bg_image[:] = (255, 0, 0)  # blue
    annotated_image = np.where(condition, image, bg_image)

    # Process the annotated image and get the pose landmarks
    pose_results = pose.process(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    

    # Draw the pose landmarks on the image
    mp_drawing.draw_landmarks(
        annotated_image,
        pose_results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    
    results = pose.process(annotated_image)
    if results.pose_landmarks:
       mp_drawing = mp.solutions.drawing_utils
       mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
       landmarks = results.pose_landmarks.landmark
       #cv2.imshow("Segmented Image", annotated_image)
       
       label2 = tk.Label(root, width=600, height=600)
       label2.place(x=800, y=70)
       image = Image.fromarray(annotated_image)
  
       iago = ImageTk.PhotoImage(image)
       label2.configure(image=iago)
       label2.image = iago
       
    cv2.waitKey(0)

def deve1( w, h):
  global label1
  mp_pose = mp.solutions.pose
  pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
  mp_drawing = mp.solutions.drawing_utils
  pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

  image = cv2.imread("C:/Users/Admin/OneDrive/Pictures/Screenshot 2023-03-13 110302.jpg")
  img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  results = pose.process(img)
  if results.pose_landmarks:
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

# Display image with landmarks
  #cv2.imshow('img',img)
  label1 = tk.Label(root, width=600, height=600)
  label1.place(x=100, y=70)
  image = Image.fromarray(img)
  
  iago = ImageTk.PhotoImage(image)
  label1.configure(image=iago)
  label1.image = iago
  cv2.waitKey(0)
  
    
  cv2.destroyAllWindows()
#################################################################################################
  #4
  
table_frame = None 
def select_file():
    global current_thread
    
    # stop the previous thread if it is currently running
    if current_thread and current_thread.is_alive():
        current_thread.stop()
    cricket_match()
    current_thread = threading.Thread(target=([cricket_match]))
    current_thread.start()

def close_window():
    global table_frame,label3,label1,label2,stop,label4,label5,a,label6
    if table_frame:
       table_frame.destroy()
    
     
    if label3:
        label3.destroy()
        root.update()

    if label1:
        label1.destroy()
        root.update()
    if label2:
        label2.destroy()
        root.update()
    if label4 and label5:
        label4.destroy()
        stop = True
    if label6:
         stop = True
    
    a=None
    label6.configure(text='FPS: {}'.format(a))

close_button = tk.Button(frame_1, text='Close', command=close_window).place(x=753, y=5)

select_button = tk.Button(frame_1, text='Cricket Match', command=select_file).place(x=1350, y=15)


def cricket_match():
    global table_frame
    match_url = "https://api.cricapi.com/v1/currentMatches"
    match_params = {"apikey": "9f8fd5e4-4e93-4ecf-b4e4-9b8fcd33b30a", "offset": 0}

    all_matches_url = "https://api.cricapi.com/v1/matches"
    all_matches_params = {"apikey": "9f8fd5e4-4e93-4ecf-b4e4-9b8fcd33b30a", "offset": 0}

    match_response = requests.get(match_url, params=match_params)
    all_matches_response = requests.get(all_matches_url, params=all_matches_params)

    if match_response.status_code == 200 and all_matches_response.status_code == 200:
        # extract the match data from the API response
        match_data = match_response.json()["data"]
        all_matches_data = all_matches_response.json()["data"]
        # convert the match data to Pandas DataFrame
        df_matches = pd.json_normalize(match_data)
        df_all_matches = pd.json_normalize(all_matches_data)

        df_matches = df_matches.drop(columns=['id'])
        df_all_matches = df_all_matches.drop(columns=['id'])
        df_matches = df_matches.drop(columns=['date'])
        df_all_matches = df_all_matches.drop(columns=['date'])
        df_matches = df_matches.drop(columns=['teams'])
        df_all_matches = df_all_matches.drop(columns=['teams'])
        df_matches = df_matches.drop(columns=['series_id'])
        df_all_matches = df_all_matches.drop(columns=['series_id'])
        df_matches = df_matches.drop(columns=['teamInfo'])
        df_all_matches = df_all_matches.drop(columns=['teamInfo'])
    else:
        print("Request failed")
    
    # create the table widget
    table_frame = tk.Frame(root)
    
    table_frame.place(x=10,y=100)
    table = Table(table_frame, dataframe=pd.concat([df_matches, df_all_matches]), width=1400, height=600)
   

    table.show()
##################################################################################################
#5
#skeleton

def select_file10():
    file_path = filedialog.askopenfilename()
    select_im(file_path, w, h)

select_button = Button(frame_1, text='HUman Pose ', command=select_file10)
select_button.place(x=30,y=15)
def select_im(file_path, w, h):
    global label4,label5,labels,stop
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose=mp_pose.Pose()
    cap = cv2.VideoCapture(file_path)
    while True:
        ret, img = cap.read()
        if not ret:
            label4.destroy()
            cv2.destroyAllWindows()
            break
        img = cv2.resize(img, (w,h))
        result=pose.process(img)
        mp_drawing.draw_landmarks(img, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec((255,0,0),2,2), DrawingSpec((255,0,255),2,2))
        h,w,c=img.shape
        opIng=np.zeros([h,w,c], dtype=np.uint8)  # convert data type to uint8
        opIng.fill(255)
        mp_drawing.draw_landmarks(opIng, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, DrawingSpec((255,0,0),2,2), DrawingSpec((255,0,255),2,2))
        if result.pose_landmarks is not None:
            label4 = tk.Label(frame_1, width=w, height=h)
            label4.place(x=2, y=160)
            image = Image.fromarray(img)
            iago = ImageTk.PhotoImage(image)
            label4.configure(image=iago)
            label4.image = iago
            labels.append(label4)

            label5 = tk.Label(frame_1, width=w, height=h)
            label5.place(x=700, y=160)
            image_2 = Image.fromarray(opIng)
            iago_2 = ImageTk.PhotoImage(image_2)
            label5.configure(image=iago_2)
            label5.image = iago_2
            labels.append(label5)

            root.update()
            root.after(10)
            if stop:
                cap.release()
                
                # destroy all labels in the list
                for label in labels:
                    label.destroy()
                    stop = False
                    
                break
####################################################################################################
#6
#vedio
def select_file11():
    file_path = filedialog.askopenfilename()
    select_i(file_path)

select_button = Button(frame_1, text='Video', command=select_file11)
select_button.place(x=1000,y=15)

def select_i(file_path):
    global label6,labels,stop,a
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(file_path)
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
      while cap.isOpened():

        success, image = cap.read()
        if not success:
            a=b
            label6.configure(text='FPS: {}'.format(a))
            break

        image = cv2.resize(image, (800, 800))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        keypoints = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
        
        if  (keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value * 3 + 1] and \
              keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value * 3 + 1] and \
              keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value * 3 + 1] and \
              keypoints[mp_pose.PoseLandmark.LEFT_HIP.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value * 3 + 1] and \
              keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value * 3 + 1] and \
              keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value * 3 + 1]) and(keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value * 3 + 1] and \
              keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value * 3 + 1] and \
              keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value * 3 + 1] and \
              keypoints[mp_pose.PoseLandmark.LEFT_HIP.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value * 3 + 1] and \
              keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value * 3 + 1] and \
              keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value * 3 + 1]):
              fps="Drive shot detected!"
              b=fps
              #label6.configure(text='FPS: {}'.format(a))
              #cv2.putText(image, 'FPS: {}'.format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
              cv2.waitKey(100)
        
        elif (keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_HIP.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value * 3] > keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value * 3] and \
               keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value * 3] > keypoints[mp_pose.PoseLandmark.LEFT_HIP.value * 3]) and(keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value * 3 + 1] > keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_HIP.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.RIGHT_ANKLE.value * 3 + 1] and \
               keypoints[mp_pose.PoseLandmark.LEFT_ANKLE.value * 3] > keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value * 3] and \
               keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value * 3] > keypoints[mp_pose.PoseLandmark.LEFT_HIP.value * 3]):
               fps="Pull shot detected!"
               b=fps
               #cv2.putText(image, 'FPS: {}'.format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
               cv2.waitKey(100) 
        elif   (keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value * 3 + 0] > keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value * 3 + 0] and \
                keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value * 3 + 1] and \
                keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value * 3 + 1]) and(keypoints[mp_pose.PoseLandmark.RIGHT_ELBOW.value * 3 + 0] > keypoints[mp_pose.PoseLandmark.RIGHT_WRIST.value * 3 + 0] and \
                keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value * 3 + 1] and \
                keypoints[mp_pose.PoseLandmark.LEFT_ELBOW.value * 3 + 1] < keypoints[mp_pose.PoseLandmark.LEFT_WRIST.value * 3 + 1]):
                fps="Leg glance/flick shot detected!"
                b=fps
                #cv2.putText(image, 'FPS: {}'.format(fps), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
                cv2.waitKey(100)
        else :
            b="Unknow shot"
        cv2.imshow('MediaPipe Pose', image)
        
        
       
        root.update()
        root.after(10)
        
    cap.release()
    cv2.destroyAllWindows()
    cv2.destroyAllWindows()

root.mainloop()
