#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


# In[2]:


def mouth_aspect_ration(mouth):
    # Vertical distance
    A = dist.euclidean(mouth[3],mouth[9])
    
    # Horizantal distance
    B = dist.euclidean(mouth[0],mouth[6])
    
    mar = A/B
    
    return mar


# In[3]:


FRAMES = 100


# In[4]:


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('facial_landmarks.dat')


# In[5]:


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouthStart,mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


# In[6]:


COUNTER = 0
n_frames = 0
TOTAL = 0
mar = 0

#for plot
m = []

vs = cv2.VideoCapture(0)

if (vs.isOpened() == False):
    print('Error')


flag = ""

time.sleep(1.0)

while True:
    
    
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    
    n_frames += 1
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)
    
    for rect in rects:

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        mouthShape = shape[mouthStart:mouthEnd]
        mar = mouth_aspect_ration(mouthShape)
        m.append(mar)
        mouthHull = cv2.convexHull(mouthShape)
        area1 = cv2.contourArea(mouthShape)
        area2 = cv2.contourArea(mouthHull)
        #bounding = cv2.boundingRect(mouthShape)
       
    
        #cv2.drawContours(frame,[mouthHull],-1,(0,0,255),1)
        
        
        
    if (n_frames == FRAMES and len(m) > 0 ):
            
        n_frames = 0
        
        # convert to numpy array
        mm = np.array(m)
        
        # thresholds may vary. Test with different values   
        
        peak1 = np.where(np.logical_and(mm >= 0.4,mm <= 0.46))[0].size
        peak2 = np.where(mm > 0.5 )[0].size
        
        # 3 is set to avoid any false positive cases
        if  peak1 > peak2 and peak1 > 3:
            flag = 'Speaking'
            
        elif peak2 > peak1 and peak2 > 3:
            flag = 'Yawning'
            
        else:
            flag = 'Quiet'
           
            # mouth analysis: mean of 10 frames
        m1 = []
        i = 0
        f = 10
        while (i < (len(m) - len(m)%f)):
            temp = m[i:i+f]
            m1.append(sum(temp)/len(temp))
            i = i+f
        
        """
        #Check these graphs to understand how threshold varies
        
        import matplotlib.pyplot as plt 

        plt.figure(figsize=(15,8))
        plt.title("Lips")
        plt.xlabel("Frames")
        plt.ylabel("MAR")
        plt.ylim(0,1)
        plt.plot(m1)
        plt.show()
        plt.show()
        """   
        
        # reinitialize
        m = []
        #end of if        
    
    cv2.putText(frame, flag, (300, 80),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    
    
    

    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


vs.release()
cv2.destroyAllWindows()

