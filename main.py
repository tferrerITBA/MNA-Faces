# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 14:41:01 2019

@author: Marcos
"""
import cv2
from facespca import image_training, input_testing_pca
#import sys

print('Training in progress...')
training = image_training()
print('Training ready.')

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(1)
print(video_capture.isOpened())
while True:

    ret, frame = video_capture.read()

    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            #flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        cmd = cv2.waitKey(1)
        if cmd==ord('s'):
            i = 0
            for (x, y, w, h) in faces:
                new_h = int(1.1 * h)
                h_padd = int(0.3*h)
                #cv2.rectangle(frame, (x, y), (x+w, y+new_h), (0, 255, 0), 2)
                face_img = gray[y-h_padd:y + new_h, x:x + w]
                face_img = cv2.resize(face_img, (92, 112))
                #gray_image = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
                print("[INFO] Object found. Saving locally.") 
                cv2.imwrite('detected_faces/faces_' + str(i) + '.pgm', face_img )
                input_testing_pca(training, face_img)
                i+= 1
        elif cmd==ord('e'):
            break
        else:
            for (x, y, w, h) in faces:
                new_h = int(1.1 * h)
                h_padd = int(0.3*h)
                cv2.rectangle(frame, (x, y-h_padd), (x+w, y+new_h), (0, 255, 0), 2)
        #status = cv2.imwrite('faces_detected.jpg', frame)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        #if cv2.waitKey(10)==ord('e'):
         #   break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()