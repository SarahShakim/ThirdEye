import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import face_recognition
import numpy as np

#link to path with file that contains classifiers for different face aspects
#loads face cascade into memory to be able to detect faces
cascPath = "haarcascade_frontalface_default.xml" 
#creating a face cascade
faceCascade = cv2.CascadeClassifier(cascPath)

log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0


print("Facial Recognition Started...")
jameela_image = face_recognition.load_image_file("faces/Jameela Shakim.jpg")
jameela_encoding = face_recognition.face_encodings(jameela_image)[0]

jiten_image = face_recognition.load_image_file("faces/Jiten Aylani.jpg")
jiten_encoding = face_recognition.face_encodings(jiten_image)[0]

known_face_encodings = [
    jameela_encoding,
    jiten_encoding
]
known_face_names = [
    "Jameela Shakim",
    "Jiten Aylani"
]
print("Facial Recognition Complete!")

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))


    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    key = cv2.waitKey(1)
    
    if anterior > 0:
        cv2.imwrite(filename='saved_img.jpg', img=frame)
       # webcam.release()
        img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
        # img_new = cv2.imshow("Captured Image", img_new)
        cv2.waitKey(1650)
        cv2.destroyAllWindows()

        saved_img = face_recognition.load_image_file("saved_img.jpg")
        saved_locations = face_recognition.face_locations(saved_img)
        saved_encodings = face_recognition.face_encodings(saved_img, saved_locations)

        face_names = []
        for saved_encoding in saved_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, saved_encoding)

            face_distances = face_recognition.face_distance(known_face_encodings, saved_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            else: 
                name = "Unsuspected"
            
            face_names.append(name)

        print(face_names)
      
        for (top, right, bottom, left), name in zip(saved_locations, face_names):
            # img_new = cv2.rectangle(img_new, (left, top), (right, bottom), (0, 0, 255), 2)
            # img_new = cv2.rectangle(img_new, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            img_new = cv2.putText(img_new, name, (left + 6, bottom + 50), font, 1.0, (255, 255, 255), 1)
            cv2.imwrite(filename='detected_img.jpg', img=img_new)
            cv2.imshow("Captured Image", img_new)

    
    '''
    if key == ord('s'): 
        cv2.imwrite(filename='saved_img.jpg', img=frame)
        webcam.release()
        img_new = cv2.imread('saved_img.jpg', cv2.IMREAD_GRAYSCALE)
        img_new = cv2.imshow("Captured Image", img_new)
        cv2.waitKey(1650)
        cv2.destroyAllWindows()
    '''

    
    if key == ord('q'):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
