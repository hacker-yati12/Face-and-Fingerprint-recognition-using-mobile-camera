# importing required libraries

from tkinter import Image
import cv2
import numpy as np
import os
import dlib 
import face_recognition

# funtion to create box around detected face using pretrained model.
def faceBox(faceNet,frame):
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            cv2.rectangle(frame, (x1,y1),(x2,y2),(0,255,0), 4)
    return frame, bboxs

# Storing models name in in variables.
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# improting model using dnn.readnet in our program.
faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

# these are the mean values for gender and age prediction.
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-6)', '(8-12)', '(12-16)', '(16-20)', '(21-25)', '(27-35)', '(36-45)', '(45-60)']
genderList = ['Male', 'Female']

# finging encodings for all images stored in database
# loading images
path= 'images'
images = []
classnames = []
myList = os.listdir(path)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classnames.append(os.path.splitext(cl)[0])

#finding encodings

def findEncodings(images):
    encodlist = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodlist.append(encode)
    return encodlist

encodeListKnown = findEncodings(images)
print("Encoding done")

# Main program starts.

while True:

    #video captures using cv2 for face recognition.
    video=cv2.VideoCapture(1)                           
    padding=20


    while True:

        # storing realtime image in frame
        ret,frame=video.read()                           
        frame,bboxs=faceBox(faceNet,frame)

        # getting values to draw box around face using above function.
        for bbox in bboxs:
            face=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
            blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

            # predicting gender.
            genderNet.setInput(blob)
            genderPred=genderNet.forward()
            gender=genderList[genderPred[0].argmax()]

            # predicting age.
            ageNet.setInput(blob)
            agePred=ageNet.forward()
            age=ageList[agePred[0].argmax()]


            label="{},{}".format(gender,age)

            # drawing box around face.
            cv2.rectangle(frame,(bbox[0], bbox[1]-30), (bbox[2], bbox[1]), (0,255,0), 4) 

            # showing text.
            cv2.putText(frame, label, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2,cv2.LINE_AA)
        cv2.imshow("Age-Gender",frame)
        
        # Taking snapshot of face for face recognition.
        k = cv2.waitKey(1) & 0xFF
        if k == 13:
            test = "test_face.png"
            cv2.imwrite(test, frame)
            break

    #release camera
    video.release()   

    cv2.destroyAllWindows()

    # code for face recognition
    tstimg = cv2.imread(f'test_face.png')
    tstimg = cv2.cvtColor(tstimg, cv2.COLOR_BGR2RGB)
    encodetstimg = face_recognition.face_encodings(tstimg)[0]
    for en in encodeListKnown:
        matches = face_recognition.compare_faces(encodeListKnown,encodetstimg)
        faceDis = face_recognition.face_distance(encodeListKnown,encodetstimg)
        matchIndex = np.argmin(faceDis)
        
    # print(faceDis[np.argmin(faceDis)])

    #checking face accuracy rate greater than 55 %
    if(faceDis[np.argmin(faceDis)]>0.55):     

        print("No face match found try again")
    else:



                                #####         FINGERPRINT RECOGNITION TESTING        ########



        name = classnames[matchIndex].upper()  

        # open secind camera for fingerprint recognition
        cap = cv2.VideoCapture(0)                                   
        while True:
            ret, img = cap.read()
            cv2.rectangle(img,pt1= (280,180),pt2 = (380,330),color = (0,255,0),thickness=5)
            

            cv2.imshow('fingerprint',img)


            # taking screenshot for fingerprint recognition
            k = cv2.waitKey(1) & 0xFF
            if k == 13:
                test = "test_finger.png"
                cv2.imwrite(test, img)
                break
        

        # release camera
        cap.release()
        cv2.destroyAllWindows()

        #converting img to gray scale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #sharpening img for fingerprint enhancement
        kernel_sharpening = np.array([[-1,-1,-1], 
                                    [-1,9,-1], 
                                    [-1,-1,-1]])
        img = cv2.filter2D(img, -1, kernel_sharpening)

        #using gaussian adaptive threshold for fingerprint extraction
        th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                            cv2.THRESH_BINARY,11,2)
        # blur = cv2.GaussianBlur(th3,(5,5),0)
        # th3 = cv2.addWeighted(blur ,1.5,th3,-0.5,0)
        # cv2.imshow('gray',th3)

        #crop fingerprint from img
        crop_img = th3[190:320, 280:380]
        test = "test_finger1.bmp"
        test2 = "final_finger.bmp"
        crop = cv2.resize(crop_img, None, fx = 1.5, fy = 1.5, interpolation = cv2.INTER_CUBIC)
        # crop = fingerprint_enhancer.enhance_Fingerprint(crop)
        cv2.imshow('f_print',crop)

        #store the fingerprint in folder
        cv2.imwrite(test, th3)
        cv2.imwrite(test2, crop)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


        #reading image for fingerprint matching
        image=cv2.imread("final_finger.bmp")

        fingerprint_database_image = cv2.imread("./saved_samples/" + name + ".bmp" )
        # cv2.imshow('gray2',fingerprint_database_image)

        #using sift algorithm for finding differents points
        sift = cv2.SIFT_create()

        keypoints_1, descriptors_1 = sift.detectAndCompute(image, None)
        keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)  

        # matching the points using flann based matcher
        matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), 
                    dict()).knnMatch(descriptors_1, descriptors_2, k=2)
        match_points = []
        
        #comparing the match points
        for p, q in matches:
            if p.distance <= 0.9*q.distance:
                match_points.append(p)
                
        keypoints = 0
        #storing matched keypoints 
        if len(keypoints_1) <  len(keypoints_2):
            keypoints = len(keypoints_1)            
        else:
            keypoints = len(keypoints_2)


        #checking fingerprint accuracy 
        if (len(match_points) / keypoints * 100 >=25):


            #if face and fingerprint matches then printing name age gender

            print("Name:- ", name)
            print("Face match accuracy : ",(1-faceDis[np.argmin(faceDis)])*100)
            print("Gender :- ", gender)
            print("Predicted Probable Age range :- ", age)
            print("Fingerprint matched \n Access Granted!!!!")
            # print("Fingerprint match accuracy: ", (len(match_points) / keypoints) * 100)
            # print("Figerprint ID: " + str(file)) 
            # result = cv2.drawMatches(image, keypoints_1, fingerprint_database_image, 
            #                         keypoints_2, match_points, None) 
            # result = cv2.resize(result, None, fx=2, fy=2)
            # cv2.imshow("result", result)  
            
        else:
            #if fingerprint doesnot match 
            print("Fingerprint does not match.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()    

    #to identify other person press enter else press Q
    var=input("Press 'Q' to exit or 'enter' to authorise other person.")
    if(var=='Q'):
        break;
    else:
        continue
