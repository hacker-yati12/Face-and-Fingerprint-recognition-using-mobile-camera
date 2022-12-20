from tkinter import Image
import cv2
import numpy as np
import os
from PIL import Image

cap = cv2.VideoCapture(0)                                   #####fingerprint testrecognition########
while True:
    ret, img = cap.read()
    cv2.rectangle(img,pt1= (280,180),pt2 = (380,330),color = (0,255,0),thickness=5)
    


    cv2.imshow('fingerprint',img)


    # taking screenshot 
    k = cv2.waitKey(1) & 0xFF
    if k == 13:
        test = "test_finger.bmp"
        cv2.imwrite(test, img)
        break

cap.release()
cv2.destroyAllWindows()


img = cv2.imread('test_finger.bmp',1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.medianBlur(img,5)
kernel_sharpening = np.array([[-1,-1,-1], 
                              [-1,9,-1], 
                              [-1,-1,-1]])
img = cv2.filter2D(img, -1, kernel_sharpening)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv2.THRESH_BINARY,11,2)
# blur = cv2.GaussianBlur(th3,(5,5),0)
# th3 = cv2.addWeighted(blur ,1.5,th3,-0.5,0)
# cv2.imshow('gray',th3)
crop_img = th3[190:320, 280:380]
test = "test_finger1.bmp"
test2 = "final_finger.bmp"
crop = cv2.resize(crop_img, None, fx = 1, fy = 1, interpolation = cv2.INTER_CUBIC)
# cv2.imshow('f_print',crop_img)
cv2.imwrite(test, th3)
cv2.imwrite(test2, crop)

cv2.waitKey(0)
cv2.destroyAllWindows()

# image=cv2.imread("final_finger.bmp")

# # dst = cv2.resize(src, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
# fingerprint_database_image = cv2.imread("./saved_samples/Abhishek.bmp" )
# # cv2.imshow('gray2',fingerprint_database_image)
# sift = cv2.SIFT_create()

# keypoints_1, descriptors_1 = sift.detectAndCompute(image, None)
# keypoints_2, descriptors_2 = sift.detectAndCompute(fingerprint_database_image, None)  
# matches = cv2.FlannBasedMatcher(dict(algorithm=1, trees=10), 
#              dict()).knnMatch(descriptors_1, descriptors_2, k=2)
# match_points = []
        
# for p, q in matches:
#     if p.distance <= 0.9*q.distance:
#         match_points.append(p)
        
# keypoints = 0
# if len(keypoints_1) <  len(keypoints_2):
#     keypoints = len(keypoints_1)            
# else:
#     keypoints = len(keypoints_2)

# if (len(match_points) / keypoints * 100 >=0):

#     # print("Name:- ", name)
#     # print("Face match accuracy : ",(1-faceDis[np.argmin(faceDis)])*100)
#     # print("Gender :- ", gender)
#     # print("perdicted Probable Age range :- ", age)
#     print("fingerprint match accuracy: ", (len(match_points) / keypoints) * 100)
#     # print("Figerprint ID: " + str(file)) 
#     result = cv2.drawMatches(image, keypoints_1, fingerprint_database_image, 
#                             keypoints_2, match_points, None) 
#     result = cv2.resize(result, None, fx=2, fy=2)
#     cv2.imshow("result", result)
# else:
#     print("Fingerprint does not match.")
# cv2.waitKey(0)
# cv2.destroyAllWindows()
