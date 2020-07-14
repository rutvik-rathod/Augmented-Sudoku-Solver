import sys
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import cv2
import numpy as np
import requests
import func as fo
import solver as ss
import copy

import pickle
import os

#ipwebcame server url
url = "http://[2401:4900:36ce:645c::b0]:8080/shot.jpg"

sudoku_size = 400

# image_folder = "//home//debz//Desktop//Deep Learning//Sudoku_project//Data Set//"
##########################################################################
######################### VIDEO SAVER ####################################
filename = '//home//debz//Desktop//Deep Learning//Sudoku_project//hello.mp4'
codec = cv2.VideoWriter_fourcc('X','V','I','D')
framerate = 10
resolution = (720,480)
VideoFileOutput = cv2.VideoWriter(filename,codec,framerate,resolution)
##########################################################################

##########################################################################
def preProcessing(image):
    image = cv2.resize(image,(32,32))
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = cv2.equalizeHist(image)
    image = image/255
    image = image.reshape(1,32,32,1)
    return image

pickle_in = open("/home/debz/Desktop/Deep Learning/Sudoku_project/Model/model_trained.p","rb")
model = pickle.load(pickle_in)
##########################################################################

board_unsolved = [[0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0]]



approx = [0]

while True:
    img_req = requests.get(url)
    img_arr = np.array(bytearray(img_req.content),dtype=np.uint8)
    image = cv2.imdecode(img_arr,-1)
    image = cv2.resize(image,(720,480))
    """ -convert the image into gray
        - apply gausian blur
        - apply adaptive thresholding 
    """
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thres = cv2.adaptiveThreshold(blur, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY,11,8)
    thres = cv2.bitwise_not(thres)
    
    lines = cv2.HoughLinesP(thres,1, np.pi/180, 150, 
                            minLineLength = 100, 
                            maxLineGap = 20)
    
    #convert image to 3 channel to print green lines
    new_thres = np.stack((thres,)*3,axis = -1)
    
    
    image_disp = copy.deepcopy(image)
    blank_color = np.zeros([sudoku_size,sudoku_size,3],dtype=np.uint8)
    blank_BW = np.zeros([sudoku_size,sudoku_size],dtype=np.uint8)
    if str(type(lines)) != '<class \'NoneType\'>' :
        for line in lines:
            x1,y1,x2,y2 = line[0]
            cv2.line(new_thres,(x1,y1),(x2,y2),(0,255,0),2)
    
        contours,hierarchy = cv2.findContours(thres,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        
        areas = [cv2.contourArea(c) for c in contours]
        max_index = np.argmax(areas)
        cnt = contours[max_index]

        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.015*peri,True)
        
        
        # print(image_disp.shape)
        blank_color = np.zeros([sudoku_size,sudoku_size,3],dtype=np.uint8)
        blank_BW = np.zeros([sudoku_size,sudoku_size],dtype=np.uint8)
        image_inverse = np.zeros([480,720],dtype=np.uint8)
    if len(approx) == 4:
        [[x1,y1]],[[x2,y2]],[[x3,y3]],[[x4,y4]] = approx
        
        src_pts = np.array([[x2,y2],
                            [x1,y1],
                            [x4,y4],
                            [x3,y3]],dtype='float32')
        blank_BW = fo.four_point_transform(thres,src_pts)
        blank_color = fo.four_point_transform(image,src_pts)
        # print(thres.shape)
        image_dict = {}
        factor = int(sudoku_size/9)
        for i in range(9):
            for j in range(9):
                image_dict[str(j+1)+str(i+1)] = blank_BW[i * factor : i * factor + factor,
                                                         j * factor : j * factor + factor]
                
                
                image_dict[str(j+1)+str(i+1)] = preProcessing(image_dict[str(j+1)+str(i+1)])
                classIndex = int(model.predict_classes(image_dict[str(j+1)+str(i+1)]))
                board_unsolved[i][j] = classIndex 
        
        board_solved = []
        board_solved  = copy.deepcopy(board_unsolved)
        
        board_solved = ss.solve(board_solved)
        if str(type(board_solved)) != "<class 'bool'>":
            for i in range(9):
                for j in range(9):        
                    origin = ((j * factor + 10, (i*factor + factor)-10))
                    if(board_unsolved[i][j] == 0):
                        blank_color = cv2.putText(blank_color,
                                            str(board_solved[i][j]),
                                            origin,
                                            cv2.FONT_HERSHEY_SIMPLEX,
                                            1.2,
                                            (255,0,255),
                                            2,
                                            cv2.LINE_AA)
                        cv2.drawContours(image_disp,[cnt],0,(0,0,0),-1)
                        image_inverse = fo.four_point_transform_inverse(blank_color,src_pts)
                        image_disp = image_disp + image_inverse
    
    cv2.imshow('Raw_feed',image_disp)
    VideoFileOutput.write(image_disp)
    cv2.imshow('thres',new_thres)
    cv2.imshow('sudoku_projection',blank_color)
    cv2.imshow('sudoku_unsolved',blank_BW)
    
    if cv2.waitKey(10) == 27:
        cv2.destroyAllWindows()
        VideoFileOutput.release()
        break