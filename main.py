import cv2
import numpy as np
import time
import math
import control
from operator import itemgetter, attrgetter, methodcaller
x_point=[]
y_point=[]
circledraw=[]
pointsdraw=[]
length = 23.5
breadth = 19.0
#load camera
cap = cv2.VideoCapture(0)
fourcc=cv2.cv.CV_FOURCC(*'FMP4')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (800,600))
def nothing(x):
    pass

def get_pixel(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print x,y
        x_point.append(x)
        y_point.append(y)

 
#create calibration window
cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Calibration',get_pixel)
cv2.resizeWindow('Calibration', 800, 600)

while(len(x_point)<=3):
    #get first frame
    ret, img = cap.read()
    cv2.putText(img,"Select points to define corners", (50,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0))
    cv2.imshow('Calibration',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if len(x_point)>3:
        cv2.destroyAllWindows()
        print "black",x_point[0],y_point[0]
        pts1 = np.float32([[x_point[0],y_point[0]],[x_point[1],y_point[1]],[x_point[2],y_point[2]],[x_point[3],y_point[3]]])
        pts2 = np.float32([[0,0],[800,0],[0,600],[800,600]])
        break

#create other windows
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800, 600)

cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
cv2.resizeWindow('mask', 800, 600)
th=0
#time variables
t2=0
t1=0
tc=time.time()
while(len(x_point)>3):
    tic=time.clock()
    # Capture frame-by-frame
    ret, img = cap.read()
    
    #image warp
    M = cv2.getPerspectiveTransform(pts1,pts2)
    imgc=cv2.warpPerspective(img,M,(800,600))
    
    #rotate image
    (h, w) = imgc.shape[:2]
    center = (w / 2, h / 2)
    obj = cv2.getRotationMatrix2D(center, 180, 1.0)
    imgc = cv2.warpAffine(imgc, obj, (w, h))
    
    #convert to hsv format
    hsv = cv2.cvtColor(imgc, cv2.COLOR_BGR2HSV)
    
    #set the values for green
    lower_green = np.array([40,100,100],dtype=np.uint8)
    upper_green = np.array([60,255,255],dtype=np.uint8) 
    
    #create a mask image 
    mask_img = cv2.inRange(hsv, lower_green, upper_green)
    
    resu = cv2.bitwise_and(imgc,imgc, mask= mask_img)                      
    result_image = cv2.cvtColor(resu, cv2.COLOR_BGR2GRAY)
    
    #image thresholding
    ret,thresh = cv2.threshold(result_image,120,255,0)
    
    #find contours            
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    num=0
    x1=0
    x2=0
    x3=0
    x4=0
    y1=0
    y2=0
    y3=0
    y4=0
    coord=list()
    for cnt in contours:
        if cv2.contourArea(cnt)>150:
            #print cv2.contourArea(cnt)
            M = cv2.moments(cnt)                                                                                                                                                                    
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #print cX,cY
            #print "x = "+str((length/800)*cX)+"y = "+str((breadth/600)*cY)
            cv2.circle(img, (cX, cY), 50, (255, 0, 0), -1)
            num=num+1
            if num==1:
                x1=cX
                y1=cY
                coord.append([x1,y1])                
            if num==2:                   
                x2=cX
                y2=cY
                coord.append([x2,y2])
            if num==3:
                x3=cX
                y3=cY
                coord.append([x3,y3])
            
    
    if num==3:
        coord=sorted(coord,key=itemgetter(0))

        
        if(coord[1][1]>coord[2][1] and coord[1][0]>400):
            coord[1],coord[2]=coord[2],coord[1]
        if(coord[1][1]>coord[0][1] and coord[1][0]<400):
            coord[1],coord[0]=coord[0],coord[1]
        
        
        cv2.putText(result_image,"1", (coord[0][0],coord[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.putText(result_image,"2", (coord[2][0],coord[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        cv2.putText(result_image,"3", (coord[1][0],coord[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
        
        #convert pixels in inches. 
        x1=coord[0][0]*length/800
        x2=coord[2][0]*length/800
        x3=coord[1][0]*length/800
        y1=coord[0][1]*breadth/600
        y2=coord[2][1]*breadth/600
        y3=coord[1][1]*breadth/600
        
        cv2.putText(result_image,str((x1,y1)), (coord[0][0]-25,coord[0][1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        cv2.putText(result_image,str((x2,y2)), (coord[2][0]-25,coord[2][1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
        cv2.putText(result_image,str((x3,y3)), (coord[1][0]-25,coord[1][1]+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

        x1=coord[0][0]*length/800
        x2=coord[2][0]*length/800
        x3=coord[1][0]*length/800
        y1=coord[0][1]*breadth/600
        y2=coord[2][1]*breadth/600
        y3=coord[1][1]*breadth/600
        x4=(coord[0][0]+coord[2][0]-coord[1][0])*length/800
        y4=(coord[0][1]+coord[2][1]-coord[1][1])*breadth/600
        #print coord
        d12=math.sqrt(((x2-x1)**2)+((y2-y1)**2))
        d13=math.sqrt(((x3-x1)**2)+((y3-y1)**2))
        d23=math.sqrt(((x3-x2)**2)+((y3-y2)**2))
        d42=math.sqrt(((x4-x2)**2)+((y4-y2)**2))
        cv2.putText(result_image,str((x4,y4)), (int(x4*800/length)-25,int(y4*600/breadth)+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

        #beta=math.acos(((d13**2)+(d12**2)-(d23**2))/(2*d13*d12))
        alpha = math.acos(((d13**2)+(d23**2)-(d12**2))/(2*d13*d23))
        beta = math.pi - alpha
        
        s1=math.sqrt(((x3-x4)**2)+((y3-y4)**2))
        #print s1
        s2=math.sqrt(((x1-x2)**2)+((y1-y2)**2))
        #print beta,alpha,s1
        t1=time.time()
        a,b=3,3
        
        xT=0.1+round(a*math.cos(int(th)*math.pi/180),2)
        yT=8.5+round(b*math.sin(int(th)*math.pi/180),2)
        th=th+0.25
        if th==360:
            th=2
        if xT==0:
            xT=-0.1
        #print xT,yT
        #xT,yT=circle(tc)
        #theta = 90 + math.asin((x4-x3)/s1)*180/math.pi
        
        s4x = x4 + 7.0
        s4y = y4 + 1.2
        s4  = math.sqrt(((s4x-x2)**2)+((s4y-y2)**2))

        s3x = x4 - 7.0
        s3y = y4 + 1.2
        s3  = math.sqrt(((s3x-x1)**2)+((s3y-y1)**2))
        #delta = math.acos(((-s4**2)+49+d42)/(2*7*d42))
        delta = math.atan((y4-y2)/(x2-x4))
        e3 = math.atan((y2-y1)/(x2-x1))
        e1    = math.atan((s3y-y1)/(x1-s3x))
        e2    = math.atan((s4y-y2)/(s4x-x2))
        theta = delta+alpha/2

        cv2.circle(imgc,(int(s3x*800/length),int(s3y*600/breadth)),10,255)
        cv2.circle(imgc,(int(s4x*800/length),int(s4y*600/breadth)),10,255)
        cv2.circle(imgc,(int(x4*800/length),int(y4*600/breadth)),10,255)
        circledraw.append((x3,y3))
        pointsdraw.append((int((xT+11.47)*800/length),int((12.65-yT)*600/breadth)))
        for i in circledraw:
            cv2.circle(imgc,(int(i[0]*800/length),int(i[1]*600/breadth)),1,(0,0,0))
        for y in pointsdraw:
            cv2.circle(imgc,(y[0],y[1]),1,(255,20,255))
        #print alpha*180/math.pi,delta*180/math.pi,s1,s2,s3,s4,d42,theta*180/math.pi,e1*180/math.pi,e2*180/math.pi,xT,yT,(t1-t2)
        control.control(alpha,delta,theta,s1,s2,s3,s4,e1,e2,e3,xT,yT,d42,(t1-t2))
        t2=time.time()
    cv2.imshow('mask',result_image)
    cv2.imshow('image',imgc)
    out.write(imgc)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out.release()
        break
    toc=time.clock()
    #print tic-toc
    
    
    

