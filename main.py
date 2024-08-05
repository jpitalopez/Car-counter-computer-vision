import cv2
import torch
import numpy as np
from tracker import *



print(torch.cuda.get_device_name(0))

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# SET PATH TO YOUR VIDEO INPUT
cap=cv2.VideoCapture('highway.mp4')


# SET NAME, FORMAT, FPS, AND DIMENSIONS OF YOUR VIDEO OUTPUT
out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'),15, (1020, 600))



count=0
tracker = Tracker()

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        
cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)


# SET THE AREA YOU WANT TO TRACK
area1=[(200,445),(200,472),(900,482),(900,449)]


area_1=set()

while True:
    ret,frame=cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame=cv2.resize(frame,(1020,600))
    
    results=model(frame)
    list=[]
    for index, rows in results.pandas().xyxy[0].iterrows():
        x=int(rows[0])
        y=int(rows[1])
        x1=int(rows[2])
        y1=int(rows[3])
        b=str(rows['name'])
        list.append([x,y,x1,y1])
    
    idx_bbox=tracker.update(list)
    for bbox in idx_bbox:
        x2,y2,x3,y3,id=bbox
        cv2.rectangle(frame,(x2,y2),(x3,y3),(0,0,255),2)
        cv2.putText(frame,str(id),(x2,y2),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
        cv2.circle(frame,(x3,y3),4,(0,255,0),-1)
        result=cv2.pointPolygonTest(np.array(area1,np.int32),((x3,y3)),False)
        if result >0:
            area_1.add(id)
        
    
    
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,255),3)
    a1=len(area_1)
    cv2.putText(frame,str(a1),(549,465),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),2)
    cv2.imshow("FRAME",frame)

    out.write(frame)

    if cv2.waitKey(0)&0xFF==27:
        break
cap.release()
out.release()


cv2.destroyAllWindows()
