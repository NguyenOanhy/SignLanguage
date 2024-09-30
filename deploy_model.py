import cv2
import pandas
import numpy as np
import torch

def detection():
    model = torch.hub.load("./yolov5","custom",path = "last.pt", source='local')
    cap = cv2.VideoCapture(0)
    while True:
            ret,img = cap.read()
            img = cv2.resize(img,(640,640))
            detection = model(img)
            rusult = detection.pandas().xyxy[0].to_dict(orient = "records")
            x = np.array(rusult)
            if len(x):
                for result in rusult:
                    confidence = round(result['confidence'],2)
                    name = result["name"]
                    clas = result["class"]
                    print(name,confidence)
                    if confidence > 0.1:
                        x1 = int(result["xmin"])
                        y1 = int(result["ymin"])
                        x2 = int(result["xmax"])
                        y2 = int(result["ymax"])

                        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
                        cv2.putText(img,name,(x1+3,y1-10),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,0),1)

            cv2.imshow('frame', img)
            cv2.waitKey(1)


detection()