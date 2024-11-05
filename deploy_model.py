import cv2
import numpy as np
import torch

def detection():
    # Load the YOLOv9 model
    model = torch.hub.load("./yolov9", "custom", path="last.pt", source='local')

    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to capture video")
            break
        
        img = cv2.resize(img, (640, 640))
        
        # Perform detection
        results = model(img)

        # Convert results to pandas DataFrame
        results_df = results.pandas().xyxy[0]

        if len(results_df):
            for index, result in results_df.iterrows():
                confidence = round(result['confidence'], 2)
                name = result["name"]
                clas = result["class"]
                print(name, confidence)
                
                if confidence > 0.1:
                    x1 = int(result["xmin"])
                    y1 = int(result["ymin"])
                    x2 = int(result["xmax"])
                    y2 = int(result["ymax"])

                    # Draw rectangle and label
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, name, (x1 + 3, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)

        cv2.imshow('frame', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

detection()
