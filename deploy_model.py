# import cv2
# import pandas as pd
# import numpy as np
# import torch

# def detection():
#     # Load the YOLOv9 model
#     model = torch.hub.load("./yolov9", "custom", path="last.pt", source='local')

#     cap = cv2.VideoCapture(0)  # Mở camera
#     while True:
#         ret, img = cap.read()
#         if not ret:
#             print("Failed to capture video")
#             break

#         # Resize frame to fit the model
#         img = cv2.resize(img, (640, 640))

#         # Perform detection
#         detection = model(img)

#         # Convert results to pandas DataFrame
#         results = detection.pandas().xyxy[0].to_dict(orient="records")
#         x = np.array(results)

#         if len(x):
#             for result in results:
#                 confidence = round(result['confidence'], 2)
#                 name = result["name"]
#                 clas = result["class"]
#                 print(name, confidence)

#                 if confidence > 0.4:
#                     x1 = int(result["xmin"])
#                     y1 = int(result["ymin"])
#                     x2 = int(result["xmax"])
#                     y2 = int(result["ymax"])

#                     # Draw rectangle around detected object and add label
#                     cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                     cv2.putText(img, name, (x1 + 3, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)

#         # Show the frame
#         cv2.imshow('frame', img)

#         # Wait for a small amount of time to display frames smoothly and check for keypress
#         if cv2.waitKey(1) & 0xFF == ord('q'):  
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# detection()
import cv2
import pandas as pd
import numpy as np
import torch

def detection():
    # Load the YOLOv9 model
    model = torch.hub.load("./yolov9", "custom", path="best.pt", source='local')

    cap = cv2.VideoCapture(0)  # Open the webcam
    while True:
        ret, img = cap.read()
        if not ret:
            print("Failed to capture video")
            break

        # Resize frame to fit the model
        img_resized = cv2.resize(img, (640, 640))

        # Perform detection
        detection = model(img_resized)

        # Convert results to pandas DataFrame
        results = detection.pandas().xyxy[0].to_dict(orient="records")
        x = np.array(results)

        # Variable to store the object with the highest confidence
        highest_confidence = 0
        best_result = None

        if len(x):
            for result in results:
                confidence = round(result['confidence'], 2)
                name = result["name"]

                # Check if the detected object's confidence is higher than the current highest confidence
                if confidence > highest_confidence:
                    highest_confidence = confidence
                    best_result = result

            # If we found a result with confidence higher than 0.4, draw a rectangle
            if best_result and highest_confidence > 0.3:
                x1 = int(best_result["xmin"])
                y1 = int(best_result["ymin"])
                x2 = int(best_result["xmax"])
                y2 = int(best_result["ymax"])

                # Draw rectangle around detected object with the highest confidence
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f"{best_result['name']} ({highest_confidence})", 
                            (x1 + 3, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 1)

        # Show the frame
        cv2.imshow('frame', img)

        # Wait for a small amount of time to display frames smoothly and check for keypress
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break

    cap.release()
    cv2.destroyAllWindows()

detection()
