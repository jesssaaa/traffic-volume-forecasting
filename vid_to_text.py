import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np
import csv


# Variables
count = 0
area=[(762,9),(762,32),(1009,32),(1009,9)]


# Functions
def Vehicle_Counter(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

def confidence_format(a):
    rounded = round(a, 2)
    conf_format = rounded * 100
    
    return conf_format

def vehicle_counter(vehicle_class_list):
    if vehicle_class_list in c:
        ocr_result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if ocr_result>=0:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
            cvzone.putTextRect(
                frame,f'ID: {track_id} {class_names[class_id]}',
                (x1,y1),2,3,
                colorT=(255, 255, 255), colorR=(0, 0, 0)
                )
            cvzone.putTextRect(
                frame,f'{confidence_format(conf)}',
                (x1,y2),2,3,
                colorT=(255, 255, 255), colorR=(0, 0, 0)
                )
            


cv2.namedWindow('Vehicle_Counter')
cv2.setMouseCallback('Vehicle_Counter', Vehicle_Counter)

# Load custom dataset class names
with open("ocr_dataset.txt", "r") as f:
    class_names = f.read().splitlines()

# Set the csv file headers
with open("results2.csv", "w", newline="") as csvfile:
    headernames = ["ID", "DIGIT", "CONFIDENCE"]
    writer = csv.DictWriter(csvfile, fieldnames=headernames)
    writer.writeheader()

# Load the model
model = YOLO("ocr.pt")

# Open the video file
cap = cv2.VideoCapture('D:/all_machine_learning_trials/cv_2_0/testing_videos/5.mp4')





while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    
    # Run YOLO tracking on the frame, persisting tracks between frames
    ocr_results = model.track(frame, conf=.80, iou = .20)
    # ocr_results = model.track(frame, tracker="bytetrack.yaml")
   
    # Check if there are any boxes in the ocr_results
    if ocr_results[0].boxes is not None and ocr_results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = ocr_results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = ocr_results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = ocr_results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = ocr_results[0].boxes.conf.cpu().tolist()  # Confidence score
       
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            cx = int(x1+x2)//2
            cy = int(y1+y2)//2

            
            vehicle_class_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            

            # Each Vehicle Counter
            for i in range(0, 9):
                vehicle_counter(vehicle_class_list[i])
            
                     
            
            with open("ocr_results.csv", "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headernames)
                writer.writerow({
                    'ID': count,
                    'DIGIT': c,
                    'CONFIDENCE': confidence_format(conf),
                }),
                      

    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,255,255),2) # vehicle counter area/box
    


    # with open("ocr_results.csv", "w", newline="") as csvfile:
    #    headernames = ["ID", "BICYCLE", "CAR", "E-TRIKE", "JEEPNEY", "MOTORCYCLE", "MULTICAB", "TRICYCLE", "TRUCK", "VAN"]
    #    writer = csv.DictWriter(csvfile, headernames)
    #    writer.writeheader()
    #    writer.writerow({
    #       'ID': track_id,
    #       'BICYCLE': bicycle_counter,
    #       'CAR': car_counter,
    #       'E-TRIKE': e_trike_counter,
    #       'JEEPNEY': jeepney_counter,
    #       'MOTORCYCLE': motorcycle_counter,
    #       'MULTICAB': multicab_counter,
    #       'TRICYCLE': tricycle_counter,
    #       'TRUCK': truck_counter,
    #       'VAN': van_counter,
    #    }),


    cv2.imshow("Vehicle_Counter", frame)
    if cv2.waitKey(0) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



with open("results2.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    columnNames = reader.fieldnames
    print(columnNames)




with open("results2.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['ID']+" - "+row['DIGIT']+" - "+row['CONFIDENCE'])