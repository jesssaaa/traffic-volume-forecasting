import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np
import pytesseract
import csv
from PIL import Image
import datetime
import time

# Variables
area = [(475,2), (14,2), (14,312), (707,488), (974,488), (974,83)]
count = 0
frame_count = 0
total_count = 0
bicycle_count = []
bus_count = []
car_count = []
jeepney_count = []
motorcycle_count = []
multicab_count = []
tricycle_count = []
truck_count = []
van_count = []

bicycle_counter=0
bus_counter=0
car_counter=0
jeepney_counter=0
motorcycle_counter=0
multicab_counter=0
tricycle_counter=0
truck_counter=0
van_counter=0



# Functions
def Vehicle_Counter(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

def confidence_format(a):
    rounded = round(a, 2)
    conf_format = int(rounded * 100)
    
    return conf_format

def vehicle_counter(vehicle_class_list, vehicle_counter_list):
    if vehicle_class_list in c:
        result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if result>=0:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
            cvzone.putTextRect(
                frame, f'ID: {track_id} {class_names[class_id]}',
                (x1,y1), 2, 3,
                colorT = (255, 255, 255), colorR = (0, 0, 0)
                )
            cvzone.putTextRect(
                frame, f'{confidence_format(conf)}%',
                (x1, y2), 2 ,3,
                colorT = (255, 255, 255), colorR = (0, 0, 0)
                )
            if vehicle_counter_list.count(track_id)==0:
                vehicle_counter_list.append(track_id)


cv2.namedWindow('Vehicle_Counter')
cv2.setMouseCallback('Vehicle_Counter', Vehicle_Counter)

# Load custom dataset class names
with open("dataset.txt", "r") as f:
    class_names = f.read().splitlines()

# Set the csv file headers
with open("results.csv", "w", newline="") as csvfile:
    headernames = ["ID", "VEHICLE", "CONFIDENCE"]
    writer = csv.DictWriter(csvfile, fieldnames=headernames)
    writer.writeheader()

with open("sequential_data4.csv", "w", newline="") as csvfile:
    headernames2 = ["DATE", "TIME", "BICYCLE", "BUS", "CAR", "JEEPNEY", "MOTORCYCLE", "MULTICAB", "TRICYCLE", "TRUCK", "VAN", "TOTAL"]
    writer = csv.DictWriter(csvfile, fieldnames=headernames2)
    writer.writeheader()

# Load the model
model = YOLO("vehicle-detection-final.pt")


# Open the video file
cap = cv2.VideoCapture('D:/0106.mp4')




index = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:
        continue
    
     # DATE AND TIME STAMP
    get_datetime = datetime.datetime.now()
    date = get_datetime.strftime("%Y-%m-%d")
    day_of_week = get_datetime.strftime("%A")
    time_hour = get_datetime.strftime("%H")
    time_min = get_datetime.strftime("%M")
    time_sec = get_datetime.strftime("%S")

    frame = cv2.resize(frame, (1020, 500))

    
    # Run YOLO tracking on the frame, persisting tracks between frames
    results = model.track(frame, conf=.85, iou = .20, tracker="bytetrack.yaml", persist = True)
    # results = model.track(frame, tracker="bytetrack.yaml")



    

   
    # Check if there are any boxes in the results
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
        boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
        class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
        track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
        confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score
       
        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            c = class_names[class_id]
            x1, y1, x2, y2 = box
            cx = int(x1+x2)//2
            cy = int(y1+y2)//2

            
            vehicle_class_list = ['BICYCLE', 'BUS', 'CAR', 'JEEP', 'MOTORCYCLE', 'MULTICAB', 'TRICYCLE', 'TRUCK', 'VAN']
            vehicle_counter_list = [bicycle_count, bus_count, car_count, jeepney_count, motorcycle_count, multicab_count, tricycle_count, truck_count, van_count]

            bicycle_counter=len(bicycle_count)
            bus_counter=len(bus_count)   
            car_counter=len(car_count)  
            jeepney_counter=len(jeepney_count)
            motorcycle_counter=len(motorcycle_count) 
            multicab_counter=len(multicab_count) 
            tricycle_counter=len(tricycle_count)
            truck_counter=len(truck_count)  
            van_counter=len(van_count)   

            total_count = bicycle_counter + bus_counter + car_counter + jeepney_counter + motorcycle_counter + multicab_counter + tricycle_counter + truck_counter + van_counter


            # Each Vehicle Counter
            for i in range(0, 9):
                vehicle_counter(vehicle_class_list[i], vehicle_counter_list[i])
                
           

            with open("results.csv", "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headernames)
                writer.writerow({
                    'ID': track_id,
                    'VEHICLE': c,
                    'CONFIDENCE': confidence_format(conf),
                }),

    if int(time_min) % 5 == 0 and int(time_sec) == 0:        
        with open("sequential_data4.csv", "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headernames2)
            writer.writerow({
                'DATE': date,
                'TIME': f"{time_hour}:{time_min}:{time_sec}",
                'BICYCLE': bicycle_counter,
                'BUS': bus_counter,
                'CAR': car_counter,
                'JEEPNEY': jeepney_counter,
                'MOTORCYCLE': motorcycle_counter,
                'MULTICAB': multicab_counter,
                'TRICYCLE': tricycle_counter,
                'TRUCK': truck_counter,
                'VAN': van_counter,
                'TOTAL': total_count
            }),

                   

    cv2.polylines(frame, [np.array(area,np.int32)], True,(0, 0, 255), 2) # vehicle counter area/box
    
    # UI Vehicle Counter                  
    cvzone.putTextRect(frame,f'Bicycle: {bicycle_counter}',(845,60),1,2,colorR=(29, 66, 31))                           
    cvzone.putTextRect(frame,f'Bus: {bus_counter}',(845,140),1,2,colorR=(29, 66, 31))               
    cvzone.putTextRect(frame,f'Car: {car_counter}',(845,100),1,2,colorR=(29, 66, 31))                               
    cvzone.putTextRect(frame,f'Jeepney: {jeepney_counter}',(845,180),1,2,colorR=(29, 66, 31))                              
    cvzone.putTextRect(frame,f'Motorcycle: {motorcycle_counter}',(845,220),1,2,colorR=(29, 66, 31))                              
    cvzone.putTextRect(frame,f'Multicab: {multicab_counter}',(845,260),1,2,colorR=(29, 66, 31))                           
    cvzone.putTextRect(frame,f'Tricycle: {tricycle_counter}',(845,300),1,2,colorR=(29, 66, 31))                            
    cvzone.putTextRect(frame,f'Truck: {truck_counter}',(845,340),1,2,colorR=(29, 66, 31))                    
    cvzone.putTextRect(frame,f'Van: {van_counter}',(845,380),1,2,colorR=(29, 66, 31))                  
    cvzone.putTextRect(frame,f'INTERVAL TOTAL: {total_count}',(845,420),1,2,colorR=(29, 66, 31))                  

    if int(time_min) % 5 == 0 and int(time_sec) == 0:
        bicycle_count.clear()
        bus_count.clear()
        car_count.clear()
        jeepney_count.clear()
        motorcycle_count.clear()
        multicab_count.clear()
        tricycle_count.clear()
        truck_count.clear()
        van_count.clear()

        total_count = 0

   

    time.sleep(0.02)
    cv2.imshow("Vehicle_Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



