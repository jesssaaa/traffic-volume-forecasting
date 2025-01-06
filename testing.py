import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np
import pytesseract
import csv
from PIL import Image
from timeit import default_timer as timer
import threading
import datetime
import time

day_of_week = "None"
congestion_status = "Light"
density_stats = False
# Variables
area = [(475,2), (14,2), (14,312), (707,488), (974,488), (974,83)]
count = 0
frame_count = 0
total_count = 0
interval_count = 0
total_count_5_min = 0
total_counter_5_min = 0
density_count = 0
density_list = []
vehicle_counter_list = []
density_counter_list = []
vehicle_interval_list = []
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
                
def density_counter(vehicle_class_list, density_counter_list):
    if vehicle_class_list in c:
        result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if result>=0:
            if density_counter_list.count(track_id)==0:
                density_counter_list.append(track_id)

def vehicle_interval_counter(vehicle_class_list, vehicle_interval_list, total_counter_5_min):
    if vehicle_class_list in c:
        result=cv2.pointPolygonTest(np.array(area,np.int32),((cx,cy)),False)
        if result>=0:
            if vehicle_interval_list.count(track_id)==0:
                vehicle_interval_list.append(track_id)
                total_counter_5_min += len(vehicle_interval_list)
    return total_counter_5_min
# def ocr_task():
#             ocr_img = cv2.imread(cropped_img_name)
#             ocr = pytesseract.image_to_string(ocr_img)
#             return ocr

def to_day_of_week(date):
    format = '%Y-%m-%d'
    date_str = datetime.datetime.strptime(date, format)

    return date_str



# checks if values of congestion list are all the same
def congestion_checker(density_list):
    return all(i == density_list[i] for i in density_list)


def congestion_status_generator(interval_count, density_status):
    interval_count += 1
    global congestion_status
    if interval_count <= 3 and density_status == True:
        congestion_status = "Moderate"
    if interval_count > 3 and density_status == True:
        congestion_status = "Heavy"


cv2.namedWindow('Vehicle_Counter')
cv2.setMouseCallback('Vehicle_Counter', Vehicle_Counter)

# Load custom dataset class names
with open("dataset.txt", "r") as f:
    class_names = f.read().splitlines()

# Set the csv file headers
with open("results.csv", "w", newline="") as csvfile:
    headernames = ["ID", "VEHICLE", "CONFIDENCE", "DATETIME", "DENSITY"]
    writer = csv.DictWriter(csvfile, fieldnames=headernames)
    writer.writeheader()


with open("sequential_data.csv", "w", newline="") as csvfile:
    headernames3 = ["DATE", "TIME", "DAY_OF_WEEK", "BICYCLE", "BUS", "CAR", "JEEPNEY", "MOTORCYCLE", "MULTICAB", "TRICYCLE", "TRUCK", "VAN", "DENSITY", "CONGENSTION_STATUS", "INTERVAL_COUNT", "TOTAL_COUNT"]
    writer = csv.DictWriter(csvfile, fieldnames=headernames3)
    writer.writeheader()

# Load the model
model = YOLO("vehicle-counter-final.pt")

# pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Open the video file
cap = cv2.VideoCapture('D:/0106.mp4')
# cap = cv2.VideoCapture('D:/all_machine_learning_trials/cv_2_0/testing_videos/6.mp4')
# heavy traffic
# cap = cv2.VideoCapture('C:/Users/Toink/Downloads/FREE STOCK FOOTAGE - Heavy traffic.mp4')



# Get initial time for FPS calculation
start_time = timer()
index = 0
q = 0


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
    results = model.track(frame, conf = 0.90, iou = 0.5, persist = True, tracker="bytetrack.yaml")
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

                # Each Vehicle Counter
            for i in range(0, 9):
                # vehicle_counter(vehicle_class_list[i], vehicle_counter_list[i])
                vehicle_counter(vehicle_class_list[i], vehicle_counter_list[i])
                density_counter(vehicle_class_list[i], density_counter_list)
                total_counter_5_min = vehicle_interval_counter(vehicle_class_list[i], vehicle_interval_list, total_counter_5_min)

                    

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
            density_count = len(density_counter_list)
                
                        
                        

            with open("results.csv", "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headernames)
                writer.writerow({
                    'ID': f'{track_id} >> {index}',
                    'VEHICLE': c,
                    'CONFIDENCE': confidence_format(conf),
                    'DENSITY': density_count
                }),
        

        
                    
    # if int(time_min) % 15 == 0 and int(time_sec) == 0:  # 15 min interval     
    if int(time_min) % 5 == 0 and int(time_sec) == 0: # 5 min interval       
        if density_count > 3:
            density_list.append(density_count)
            density_stats = congestion_checker(density_list)       

            congestion_status_generator(interval_count, density_stats)

        total_count_5_min += total_counter_5_min

        with open("sequential_data.csv", "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headernames3)
            writer.writerow({
                'DATE': date,
                'TIME': f"{time_hour}:{time_min}:{time_sec}",
                'DAY_OF_WEEK': day_of_week,
                'BICYCLE': bicycle_counter,
                'BUS': bus_counter,
                'CAR': car_count,
                'JEEPNEY': jeepney_counter,
                'MOTORCYCLE': motorcycle_counter,
                'MULTICAB': multicab_counter,
                'TRICYCLE': tricycle_counter,
                'TRUCK': truck_counter,
                'VAN': van_counter,
                'DENSITY': density_count,
                'CONGENSTION_STATUS': congestion_status,
                'INTERVAL_COUNT': total_count,
                'TOTAL_COUNT': total_count_5_min
            }),

         
        for i in range(0, 9):
            # vehicle_counter(vehicle_class_list[i], vehicle_counter_list[i])
            vehicle_counter_list[i].clear()
            
        vehicle_interval_list.clear()
        total_count_5_min = 0  
        total_count = 0   
                

            
        
        
                
        
    cv2.polylines(frame, [np.array(area,np.int32)], True,(0, 255, 0), 2) # vehicle counter area/box
        
                        
    cvzone.putTextRect(frame,f'Bicycle: {bicycle_counter}',(845,60),1,2,colorR=(29, 66, 31))                  
    
                    
    cvzone.putTextRect(frame,f'Bus: {bus_counter}',(845,100),1,2,colorR=(29, 66, 31))    

                        
    cvzone.putTextRect(frame,f'Car: {car_counter}',(845,140),1,2,colorR=(29, 66, 31))                  

                        
    cvzone.putTextRect(frame,f'Jeepney: {jeepney_counter}',(845,180),1,2,colorR=(29, 66, 31))                  

                        
    cvzone.putTextRect(frame,f'Motorcycle: {motorcycle_counter}',(845,220),1,2,colorR=(29, 66, 31))                  

                        
    cvzone.putTextRect(frame,f'Multicab: {multicab_counter}',(845,260),1,2,colorR=(29, 66, 31))                  
        
                        
    cvzone.putTextRect(frame,f'Tricycle: {tricycle_counter}',(845,300),1,2,colorR=(29, 66, 31))                  

                        
    cvzone.putTextRect(frame,f'Truck: {truck_counter}',(845,340),1,2,colorR=(29, 66, 31))                  
                    
    cvzone.putTextRect(frame,f'Van: {van_counter}',(845,380),1,2,colorR=(29, 66, 31))                  
        
    cvzone.putTextRect(frame,f'Total Count: {total_count}',(845,420),1,2,colorR=(29, 69, 31))                  
        
    # Reset values every interval
        
    density_counter_list.clear()
    density_count = len(density_counter_list)
    index += 1

    # Calculate and display FPS
    frame_count += 1
    end_time = timer()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    time.sleep(0.014)

    # Displays vehicle counter window
    cv2.imshow("Vehicle_Counter", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



# with open("results.csv") as csvfile:
#     reader = csv.DictReader(csvfile)
#     columnNames = reader.fieldnames
#     print(columnNames)


# with open("results.csv") as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         print(row['ID']+" - "+row['VEHICLE']+" - "+row['CONFIDENCE']+" - "+row['OCR']+" - "+row['DENSITY'])


# with open("sequential_data4.csv") as csvfile:
#     reader = csv.DictReader(csvfile)
#     columnNames2 = reader.fieldnames
#     print(columnNames2)


# with open("sequential_data4.csv") as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         print(row['DATE']+" - "+row['HOUR']+" - "+row['MINUTE']+" - "+row['SECOND']+" - "+row['BICYCLE']+" - "+row['BUS']+" - "+row['CAR']+" - "+row['JEEPNEY']+" - "+row['MOTORCYCLE']+" - "+row['MULTICAB']+" - "+row['TRICYCLE']+" - "+row['TRUCK']+" - "+row['VAN']+" - "+row['TOTAL'])

