import cv2
import numpy as np
from ultralytics import YOLO
import cvzone
import numpy as np
import pytesseract
import csv
from PIL import Image

# Variables
area=[(550,17),(11,13),(11,488),(974,488),(974,83)]
count=0
bicycle_count = []
bus_count = []
car_count = []
jeepney_count = []
motorcycle_count = []
multicab_count = []
tricycle_count = []
truck_count = []
van_count = []


# Functions
def Vehicle_Counter(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

def confidence_format(a):
    rounded = round(a, 2)
    conf_format = rounded * 100
    
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
                frame, f'{confidence_format(conf)}',
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
    headernames = ["ID", "VEHICLE", "CONFIDENCE", "OCR"]
    writer = csv.DictWriter(csvfile, fieldnames=headernames)
    writer.writeheader()

with open("sequential_data.csv", "w", newline="") as csvfile:
    headernames2 = ["FRAME", "DATE", "TIME", "BICYCLE", "BUS", "CAR", "JEEPNEY", "MOTORCYCLE", "MULTICAB", "TRICYCLE", "TRUCK", "VAN"]
    writer = csv.DictWriter(csvfile, fieldnames=headernames2)
    writer.writeheader()

# Load the model
model = YOLO("vehicle_counter-2-3.pt")

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# Open the video file
cap = cv2.VideoCapture('D:/all_machine_learning_trials/cv_2_0/testing_videos/6.mp4')




index = 0
while True:
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 2 != 0:
        continue


    frame = cv2.resize(frame, (1020, 500))

    # OCR of the date and time
   

    # save every 100th frame
    if index % 60 == 0:
        ocr_img_name = r'ocr_saved_img/frame' + str(index) + r'.png'
        cropped_img_name = r'cropped_images/frame' + str(index) + r'.png'

        print('Extracting frames...' + ocr_img_name)
        cv2.imwrite(ocr_img_name, frame)

        im = Image.open(ocr_img_name)
        # Setting the points for cropped image
        left = 750
        top = 7
        right = 1015
        bottom = 35
    
        # Cropped image of above dimension
        # (It will not change original image)
        cropped_image = im.crop((left, top, right, bottom))
        cropped_image.save(cropped_img_name)
        print('Cropping frames...' + cropped_img_name)
        # cv2.imwrite(cropped_img_name, cropped_image)
    
    

    ocr_img = cv2.imread(cropped_img_name)
    ocr = pytesseract.image_to_string(ocr_img)

    
    # Run YOLO tracking on the frame, persisting tracks between frames
    results = model.track(frame, iou = .70, tracker="bytetrack.yaml", persist = True)
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

            
            vehicle_class_list = ['Bicycle', 'Bus', 'Car', 'Jeepney', 'Motorcycle', 'Multicab', 'Tricycle', 'Truck', 'Van']
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


            # Each Vehicle Counter
            for i in range(0, 9):
                vehicle_counter(vehicle_class_list[i], vehicle_counter_list[i])
            
           

            with open("results.csv", "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headernames)
                writer.writerow({
                    'ID': track_id,
                    'VEHICLE': c,
                    'CONFIDENCE': confidence_format(conf),
                    'OCR': ocr,
                }),
            
            split_date_time = ocr.split()
            if len(split_date_time) is None:
                date = time
            elif len(split_date_time) != 2:
                date, time = 'None', 'None'
            else:
                date =  split_date_time[0]
                time =  split_date_time[1]
           
            with open("sequential_data.csv", "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=headernames2)
                writer.writerow({
                    'FRAME': index,
                    'DATE': date,
                    'TIME': time,
                    'BICYCLE': bicycle_counter,
                    'BUS': ocr,
                    'CAR': ocr,
                    'JEEPNEY': ocr,
                    'MOTORCYCLE': ocr,
                    'MULTICAB': ocr,
                    'TRICYCLE': ocr,
                    'TRUCK': ocr,
                    'VAN': ocr,
                }),

                   

    cv2.polylines(frame,[np.array(area,np.int32)],True,(255,255,255),2) # vehicle counter area/box
    
                      
    cvzone.putTextRect(frame,f'Bicycle: {bicycle_counter}',(845,60),1,2,colorR=(29, 66, 31))                  
   
                   
    cvzone.putTextRect(frame,f'E-trike: {bus_counter}',(845,140),1,2,colorR=(29, 66, 31))    

                    
    cvzone.putTextRect(frame,f'Car: {car_counter}',(845,100),1,2,colorR=(29, 66, 31))                  


                      
    cvzone.putTextRect(frame,f'Jeepney: {jeepney_counter}',(845,180),1,2,colorR=(29, 66, 31))                  

                    
    cvzone.putTextRect(frame,f'Motorcycle: {motorcycle_counter}',(845,220),1,2,colorR=(29, 66, 31))                  

                    
    cvzone.putTextRect(frame,f'Multicab: {multicab_counter}',(845,260),1,2,colorR=(29, 66, 31))                  

                      
    cvzone.putTextRect(frame,f'Tricycle: {tricycle_counter}',(845,300),1,2,colorR=(29, 66, 31))                  

                    
    cvzone.putTextRect(frame,f'Truck: {truck_counter}',(845,340),1,2,colorR=(29, 66, 31))                  

                   
    cvzone.putTextRect(frame,f'Van: {van_counter}',(845,380),1,2,colorR=(29, 66, 31))                  

    index = index + 1 

    # with open("results.csv", "w", newline="") as csvfile:
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
    if cv2.waitKey(1) & 0xFF == ord("q"):
       break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()



with open("results.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    columnNames = reader.fieldnames
    print(columnNames)


with open("results.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['ID']+" - "+row['VEHICLE']+" - "+row['CONFIDENCE']+row['OCR'])


with open("sequential_data.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    columnNames2 = reader.fieldnames
    print(columnNames2)


with open("sequential_data.csv") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        print(row['FRAME']+" - "+row['DATE']+" - "+row['TIME']+row['BICYCLE']+row['BUS']+row['CAR']+row['JEEPNEY']+row['MOTORCYCLE']+row['MULTICAB']+row['TRICYCLE']+row['TRUCK']+row['VAN'])