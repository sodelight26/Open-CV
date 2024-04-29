import cv2 # opencv
import numpy as np # การเรียกใช้ numpy เป็น np 
import pytesseract # ตัวแปลงรูปเป็น text

from connect import mydb

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from datetime import date, datetime
from tracker import * #การอิมพอตไลบรารี่เเละสคริปต์ที่ต้องใช้

        
#เรียกใช้ฟังก์ชั่น Tracker คือ โค้ดที่ใช้ในการบันทึกไอดีของสิ่งที่ตรวจจับได้ในแต่ล่ะเฟรม
Tracker = EuclideanDistTracker() 
#การกำหนดกล้องที่จะใช้งาน
cap = cv2.VideoCapture(0)
current_time = datetime.now().time()
current_time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
#การสร้างหน้าต่างสำหรับการปรับค่า Traeshold         
def empty(a):#สร้างฟังก์ชั้น ชื่อ empty 
        pass

cv2.namedWindow("Parameter") #ตั้งชื่อวินโดว์ว่า Parameter
cv2.resizeWindow("Parameter", 640, 240) #ปรับขนาดของWindowชื่อพารามิเตอร์ให้เป็น 640,240
cv2.createTrackbar("Threshold1","Parameter",150,255,empty) #สร้างแทรคบาร์ชื่อ Treshold1 บน Window Parameter มีค่า 0 - 225
cv2.createTrackbar("Threshold2","Parameter",255,255,empty) #สร้างแทรคบาร์ชื่อ Treshold2 บน Window Parameter มีค่า 0 - 225

#สร้างตัวเก็บนับจำนวน
totalcount = []

while True: #ในระหว่างที่เป็นจริง
        ret, frame = cap.read() #กำหนดให้เฟรมมีค่าเท่ากับภาพที่อ่านได้จากตัวแปร cap 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #สร้างตัวแปร gray เพื่อเก็บภาพ gray scale จากฟังก์ชั้น cvtColer
        gray = cv2.bilateralFilter(gray, 10, 10, 20) #ใส่ฟิลเตอร์เบลอลงบนภาพที่อยู่บนตัวแปร gray เพื่อลบรายละเอียดเล็กๆที่ไม่ต้องการ
        threshold1 = cv2.getTrackbarPos("Threshold1","Parameter") #สร้างตัวแปร Treshold1 เพื่อรับค่าจากแทรคบาร์ชื่อ Treshold1 บน "Parameter"
        threshold2 = cv2.getTrackbarPos("Threshold2","Parameter") #สร้างตัวแปร Treshold2 เพื่อรับค่าจากแทรคบาร์ชื่อ Treshold2 บน "Parameter"
        edged = cv2.Canny(gray, threshold1, threshold2) #สร้างตัวแปร edged เพื่แเก็บภาพ edged จากฟังก์ชั่น Canny

        imgcontour = frame.copy() #สร้างตัวแปร imgcontour เพื่อเก็บภาพ copy จาก frame  
        
        contours, hierarchy  = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #สร้างตัวแปร contours, hierarchy เพื่อเก็บค่าจากฟังก์ชั้น findContours จากตัวแปร edged
        detection = [] #สร้างตัวแปร detection
        for cnt in contours: #การเข้าลูปหาป้ายทะเบียน
                area = cv2.contourArea(cnt) #การสร้าวตัวแปร area เพื่อหาพื้อนที่ของ cnt 
                if area> 1200: #ถ้าพื้นที่ของ cnt มากกว่า 1200 พิกเซล 
                        cv2.drawContours(imgcontour, [cnt], -1,(255,0,255), 3) #วาดเส้นขอบลงบนตัวแปร  imgcontour อ้างอิงจากพิกัดของ cnt
                        
                        peri = cv2.arcLength(cnt, True) #กำหนดค่า peri เพื่อเอาไปคำนวนการหาป้ายทะเบียน
                        appox = cv2.approxPolyDP(cnt, 0.02*peri, True) #สร้างตัวแปร appox เพื่อหารูปร่างในภาพ cnt                      
                        
                        if len(appox) == 4: #ถ้า appox มีมุมเท่ากับ 4
                                screenCnt = appox #ให้ตัวแปรสกรีนมีค่าเท่ากับตัวแปร appox
                                x, y, w, h, = cv2.boundingRect(appox) #ให้ตัวเเปร x y w h มีค่าเท่ากับพิกัดของ appox 
                                cv2.rectangle(imgcontour,(x,y), (x + w, y + h), (0,255,0), 5) #วาดสี่เหลี่ยมบนภาพจากตัวแปร imgcontour
                                detection.append([x, y, w, h]) #การเก็บค่าพิกัดลงในตัวแปร detection

                                # Masking คือการ mask ส่วนที่เราต้องการ 
                                mask = np.zeros(gray.shape,np.uint8) #การปรับภาพให้ส่วนที่ต้องการเป็นสีขาวเเละที่เหลือเป็นสีดำจากภาพบนตัวแปร gray 
                                cv2.drawContours(mask,[screenCnt],0,255,-1,) #วาด Contours ลงในตัวแปร mask
                                cv2.bitwise_and(frame,frame,mask=mask) #การรวมภาพ mask กับภาพ frame เข้าด้วยกัน
                                
                                # Now crop ตัดภาพ
                                (x, y) = np.where(mask == 255)  #กำหนด x y ให้เท่ากับพิกัดของ mask
                                (topx, topy) = (np.min(x), np.min(y))  #กำหนดให้ topx topy เป็นค่าน้อยที่สุดของ x y 
                                (bottomx, bottomy) = (np.max(x), np.max(y)) #กำหนดให้ bottomx bottomy เป็นค่ามากที่สุดของ x y
                                Cropped = gray[topx:bottomx+1, topy:bottomy+1] #ทำการตัดภาพจากพิกัดที่ได้ ลงในตัวแปร Cropped
                                cv2.imshow("Crop", Cropped) #โชว์ภาพที่อยู่บนตัวแปร Cropped ออกมาบน Window ชื่อ cam1 
                                text = pytesseract.image_to_string(Cropped, config='--psm 7 -c tessedit_char_whitelist=0123456789' , lang='tha+eng') 
                                #อ่านภาพบนตัวแปร Cropped และเก็บภาพลงในตัวแปร text 
                                
        #line กำหนดพิกัดเส้นsensor
        limit = [145, 0, 145, 480]
                            
        #box tracking สร้างตัวแปร box_id ไว้เก็บไอดีของวัตถุที่ตรวจจับ
        box_ids = Tracker.update(detection)

       
        cv2.line(imgcontour, (limit[0], limit[1]), (limit[2], limit[3]), (0,0,255), 3) #สร้างเส้นไว้สำหรับเป็นsensor ในการนับรถ       
        for box_id in box_ids: #สร้างเงื่อนไขการนับรถ
                x, y, w, h, id = box_id #เก็บค่าพิกัดและไอดี
                cv2.putText(imgcontour, str(id), (x,y - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2) #แสดงไอดี             
                cv2.putText(imgcontour,"Number plate is :" + str(text),(x,y - 50),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2) #แสดงเลขป้ายทะเบียนที่อ่านได้
                print(text) #แสดงเลขป้ายทะเบียนที่อ่านได้ออกทาง terminal

                cx,cy = x+w//2, y+h//2 #กำหนดจุกึ่งกลางของป้ายทะเบียน
                cv2.circle(imgcontour,(cx, cy), 5,(255,0,255),cv2.FILLED) #สร้างจุดวงกลมตรงกลางป้ายทะเบียน

                if limit[0] -20 <cx< limit[0] +20 and limit[1] <cy< limit[3]: #ถ้าจุดวงกลมที่สร้างอยู่ระหว่างเส้น sensor 
                        if totalcount.count(id) == 0: #ถ้าไอดีที่ตรวจเจอไม่ใช่ไอดีที่มีอยู่แล้ว
                                totalcount.append(id) #เพิ่มจำนวนรถที่นับได้
                                if text: #ถ้ามีการอ่านป้ายทะเบียน
                                        mycursor = mydb.cursor() #สร้างตัวแปร mycursor 
                                        current_date = date.today() #สร้างตัวแปรเพื่อเก็บค่าวันที่ปัจจุบัน
                                        current_time = datetime.now().time() #สร้างตัวแปรเพื่อเก็บค่าเวลาปัจจุบัน
                                        sql = "INSERT INTO car_records (license_plate_image, license_plate_text, date, time, total_cars) VALUES (%s, %s, %s, %s, %s)" #คำสั่ง SQL ที่ใช้ในการเพิ่มข้อมูล
                                        # อ่านภาพป้ายทะเบียนจาก frame ตัดบางส่วนของภาพออกเพื่อเฉพาะป้ายทะเบียน และแปลงเป็น binary string ก่อนบันทึกลงในฐานข้อมูล
                                        ret, buffer = cv2.imencode('.jpg', frame[y:y+h, x:x+w]) #เข้ารหัสเฟรมภาพ  
                                        img_str = buffer.tobytes()#แปลงข้อมูลที่เข้ารหัสเป็นรูปแบบ bytearray ให้กลายเป็นสตริงของไบนารี
                                        val = (img_str, text, current_date, current_time, len(totalcount)) #สร้างตัวแปร เพื่อเก็บข้อมูลที่ต้องการจะเพิ่มลงในแต่ละคอลัมน์
                                        mycursor.execute(sql, val) #เพิ่มเข้าไปในตาราง
                                        mydb.commit() #ยืนยันการเปลี่ยนแปลง
                                        print("Record inserted successfully.")
        
        cv2.putText(imgcontour, f'CarCount :  {len(totalcount)}', (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2) #สร้างตัวอักษรแสดงจำนวนรถที่นับได้
        
        
        cv2.imshow("cam1", imgcontour) #โชว์ภาพจาก frame บน cam1
        cv2.imshow("cam2", edged) #โชว์ภาพจาก edged บน cam2
        key = cv2.waitKey(1) #กำหนดปุ่มหยุดการทำงาน
        if key == 27: #ถ้าตัวแปร key มีค่าเท่ากับปุ่ม ESC 
                break



cap.release() #คืนค่า ram 
cv2.destroyAllWindows() #ปิด Window ทุกอัน