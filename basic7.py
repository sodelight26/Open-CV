import cv2
import numpy as np
import pytesseract
from tracker import *

#create tracker
Tracker = EuclideanDistTracker()

#load cam
cap = cv2.VideoCapture(0)
        
def empty(a):
    pass

cv2.namedWindow("Parameter")
cv2.resizeWindow("Parameter", 640, 240)
cv2.createTrackbar("Threshold1","Parameter",150,255,empty)
cv2.createTrackbar("Threshold2","Parameter",255,255,empty)


while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert to grey scale
    gray = cv2.bilateralFilter(gray, 50, 50, 60)

    threshold1 = cv2.getTrackbarPos("Threshold1","Parameter")
    threshold2 = cv2.getTrackbarPos("Threshold2","Parameter")
    edged = cv2.Canny(gray, threshold1, threshold2) #Perform Edge detection

    imgcontour = frame.copy()
    
    contours, hierarchy  = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detection = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(imgcontour, [cnt], -1, (255, 0, 255), 3)
            
            peri = cv2.arcLength(cnt, True)
            appox = cv2.approxPolyDP(cnt, 0.02 * peri, True)                     
            if len(appox) == 4:
                screenCnt = appox
                x, y, w, h = cv2.boundingRect(appox)
                cv2.rectangle(imgcontour, (x,y), (x + w, y + h), (0, 255, 0), 5)
                detection.append([x, y, w, h])
                # Masking the part other than the number plate
                mask = np.zeros(gray.shape, np.uint8)
                cv2.drawContours(mask, [screenCnt], 0, 255, -1)
                Cropped = cv2.bitwise_and(gray, gray, mask=mask)   
                Cropped = gray[y:y+h, x:x+w]
                cv2.imshow("Cropped", Cropped)
                text = pytesseract.image_to_string(Cropped, config='--psm 7 -c tessedit_char_whitelist=0123456789', lang='tha')
    
    #line
    limit = [185, 300, 440, 300]
    
    #box tracking
    box_ids = Tracker.update(detection)

    #carcount
    totalcount = 0
    cv2.line(imgcontour, (limit[0], limit[1]), (limit[2], limit[3]), (0,0,255), 3)             
    for box_id in box_ids:
        x, y, w, h, id = box_id
        cv2.putText(imgcontour, str(id), (x,y - 15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)              
        cv2.putText(imgcontour, "Number plate is :" + str(text), (x,y - 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)

        cx,cy = x+w//2, y+h//2
        cv2.circle(imgcontour, (cx, cy), 5, (255,0,255), cv2.FILLED)

        if limit[0] < cx < limit[2] and limit[1] - 20 < cy < limit[1] + 20:
            totalcount += 1
    
    cv2.putText(imgcontour, "CarCount : "+ str(totalcount), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,255), 2)
    
    cv2.imshow("imgcontour", imgcontour)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
