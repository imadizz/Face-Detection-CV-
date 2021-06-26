import cv2
import numpy as np
import os
import time
import smtplib 
from email.message import EmailMessage

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
model = cv2.face_LBPHFaceRecognizer.create()
model.read("face_detector_2_faces.yml")

def send_mail(name):

    EmailAdd = "ml.prathamesh.master@gmail.com"
    Pass = "lucifer0118"

    msg = EmailMessage()
    msg['Subject'] = "Access Notification"
    msg['From'] = EmailAdd
    msg['To'] = 'mistryprathamesh@gmail.com'
    msg.set_content(f'{name} has accessed the device!')

    with smtplib.SMTP_SSL('smtp.gmail.com',465) as smtp:
        smtp.login(EmailAdd,Pass)
        smtp.send_message(msg)


def crop_image(img):
    offset = 10
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img,1.1,5)
    print(face_rects)

    if face_rects != ():
        for (x,y,w,h) in face_rects:
            cropped_image = face_img[y-offset:y+h+offset,x-offset:x+w+offset,:].copy()
            cropped_image = cv2.resize(cropped_image,(200,200))
            cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)

    else:
        cropped_image = cv2.resize(img,(200,200))
        cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)

    return cropped_image

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
counter_p = 0
counter_a = 0
while True:
    try:
        ret, image = cap.read()
        cropped_image = crop_image(image)
        label, error = model.predict(np.asarray(cropped_image))
        print(label,error)

        if label == 0 and error < 55:
            print('Inside P')
            counter_p += 1
            text = "Prathamesh Detected!"
            fcolor = (0,255,0)
            if counter_p > 10:
                os.system("chrome google.com")
                send_mail("Prathamesh Mistry")
                break

        elif label == 1 and error < 55:
            text = "Aditya Detected!"
            fcolor = (0,255,0)
            counter_a += 1
            if counter_a > 10:
                os.system("chrome facebook.com")
                send_mail("Aditya Lokhande")
                break
        else: 
            text = "Not Found!"
            fcolor = (0,0,255)

        cv2.putText(image,text,(50,100),cv2.FONT_HERSHEY_TRIPLEX,1,fcolor,1)
        if counter_a > 5:
            cv2.putText(image,"Aditya Authorized",(300,450),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        elif counter_p > 5:
            cv2.putText(image,"Prathamesh Authorized",(300,450),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
        else:
            cv2.putText(image,"Verification in Progress", (50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),1)

        cv2.imshow("window",image)

    except:
        pass

    if cv2.waitKey(25) & 0Xff == ord('q'):
        cap.release()
        break 

cv2.destroyAllWindows()


