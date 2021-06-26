import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detect_face(img):
    face_img = img.copy()

    face_rects = face_cascade.detectMultiScale(face_img)

    for (x,y,w,h) in face_rects:
        cv2.rectangle(face_img, (x,y) , (x+w,y+h), (0,255,0),5)
    
    return face_img

def crop_image(img):
    offset = 10
    face_img = img.copy()
    face_rects = face_cascade.detectMultiScale(face_img,1.1,5)

    for (x,y,w,h) in face_rects:
        cropped_image = face_img[y-offset:y+h+offset,x-offset:x+w+offset,:].copy()
        cropped_image = cv2.resize(cropped_image,(200,200))
        cropped_image = cv2.cvtColor(cropped_image,cv2.COLOR_BGR2GRAY)
        # print(cropped_image.shape)

    return cropped_image

n = 0
while True:
    ret, image = cap.read()

    try:
        cv2.imshow("window",crop_image(image))
        cv2.imwrite(f"images/{n}.jpg",crop_image(image))
    except:
        pass


    if cv2.waitKey(25) & 0Xff == ord('q'):
        cap.release()
        break 
    if n > 150:
        break
    n += 1


cv2.destroyAllWindows()
