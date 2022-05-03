import cv2

cap = cv2.VideoCapture(0) # ID of my webcam
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
    ret, frame = cap.read()
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue
    
    faces = face_cascade.detectMultiScale(gray_frame,1.2, 3) # returns coordinates of detected faces

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow("Video Frame", frame)
    # cv2.imshow("Gray Frame" ,gray_frame)

    # Wait for user input - q, then this will break the loop

    keyPressed = cv2.waitKey(1) & 0xFF # Masking cv2.waitkey(1), which is 32 bit to  bit number

    if keyPressed == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()