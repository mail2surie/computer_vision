import cv2

cap = cv2.VideoCapture(0)
# set id's and params 3-height, 4-width, 10-brightness
cap.set(3, 340)
cap.set(4, 480)
cap.set(10, 100)

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow('video', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()



