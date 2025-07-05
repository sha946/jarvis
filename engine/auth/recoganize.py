import cv2


def AuthenticateFace():
    """
    Authenticates a face using LBPH model and OpenCV's Haar cascades.
    Returns:
        int: 1 if face is recognized with accuracy > 65%, 0 otherwise.
    """
    flag = 0

    # Load trained LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('engine\\auth\\trainer\\trainer.yml')

    # Load Haar cascade for face detection
    cascadePath = "engine\\auth\\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Define known names; index should match label in training
    names = ['', 'Chaima']  # first element empty because LBPH labels start at 1

    # Open camera
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)  # width
    cam.set(4, 480)  # height

    # Minimum face size to detect
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, distance = recognizer.predict(gray[y:y + h, x:x + w])
            accuracy_percentage = 100 - distance  # higher is better

            if accuracy_percentage > 49:
                name = names[id]
                text_accuracy = f"{round(accuracy_percentage)}%"
                flag = 1
            else:
                name = "unknown"
                text_accuracy = f"{round(accuracy_percentage)}%"
                flag = 0

            cv2.putText(img, str(name), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(text_accuracy), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('Camera', img)

        k = cv2.waitKey(10) & 0xff
        if k == 27:  # ESC to exit
            break
        if flag == 1:  # recognized with enough accuracy, exit loop
            break

    # Cleanup
    cam.release()
    cv2.destroyAllWindows()
    return flag


if __name__ == '__main__':
    result = AuthenticateFace()
    if result == 1:
        print("Face recognized with sufficient accuracy. Access granted.")
    else:
        print("Face not recognized or accuracy too low. Access denied.")
