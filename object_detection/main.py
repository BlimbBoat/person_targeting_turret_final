from ultralytics import YOLO
import cv2
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

model = YOLO("bestnanoreal.pt")

classNames = ["person"]


def calculate_centre_point(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    return ((x2 - x1) // 2) + x1, ((y2 - y1) // 2) + y1


while True:
    success, img = cap.read()
    results = model(img, stream=True)
    list_of_centre_points = []
    # coordinates
    for r in results:
        boxes = r.boxes

        for index, box in enumerate(boxes):
            # confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100
            print("Confidence --->", confidence)

            if confidence >= 0.5:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # calculate centre points and put them into a list
                list_of_centre_points.append(calculate_centre_point(x1, y1, x2, y2))

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # draw centre point
                cv2.circle(img, list_of_centre_points[index], 1, (255, 0, 0), 3)
                print("X1Y1X2Y2 --->", x1, y1, x2, y2)
                print("Centre point --->", list_of_centre_points[index])


    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
