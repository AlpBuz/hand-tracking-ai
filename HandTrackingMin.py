# This file is just a basic hand tracking file to help understand how the hand tracking works

import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands= mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    success, img = cap.read()

    #converts the image into an rgb image as the hands class only uses rgb images
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)#prcoess the given image and finds any hands and there points

    if results.multi_hand_landmarks: # multi_hand_landmarks tells us if any hands are in frame
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                height, width, channels = img.shape
                cx, cy = int(lm.x*width), int(lm.y*height)

                print(f"ID: {id}")
                print(f"x: {cx}, y:{cy}")
                print("------------------")
                if id == 4:
                    cv2.circle(img, (cx,cy), 15, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)