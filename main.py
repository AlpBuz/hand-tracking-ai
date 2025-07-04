from HandTrackingModule import *

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmDict = detector.findPositions(img, draw=False)

        #print(detector.findDistance((0,8), (1,8), lmDict, img))
        print(detector.detect_Gesture(lmDict))
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)






if __name__ == "__main__":
    main()