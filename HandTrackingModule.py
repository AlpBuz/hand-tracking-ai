import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands= mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHands(self, img, draw=True):
        #converts the image into an rgb image as the hands class only uses rgb images
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB) #process the rgb image to find any hands on the image

        if self.results.multi_hand_landmarks: 
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) #this draws the 21 dots on the hands, HAND__CONNECTIONS connects the dots if wanted
        
        return img
    
    def findPosition(self, img, PositionNumber: int=0, draw=True):
        if PositionNumber < 0 or PositionNumber > 20:
            return []
        
        lmDict = {}
        counter = 0
        if self.results.multi_hand_landmarks: 
            for handLms in self.results.multi_hand_landmarks:
                lmDict[counter] = []
                for id, lm in enumerate(handLms.landmark):
                    height, width, channels = img.shape
                    cx, cy = int(lm.x*width), int(lm.y*height)
                    lmDict[counter].append([id, cx, cy])
                    if draw:
                        if id == PositionNumber:
                            cv2.circle(img, (cx,cy), 10, (255, 0, 255), cv2.FILLED) 
                
                counter += 1

        return lmDict
    
   



def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        results = detector.findPosition(img, 4)
        
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)






if __name__ == "__main__":
    main()