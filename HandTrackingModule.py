import cv2
import mediapipe as mp
import time
from helper import *


class landMark:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y


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
    
    def findPositions(self, img, draw=True) -> dict[int, list[landMark]]:
        lmDict = {}
        counter = 0
        if self.results.multi_hand_landmarks: 
            for handLms in self.results.multi_hand_landmarks:
                lmDict[counter] = []
                for id, lm in enumerate(handLms.landmark):
                    height, width, channels = img.shape
                    cx, cy = int(lm.x*width), int(lm.y*height)
                    newLandMark = landMark(id, cx, cy)
                    lmDict[counter].append(newLandMark)
                    if draw:
                        cv2.circle(img, (cx,cy), 10, (255, 0, 255), cv2.FILLED) 
                
                counter += 1

        return lmDict
        # Returns a dictionary with the hand number as key and a class landmark of [id, x, y] for each landmark as value
        # If no hands are detected, returns an empty dictionary {}. 
    
    def find_single_position(self, img, positionNumber: int, draw=True) -> dict[int, list[landMark]]:
        if positionNumber < 0 or positionNumber > 20:
            return {}

        lmDict = {}
        counter = 0
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                lmDict[counter] = []
                for id, lm in enumerate(handLms.landmark):
                    if id == positionNumber:
                        height, width, channels = img.shape
                        cx, cy = int(lm.x*width), int(lm.y*height)
                        newLandMark = landMark(id, cx, cy)
                        lmDict[counter].append(newLandMark)
                        if draw:
                            cv2.circle(img, (cx,cy), 10, (255,0,255), cv2.FILLED)
                
                counter += 1

        return lmDict
        #returns a dictionary with the hand number as key and a list of [id, cx, cy] for the wanted landmark
        #if no hands are detected it returns an empty dictionary

    
    def findDistance(self, p1, p2, lmDict, img=None, draw=True) -> int:
        # p1 and p2 are tuples of (hand number, landmark id)
        # Example: findDistance((0, 4), (0, 8)) for distance between thumb tip and index finger tip of the first hand
        if len(p1) != 2 or len(p2) != 2:
            return None
        
        hand1, id1 = p1
        hand2, id2 = p2

        if id1 < 0 or id1 > 20:
            return None
        
        if id2 < 0 or id2 > 20:
            return None

        if hand1 not in lmDict or hand2 not in lmDict:
            return None
        
        x1, y1 = lmDict[hand1][id1].cx, lmDict[hand1][id1].cy
        x2, y2 = lmDict[hand2][id2].cx, lmDict[hand2][id2].cy

        length = euclidean_distance(x1, y1, x2, y2)

        if draw and img is not None:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 255, 0), cv2.FILLED)

        return length
        

    def check_thumbs_up(self, handLM):
        """
        Determine if hand shows a thumbs up gesture.
        """
         
        thumbs_up = False

        # Thumb extended upward
        if handLM[4].y < handLM[3].y and handLM[3].y < handLM[2].y:
            thumbs_up = True

        # Other fingers folded (tip below PIP)
        index_folded = handLM[8].y > handLM[6].y
        middle_folded = handLM[12].y > handLM[10].y
        ring_folded = handLM[16].y > handLM[14].y
        pinky_folded = handLM[20].y > handLM[18].y

        if thumbs_up and index_folded and middle_folded and ring_folded and pinky_folded:
            return "Thumbs Up"
        
        return "Unknown"
    

    def check_thumbs_down(self, handLM):
        """
        Detects a thumbs down gesture:
        """
        thumbs_down = False

        # Thumb extended downward
        if handLM[4].y > handLM[3].y and handLM[3].y > handLM[2].y:
            thumbs_down = True

        # Other fingers folded (tip below PIP)
        index_folded = handLM[8].y > handLM[6].y
        middle_folded = handLM[12].y > handLM[10].y
        ring_folded = handLM[16].y > handLM[14].y
        pinky_folded = handLM[20].y > handLM[18].y

        if thumbs_down and index_folded and middle_folded and ring_folded and pinky_folded:
            return "Thumbs Down"

        return "Unknown"


    def detect_Gesture(self, lmDict):
        if not lmDict:
            return None

        gesture = "Unknown"
        hand = lmDict[0] #only does one hand for now
        
        #detects Thumbs Down
        gesture = self.check_thumbs_up(hand)
        if gesture != "Unknown":
            return gesture
        
        gesture = self.check_thumbs_down(hand)
        if gesture != "Unknown":
            return gesture

        return gesture
