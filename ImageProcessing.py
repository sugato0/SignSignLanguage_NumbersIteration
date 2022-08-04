import cv2
import numpy as np
import mediapipe as mp
class HandDetector:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, minTrackCon=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.minTrackCon = minTrackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands,
                                        min_detection_confidence=self.detectionCon,
                                        min_tracking_confidence=self.minTrackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.fingers = []
        self.lmList = []



    def findHands(self, img, draw=True,flipType = True):


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.results = self.hands.process(imgRGB)

        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []

                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])

                myHand["lmList"] = mylmList

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Right"
                    else:
                        myHand["type"] = "Left"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)
                if draw:
                    self.mpDraw.draw_landmarks(imgRGB, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        if draw:
            return allHands, imgRGB
        else:
            return allHands
detector = HandDetector(detectionCon=0.8, maxHands=2)
def crop_res_img(image):

    coef: int
    crop_image: image
    #make square for picture

    if np.shape(image)[0] > np.shape(image)[1]:

        coef = np.shape(image)[0] - np.shape(image)[1]
        if coef % 2 != 0:
            coef -= 1
        #print(coef)
        crop_image = image[int(coef/2):int(np.shape(image)[0] - coef/2)]
    elif np.shape(image)[0] < np.shape(image)[1]:
        coef = np.shape(image)[1] - np.shape(image)[0]
        if coef % 2 != 0:
            coef -= 1
        #print(coef)
        crop_image = image[:, int(coef/2):int(np.shape(image)[1] - coef/2):]
    else:

        crop_image = image

    res_image = cv2.resize(crop_image,(100,100))
    return res_image

#with mediapipe and function crop_res_img process our photo
def GetLmListFromImg(image):
    image = cv2.imread(image)
    crop_image = crop_res_img(image)
    hands, image = detector.findHands(crop_image)
    #have we got one of hands on our frame

    if hands:
        # Hand 1
        hand1 = hands[0]


        lmList1 = hand1["lmList"]

        return np.array([lmList1])
    else:
        return None