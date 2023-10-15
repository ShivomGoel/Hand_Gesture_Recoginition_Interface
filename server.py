from flask import Flask, render_template, redirect, url_for
from flask_caching import Cache
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import math
import time
import gtts
import os
import tensorflow
from playsound import playsound
import threading
def play_sound():
  t3 = gtts.gTTS("Welcome to Hand Gesture Recognition Interface")
  t3.save("intro.mp3")
  playsound('intro.mp3')
  os.remove("intro.mp3")
# Hand detector class for Virtual Mouse --------------------------------------------------------
class HandDetector:
  def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
    self.mode = mode
    self.maxHands = maxHands
    self.detectionCon = detectionCon
    self.modelComplex = modelComplexity
    self.trackCon = trackCon
    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(
      self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon
    )
    self.mpDraw = mp.solutions.drawing_utils
    self.tipIds = [4, 8, 12, 16, 20]

  def findHands(self, img, draw=True):


    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)

    if self.results.multi_hand_landmarks:
      for handLms in self.results.multi_hand_landmarks:
        if draw:
          self.mpDraw.draw_landmarks(
            img, handLms, self.mpHands.HAND_CONNECTIONS
          )

    return img

  def findPosition(self, img, handNo=0, draw=True):
    xList = []
    yList = []
    bbox = []
    self.lmList = []
    if self.results.multi_hand_landmarks:
      myHand = self.results.multi_hand_landmarks[handNo]
      for id, lm in enumerate(myHand.landmark):
        # print(id, lm)
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        xList.append(cx)
        yList.append(cy)
        # print(id, cx, cy)
        self.lmList.append([id, cx, cy])
        if draw:
          cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

      xmin, xmax = min(xList), max(xList)
      ymin, ymax = min(yList), max(yList)
      bbox = xmin, ymin, xmax, ymax

      if draw:
        cv2.rectangle(
          img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2
        )

    return self.lmList, bbox

  def fingersUp(self):
    fingers = []

    try:
      # Thumb
      if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
        fingers.append(1)
      else:
        fingers.append(0)
      # Fingers
      for id in range(1, 5):

        if (
                self.lmList[self.tipIds[id]][2]
                < self.lmList[self.tipIds[id] - 2][2]
        ):
          fingers.append(1)
        else:
          fingers.append(0)

      # totalFingers = fingers.count(1)
    except:
      pass

    return fingers

  def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
    x1, y1 = self.lmList[p1][1:]
    x2, y2 = self.lmList[p2][1:]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    if draw:
      cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
      cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
      cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
      cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
    length = math.hypot(x2 - x1, y2 - y1)

    return length, img, [x1, y1, x2, y2, cx, cy]
####### Hand Detector class for Sign language--------------------------------
class HandDetectorSLD:

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
    self.tipIds = [4, 8, 12, 16, 20]
    self.fingers = []
    self.lmList = []

  def findHands(self, img, draw=True, flipType=True):

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)
    allHands = []
    h, w, c = img.shape
    if self.results.multi_hand_landmarks:
      for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
        myHand = {}
        mylmList = []
        xList = []
        yList = []
        for id, lm in enumerate(handLms.landmark):
          px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
          mylmList.append([px, py, pz])
          xList.append(px)
          yList.append(py)

        xmin, xmax = min(xList), max(xList)
        ymin, ymax = min(yList), max(yList)
        boxW, boxH = xmax - xmin, ymax - ymin
        bbox = xmin, ymin, boxW, boxH
        cx, cy = bbox[0] + (bbox[2] // 2), \
                 bbox[1] + (bbox[3] // 2)

        myHand["lmList"] = mylmList
        myHand["bbox"] = bbox
        myHand["center"] = (cx, cy)

        if flipType:
          if handType.classification[0].label == "Right":
            myHand["type"] = "Left"
          else:
            myHand["type"] = "Right"
        else:
          myHand["type"] = handType.classification[0].label
        allHands.append(myHand)

        if draw:
          self.mpDraw.draw_landmarks(img, handLms,
                                     self.mpHands.HAND_CONNECTIONS)
          cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                        (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                        (255, 0, 255), 2)
          cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                      2, (255, 0, 255), 2)
    if draw:
      return allHands, img
    else:
      return allHands

  def fingersUp(self, myHand):
    """
    Finds how many fingers are open and returns in a list.
    Considers left and right hands separately
    :return: List of which fingers are up
    """
    myHandType = myHand["type"]
    myLmList = myHand["lmList"]
    if self.results.multi_hand_landmarks:
      fingers = []
      if myHandType == "Right":
        if myLmList[self.tipIds[0]][0] > myLmList[self.tipIds[0] - 1][0]:
          fingers.append(1)
        else:
          fingers.append(0)
      else:
        if myLmList[self.tipIds[0]][0] < myLmList[self.tipIds[0] - 1][0]:
          fingers.append(1)
        else:
          fingers.append(0)

      for id in range(1, 5):
        if myLmList[self.tipIds[id]][1] < myLmList[self.tipIds[id] - 2][1]:
          fingers.append(1)
        else:
          fingers.append(0)
    return fingers

  def findDistance(self, p1, p2, img=None):

    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)
    info = (x1, y1, x2, y2, cx, cy)
    if img is not None:
      cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
      cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
      cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
      cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
      return length, info, img
    else:
      return length, info

class Classifier:

  def __init__(self, modelPath, labelsPath=None):
    self.model_path = modelPath

    np.set_printoptions(suppress=True)

    self.model = tensorflow.keras.models.load_model(self.model_path)

    self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    self.labels_path = labelsPath
    if self.labels_path:
      label_file = open(self.labels_path, "r")
      self.list_labels = []
      for line in label_file:
        stripped_line = line.strip()
        self.list_labels.append(stripped_line)
      label_file.close()

  # print("No Labels Found")

  def getPrediction(self, img, draw=True, pos=(50, 50), scale=2, color=(0, 255, 0)):

    imgS = cv2.resize(img, (224, 224))
    image_array = np.asarray(imgS)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    self.data[0] = normalized_image_array

    prediction = self.model.predict(self.data)
    indexVal = np.argmax(prediction)

    if draw and self.labels_path:
      cv2.putText(img, str(self.list_labels[indexVal]),
                  pos, cv2.FONT_HERSHEY_COMPLEX, scale, color, 2)

    return list(prediction[0]), indexVal
app = Flask(__name__)
# cache = Cache(app,config)

@app.route('/')
# @cache.memoize(timeout=3600)
def index():
  # t3 = gtts.gTTS("Welcome to Hand Gesture Recognition Interface")
  # t3.save("intro.mp3")
  # playsound("intro.mp3")
  # os.remove("intro.mp3")
  delay=3
  timer = threading.Timer(delay, play_sound)
  timer.start()

  return render_template('index.html')

@app.route('/Virtual_Mouse/')
# @cache.memoize(timeout=3600)
def Virtual_Mouse():

# Virtual Mouse Simulation---------------------------------------------------------------------
  psound=1
  hCam, wCam = 480, 640
  frameR = 100  # Frame Reduction
  smoothening = 7

  pTime = 0
  plocX, plocY = 0, 0
  clocX, clocY = 0, 0

  cap = cv2.VideoCapture(0)
  cap=cv2.VideoCapture(0)

  cap.set(3, wCam)
  cap.set(4, hCam)
  detector = HandDetector(maxHands=1)
  wScr, hScr = pyautogui.size()

  while True:
    # 1. Find hand Landmarks
    success, img = cap.read()
    if img is None:
      continue
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img)

    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
      x1, y1 = lmList[8][1:]
      x2, y2 = lmList[12][1:]
      # print(x1, y1, x2, y2)

    # 3. Check which fingers are up
    fingers = detector.fingersUp()

    #     print(fingers)
    cv2.rectangle(
      img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2
    )
    #     left_click=False
    #     double_click=0
    if len(fingers) > 0:

      # 4. Only Index Finger : Moving Mode
      if fingers[1] == 1 and fingers[2] == 0 and fingers[0] == 0:
        # 5. Convert Coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

        # 6. Smoothen Values
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        # 7. Move Mouse
        pyautogui.moveTo(wScr - clocX, clocY, _pause=False)

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY

      # 8. Right Click: Both Index and middle fingers are up
      if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 0:

        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(8, 12, img)
        #             print(length)

        # 10. Click mouse if distance short
        if length < 27:
          cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

          pyautogui.click(button="right")
          t1 = gtts.gTTS("Right Click")
          t1.save("right.mp3")
          playsound("right.mp3")
          os.remove("right.mp3")

      # Left Click: Thumb and Index finger
      if fingers[0] == 1 and fingers[1] == 1:

        # 9. Find distance between fingers
        length, img, lineInfo = detector.findDistance(4, 8, img)
        #             print(length)

        # 10. Click mouse if distance short
        if length < 27:
          cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)

          pyautogui.click(button='left')

          t1 = gtts.gTTS("Left Click")
          t1.save("Left.mp3")
          playsound("Left.mp3")
          os.remove("Left.mp3")

      # Double Click: Index, Middle and Ring finger

      if fingers[1] == 1 and fingers[4] == 1 and fingers[0] == 0 and fingers[2] == 0 and fingers[
        3] == 0:  # and fingers[2]==1:
        # #             length1, img, lineInfo = detector.findDistance(8, 12, img)
        # #             print(length)
        # #             length2, img, lineInfo = detector.findDistance(12, 16, img)
        # #             print(length)
        # #             if length1 < 25 and length2 < 25:
        # #                 cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
        pyautogui.click(button='left')
        pyautogui.click(button='left')
        t1 = gtts.gTTS("Double Click")
        t1.save("Double.mp3")
        playsound("Double.mp3")
        os.remove("Double.mp3")
        window_name = pyautogui.getActiveWindow().title
        if window_name=="":
          curr_wind = gtts.gTTS("Cursor on Desktop")
          curr_wind.save("Desktop.mp3".format(window_name))
          playsound("Desktop.mp3".format(window_name))
          os.remove("Desktop.mp3".format(window_name))
        else:
          curr_wind = gtts.gTTS("Opened {}".format(window_name))
          curr_wind.save("{}.mp3".format(window_name))
          playsound("{}.mp3".format(window_name))
          os.remove("{}.mp3".format(window_name))



      # if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 1:
      #   break


    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("Image", img)
    if psound==1:
      intro_vm = gtts.gTTS("Welcome to Virtual Mouse")
      intro_vm.save("vm.mp3")
      playsound("vm.mp3")
      os.remove("vm.mp3")

      psound=0

    if cv2.waitKey(1) & 0xFF == ord("q"):
      break

  cap.release()
  cv2.destroyAllWindows()
  curr_wind = gtts.gTTS("Virtual Mouse closed ")
  curr_wind.save("ss.mp3")
  playsound("ss.mp3")
  os.remove("ss.mp3")
  # t3 = gtts.gTTS("Welcome to Hand Gesture Recognition Interface")
  # t3.save("intro.mp3")
  # playsound("intro.mp3")
  # os.remove("intro.mp3")
  # return render_template('index.html')
  return redirect('/')

# Sign -language Detection -----------------------------------------------------

@app.route('/Sign_Language_Detection/')
# @cache.memoize(timeout=3600)
def Sign_Language_Detection():
  psound=1
  cap = cv2.VideoCapture(0)
  detector = HandDetectorSLD(maxHands=1)
  classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

  offset = 20
  imgSize = 300

  folder = "Data/C"
  counter = 0

  labels = ["No", "Yes", "Thanks","Hello","Sorry"]

  while True:
    index = None
    success, img = cap.read()
    if img is None:
      continue
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
      hand = hands[0]
      x, y, w, h = hand['bbox']

      imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
      imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
      imgCropShape = list(imgCrop.shape)
      if imgCropShape[0]*imgCropShape[1]!=0:

        aspectRatio = h / w
        # if h==0 or w==0:
        #   continue
        try:
          if aspectRatio > 1:
            # k = imgSize / h
            wCal = math.ceil(imgSize/aspectRatio)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            # print(prediction, index)

          else:
            # k = imgSize / w
            hCal = math.ceil(imgSize*aspectRatio)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        except:
          print(imgCrop, len(imgCrop), imgCrop.shape)

        cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                      (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
        if index is not None:
          cv2.putText(imgOutput, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

      # cv2.imshow("ImageCrop", imgCrop)
      # cv2.imshow("ImageWhite", imgWhite)


    cv2.imshow("Image", imgOutput)
    if psound==1:
      intro_sld = gtts.gTTS("Welcome to Sign Language Detection")
      intro_sld.save("SLD.mp3")
      playsound("SLD.mp3")
      os.remove("SLD.mp3")
      psound=0
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
  cap.release()
  cv2.destroyAllWindows()
  close_sld = gtts.gTTS("Sign Language Detector Closed")
  close_sld.save("ss.mp3")
  playsound("ss.mp3")
  os.remove("ss.mp3")
  # t3 = gtts.gTTS("Welcome to Hand Gesture Recognition Interface")
  # t3.save("intro.mp3")
  # playsound("intro.mp3")
  # os.remove("intro.mp3")

  # return render_template('index.html')
  return redirect(url_for('index'))

if __name__ == '__main__':
  app.run(debug=True)	