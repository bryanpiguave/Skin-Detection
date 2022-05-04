import cv2
import numpy as np
import imutils
def extractSkin(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(img, img, mask=skinMask)
    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def get_hand_area(image,threshold):
    gray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    if len(cnt) ==0:
        return None
    cnt = cnt[0]
    hand_area=cv2.contourArea(cnt)
    if hand_area < threshold:
        return None
    return hand_area

def get_hand(image,threshold=15000):
    gray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnt = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    if len(cnt) ==0:
        return None
    cnt = cnt[0]
    hand_area=cv2.cv2.contourArea(cnt)
    if hand_area < threshold:
        return None
    print("Hand_area", hand_area)
    epsilon = 0.001 * cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    
    return cv2.drawContours(image, [approx], -1, (0, 0, 255), 2)




def extract_skin_complement(image):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Defining HSV Threadholds
    lower_threshold = np.array([0, 48, 80], dtype=np.uint8)
    upper_threshold = np.array([20, 255, 255], dtype=np.uint8)
    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    return skinMask