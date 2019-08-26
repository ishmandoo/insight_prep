import cv2
import numpy as np
import sys
import os
import collections as c

path, ext = os.path.splitext(sys.argv[1])

in_path = path + ".avi"
mid_path = path + ".mid.avi"
out_path = path + ".processed.avi"

cap = cv2.VideoCapture(in_path)
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(out_path,fourcc, 20.0, (512,512))
mid_out = cv2.VideoWriter(mid_path,fourcc, 20.0, (512,512))


r = 25

nextDotId = 0

dots = []

imgQueue = c.deque(maxlen=10)
class Dot:
    def __init__(self, startPos):
        self.id = len(dots)
        self.posHistory = [startPos]
        self.brightnessHistory = [0]
        self.active = True
    def findInROI(self, img):

        pos = self.posHistory[-1]

        height, width = img.shape[:2]

        x_min = max(0, pos[0]-r)
        x_max = min(width, pos[0]+r)
        y_min = max(0, pos[1]-r)
        y_max = min(height, pos[1]+r)

        print(x_min, x_max, y_min, y_max)

        roi = img[y_min:y_max, x_min:x_max]
        roiHeight, roiWidth = roi.shape[:2]

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(roi)

        if (maxVal > 2 * minVal):
            maxLoc = np.array(maxLoc) + pos - np.array([roiHeight//2,roiWidth//2])
        else:
            maxLoc = pos

        if pos[0] > width or pos[0] < 0 or pos[1] > height or pos[1] < 0:
            self.active = False

        self.posHistory.append(np.array(maxLoc))
        self.brightnessHistory.append(np.sum(roi))



def clickListener(event, x, y, flags, param):
    global dots
    if event == cv2.EVENT_LBUTTONDOWN:
        pos = np.array([x, y])
        print(pos)
        activeDots = filter(lambda dot: dot.active, dots)
        for dot in activeDots:
            if  len(dot.posHistory) > 0 and np.linalg.norm(dot.posHistory[-1] - pos) < r:
                dot.active = False
                return

        newDot = Dot(np.array([x, y]))
        dots.append(newDot)

def getImg():
    # capture image
    ret, cimg = cap.read()

    # convert to grascale
    gimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2GRAY).astype(float)

    # add to the image queue
    imgQueue.append(gimg)

    # calculate the mean image in the queue
    meanImg = np.median(np.array(imgQueue),axis=0)

    # blur the image and the mean image
    blurred = cv2.GaussianBlur(gimg, (5, 5), 10)
    blurredMean = cv2.GaussianBlur(meanImg, (5, 5), 10)

    #subtract the mean
    diff = blurred-blurredMean
    
    diff = cv2.GaussianBlur(diff, (5, 5), 10)

    # scale up the brightness to span the full dynamic range and clip
    #pimg = ((255./np.max(diff)) * diff).clip(min=0, max=255).astype(np.uint8)
    diff -= np.min(diff)
    diff *= 255 / np.max(diff)
    pimg = diff.astype(np.uint8)

    return (cimg, gimg, pimg)

while len(dots) == 0:
    # show the user frames until the select a dot

    # get the color image, greyscale image, and processed image
    cimg, gimg, pimg = getImg()

    # show the user the image and wait for a click
    cv2.imshow('pimg',pimg)
    cv2.setMouseCallback("pimg", clickListener)
    key = cv2.waitKey(0)
    if key == 27:
        break

    cv2.destroyAllWindows()

    for dot in dots:
        dot.findInROI(pimg)


#while(cap.isOpened()):
for i in range(100):
    # get the color image, grayscale image, and processed image
    cimg, gimg, pimg = getImg()

    # copy the processed image for saving to file
    outImg = pimg.copy()

    # find the dots that are currently active
    activeDots = filter(lambda dot: dot.active, dots)
    for dot in activeDots:
        # find each active dot
        dot.findInROI(pimg)

        # draw annotations on the output image
        cv2.circle(outImg, tuple(dot.posHistory[-1]), 20, (255, 0, 0), 2)
        cv2.putText(outImg, str(dot.id), tuple(dot.posHistory[-1] + np.array([20,-15])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)

        cv2.circle(cimg, tuple(dot.posHistory[-1]), 20, (255, 0, 0), 2)
        cv2.putText(cimg, str(dot.id), tuple(dot.posHistory[-1] + np.array([20,-15])), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255),2)
    # write the frame to the processed video
    out.write(cimg)
    mid_out.write(cv2.cvtColor(pimg,cv2.COLOR_GRAY2BGR))

    # wait for a click for a new dot or any other key to advance
    cv2.imshow('click',outImg)
    cv2.setMouseCallback("click", clickListener)
    key = cv2.waitKey(0)
    if key == 27:
        break

for dot in dots:
    posHistory = np.array(dot.posHistory)
    brightnessHistory = np.reshape(dot.brightnessHistory,(-1,1))
    np.savetxt(f"{path}_dot_{dot.id}.txt", np.hstack((posHistory, brightnessHistory)), fmt="%d %d %d")

cap.release()
out.release()
mid_out.release()

cv2.destroyAllWindows()
