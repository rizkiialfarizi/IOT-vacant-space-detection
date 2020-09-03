from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import imutils
import time
import cv2
from tkinter import *
from PIL import Image , ImageTk
from tkinter import Tk , BOTH
from tkinter . ttk import Frame , Button
import tkinter . messagebox as mb

class SplashScreen:
    def __init__(self, parent):
        self.parent = parent
 
        self.aturSplash()
        self.aturWindow()
 
    def aturSplash(self):
        # import image menggunakan Pillow
        self.gambar = Image.open('carpark.png')
        self.imgSplash = ImageTk.PhotoImage(self.gambar)
 
    def aturWindow(self):
        # ambil ukuran dari file image
        lebar, tinggi = self.gambar.size
 
        setengahLebar = (self.parent.winfo_screenwidth()-lebar)//2
        setengahTinggi = (self.parent.winfo_screenheight()-tinggi)//2
 
        # atur posisi window di tengah-tengah layar
        self.parent.geometry("%ix%i+%i+%i" %(lebar, tinggi,
                                             setengahLebar,setengahTinggi))
 
        # atur Image via Komponen Label
        Label(self.parent, image=self.imgSplash).pack()
         
if __name__ == '__main__':
    root = Tk()
 
    # menghilangkan judul dan batas frame Window
    root.overrideredirect(True)
 
    app = SplashScreen(root)
 
    # menutup window setelah 3 detik
    root.after(3000, root.destroy)
     
    root.mainloop()
    
def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)



# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (320, 320)
(rW, rH) = (None, None)

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
#print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

# car detection cascade
#print("[INFO] loading car cascade detector...")
cascade = cv2.CascadeClassifier('cars.xml')


#print("[INFO] starting video stream...")
vs = VideoStream("media/test.mp4").start()


# start the FPS throughput estimator
fps = FPS().start()


# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	#frame = frame[1] if args.get("video", False) else frame

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame, maintaining the aspect ratio
	frame = imutils.resize(frame, width=1000)
	orig = frame.copy()

	# car detection
	gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
	# haar detection.
	cars = cascade.detectMultiScale(gray, 1.2, 3)

	for (a, b, c, d) in cars:
	  cv2.rectangle(orig, (a, b), (a+c, b+d), (0, 0, 255), 2)
	
	# end of car detection

	# if our frame dimensions are None, we still need to compute the
	# ratio of old frame dimensions to new frame dimensions
	if W is None or H is None:
		(H, W) = frame.shape[:2]
		rW = W / float(newW)
		rH = H / float(newH)

	# resize the frame, this time ignoring aspect ratio
	frame = cv2.resize(frame, (newW, newH))

	# construct a blob from the frame and then perform a forward pass
	# of the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(frame, 1.0, (newW, newH),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)

	# decode the predictions, then  apply non-maxima suppression to
	# suppress weak, overlapping bounding boxes
	(rects, confidences) = decode_predictions(scores, geometry)
	boxes = non_max_suppression(np.array(rects), probs=confidences)
	

	# loop over the bounding boxes
	for (startX, startY, endX, endY) in boxes:
		# scale the bounding box coordinates based on the respective
		# ratios
		startX = int(startX * rW)
		startY = int(startY * rH)
		endX = int(endX * rW)
		endY = int(endY * rH)

		# draw the bounding box on the frame
		cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
	
	# set parking data
	parkingLot = 14
	available = len(boxes)
	used = parkingLot - available
	carCount = len(cars)

	# add label for parking lot information
	font = cv2.FONT_HERSHEY_SIMPLEX
	cv2.putText(orig,"Parking Lot: " + str(parkingLot), (30,50),font, 1.1, (0,255,0),2)
	cv2.putText(orig,"Available: " + str(available), (30,100),font, 1.1, (0,255,0),2)
	cv2.putText(orig,"Used: " + str(used), (30,150),font, 1.1, (0,0,255),2)
	cv2.putText(orig,"Cars: " + str(carCount), (30,200),font, 1.1, (0,0,255),2)

	# update the FPS counter
	fps.update() 

	# show the output frame
	cv2.imshow("Parking Lot", orig)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# stop the timer and display FPS information
fps.stop()
#print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# release the pointer
vs.stop()


# close all windows
cv2.destroyAllWindows()
