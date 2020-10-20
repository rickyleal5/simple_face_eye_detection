'''
File: face_eye_recognition.py
Description: Simple face and eye detection using Haar Cascades calssifiers with OpenCV
'''
#Libraries
import cv2 

#Main
def main():
	#Start camera
	camera = cv2.VideoCapture(0)
	
	#Cascade classifiers
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
	
	#Display message
	print('Press ESC key to stop the program')

	while(camera.isOpened()):
		#Read a frame
		ret, frame = camera.read()
		
		#Convert frame to gray color
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		#Detect face
		faces = face_cascade.detectMultiScale(gray_frame, 1.1, 3)
		
		#Create a rectangle for each face detected
		for (x, y, w, h) in faces:
			#Draw rectangles around  faces
			cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 3)
			
			#Create region of interest around the detected face in color and gray scale
			gray_face_roi = gray_frame[y:y+h, x:x+w]
			color_face_roi = frame[y:y+h, x:x+w]
			
			#Detect eyes
			eyes = eyes_cascade.detectMultiScale(gray_face_roi)
			#Create a rectangle for each eye detected
			for (x, y, w, h) in eyes:
				#Draw rectangles around eyes
				cv2.rectangle(color_face_roi, (x,y), (x+w, y+h), (255,0, 0), 3)
			
				
		#Display frame
		cv2.imshow('camera', frame)
		
		#If ESC key is pressed, stop using the camera
		if cv2.waitKey(1) & 0xFF == 27:
			camera.release()
	
	#Close windows
	cv2.destroyAllWindows()
	
main()
