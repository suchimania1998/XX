
import cv2


# define a video capture object
cap = cv2.VideoCapture(0)
start_frame_number = 50
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

while(True):
	
	# Capture the video frame
	# by frame
	ret, frame = cap.read()

	# Display the resulting frame
	cv2.imshow('frame', frame)
	
	# the 'q' button is set as the
	# quitting button you may use any
	# desired button of your choice
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# After the loop release the cap object
cap.release()
# Destroy all the windows
cv2.destroyAllWindows()
