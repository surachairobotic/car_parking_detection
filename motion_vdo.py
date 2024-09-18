import cv2
import numpy as np
import urllib.request
from datetime import datetime
from carparkingdatabase import CarParkingDatabase
from common_functions import *

# Replace with your image URL
urls = ['C:\\Users\\admin\\Downloads\\VR-20240826-152905.avi',
        'C:\\Users\\admin\\Downloads\\VR-20240827-113622.avi',
        'C:\\Users\\admin\\Downloads\\VR-20240827-113955.avi',
        'C:\\Users\\admin\\Downloads\\VR-20240826-153523.avi']
url = urls[1]

contour_positions = np.array([
    [10, 447],
    [72, 302],
    [153, 211],
    [258, 141],
    [325, 108],
    [353, 101],
    [433, 85],
    [528, 68],
    [618, 52],
    [683, 41],
    [685, 0],
    [800, 0],
    [800, 447]
])

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(url)

# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video file")

# Initialize the background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=1024)
fgbg2 = cv2.createBackgroundSubtractorMOG2(varThreshold=1024*4)
fgbg3 = cv2.createBackgroundSubtractorMOG2(varThreshold=1024*8)

# Initialize a global variable to store the mouse position
mouse_position = (0, 0)

kErode3 = np.ones((3, 3), np.uint8)
kDilate3 = np.ones((3, 3), np.uint8)
kErode5 = np.ones((5, 5), np.uint8)
kDilate5 = np.ones((5, 5), np.uint8)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Variables to store the previous frame and points
prev_gray = None
prev_points = None
count=0

# Define a mouse callback function to update the mouse position
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y], flush=True)

# Create a named window and set the mouse callback function
cv2.namedWindow('Motion Tracking')
cv2.setMouseCallback('Motion Tracking', mouse_callback)

while cap.isOpened():
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            ori_frame = frame
    except Exception as e:
        print(e)

    # Check if the frame is valid
    if frame is None:
        print("Failed to retrieve image")
        break

    height, width = ori_frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = np.ones_like(frame) * 255
    cv2.fillPoly(mask, [contour_positions], (0, 0, 0))
    mask_frame = cv2.bitwise_and(frame, mask)

    # Apply the background subtractor to get the foreground mask
    fgmask = fgbg.apply(mask_frame)
    fgmask2 = fgbg2.apply(mask_frame)
    fgmask3 = fgbg3.apply(mask_frame)
    
    #fgmask = fgbg.apply(mask_frame)
    fgmask = cv2.dilate(fgmask, kDilate3, iterations=1)
    fgmask = cv2.erode(fgmask, kErode5, iterations=1)
    fgmask = cv2.dilate(fgmask, kDilate5, iterations=1)

    # Find contours in the mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_points = []
    #count=0
    timestamp = get_date_time_string()

    # Draw bounding boxes around detected motions
    for contour in contours:
        colour = (200, 0, 0)
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        threshold = 10000
        if x > width/2:
            threshold = 1000
        if area > threshold:  # Filter out small movements
            colour = (0, 255, 0)
            #crop_image(frame, x, y, w, h, format_to_three_digits(count), timestamp=timestamp)
            #count = count+1
            cv2.rectangle(frame, (x, y), (w + x, h + y), colour, 2)
            text_position = (x, y)
            cv2.putText(frame, f'{int(area)}', text_position, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

            # Track the center of the contour
            center = np.array([[x + w // 2, y + h // 2]], dtype=np.float32)
            current_points.append(center)

        #elif area > 100:  # Filter out small movements
        #    colour = (255, 255, 0)
        #cv2.drawContours(frame, [contour], -1, colour, 1)
        #cv2.rectangle(frame, (x, y), (w + x, h + y), colour, 1)

    #cv2.putText(frame, f'Position: {mouse_position}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    current_points = np.array(current_points)

    if prev_gray is not None and len(current_points) > 0 and len(prev_points) > 0:
        # Calculate optical flow to predict new positions
        new_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_points, current_points, **lk_params)

        for i, new_point in enumerate(new_points):
            if status[i]:
                predicted_center = new_point.ravel()
                cv2.circle(frame, (int(predicted_center[0]), int(predicted_center[1])), 5, (255, 0, 0), -1)
                cv2.putText(frame, 'Predicted', (int(predicted_center[0]), int(predicted_center[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Update previous frame and points
    prev_gray = gray.copy()
    prev_points = current_points.reshape(-1, 1, 2)

    fgmask = cv2.cvtColor(fgmask,cv2.COLOR_GRAY2RGB)
    fgmask2 = cv2.cvtColor(fgmask2,cv2.COLOR_GRAY2RGB)
    fgmask3 = cv2.cvtColor(fgmask3,cv2.COLOR_GRAY2RGB)
    #mask_frame = cv2.cvtColor(mask_frame,cv2.COLOR_GRAY2RGB)

    vis1 = np.concatenate((frame, fgmask), axis=1)
    vis2 = np.concatenate((fgmask2, fgmask3), axis=1)
    vis = np.concatenate((vis1, vis2), axis=0)
    frame = image_resize(vis, height=400)

    output_name = format_to_three_digits(count)
    cv2.imwrite("images/"+timestamp+"_"+output_name+".png", frame)
    count = count+1

    # Display the resulting frame
    cv2.imshow('Motion Tracking', frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit
    if key == ord('q'):
        break
    elif key == ord('s'):  # Press 's' to save the raw image
        filename = 'saved_image.png'
        cv2.imwrite(filename, ori_frame)
        print(f"Image saved as {filename}")

cap.release()

# Close the window
cv2.destroyAllWindows()
