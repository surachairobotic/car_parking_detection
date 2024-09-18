import cv2
from ultralytics import YOLO
from common_functions import *

urls = ['C:\\Users\\admin\\Downloads\\VR-20240826-152905.avi',
        #'C:\\Users\\admin\\Downloads\\VR-20240827-113622.avi',
        'C:\\Users\\admin\\Downloads\\VR-20240827-113955.avi',
        'C:\\Users\\admin\\Downloads\\VR-20240826-153523.avi']

# Open the video file
video_path = urls[2]

def process(video_path):
    # Load the YOLOv8 model
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create a VideoWriter object
    output_path = video_path.split('.')[0] + "_result.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for .mp4 files
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)
            print(type(results[0]))

            # Visualize the results on the frame
            annotated_frame = results[0].plot(line_width=1)
            
            # Write the annotated frame to the output video
            out.write(annotated_frame)

            #frame = image_resize(annotated_frame, height=400)

            # Display the annotated frame
            #cv2.imshow("YOLOv8 Tracking", frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture and writer objects, and close the display window
    cap.release()
    out.release()
    cv2.destroyAllWindows()

for url in urls:
    process(url)