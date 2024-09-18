# Ultralytics YOLO 🚀, AGPL-3.0 license
# python yolo_region.py --source "D:\\tmp\\vdo\\VR-20240826-152905.avi" --view-img
# 862, 366  943, 326  1001, 330  972, 377
# python yolo_region.py --source "D:\\tmp\\vdo\\VR-20240826-153523.avi" --save-img
# (967, 450), (982, 397), (1153, 362), (1141, 458)
# (871, 691), (977, 483), (1153, 362), (1141, 458)
# python yolo_region.py --source "D:\\tmp\\vdo\\VR-20240827-113622.avi" --save-img --view-img
# python yolo_region.py --source "D:\\tmp\\vdo\\VR-20240827-113955.avi" --save-img
# (948, 314), (958, 237), (1115, 220), (1095, 306)
# (804, 220), (930, 174), (992, 179), (953, 237)

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.point import Point

from ultralytics import YOLO
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors

from common_functions import *

track_history = defaultdict(list)

current_region = None
r20240826 = [
    {
        "name": "Area B",
        "polygon": Polygon([(862, 366), (943, 326), (1001, 330), (972, 377)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (0, 255, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
    {
        "name": "Area A",
        "polygon": Polygon([(967, 450), (982, 397), (1153, 362), (1141, 458)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (255, 0, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]

r20240827 = [
    {
        "name": "Area B",
        "polygon": Polygon([(804, 220), (930, 174), (992, 179), (953, 237)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (0, 255, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
    {
        "name": "Area A",
        "polygon": Polygon([(948, 314), (958, 237), (1115, 220), (1095, 306)]),  # Polygon points
        "counts": 0,
        "dragging": False,
        "region_color": (255, 0, 225),  # BGR Value
        "text_color": (0, 0, 0),  # Region Text Color
    },
]
# (948, 314), (958, 237), (1115, 220), (1095, 306)
# (804, 220), (930, 174), (992, 179), (953, 237)

def run(
    weights="yolov8n.pt",
    source=None,
    device="cpu",
    view_img=False,
    save_img=False,
    exist_ok=False,
    classes=None,
    line_thickness=1,
    track_thickness=1,
    region_thickness=1,
):
    """
    Run Region counting on a video using YOLOv8 and ByteTrack.

    Supports movable region for real time counting inside specific area.
    Supports multiple regions counting.
    Regions can be Polygons or rectangle in shape

    Args:
        weights (str): Model weights path.
        source (str): Video file path.
        device (str): processing device cpu, 0, 1
        view_img (bool): Show results.
        save_img (bool): Save results.
        exist_ok (bool): Overwrite existing files.
        classes (list): classes to detect and track
        line_thickness (int): Bounding box thickness.
        track_thickness (int): Tracking line thickness
        region_thickness (int): Region thickness.
    """
    global current_region, track_history

    vid_frame_count = 0

    if '20240826' in source:
        counting_regions = r20240826
    elif '20240827' in source:
        counting_regions = r20240827

    frame_start = 0
    frame_end = 99999
    # if '113955' in source:
    #     frame_start = 800
    #     frame_end = 1000
    # elif '153523' in source:
    #     frame_start = 170

    # Check source path
    if not Path(source).exists():
        raise FileNotFoundError(f"Source path '{source}' does not exist.")

    # Setup Model
    model = YOLO(f"{weights}")
    model.to("cuda") if device == "0" else model.to("cpu")

    # Extract classes names
    names = model.model.names

    # Video setup
    videocapture = cv2.VideoCapture(source)
    frame_width, frame_height = int(videocapture.get(3)), int(videocapture.get(4))
    fps, fourcc = int(videocapture.get(5)), cv2.VideoWriter_fourcc(*"mp4v")

    # Desired width or height
    desired_width = 1280  # Set this to your desired width or None
    desired_height = None  # Set this to your desired height or None

    # Calculate new size
    new_width, new_height = calculate_new_size(width=desired_width, height=desired_height, original_width=frame_width, original_height=frame_height)

    # Output setup
    save_dir = increment_path(Path("ultralytics_rc_output") / "exp", exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    video_writer = cv2.VideoWriter(str(save_dir / f"{Path(source).stem}.mp4"), fourcc, fps, (new_width, new_height))

    frame_count = 0

    # Iterate over video frames
    while videocapture.isOpened():
        success, frame = videocapture.read()
        if not success or frame_count > frame_end:
            break
        vid_frame_count += 1

        frame_count += 1
        if frame_count < frame_start:
            continue

        print(frame.shape)
        frame = image_resize(frame, width=new_width, height=None)
        frame_process = frame.copy()
        # Get the dimensions of the image
        height, width, _ = frame_process.shape

        # Create a mask filled with zeros (black)
        # mask = np.zeros_like(frame)
        if '20240826' in source:
            polygon = [[0, 0], [width, 0], [width, (height//3)+110], [0, height//3]]
        elif '20240827' in source:
            polygon = [[0, 0], [width, 0], [width, (height//4)], [0, height//4]]


        # Define the polygon points (upper half of the image)
        pts = np.array(polygon)

        # Fill the polygon with black on the mask
        cv2.fillPoly(frame_process, [pts], color=(0, 0, 0))

        # Apply the mask to the image
        # frame = cv2.bitwise_and(frame, mask)

        # frame = fillBlack(frame, blackPolygon)
        print(frame_process.shape)

        # Extract the results
        results = model.track(frame_process, persist=True, classes=classes)

        if results[0].boxes.id is not None:
            # print(results[0])
            boxes = results[0].boxes.xyxy.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.cpu().tolist()

            annotator = Annotator(frame_process, line_width=line_thickness, example=str(names))

            for box, track_id, cls in zip(boxes, track_ids, clss):
                if not (cls in [2, 5, 6, 7]):
                    continue
                cls = 2
                annotator.box_label(box, str(names[cls]), color=colors(cls, True))
                bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # Bbox center

                track = track_history[track_id]  # Tracking Lines plot
                track.append((float(bbox_center[0]), float(bbox_center[1])))
                if len(track) > 1000:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                # Calculate the total distance
                total_distance = calculate_total_distance(points)
                print('total_distance: ', total_distance)
                
                # Define the position to print the distance
                last_point = points[-1][0]
                distance_text_position = (int(last_point[0]), int(last_point[1]) - 10)
                

                if total_distance > 100:
                    # Check if detection inside region
                    for region in counting_regions:
                        # if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                        #     region["counts"] += 1

                        # Check the last half of points in the track history
                        if len(points) >= 2:  # Ensure there are enough points
                            # Determine the range of the last half of the points
                            half_index = len(points) // 2
                            last_half_points = points[half_index:]
                            
                            # Check if any points in the last half are inside the region polygon
                            any_point_inside = any(region["polygon"].contains(Point(point[0])) for point in last_half_points)
                            
                            # If any point in the last half is inside the polygon, print "SOME_IN"
                            if any_point_inside:
                                cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=track_thickness)
                                # cv2.putText(frame, f'D: {total_distance:.2f}', distance_text_position, 
                                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors(cls, True), 2)
                            else:
                                cv2.polylines(frame, [points], isClosed=False, color=(255, 255, 0), thickness=track_thickness)

        # Plot polylines for all tracks in track_history that were not updated in the loop above
        for track_id, track in track_history.items():
            # Skip track_ids that were updated above
            if track_id in track_ids:
                continue

            # Convert track points to an array and draw the polyline
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            if len(points) > 1:
                # cls = some_default_class  # Set this to a default class/color if needed
                color = (0, 0, 0)  # RGB color
                # black_color = (0, 0, 0)

                # Calculate the total distance
                total_distance = calculate_total_distance(points)
                
                # Use the last point in points as the position for the distance text
                last_point = points[-1][0]  # Extract the last point
                distance_text_position = (int(last_point[0]), int(last_point[1]) - 10)
                
                if total_distance > 500:
                    for region in counting_regions:
                        if region["polygon"].contains(Point((bbox_center[0], bbox_center[1]))):
                            region["counts"] += 1
                        # Check the last half of points in the track history
                        if len(points) >= 2:  # Ensure there are enough points
                            # Determine the range of the last half of the points
                            half_index = len(points) // 2
                            last_half_points = points[half_index:]
                            
                            # Check if any points in the last half are inside the region polygon
                            any_point_inside = any(region["polygon"].contains(Point(point[0])) for point in last_half_points)
                            
                            # If any point in the last half is inside the polygon, print "SOME_IN"
                            if any_point_inside:
                                color = (255, 0, 0)

                    cv2.polylines(frame, [points], isClosed=False, color=color, thickness=track_thickness)
                    cv2.putText(frame, f'D: {total_distance:.2f}', distance_text_position, 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


        # Draw regions (Polygons/Rectangles)
        for region in counting_regions:
            if region["name"] != "Area B" and region["name"] != "Area A":
                region["counts"] = region["polygon"]
            region_label = str(region["counts"])
            region_color = region["region_color"]
            region_text_color = region["text_color"]

            polygon_coords = np.array(region["polygon"].exterior.coords, dtype=np.int32)
            centroid_x, centroid_y = int(region["polygon"].centroid.x), int(region["polygon"].centroid.y)

            text_size, _ = cv2.getTextSize(
                region_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.7, thickness=line_thickness
            )
            text_x = centroid_x - text_size[0] // 2
            text_y = centroid_y + text_size[1] // 2
            cv2.rectangle(
                frame,
                (text_x - 5, text_y - text_size[1] - 5),
                (text_x + text_size[0] + 5, text_y + 5),
                region_color,
                -1,
            )
            cv2.putText(
                frame, region_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, region_text_color, line_thickness
            )
            cv2.polylines(frame, [polygon_coords], isClosed=True, color=region_color, thickness=region_thickness)

        if view_img:
            if vid_frame_count == 1:
                cv2.namedWindow("Ultralytics YOLOv8 Region Counter Movable")
            cv2.imshow("Ultralytics YOLOv8 Region Counter Movable", frame)

        if save_img and frame_count > frame_start:
            video_writer.write(frame)

        for region in counting_regions:  # Reinitialize count for each region
            region["counts"] = 0

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    del vid_frame_count
    video_writer.release()
    videocapture.release()
    cv2.destroyAllWindows()


def parse_opt():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="initial weights path")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--source", type=str, required=True, help="video file path")
    parser.add_argument("--view-img", action="store_true", help="show results")
    parser.add_argument("--save-img", action="store_true", help="save results")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class: --classes 0, or --classes 0 2 3")
    parser.add_argument("--line-thickness", type=int, default=2, help="bounding box thickness")
    parser.add_argument("--track-thickness", type=int, default=2, help="Tracking line thickness")
    parser.add_argument("--region-thickness", type=int, default=4, help="Region thickness")

    return parser.parse_args()


def main(opt):
    """Main function."""
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
