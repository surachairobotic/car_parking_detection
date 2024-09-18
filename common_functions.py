import cv2
from datetime import datetime
import numpy as np

def fillBlack(image, polygon):
    # Create a mask filled with zeros (black)
    mask = np.zeros_like(image)

    # Define the polygon points (upper half of the image)
    pts = np.array(polygon)

    # Fill the polygon with black on the mask
    cv2.fillPoly(mask, [pts], color=(0, 0, 0))

    # Apply the mask to the image
    image = cv2.bitwise_and(image, mask)

    return image

def calculate_total_distance(points):
    # Initialize total distance
    total_distance = 0.0
    
    min_x, max_x, min_y, max_y = find_min_max(points)
    total_distance = (max_x-min_x) + (max_y-min_y)
    
    return total_distance

def find_min_max(points):
    # Initialize min and max with the first point's x and y values
    # min_x = max_x = points[0][0]
    # min_y = max_y = points[0][1]
    min_x = min_y = 99999
    max_x = max_y = -9999

    # Iterate through all the points to find the min and max x, y values
    for point in points:
        # print(point)
        # print(type(point))
        x, y = point[0][0], point[0][1]

        # Update min and max for x and y
        if x < min_x:
            min_x = x
        if x > max_x:
            max_x = x
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    return min_x, max_x, min_y, max_y

def calculate_total_distance2(points):
    # Initialize total distance
    total_distance = 0.0
    
    # Iterate through the points and calculate the distance between consecutive points
    for i in range(1, len(points)):
        # Get the current point and the previous point
        point1 = points[i-1][0]
        point2 = points[i][0]
        
        # Calculate Euclidean distance between the points
        distance = np.linalg.norm(np.array(point2) - np.array(point1))
        
        # Add to the total distance
        total_distance += distance
    
    return total_distance

def calculate_new_size(width=None, height=None, original_width=None, original_height=None):
    if width is None and height is None:
        raise ValueError("At least one of width or height must be provided")

    # Determine new dimensions
    if width is None:
        # Calculate the new width based on height
        ratio = height / float(original_height)
        new_width = int(original_width * ratio)
        new_height = height
    elif height is None:
        # Calculate the new height based on width
        ratio = width / float(original_width)
        new_width = width
        new_height = int(original_height * ratio)
    else:
        # Calculate the ratio of width and height to maintain aspect ratio
        ratio_width = width / float(original_width)
        ratio_height = height / float(original_height)
        ratio = min(ratio_width, ratio_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
    
    return (new_width, new_height)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    dim = calculate_new_size(width=width, height=height, original_width=w, original_height=h)

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized

def get_date_time_string():
    # Get the current date and time
    now = datetime.now()
    
    # Format it as a string (e.g., "2024-08-15 13:45:30")
    formatted_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    return formatted_string

def crop_image(image, x, y, w, h, output_name, timestamp):
    # Crop the image using the provided x, y, width (w), and height (h)
    cropped_image = image[y:y+h, x:x+w]
    
    # Save the cropped image to the specified output path
    cv2.imwrite("images/"+timestamp+"_"+output_name+".png", cropped_image)
    
    return cropped_image

def format_to_three_digits(number):
    # Format the number to a string with leading zeros (up to 3 digits)
    return f"{number:03}"
