import os
import cv2
import sys
import time
import numpy as np
import datetime
from math import sqrt

import libardrone
from tracker import Tracker
from mvnc import mvncapi as mvnc
import random

r = lambda: random.randint(0,255)

box_color00 = (r(),r(),r())
box_color01 = (r(),r(),r())
box_color10 = (r(),r(),r())
box_color11 = (r(),r(),r())
global_box_color = (r(),r(),r())

dim=(300,300)

# LABELS = ('background', 'heli')
LABELS = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor',
           'heli')

SPLIT = False
CLUSTERING = True

width = 640
height = 320

command_time = False
min_score_percent = 8

def distance(p1, p2):
    return (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2


def cluster(centers, d):
    clusters = []
    d2 = d * d
    n = len(centers)
    used = [False] * n
    for i in range(n):
        if not used[i]:
            cluster_center_count = 1

            center = [centers[i][0], centers[i][1], centers[i][2], centers[i][3], centers[i][4]]
            used[i] = True

            for j in range(i+1, n):
                if distance(centers[i], centers[j]) < d2:
                    center[0] = max(center[0], centers[j][0])
                    center[1] += centers[j][1]
                    center[2] += centers[j][2]
                    cluster_center_count+=1

                    used[j] = True

            center[1] /= cluster_center_count
            center[2] /= cluster_center_count

            clusters.append((center[0], center[1], center[2], center[3], center[4]))

    return clusters

def run_inference(image_to_classify, ssd_mobilenet_graph, width, height, w_base, h_base, box_color=(255, 128, 0)):
    resized_image = preprocess_image(image_to_classify)

    # Send the image to the NCS
    ssd_mobilenet_graph.LoadTensor(resized_image.astype(np.float16), None)

    # Get the result from the NCS
    output, userobj = ssd_mobilenet_graph.GetResult()

    #   a.  First fp16 value holds the number of valid detections = num_valid.
    #   b.  The next 6 values are unused.
    #   c.  The next (7 * num_valid) values contain the valid detections data
    #       Each group of 7 values will describe an object/box These 7 values in order.
    #       The values are:
    #         0: image_id (always 0)
    #         1: class_id (this is an index into labels)
    #         2: score (this is the probability for the class)
    #         3: box left location within image as number between 0.0 and 1.0
    #         4: box top location within image as number between 0.0 and 1.0
    #         5: box right location within image as number between 0.0 and 1.0
    #         6: box bottom location within image as number between 0.0 and 1.0

    # number of boxes returned
    num_valid_boxes = int(output[0])

    detectedBoxes = []

    for box_index in range(num_valid_boxes):
        base_index = 7+ box_index * 7
        if (not np.isfinite(output[base_index]) or
                not np.isfinite(output[base_index + 1]) or
                not np.isfinite(output[base_index + 2]) or
                not np.isfinite(output[base_index + 3]) or
                not np.isfinite(output[base_index + 4]) or
                not np.isfinite(output[base_index + 5]) or
                not np.isfinite(output[base_index + 6])):
            # print('box at index: ' + str(box_index) + ' has nonfinite data, ignoring it')
            continue

        # We only want to use the human detection results
        if (output[base_index + 1] != 15):
            continue

        # ignore boxes less than the minimum score
        if (int(output[base_index + 2] * 100) <= min_score_percent):
            continue

        # clip the boxes to the image size incase network returns boxes outside of the image
        x1 = max(0, int(output[base_index + 3] * width)) + w_base
        y1 = max(0, int(output[base_index + 4] * height)) + h_base
        x2 = min(width, int(output[base_index + 5] * width)) + w_base
        y2 = min(height, int(output[base_index + 6] * height)) + h_base

        # ignore detections smaller than 20x20
        # if (x1 > (x2 - 20)):
        #     continue

        # if (y1 > (y2 - 20)):
        #     continue

        centerX = (x1 + x2) / 2
        centerY = (y1 + y2) / 2
        box_width = x2 - x1
        box_height = y2 - y1

        detectedBoxes.append(((output[base_index + 2]*100), centerX, centerY, box_width, box_height))

        overlay_on_image(image_to_classify, output[base_index:base_index + 7], min_score_percent, box_color)

    return detectedBoxes;


# overlays the boxes and labels onto the display image.
# display_image is the image on which to overlay the boxes/labels
# object_info is a list of 7 values as returned from the network
#     These 7 values describe the object found and they are:
#         0: image_id (always 0 for myriad)
#         1: class_id (this is an index into labels)
#         2: score (this is the probability for the class)
#         3: box left location within image as number between 0.0 and 1.0
#         4: box top location within image as number between 0.0 and 1.0
#         5: box right location within image as number between 0.0 and 1.0
#         6: box bottom location within image as number between 0.0 and 1.0
# returns None
def overlay_on_image(display_image, object_info, min_score_percent, box_color):
    source_image_width = display_image.shape[1]
    source_image_height = display_image.shape[0]

    base_index = 0
    class_id = object_info[base_index + 1]
    percentage = int(object_info[base_index + 2] * 100)
    if (percentage <= min_score_percent):
        # ignore boxes less than the minimum score
        return

    label_text = LABELS[int(class_id)] + " (" + str(percentage) + "%)"
    box_left = int(object_info[base_index + 3] * source_image_width)
    box_top = int(object_info[base_index + 4] * source_image_height)
    box_right = int(object_info[base_index + 5] * source_image_width)
    box_bottom = int(object_info[base_index + 6] * source_image_height)

    # box_color = (255, 128, 0)  # box color
    box_thickness = 2
    cv2.rectangle(display_image, (box_left, box_top), (box_right, box_bottom), box_color, box_thickness)

    # draw the classification label string just above and to the left of the rectangle
    label_background_color = (125, 175, 75)
    label_text_color = (255, 255, 255)  # white text

    label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    label_left = box_left
    label_top = box_top - label_size[1]
    if (label_top < 1):
        label_top = 1
    label_right = label_left + label_size[0]
    label_bottom = label_top + label_size[1]
    cv2.rectangle(display_image, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                  label_background_color, -1)

    cv2.putText(display_image, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)


# create a preprocessed image from the source image that complies to the
# network expectations and return it
def preprocess_image(src):
    NETWORK_WIDTH = 300
    NETWORK_HEIGHT = 300
    img = cv2.resize(src, (NETWORK_WIDTH, NETWORK_HEIGHT))
    #cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)

    # adjust values to range between -1.0 and + 1.0
    img = img - 127.5
    img = img * 0.007843
    return img

required_height_ratio_upper = 0.95
required_height_ratio_lower = 0.6
def updatePID(center, tracker, tracker_width, tracker_height, drone):
    x_raw_distance = tracker[0] - center[0] 
    y_raw_distance = tracker[1] - center[1] 
    
    x_required_speed = (abs(x_raw_distance) / center[0])[0]
    y_required_speed = (abs(y_raw_distance) / center[1])[0]

    if (x_raw_distance > 70):
        drone.turn_right(x_required_speed)
        command_time = time.time()
    elif (x_raw_distance < -70):
        drone.turn_left(x_required_speed)
        command_time = time.time()

    # disabled vertical centering for now
    # if (y_raw_distance > 60):
    #     drone.move_down(y_required_speed)
    #     print ("AUTO MOVE DOWN")
    # if (y_raw_distance < -60):
    #     drone.move_up(y_required_speed)
    #     print ("AUTO MOVE UP")


    # disabled forward / backward movement for now
    # if (tracker_height):
    #     height_raw_ratio = tracker_height / height

    #     if (height_raw_ratio < required_height_ratio_lower):
    #         # TODO use ratio somehow
    #         drone.move_forward()
    #         command_time = time.time()
    #         print ("AUTO move_forward")
    #     elif (height_raw_ratio > required_height_ratio_upper):
    #         # TODO use ratio somehow
    #         drone.move_backward()
    #         print ("AUTO move_backward")
    #         command_time = time.time()

def main():
    command_time = False
    # Get a list of ALL the sticks that are plugged in
    # we need at least one
    devices = mvnc.EnumerateDevices()
    if len(devices) == 0:
        print('No devices found')
        quit()

    # Pick the first stick to run the network
    device = mvnc.Device(devices[0])

    # Open the NCS
    device.OpenDevice()

    # The graph file that was created with the ncsdk compiler
    # graph_file_name = 'graph_heli'
    graph_file_name = 'graph_all'

    # read in the graph file to memory buffer
    with open(graph_file_name, mode='rb') as f:
        graph_in_memory = f.read()

    # create the NCAPI graph instance from the memory buffer containing the graph file.
    graph = device.AllocateGraph(graph_in_memory)

    drone = libardrone.ARDrone()
    # drone = False

    # cap = cv2.VideoCapture('unobstructed.m4v')
    cap = cv2.VideoCapture('tcp://192.168.1.1:5555')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    center = [width / 2, height / 2]

    print ("width " , width)
    print ("height " , height)

    fps = 0.0
    i = 0

    track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (0, 255, 255), (255, 0, 255), (255, 127, 255),
                (127, 0, 255), (127, 0, 127)]

    tracker = Tracker(dist_thresh=50, max_frames_to_skip=5, max_trace_length=10, trackIdCount=100)

    detect = True
    AUTO = False

    while cap.isOpened():
        start = time.time()

        for j in range(10):
            ret = cap.grab()

        ret, img = cap.retrieve()
        ts = datetime.datetime.utcnow().isoformat()

        if ret == True and detect:
            i += 1
            # fileName = datetime.datetime.utcnow().isoformat().replace(":","").replace("-","").replace(".","");
            # cv2.imwrite('images/' + fileName + '.jpg', img) 

            if (SPLIT):
                split_width = 340
                split_height = 300

                #we do not need to copy the image, as we can pass it by reference to the NCS
                #we should make copies if we debug / want to separately redraw detections
                img00 = img[0:split_height, 0:split_width]#.copy()
                img01 = img[0:split_height, (width-split_width):width]#.copy()
                img10 = img[(height-split_height):height, 0:split_width]#.copy()
                img11 = img[(height-split_height):height, (width-split_width):width]#.copy()

                cv2.imshow('img00', img00)
                cv2.imshow('img01', img01)
                cv2.imshow('img10', img10)
                cv2.imshow('img11', img11)

                detectedBoxes = run_inference(img00, graph, split_width, split_height, 0, 0, box_color=box_color00) + \
                    run_inference(img01, graph, split_width, split_height, width - split_width, 0, box_color=box_color01) + \
                    run_inference(img10, graph, split_width, split_height, 0, height-split_height, box_color=box_color10) + \
                    run_inference(img11, graph, split_width, split_height, width - split_width, height-split_height, box_color=box_color11)
            else:
                detectedBoxes = run_inference(img, graph, width, height, 0, 0, global_box_color)

            if (CLUSTERING):
                distance = 35
                detectedBoxes = cluster(detectedBoxes, distance)

            tracker.Update(detectedBoxes)

            for i in range(len(tracker.tracks)):
                if (len(tracker.tracks[i].trace) > 1):
                    for j in range(len(tracker.tracks[i].trace)-1):
                        # Draw trace line
                        x1 = tracker.tracks[i].trace[j][0]
                        y1 = tracker.tracks[i].trace[j][1]
                        x2 = tracker.tracks[i].trace[j+1][0]
                        y2 = tracker.tracks[i].trace[j+1][1]
                        clr = tracker.tracks[i].track_id % 9
                        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                                 track_colors[clr], 2)

            for box in detectedBoxes:
                cv2.circle(img, (int(box[1]), int(box[2])), distance, (255, 255, 255), thickness=3, lineType=8, shift=0)

            if (len(tracker.tracks) > 0):
                tracker_width = False
                tracker_height = False

                # if (tracker.tracks[0].last_detection_assigment is not None and tracker.tracks[0].skipped_frames == 0 and tracker.tracks[0].age > 1):
                if (tracker.tracks[0].last_detection_assigment is not None):
                    tracker_width = detectedBoxes[tracker.tracks[0].last_detection_assigment][3]
                    tracker_height = detectedBoxes[tracker.tracks[0].last_detection_assigment][4]

                if (AUTO):
                    updatePID(center, tracker.tracks[0].prediction, tracker_width, tracker_height, drone)
            elif (AUTO):
                drone.turn_left()

        end = time.time()
        seconds = end - start
        fps = 1 / seconds

        fpsImg = cv2.putText(img, "%.2f fps" % (fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        fpsImg = cv2.putText(img, "AUTO MODE: ", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

        auto_mode_color = (0, 255, 0)
        if (AUTO):
            auto_mode_color = (0, 0, 255)
        fpsImg = cv2.putText(img, "%s" % (str(AUTO)), (110, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, auto_mode_color, 2)

        battery_text_color = (0, 255, 0)
        battery_text_font = 0.4
        if (drone):
            if (drone.navdata[0]['battery'] < 20):
                battery_text_color = (0, 0, 255)
                battery_text_font = 0.7
            fpsImg = cv2.putText(img, "%s battery" % (str(drone.navdata[0]['battery'])), (20, 310), cv2.FONT_HERSHEY_SIMPLEX, battery_text_font, battery_text_color, 2)

        cv2.imshow("detected", fpsImg)

        key = cv2.waitKey(33)

        if key == ord('t'):
            command_time = time.time()
            drone.takeoff()
        if key == ord('l'):
            command_time = time.time()
            drone.land()
        if key == ord('h'):
            command_time = time.time()
            drone.hover()

        if key == ord('r'):
            tracker = Tracker(dist_thresh=50, max_frames_to_skip=5, max_trace_length=10, trackIdCount=100)
            print ('RESETTING TRACKER')

        # left joystick
        if key == ord('a'):
            command_time = time.time()
            drone.move_left()
        if key == ord('d'):
            command_time = time.time()
            drone.move_right()
        if key == ord('w'):
            command_time = time.time()
            drone.move_forward()
        if key == ord('s'):
            command_time = time.time()
            drone.move_backward()


        # right joystick
        if key == ord(';'):
            command_time = time.time()
            drone.turn_left()
        if key == ord('\\'):
            command_time = time.time()
            drone.turn_right()
        if key == ord('['):
            command_time = time.time()
            drone.move_up()
        if key == ord('\''):
            command_time = time.time()
            drone.move_down()
        if key == ord('z'):
            AUTO = not AUTO
            print ("AUTO MODE ", AUTO)

        if key == ord('q'):
            break

        if (command_time):
            command_age = time.time() - command_time
            print (command_age)
            if (command_age > 0.7):
                drone.hover()
                command_time = False
                print ('hovering again')
        else:
            if (drone):
                drone.hover()


    # Clean up the graph and the device
    graph.DeallocateGraph()
    device.CloseDevice()

# main entry point for program. we'll call main() to do what needs to be done.
if __name__ == "__main__":
    sys.exit(main())
