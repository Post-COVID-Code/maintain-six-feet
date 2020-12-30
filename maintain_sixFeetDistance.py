import numpy as np
import argparse
import cv2
from math import pow
from math import sqrt


# Parsing the arguments from command line
arg = argparse.ArgumentParser()

arg.add_argument('--video', type = str, default = 'sample_video.mp4', help = 'Video file path')

arg.add_argument('--model', required = True, help = "Path to the pretrained model")

arg.add_argument('--prototxt', required = True, help = 'Prototext of the model.')

arg.add_argument('--labels', required = True, help = 'Labels of the dataset')

arg.add_argument('--confidence', type = float, default = 0.2, help='Sets confidence/probability for detecting objects')

# Converting the arguments into a dictionary for easy accessibility
args = vars(arg.parse_args())


labels = [line.strip() for line in open(args['labels'])]

# Load model
network = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

print("\nVideo Streaming \n")

# Capture video from file or through device
if args['video']:
    cap = cv2.VideoCapture(args['video'])
else:
    cap = cv2.VideoCapture(0)

while True:

    # Read the current frame into an object
    ret, frame = cap.read()

    # Storing height and width
    (h, w) = frame.shape[:2]

    # Resize the frame to suit the model requirements and store it in a blob. The frame is resized to 300X300 pixels
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),0.007843, (300, 300), 127.5)

    # Passing blob to the network (forward propagation) and store the result in detections
    network.setInput(blob)
    detections = network.forward()

    # x,y,z in cm
    pos_dict = dict()
    # x,y offset coordinates 
    coordinates = dict()

    # Average Focal length
    F = 615
    
    # Looping over the detections
    for i in range(detections.shape[2]):

        # Probability/Confidence associated with the prediction
        confidence = detections[0, 0, i, 2]

        # if confidence > 0.2 then, allow
        if confidence > args["confidence"]:

            class_id = int(detections[0, 0, i, 1])

            # Compute the (x, y) coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            # Filtering only persons detected in the frame. Class Id of 'person' is 15
            if class_id == 15:

                # Draw general box for the object
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)

                label = "{}: {:.2f}%".format(labels[class_id], confidence * 100)
                print(f"{label}")

                coordinates[i] = (startX, startY, endX, endY)

                # Mid point of bounding box
                x_mid = round((startX+endX)/2,4)
                y_mid = round((startY+endY)/2,4)

                height = round(endY-startY,4)

                # Distance from camera on the basis of triangle similarity
                distance = (165 * F)/height
                print(f"Distance(cm):{distance}\n")
             
                # Mid-point of bounding boxes (in cm) based on triangle similarity technique
                x_mid_cm = (x_mid * distance) / F
                y_mid_cm = (y_mid * distance) / F
                pos_dict[i] = (x_mid_cm,y_mid_cm,distance)

    # Distance between every object detected in a frame
    near_objects = set()
    for i in pos_dict.keys():
        for j in pos_dict.keys():
            if i < j:
                dist = sqrt(pow(pos_dict[i][0]-pos_dict[j][0],2) + pow(pos_dict[i][1]-pos_dict[j][1],2) + pow(pos_dict[i][2]-pos_dict[j][2],2))

                # Check if distance less than 2 metres or 200 centimetres
                if dist < 200:
                    near_objects.add(i)
                    near_objects.add(j)

    for i in pos_dict.keys():
        (startX, startY, endX, endY) = coordinates[i]

        if i in near_objects:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            # Convert cms to feet
            cv2.putText(frame, 'Depth: {i} ft'.format(i=round(pos_dict[i][2]/30.48,4)), (startX, y),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,255,0), 2)
        else:
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,0,255), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            # Convert cms to feet
            cv2.putText(frame, 'Depth: {i} ft'.format(i=round(pos_dict[i][2]/30.48,4)), (startX, y),cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0,0,255), 2)
        
    cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)

    # Show frame
    cv2.imshow('Frame', frame)
    cv2.resizeWindow('Frame',800,600)

    key = cv2.waitKey(1) & 0xFF

    # Press q to exit
    if key == ord("q"):
        break

# Clear all the windows
cap.release()
cv2.destroyAllWindows()
