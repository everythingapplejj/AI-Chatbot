# how to speak? Okay let me try to use gtts
#import Speech -> import speech code. 
import Speech
#object detection part 

modelpath = "./yolo/yolo.h5"
#yolo.h5 has more pre-determined models vs yolo.h3...
from imageai import Detection
yolo = Detection.ObjectDetection()
yolo.setModelTypeAsYOLOv3()
yolo.setModelPath(modelpath)
yolo.loadModel(detection_speed = "fastest and flash")

import cv2
cam = cv2.VideoCapture(0) #Laptop only has front cam so set 0. 
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1300)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)

while True:
    ## read frames
    ret, img = cam.read()
    ## predict yolo
    img, preds = yolo.detectCustomObjectsFromImage(input_image=img, 
                      custom_objects=None, input_type="array",
                      output_type="array",
                      minimum_percentage_probability=70,
                      display_percentage_probability=False,
                      display_object_name=True)
    ## display predictions
    cv2.imshow("", img)
    ## if it detects new object -> say it. 
    ## Say the object 
    ## press q or Esc to quit    
    if (cv2.waitKey(1) & 0xFF == ord("q")) or (cv2.waitKey(1)==27):
        break
## close camera
cam.release()
cv2.destroyAllWindows()
