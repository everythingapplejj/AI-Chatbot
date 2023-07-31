# importing the required library  
from imageai.Detection import ObjectDetection 
# this one requires high version of python (3. above...)

  
# instantiating the class  
recognizer = ObjectDetection()  
  
# defining the paths  
path_model = "./Models/yolo-tiny (1).h5"  
path_input = "./Input/images.jpg"  
path_output = "./Output/ouput.jpg"  
  
# using the setModelTypeAsTinyYOLOv3() function  
recognizer.setModelTypeAsTinyYOLOv3()  
# set the path for model 
recognizer.setModelPath(path_model)  
# load model  
recognizer.loadModel()  
# calling the detectObjectsFromImage() function
# Search it up on imageAI API....  
recognition = recognizer.detectObjectsFromImage(  
    input_image = path_input,  
    output_image_path = path_output  
    )  
  
# iterating items...  
for eachItem in recognition:  
    print(eachItem["name"] , " : ", eachItem["percentage_probability"])  