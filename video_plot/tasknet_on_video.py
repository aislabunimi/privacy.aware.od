from torchvision.models import detection
#from imutils.video import VideoStream  #for webcam version
#from imutils.video import FPS
import numpy as np
#import imutils
import torch
import cv2

#My import
from model_utils_and_functions import *
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
#Config
conf_threshold = 0.75
device = 'cuda'
tasknet_weights_load= "tasknet_weights/tasknet_10.pt"
video_path='parasite380x256.mp4'
output_video_path = 'tasknet.avi'
#Importing tasknet
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
tasknet = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
num_classes = 2  # 1 class (person) + background
in_features = tasknet.roi_heads.box_predictor.cls_score.in_features
tasknet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
tasknet.to(device)
tasknet_optimizer = torch.optim.SGD(tasknet.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)
tasknet_scheduler = torch.optim.lr_scheduler.StepLR(tasknet_optimizer, step_size=3, gamma=0.1)
#loading checkpoint, putting in eval mode, normalization
load_checkpoint(tasknet, tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
tasknet.eval()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #norm imagenet

cap = cv2.VideoCapture(video_path) #represents video
if not cap.isOpened():
   print("Error: Could not open the video file.")
   exit()
#fps = FPS().start() # for webcam
fourcc = cv2.VideoWriter_fourcc(*'XVID')  #codec
fps = cap.get(cv2.CAP_PROP_FPS) #fps, width and from video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

color = (255, 0, 0) 
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height)) #file output
while True:
   ret, frame = cap.read() #grab frame
   if not ret:
      break
   #frame = imutils.resize(frame, width=400) #useful for resizing webcame and for resize
   orig = frame.copy()
   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #from cv2 bgr format to rgb
   frame = frame.transpose((2, 0, 1)) #inverting channels
   frame = torch.from_numpy(frame).float().to(device)
   frame = frame.unsqueeze(0) #added first dimension to simulate batch
   frame = frame / 255.0 #first normalize 0,1 then with imagenet
   frame = norm(frame)
   detections = tasknet(frame)[0]	#grab every prediction and plot it
   for i in range(0, len(detections["boxes"])):
      confidence = detections["scores"][i] #for every pred grab confidence
      if confidence > conf_threshold: #if confidence > thresh, plot pred
         #idx = int(detections["labels"][i]) #needed for labels; don't need it as i have only person
         box = detections["boxes"][i].detach().cpu().numpy() # bbox
         (startX, startY, endX, endY) = box.astype("int")
         label = "person: {:.2f}%".format(confidence * 100)
         cv2.rectangle(orig, (startX, startY), (endX, endY), color, 2) #0 black color
         y = startY - 15 if startY - 15 > 15 else startY + 15
         cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)				
   # show tasknet output
   cv2.imshow("Frame", orig)
   out.write(orig)
   key = cv2.waitKey(1) & 0xFF
   if key == ord("q"): #close video if q is pressed
      break
   #fps.update()  #for webcam
cap.release()
out.release()
cv2.destroyAllWindows()
