from torchvision.models import detection
#from imutils.video import VideoStream
#from imutils.video import FPS
import numpy as np
#import imutils
import torch
import time
import cv2

from unet_model import UNet
from model_utils_and_functions import *
from torchvision import transforms
#config
conf_threshold = 0.75
device = 'cuda'
unet_weights_load= "model_weights/model_50.pt"
video_path='parasite380x256.mp4'
output_video_path = 'unet_fw.avi'
#FOR BACKWARD
#unet_weights_load= "model_weights/model_100.pt"
#video_path='unet.avi'
#output_video_path = 'unet_bw.avi'
#unet
unet = UNet(3, False)
unet_optimizer = torch.optim.SGD(unet.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4, nesterov=True)
unet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(unet_optimizer, mode='min', factor=0.5, patience=4)
load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
unet.to(device)
unet.eval()
#norm and unnorm
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
   print("Error: Could not open the video file.")
   exit()
#fps = FPS().start()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)
#width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#On raspberry pi 3 model B (not B+)
#4:3			#removing last layer
#256x192 -> 0.20 fps 	0.26
#200x150 -> 0.33 fps	0.43	
#160x120 -> 0.50 fps	0.68
#16:9			#removing last layer
#256x144 -> 0.27 fps	0.35
#192x108 -> 0.48 fps	0.64
#160x90 -> 0.70 fps	0.95 fps
#For validation. Other resolution under
width = 380 #256
height = 256 #192
#width = 320	#width = 384	#width = 512	#width = 640
#height = 240	#height = 288	#height = 384	#height = 480

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height)) 
while True:
   ret, frame = cap.read()
   if not ret:
      break
   #frame = imutils.resize(frame, width=400)
   orig = frame.copy()
   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   frame = cv2.resize(frame, (width, height))
   frame = frame.transpose((2, 0, 1)) #inverting channels
   frame = torch.from_numpy(frame).float().to(device)
   frame = frame.unsqueeze(0) #adding first dim to simulate batch
   frame = frame / 255.0 #first normalize between 0,1; then normalize w.r.t. ImageNet
   frame = norm(frame)
   reconstructed = unet(frame)
   reconstructed = unnormalize(reconstructed)
   reconstructed = reconstructed.squeeze(0) #remove first dim batch, don't needed
   numpy_array = reconstructed.cpu().detach().numpy() #obtain numpy version for opencv
   reconstructed = numpy_array.transpose(1, 2, 0) #inverting channels again
   reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min()) #normalize min-max; needed for converting then in uint8
   reconstructed = (reconstructed * 255).astype(np.uint8) #convert in uint8
   reconstructed = cv2.resize(reconstructed, (width, height)) #guard against loss of size related to assence of skip connection
   reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR) #convert in bgr for cv2
   cv2.imshow("Frame", reconstructed)
   out.write(reconstructed)
   key = cv2.waitKey(1) & 0xFF
   if key == ord("q"):
      break
   #fps.update()
cap.release()
out.release()
cv2.destroyAllWindows()
