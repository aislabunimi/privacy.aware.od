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
video_path='parasite.mp4'
output_video_path = 'unet.avi'
#unet
unet = UNet(3, 3, False)
unet_optimizer = torch.optim.SGD(unet.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005, nesterov=True)
unet_scheduler = torch.optim.lr_scheduler.StepLR(unet_optimizer,
                                                   step_size=10,
                                                   gamma=0.5)
load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
unet.to(device)
unet.eval()
#norm e unnorm
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
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height)) 
while True:
	ret, frame = cap.read()
	if not ret:
		break
	#frame = imutils.resize(frame, width=400)
	orig = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = frame.transpose((2, 0, 1)) #inverto canali
	frame = torch.from_numpy(frame).float().to(device)
	frame = frame.unsqueeze(0) #aggiungo prima dim per simulare il batch
	frame = frame / 255.0 #prima normalizzo fra 0 e 1, e poi normalizzo rispetto a ImageNet
	frame = norm(frame)
	reconstructed = unet(frame)
	reconstructed = unnormalize(reconstructed)
	reconstructed = reconstructed.squeeze(0) #tolgo la batch dim che non mi serve
	numpy_array = reconstructed.cpu().detach().numpy() #ottengo la sua versione in numpy
	reconstructed = numpy_array.transpose(1, 2, 0) #inverto i canali
	reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min()) #normalizzo rispetto il min e max, serve poi per quando converto in uint8
	reconstructed = (reconstructed * 255).astype(np.uint8) #converto in uint8
	reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2BGR)	#converto in bgr per cv2
	
	cv2.imshow("Frame", reconstructed)
	out.write(reconstructed)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break
	#fps.update()
cap.release()
out.release()
cv2.destroyAllWindows()
