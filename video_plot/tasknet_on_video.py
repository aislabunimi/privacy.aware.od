from torchvision.models import detection
#from imutils.video import VideoStream  #Utili se poi voglio farne versione webcam, se no useless
#from imutils.video import FPS
import numpy as np
#import imutils
import torch
import cv2

#Mie import
from model_utils_and_functions import *
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
#Config
conf_threshold = 0.75
device = 'cuda'
tasknet_weights_load= "tasknet_weights/tasknet_10.pt"
video_path='unet.avi'
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
#Carico checkpoint, metto in eval, faccio trans di normalizzazione
load_checkpoint(tasknet, tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
tasknet.eval()
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #norm imagenet

cap = cv2.VideoCapture(video_path) #oggetto che rappresenta il video
if not cap.isOpened(): #controllo se file è aperto
    print("Error: Could not open the video file.")
    exit()
#fps = FPS().start() #utile per webcam
fourcc = cv2.VideoWriter_fourcc(*'XVID')  #codec
fps = cap.get(cv2.CAP_PROP_FPS) #fps, width e height li prendo dal video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height)) #video del file di output
while True: #in loop su tutti i frame del video
	ret, frame = cap.read() #recupero il frame
	if not ret: #video finito se no
		break
	#frame = imutils.resize(frame, width=400) #utile per webcam e in generale se voglio fare il resize
	orig = frame.copy()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converto da formato di cv2 bgr a rgb
	frame = frame.transpose((2, 0, 1)) #inverto canali
	frame = torch.from_numpy(frame).float().to(device)
	frame = frame.unsqueeze(0) #aggiungo prima dim per simulare il batch
	frame = frame / 255.0 #prima normalizzo fra 0 e 1, e poi normalizzo rispetto a ImageNet
	frame = norm(frame)
	detections = tasknet(frame)[0]	#recupero prediction e ci itero per disegnarle
	for i in range(0, len(detections["boxes"])):
		confidence = detections["scores"][i] #per ogni pred recupero la prob
		if confidence > conf_threshold: #filtro e disegno solo le pred con confidence > della thresh
			#idx = int(detections["labels"][i]) #serve per le label, a me non serve visto che ho solo persone
			box = detections["boxes"][i].detach().cpu().numpy() #le bbox
			(startX, startY, endX, endY) = box.astype("int")
			label = "person: {:.2f}%".format(confidence * 100)
			cv2.rectangle(orig, (startX, startY), (endX, endY), 0, 2) #0 è colore nero
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 2)				
	# mostro l'output della tasknet e nel frattempo salvo
	cv2.imshow("Frame", orig)
	out.write(orig)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"): #chiudi il video e salva se premi q
		break
	#fps.update()  #necessario per webcam
# Chiudo tutto
cap.release()
out.release()
cv2.destroyAllWindows()
