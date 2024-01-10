#Importo le librerie necessarie
import numpy as np  #Numpy
import torchvision #libreria principale di Pytorch per riconoscere le immagini
import torch #Pytorch
from torch import nn #Modulo prinicipale di torch che definisce la maggior parte delle cose relative alle reti, come i layer, funzioni di loss...
from torch import optim #Modulo con i vari optimizer

############ QUESTO CODICE MIO
from unet_model import UNet
from dataset import *
from model_utils_and_functions import *
from train_and_test_loops import *
from plot_utils import *
############

seed_everything(0) #per rendere deterministico l'esperimento

###### CONFIG
plt.rcParams['figure.figsize'] = 15, 10
device='cuda' #cuda -> nvidia gpu
unet_save_path = "model_weights/model"
tasknet_save_path = "tasknet_weights/tasknet"
unet_weights_load= "model_weights/model_50.pt"
unet_weights_to_compare= "model_weights/model_50.pt"
tasknet_weights_load= "tasknet_weights/tasknet_20.pt"
ap_log_path="plot/ap_log.txt"
loss_log_path='plot/loss_log.txt'
train_img_folder = '/home/alberti/coco_people_indoor/train/images'
train_ann_file = '/home/alberti/coco_people_indoor/train/train.json'
val_img_folder = '/home/alberti/coco_people_indoor/val/images'
val_ann_file = '/home/alberti/coco_people_indoor/val/val.json'
train_batch_size=4
val_batch_size=4
resume_training=False
train_only_tasknet=False
num_epochs = 50 #setto numero delle epoche

###### MODELLI
#Instanzio il modello e gli iperparametri; lo muovo poi al device
#primo parametro n canali, secondo n_classes is the number of probabilities you want to get per pixel
unet = UNet(3, 3, False)
unet.to(device) 

#from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
#from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#from faster_modificata import fasterrcnn_resnet50_fpn_modificata, FasterRCNN_ResNet50_FPN_Weights
from faster_modificata.faster_rcnn import fasterrcnn_resnet50_fpn_modificata, FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
#tasknet = fasterrcnn_resnet50_fpn_modificata(weights=weights, progress=False)

tasknet = fasterrcnn_resnet50_fpn_modificata(weights=weights, progress=False,
	rpn_pre_nms_top_n_train=3, rpn_pre_nms_top_n_test=1000,
	rpn_post_nms_top_n_train=3, rpn_post_nms_top_n_test=1000,
	rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
	rpn_score_thresh=0.0)

#setup della tasknet

num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = tasknet.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
tasknet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

tasknet.to(device)

#Ora l'optimizer
tasknet_optimizer = torch.optim.SGD(tasknet.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005, nesterov=True)
tasknet_scheduler = torch.optim.lr_scheduler.StepLR(tasknet_optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

unet_optimizer = torch.optim.SGD(unet.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005, nesterov=True)
#unet_optimizer = torch.optim.Adam(unet.parameters(), lr=0.001, weight_decay=0)
unet_scheduler = torch.optim.lr_scheduler.StepLR(unet_optimizer,
                                                   step_size=10,
                                                   gamma=0.5)

train_loss = [] # Lista che conserva la training loss. Mi serve se voglio vedere l'andamento della loss
val_loss = [] #Lista che conversa la test loss
log = {'TRAIN_LOSS': [], 'VAL_LOSS': []}
completed_epochs = 0
train_dataloader, val_dataloader = load_dataset(train_img_folder, train_ann_file, val_img_folder, val_ann_file, train_batch_size, val_batch_size)

if resume_training:
	completed_epochs = load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler) #3 arg, il modello, il save path e l'optimizer

tasknet.train()

#load_checkpoint_encoder(unet, unet_weights_load, unet_optimizer, unet_scheduler)
#freeze_encoder(unet, unet_weights_load, unet_optimizer, unet_scheduler)
#load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
#freeze_encoder(unet)
#freeze_decoder(unet)

if(train_only_tasknet):
	for epoch in range(1, num_epochs+1): #itero ora facendo un train e un test per ogni epoca
    		log['TRAIN_LOSS'].append(train_tasknet(train_dataloader, epoch, device, train_loss, tasknet_save_path, tasknet, tasknet_optimizer))
    		tasknet_scheduler.step()
    		log['VAL_LOSS'].append(val_and_ap_tasknet(val_dataloader, epoch, device, val_loss, tasknet_save_path, tasknet, tasknet_optimizer, tasknet_scheduler, ap_log_path))
    		print(f'EPOCH {epoch} SUMMARY: ' + ', '.join([f'{k}: {v[epoch-1]}' for k, v in log.items()]))
    		with open(loss_log_path, 'a') as file:
    			if completed_epochs==0:
    				loss_log_append = f"{completed_epochs+epoch} {log['TRAIN_LOSS'][epoch-1]} {log['VAL_LOSS'][epoch-1]}\n"
   		 	else:
   		 		loss_log_append = f"{completed_epochs+epoch+1} {log['TRAIN_LOSS'][epoch-1]} {log['VAL_LOSS'][epoch-1]}\n"
    			#tengo traccia delle epoche già passate scrivendo nel file in quel modo
    			file.write(loss_log_append)
else:
	load_checkpoint(tasknet, tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
	tasknet.train() #faster ci deve rimanere sempre per avere la loss, freezo poi tutto qui sotto
#qui sotto freezo tutti i layer, visto che fastrcnn non mi permette di ottenere la loss
#a meno che il modello sia in training
	for param in tasknet.parameters():
		param.requires_grad = False
	for epoch in range(1, num_epochs+1): #itero ora facendo un train e un test per ogni epoca
    		log['TRAIN_LOSS'].append(train_model(train_dataloader, epoch, device, train_loss, unet, tasknet, unet_optimizer))
    		unet_scheduler.step()
    		log['VAL_LOSS'].append(val_and_ap_model(val_dataloader, epoch, device, val_loss, unet, unet_save_path, tasknet, unet_optimizer, unet_scheduler, ap_log_path))
    		print(f'EPOCH {epoch} SUMMARY: ' + ', '.join([f'{k}: {v[epoch-1]}' for k, v in log.items()]))
    		with open(loss_log_path, 'a') as file:
    			if completed_epochs==0:
    				loss_log_append = f"{completed_epochs+epoch} {log['TRAIN_LOSS'][epoch-1]} {log['VAL_LOSS'][epoch-1]}\n"
    			else:
    				loss_log_append = f"{completed_epochs+epoch+1} {log['TRAIN_LOSS'][epoch-1]} {log['VAL_LOSS'][epoch-1]}\n"
    	#tengo traccia delle epoche già passate scrivendo nel file in quel modo
    			file.write(loss_log_append)
print("Done!")

#load_checkpoint(tasknet, tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
#load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)

#show_res_test_unet(unet, tasknet, device, 'plot/val.jpg', True, 'plot/reconstructed_person.png')
#compare_two_results_unet(unet, tasknet, device, 'plot/val.jpg', 'plot/reconstructed_person.png', unet_weights_load, unet_weights_to_compare, unet_optimizer, unet_scheduler)
