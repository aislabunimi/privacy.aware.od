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

###### CONFIG
seed_everything(0) #per rendere deterministico l'esperimento
device='cuda'
#Config Plotting and Save files
plt.rcParams['figure.figsize'] = 15, 10
ap_log_path="plot/ap_log.txt"
loss_log_path='plot/loss_log.txt'
ap_score_threshold=0.75
my_ap_log_path="plot/my_ap_log.txt"
unet_save_path = "model_weights/model"
tasknet_save_path = "tasknet_weights/tasknet"
unet_weights_load= "model_weights/model_50.pt"
unet_weights_to_compare= "model_weights/model_50.pt"
tasknet_weights_load= "tasknet_weights/tasknet_20.pt"
#Config DATASET
train_img_folder = '/home/alberti/coco_people_indoor/train/images'
train_ann_file = '/home/alberti/coco_people_indoor/train/train.json'
val_img_folder = '/home/alberti/coco_people_indoor/val/images'
val_ann_file = '/home/alberti/coco_people_indoor/val/val.json'
disturbed_train_img_folder='disturbed_dataset/train'
disturbed_train_ann='disturbed_dataset/train.json'
disturbed_val_img_folder='disturbed_dataset/val'
disturbed_val_ann='disturbed_dataset/val.json'
train_batch_size=2
val_batch_size=2
resize_scales_transform = [200, 300, 400, 500, 600]
use_dataset_subset=0
#resize_scales_transform = [200]
#use_dataset_subset=10 #se è 0 uso tutto il dataset, se è n uso esattamente n elementi dal dataset
#Config Execution mode of the Architecture
resume_training=False
train_only_tasknet=False
save_disturbed_dataset=False
train_backward_on_disturbed_sets=False
use_custom_filter_proposals=False #se usare il mio filter proposal custom o meno nell'rpn della faster
num_epochs = 1 #setto numero delle epoche

###### MODELLI
#Instanzio il modello e gli iperparametri; lo muovo poi al device
#primo parametro n canali, secondo n_classes is the number of probabilities you want to get per pixel
unet = UNet(3, 3, False)
unet.to(device) 

if train_only_tasknet:
	from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
	from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
	weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
	tasknet = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
	num_classes = 2  # 1 class (person) + background
	# get number of input features for the classifier
	in_features = tasknet.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	tasknet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	tasknet.to(device)
	tasknet_optimizer = torch.optim.SGD(tasknet.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)
	tasknet_scheduler = torch.optim.lr_scheduler.StepLR(tasknet_optimizer, step_size=3, gamma=0.1)
elif not train_backward_on_disturbed_sets:
	from faster_modificata.faster_rcnn import fasterrcnn_resnet50_fpn_modificata, FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor
	weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
	tasknet = fasterrcnn_resnet50_fpn_modificata(weights=weights, progress=False,
		rpn_pre_nms_top_n_train=100, rpn_pre_nms_top_n_test=1000,
		rpn_post_nms_top_n_train=100, rpn_post_nms_top_n_test=1000,
		rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
		rpn_score_thresh=0.0, use_custom_filter_proposals=use_custom_filter_proposals)

	num_classes = 2  # 1 class (person) + background
	# get number of input features for the classifier
	in_features = tasknet.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	tasknet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	tasknet.to(device)
	tasknet_optimizer = torch.optim.SGD(tasknet.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)
	tasknet_scheduler = torch.optim.lr_scheduler.StepLR(tasknet_optimizer, step_size=3, gamma=0.1)

unet_optimizer = torch.optim.SGD(unet.parameters(), lr=0.005,
                                momentum=0.9, weight_decay=0.0005, nesterov=True)
#unet_optimizer = torch.optim.Adam(unet.parameters(), lr=0.001, weight_decay=0)
unet_scheduler = torch.optim.lr_scheduler.StepLR(unet_optimizer,
                                                   step_size=10,
                                                   gamma=0.5)

train_loss = [] # Lista che conserva la training loss. Mi serve se voglio vedere l'andamento della loss
val_loss = [] #Lista che conversa la test loss
log = {'TRAIN_LOSS': [], 'VAL_LOSS': []}

if save_disturbed_dataset:
	train_dataloader_gen_disturbed, val_dataloader_gen_disturbed = load_dataset_for_generating_disturbed_set(train_img_folder, train_ann_file, val_img_folder, val_ann_file, use_dataset_subset)

if train_backward_on_disturbed_sets: #carico i dataloader appositi del dataset disturbato
	disturbed_train_dataloader, disturbed_val_dataloader = load_disturbed_dataset(disturbed_train_img_folder, disturbed_train_ann, disturbed_val_img_folder, disturbed_val_ann, train_img_folder, val_img_folder, train_batch_size, val_batch_size, resize_scales_transform, use_dataset_subset)
else:
	train_dataloader, val_dataloader= load_dataset(train_img_folder, train_ann_file, val_img_folder, val_ann_file, train_batch_size, val_batch_size, save_disturbed_dataset, train_only_tasknet, resize_scales_transform, use_dataset_subset)


if resume_training:
	completed_epochs = load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler) #3 arg, il modello, il save path e l'optimizer

#load_checkpoint_encoder(unet, unet_weights_load, unet_optimizer, unet_scheduler)
#freeze_encoder(unet, unet_weights_load, unet_optimizer, unet_scheduler)
#load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
#freeze_encoder(unet)
#freeze_decoder(unet)

#BLOCCO TRAINING TASKNET
if(train_only_tasknet and not train_backward_on_disturbed_sets):
	tasknet.train()
	for epoch in range(1, num_epochs+1): #itero ora facendo un train e un test per ogni epoca
    		log['TRAIN_LOSS'].append(train_tasknet(train_dataloader, epoch, device, train_loss, tasknet_save_path, tasknet, tasknet_optimizer))
    		tasknet_scheduler.step()
    		log['VAL_LOSS'].append(val_tasknet(val_dataloader, epoch, device, val_loss, tasknet_save_path, tasknet, tasknet_optimizer, tasknet_scheduler, ap_log_path, ap_score_threshold, my_ap_log_path))
    		print(f'EPOCH {epoch} SUMMARY: ' + ', '.join([f'{k}: {v[epoch-1]}' for k, v in log.items()]))
    		with open(loss_log_path, 'a') as file:
    			loss_log_append = f"{epoch} {log['TRAIN_LOSS'][epoch-1]} {log['VAL_LOSS'][epoch-1]}\n"
    			file.write(loss_log_append)
#BLOCCO TRAINING MODEL
elif (not train_backward_on_disturbed_sets):
	load_checkpoint(tasknet, tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
	tasknet.train() #faster ci deve rimanere sempre per avere la loss, freezo poi tutto qui sotto
#qui sotto freezo tutti i layer, visto che fastrcnn non mi permette di ottenere la loss
#a meno che il modello sia in training
	for param in tasknet.parameters():
		param.requires_grad = False
	for epoch in range(1, num_epochs+1): #itero ora facendo un train e un test per ogni epoca
    		log['TRAIN_LOSS'].append(train_model(train_dataloader, epoch, device, train_loss, unet, tasknet, unet_optimizer))
    		unet_scheduler.step()
    		log['VAL_LOSS'].append(val_model(val_dataloader, epoch, device, val_loss, unet, unet_save_path, tasknet, unet_optimizer, unet_scheduler, ap_log_path, ap_score_threshold, my_ap_log_path))
    		if((num_epochs-epoch)==0 and save_disturbed_dataset): #serve se sono arrivato all'ultima epoca e voglio salvare il dataset disturbato
    			generate_disturbed_dataset(train_dataloader_gen_disturbed, val_dataloader_gen_disturbed, epoch, device, unet, disturbed_train_img_folder, disturbed_train_ann, disturbed_val_img_folder, disturbed_val_ann)
    		print(f'EPOCH {epoch} SUMMARY: ' + ', '.join([f'{k}: {v[epoch-1]}' for k, v in log.items()]))
    		with open(loss_log_path, 'a') as file:
    			loss_log_append = f"{epoch} {log['TRAIN_LOSS'][epoch-1]} {log['VAL_LOSS'][epoch-1]}\n"
    			file.write(loss_log_append)
#BLOCCO TRAINING MODEL ON DISTURBED SET   			
else:
	for epoch in range(1, num_epochs+1): #itero ora facendo un train e un test per ogni epoca
    		log['TRAIN_LOSS'].append(train_model_on_disturbed_images(disturbed_train_dataloader, epoch, device, train_loss, unet, unet_optimizer))
    		unet_scheduler.step()
    		log['VAL_LOSS'].append(val_model_on_disturbed_images(disturbed_val_dataloader, epoch, device, val_loss, unet, unet_save_path, unet_optimizer, unet_scheduler))
    		print(f'EPOCH {epoch} SUMMARY: ' + ', '.join([f'{k}: {v[epoch-1]}' for k, v in log.items()]))
    		with open(loss_log_path, 'a') as file:
    			loss_log_append = f"{epoch} {log['TRAIN_LOSS'][epoch-1]} {log['VAL_LOSS'][epoch-1]}\n"
    			file.write(loss_log_append)
    			
print("Done!")

#load_checkpoint(tasknet, tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
#load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)

#show_res_test_unet(unet, tasknet, device, 'plot/val.jpg', True, 'plot/reconstructed_person.png')
"""
image_name_list=['val', 'cat', 'lenna', 'people']
#image_name_list=['disturbed_val', 'disturbed_cat', 'disturbed_lenna', 'disturbed_people']

load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
for img in image_name_list:
	#image_path=f'plot/{img}.jpg'
	image_path=f'plot/disturbed_{img}.png'
	image_save_name=f'plot/test_{img}_rec_msssim.png'
	compare_two_results_unet(unet, tasknet, device, image_path, image_save_name, unet_weights_load, unet_weights_to_compare, unet_optimizer, unet_scheduler)
"""
"""
load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
for img in image_name_list:
	image_path=f'plot/{img}.jpg'
	image_save_name=f'plot/disturbed_{img}.png'
	save_disturbed_pred(unet, device, image_path, image_save_name)
"""
