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
############

###### CONFIG
seed_everything(0) #per rendere deterministico l'esperimento
#nota: upsample_bilinear2d_backward_out_cuda della unet non è deterministico
torch.use_deterministic_algorithms(mode=True, warn_only=True)
device='cuda'
#Config Plotting and Save files
loss_log_path='results/loss_log.txt'
ap_score_threshold=0.75
ap_log_path="results/ap_log.txt" #standard AP
my_ap_log_path="results/my_ap_log.txt" #mia ap con interpolation senza filtrare i result
my_ap_nointerp_thresh_path="results/my_ap_nointerp_thresh_log.txt" #mia ap senza interpolation ma filtrando i result
my_ap_interp_thresh_path="results/my_ap_interp_thresh_log.txt" #mia Ap con interpolation ma filtrando i result
michele_metric_folder="results"
unet_save_path = "model_weights/model"
tasknet_save_path = "tasknet_weights/tasknet"
unet_weights_load= "model_weights/model_50.pt"
unet_weights_to_compare= "model_weights/model_50.pt"
tasknet_weights_load= "tasknet_weights/tasknet_10.pt"
#Config DATASET
use_coco_train_for_generating_disturbed_set=False #se metti questo a true, allora stai usando il dataset di train di coco per generare il disturbato. Vanno cambiati i due path sopra
if use_coco_train_for_generating_disturbed_set:
	disturbed_train_img_gen='/home/alberti/coco_people_indoor/train/images'
	disturbed_train_ann_gen='/home/alberti/coco_people_indoor/train/train.json'
else:
	disturbed_train_img_gen='/home/math0012/Tesi_magistrale/open_images_v7/images'
	disturbed_train_ann_gen='/home/math0012/Tesi_magistrale/open_images_v7/open_images_id_list.json'
disturbed_train_img_folder='disturbed_dataset/train'
disturbed_train_ann='disturbed_dataset/train.json'
disturbed_val_img_folder='disturbed_dataset/val'
disturbed_val_ann='disturbed_dataset/val.json'

train_only_tasknet=False #se voglio trainare solo faster rcnn
if train_only_tasknet:
	train_batch_size=8 #8 batch size usata nella tasknet
	val_batch_size=8 #8 batch size usata nella tasknet
	train_img_folder = '/home/alberti/coco_person/train/images' #Questi per trainare la tasknet su 65k
	train_ann_file = '/home/alberti/coco_person/train/train.json'
	val_img_folder = '/home/alberti/coco_person/val/images'
	val_ann_file = '/home/alberti/coco_person/val/val.json'
else:
	train_batch_size=4
	val_batch_size=4
	train_img_folder = '/home/alberti/coco_people_indoor/train/images'
	#train_img_folder = '/home/math0012/Tesi_magistrale/open_images_v7/images'
	train_ann_file = '/home/alberti/coco_people_indoor/train/train.json'
	val_img_folder = '/home/alberti/coco_people_indoor/val/images'
	val_ann_file = '/home/alberti/coco_people_indoor/val/val.json'
resize_scales_transform = [200, 300, 400, 500, 600]
use_dataset_subset=0
#use_dataset_subset=10 #se è 0 uso tutto il dataset, se è n uso esattamente n elementi dal dataset
#Config Execution mode of the Architecture
resume_training=False
save_disturbed_dataset=False
keep_original_size=False #quando generi il dataset disturbato, deve essere a True se mi serve per il comparison con la tasknet plain, se no a False se devo fare train backward
#split_size_train_set = 0 #necessario per quando si fa il train backward. Se a 0 nessuno split. Non è possibile usare lo stesso dataset per train perché le img ricostruite dal train potrebbero contenere più info rispetto a quelle generate dal validation. Quindi tengo i primi n elementi per train del modello con la tasknet, poi genero di disturbati quelli da n in poi, e userò solo quelli per trainare il backward. Questo se voglio usare coco indoor.
train_backward_on_disturbed_sets=False
num_epochs = 50 #setto numero delle epoche

###### MODELLI
#Instanzio il modello e gli iperparametri; lo muovo poi al device
#primo parametro n canali, secondo n_classes is the number of probabilities you want to get per pixel
if not train_only_tasknet:
	unet = UNet(3, 3, False)
	unet.to(device)
	unet_optimizer = torch.optim.SGD(unet.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)
	unet_scheduler = torch.optim.lr_scheduler.StepLR(unet_optimizer, step_size=10, gamma=0.5)

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
		rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
		rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
		rpn_nms_thresh=0.7, rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
		rpn_score_thresh=0.0, rpn_use_custom_filter_anchors=False,
		rpn_n_top_pos_to_keep=1, rpn_n_top_neg_to_keep=5,
		rpn_n_top_bg_to_keep=1, rpn_absolute_bg_score_thresh=0.75, rpn_objectness_bg_thresh=0.0,
		rpn_use_not_overlapping_proposals=False, rpn_overlapping_prop_thresh=0.6,
		box_use_custom_filter_proposals=True, box_n_top_pos_to_keep=4, box_n_top_neg_to_keep=8, 
		box_n_top_bg_to_keep=0, box_obj_bg_score_thresh=0.9,
		box_batch_size_per_image=512, box_positive_fraction=0.25, box_bg_iou_thresh=0.5)
	"""
	comemnto su rpn_objectness_bg_thresh: la objectness varia da -25 a 10 o più. Se è 0 o superiore rappresenta la confidence dell'ancora che contenga un oggetto.
	commento su ultimo campo: rpn_iou_neg_thresh must be equal to box_bg_iou_thresh. Otherwise what is selected by the custom filter proposal may not be considered negative by the ROI heads.
	commento su penultimi due campi: sono del sampler degli indici delle proposal. Valore di default rispettivi: 512, 0.25. Il primo ti dice quante proposal tenere al massimo, scelte a caso. Il secondo ti dice: di quelle 512 proposal quante al massimo sono positive? Il sampler nel codice fa una roba del genere. Immaginiamo di avere 1 positiva (con IoU >0.5, il Roi head le discrimina così) e tenere 1000 negative (IoU<0.5).
num_pos = int(self.batch_size_per_image * self.positive_fraction)   #num_pos = 512*0.25 -> 128
# protect against not enough positive examples
num_pos = min(positive.numel(), num_pos)   # num_pos = min(1, 128) -> 1
num_neg = self.batch_size_per_image - num_pos    # num_neg = 512 - 1 -> 511
# protect against not enough negative examples
num_neg = min(negative.numel(), num_neg)    # num_neg = min(1000, 511) -> 511

# randomly select positive and negative examples
perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos] #1 a caso fra le 1 pos che ho
perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg] #511 a caso fra le 1000 neg che ho

Quindi:se voglio tenere tutte le negative, devo aumentare il 512 a un valore maggiore, ad esempio 2000, visto che di negative con basso IoU ne ho molte
	"""

	num_classes = 2  # 1 class (person) + background
	# get number of input features for the classifier
	in_features = tasknet.roi_heads.box_predictor.cls_score.in_features
	# replace the pre-trained head with a new one
	tasknet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
	tasknet.to(device)
	tasknet_optimizer = torch.optim.SGD(tasknet.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)
	tasknet_scheduler = torch.optim.lr_scheduler.StepLR(tasknet_optimizer, step_size=3, gamma=0.1)

train_loss = [] # Lista che conserva la training loss. Mi serve se voglio vedere l'andamento della loss
val_loss = [] #Lista che conversa la test loss
log = {'TRAIN_LOSS': [], 'VAL_LOSS': []}

if save_disturbed_dataset:
	train_dataloader_gen_disturbed, val_dataloader_gen_disturbed = load_dataset_for_generating_disturbed_set(disturbed_train_img_gen, disturbed_train_ann_gen, val_img_folder, val_ann_file, use_dataset_subset, use_coco_train_for_generating_disturbed_set) #, split_size_train_set)

if train_backward_on_disturbed_sets: #carico i dataloader appositi del dataset disturbato
	disturbed_train_dataloader, disturbed_val_dataloader = load_disturbed_dataset(disturbed_train_img_folder, disturbed_train_ann, disturbed_val_img_folder, disturbed_val_ann, train_img_folder, val_img_folder, train_batch_size, val_batch_size, resize_scales_transform, use_dataset_subset)
else:
	train_dataloader, val_dataloader= load_dataset(train_img_folder, train_ann_file, val_img_folder, val_ann_file, train_batch_size, val_batch_size, save_disturbed_dataset, train_only_tasknet, resize_scales_transform, use_dataset_subset) #, split_size_train_set)

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
    		log['VAL_LOSS'].append(val_tasknet(val_dataloader, epoch, device, val_loss, tasknet_save_path, tasknet, tasknet_optimizer, tasknet_scheduler, ap_log_path, ap_score_threshold, my_ap_log_path, my_ap_nointerp_thresh_path, my_ap_interp_thresh_path, michele_metric_folder))
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
    		log['VAL_LOSS'].append(val_model(val_dataloader, epoch, device, val_loss, unet, unet_save_path, tasknet, unet_optimizer, unet_scheduler, ap_log_path, ap_score_threshold, my_ap_log_path, my_ap_nointerp_thresh_path, my_ap_interp_thresh_path, michele_metric_folder))
    		if((num_epochs-epoch)==0 and save_disturbed_dataset): #serve se sono arrivato all'ultima epoca e voglio salvare il dataset disturbato
    			generate_disturbed_dataset(train_dataloader_gen_disturbed, val_dataloader_gen_disturbed, epoch, device, unet, disturbed_train_img_folder, disturbed_train_ann, disturbed_val_img_folder, disturbed_val_ann, keep_original_size, use_coco_train_for_generating_disturbed_set)
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

if save_disturbed_dataset:
	generate_disturbed_dataset(train_dataloader_gen_disturbed, val_dataloader_gen_disturbed, device, unet, disturbed_train_img_folder, disturbed_train_ann, disturbed_val_img_folder, disturbed_val_ann, keep_original_size, use_coco_train_for_generating_disturbed_set)
   			
print("Done!")
