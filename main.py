import numpy as np
import torchvision
import torch
import os
import shutil
############ Import from my code
from unet_model import UNet
from dataset import *
from model_utils_and_functions import *
from train_and_test_loops import *
############
###### CONFIG
seed_everything(0) #Deterministic experiments
torch.use_deterministic_algorithms(mode=True, warn_only=True) #To force CUDA deterministic algorithms if possible
device='cuda'
#Config Plotting and Save files.
results_dir='results' #Root for results will be saved in results folder
ap_score_threshold=0.75
unet_save_path = "model_weights/model"
tasknet_save_path = "tasknet_weights/tasknet" #Root and name for U-Net weights
unet_weights_load= "model_weights/model_50_notback.pt" #Root and name for Tasknet weights
tasknet_weights_load= "tasknet_weights/tasknet_10.pt" #Loading model path to resume training
my_recons_classifier_weights='my_recons_classifier/my_recons_classifier_weights.pt'
my_regressor_weights='my_recons_classifier/my_regressor_weights.pt'
#Config Dataset. Paths to image folders and annotations. Paths to disturbed dataset and annotations.
use_coco_for_disturbed_set=False #Set to True if you want to use COCO dataset for using and generating the disturbed set instead of default Open Images one
if use_coco_for_disturbed_set:
   disturbed_train_img_gen='/home/alberti/coco_people_indoor/train/images'
   disturbed_train_ann_gen='/home/alberti/coco_people_indoor/train/train.json'
else:
   disturbed_train_img_gen='/home/alberti/open_images/images'
   disturbed_train_ann_gen='/home/alberti/open_images/open_images_id_list.json'
disturbed_train_img_folder='disturbed_dataset/train'
disturbed_train_ann='disturbed_dataset/train.json'
disturbed_val_img_folder='disturbed_dataset/val'
disturbed_val_ann='disturbed_dataset/val.json'

tasknet_get_AP_for_comparison=False #needed for getting the AP for comparison with UNet
train_only_tasknet=False #If you want to first train only the Tasknet; if False the code assumes you want to train the model
if train_only_tasknet:
   train_batch_size=8 #8 batch size used for Tasknet for experiments
   val_batch_size=8
   train_img_folder = '/home/alberti/coco_person/train/images' #For training the tasknet on 65k people dataset
   train_ann_file = '/home/alberti/coco_person/train/train.json'
   val_img_folder = '/home/alberti/coco_person/val/images'
   val_ann_file = '/home/alberti/coco_person/val/val.json'
   resize_scales_transform = [200, 300, 400, 500, 600]
   #AP is computed with lowest resize (in this case 200) on the 65k dataset.
   #For having comparable results with UNet, it's best if you execute a validation epoch with Indoor dataset
   #To do so, leave train_only_tasknet to True and set tasknet_get_AP_for_comparison to True
   if tasknet_get_AP_for_comparison:
      resize_scales_transform = [256] #same resize used for UNet AP 
      val_img_folder = '/home/alberti/coco_people_indoor/val/images'
      val_ann_file = '/home/alberti/coco_people_indoor/val/val.json'
else:
   train_batch_size=4 #4 batchs size used for UNet for experiments.
   val_batch_size=4  
   train_img_folder = '/home/alberti/coco_people_indoor/train/images'
   open_train_img_folder = '/home/alberti/open_images/images'
   train_ann_file = '/home/alberti/coco_people_indoor/train/train.json'
   val_img_folder = '/home/alberti/coco_people_indoor/val/images'
   val_ann_file = '/home/alberti/coco_people_indoor/val/val.json'
   resize_scales_transform = [256, 288, 320, 352, 384, 416]

use_dataset_subset=0 #If you want to use a subset of data (0 is the default: whole dataset)
#Config Execution mode of the Architecture
save_disturbed_dataset=False #To use for creating the disturbed dataset for Backward training
keep_original_size=False #Must be True if you want to fine tune the Tasknet on the disturbed COCO set; can be false for Backward training
train_model_backward=True #For Backward training on disturbed dataset
resume_training=False #Set to True if you want to resume training
if train_only_tasknet:
   num_epochs = 10 #10 are enough for Tasknet as we just need to fine tune it
elif not train_model_backward:
   num_epochs = 50 #50 are enough for UNet training forward, but more epochs can show additional performance improvement
else:
   num_epochs = 100 #are enough for Backward training and simulatin an attack

###### MODELS import and Hyperparameters
if not train_only_tasknet:
   unet = UNet(n_channels=3, bilinear=False) #UNet modified without skip connection
   unet.to(device)
   unet_optimizer = torch.optim.SGD(unet.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4, nesterov=True)
   #unet_scheduler = torch.optim.lr_scheduler.StepLR(unet_optimizer, step_size=10, gamma=0.5)
   if train_model_backward: #more patience for Backward as we have slightly more data and augmentation
      unet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(unet_optimizer, mode='min', factor=0.5, patience=4, verbose=True)
   else:
      unet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(unet_optimizer, mode='min', factor=0.5, patience=2, verbose=True)

if train_only_tasknet: #Default Tasknet
   from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
   from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
   weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
   tasknet = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
   num_classes = 2  # 1 class (person) + background
   in_features = tasknet.roi_heads.box_predictor.cls_score.in_features
   tasknet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
   tasknet.to(device)
   tasknet_optimizer = torch.optim.SGD(tasknet.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
   tasknet_scheduler = torch.optim.lr_scheduler.StepLR(tasknet_optimizer, step_size=3, gamma=0.1)
   #tasknet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(tasknet_optimizer, mode='min', factor=0.5, patience=2, verbose=True)
elif not train_model_backward: #Modified Tasknet with custom proposal method
   from faster_modificata.faster_rcnn import fasterrcnn_resnet50_fpn_modificata, FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor
   weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
   use_custom_filter_proposals_objectness = True #By default use this method, as it's faster
   if use_custom_filter_proposals_objectness:
      tasknet = fasterrcnn_resnet50_fpn_modificata(weights=weights, progress=False, 
         rpn_use_custom_filter_anchors=False, rpn_n_top_pos_to_keep=2, rpn_n_top_neg_to_keep=2,
         rpn_n_top_bg_to_keep=0, rpn_objectness_bg_thresh=0.0, box_use_custom_filter_proposals_objectness=True, 
         box_n_top_pos_to_keep=4, box_n_top_neg_to_keep=4, box_n_top_bg_to_keep=0, box_obj_bg_score_thresh=0.9)
   else: #Based on score, slower and not necessarily better resutls
      tasknet = fasterrcnn_resnet50_fpn_modificata(weights=weights, progress=False,
         rpn_post_nms_top_n_train=5000, #5000 to keep all proposals, 500 for tradeoff with performance
         rpn_use_custom_filter_anchors=False, rpn_n_top_pos_to_keep=1, rpn_n_top_neg_to_keep=2,
         rpn_n_top_bg_to_keep=0, rpn_objectness_bg_thresh=0.0, box_use_custom_filter_proposals_scores=True, 
         box_n_top_pos_to_keep=3, box_n_top_neg_to_keep=6, box_n_top_bg_to_keep=0, box_obj_bg_score_thresh=0.9,
         box_batch_size_per_image=10000, box_positive_fraction=0.25) #10000 for be sure that sampler keep all proposals
      #rpn_objectness_bg_thresh is set to 0, as it's objectness score (so 0 is already a decent high confidence that the anchor contains an object.
   num_classes = 2
   in_features = tasknet.roi_heads.box_predictor.cls_score.in_features
   tasknet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
   tasknet.to(device)
   tasknet_optimizer = torch.optim.SGD(tasknet.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)
   tasknet_scheduler = torch.optim.lr_scheduler.StepLR(tasknet_optimizer, step_size=3, gamma=0.1)

if save_disturbed_dataset: #If it's True, we save the dataset and exit the script.
   train_dataloader_gen_disturbed, val_dataloader_gen_disturbed = load_dataset_for_generating_disturbed_set(
      disturbed_train_img_gen, disturbed_train_ann_gen, val_img_folder, val_ann_file, 
      use_dataset_subset, use_coco_for_disturbed_set)
   if not os.path.exists(disturbed_train_img_folder):
      os.makedirs(disturbed_train_img_folder)
   if not os.path.exists(disturbed_val_img_folder):
      os.makedirs(disturbed_val_img_folder)
   load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
   generate_disturbed_dataset(train_dataloader_gen_disturbed, val_dataloader_gen_disturbed, device, unet,
      disturbed_train_img_folder, disturbed_train_ann, disturbed_val_img_folder, disturbed_val_ann, 
      keep_original_size, use_coco_for_disturbed_set)
   print("Generated disturbed dataset")
   sys.exit()

if train_model_backward: #Disturbed dataset are different
   if use_coco_for_disturbed_set:
      orig_train_img_folder = train_img_folder
   else:
      orig_train_img_folder = open_train_img_folder
   disturbed_train_dataloader, disturbed_val_dataloader, example_dataloader = load_disturbed_dataset(
      disturbed_train_img_folder, disturbed_train_ann, disturbed_val_img_folder, disturbed_val_ann, 
      orig_train_img_folder, val_img_folder, train_batch_size, val_batch_size, resize_scales_transform, 
      use_dataset_subset, val_ann_file)
else: #Same datasets for training Tasknet and training Unet forward
   train_dataloader, val_dataloader, example_dataloader= load_dataset(
      train_img_folder, train_ann_file, val_img_folder, val_ann_file, train_batch_size, val_batch_size,
      save_disturbed_dataset, train_only_tasknet, resize_scales_transform, use_dataset_subset)

if not os.path.exists(results_dir):
   os.makedirs(results_dir)
starting_epoch=1 #Used for counting epoch from 1
if resume_training:
   if train_only_tasknet: #+1 to start at next epoch
      starting_epoch = load_checkpoint(tasknet, tasknet_weights_load, tasknet_optimizer, tasknet_scheduler) + 1
   else:
      starting_epoch = load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler) + 1
      #load_checkpoint_encoder(unet, unet_weights_load, unet_optimizer, unet_scheduler, load_optim_scheduler=False)
      #load_checkpoint_decoder(unet, unet_weights_load, unet_optimizer, unet_scheduler, load_optim_scheduler=False)
      #freeze_encoder(unet)
      #freeze_decoder(unet)
else: #Remove results folder as it's from old experiment
   shutil.rmtree(results_dir)
   os.makedirs(results_dir)
#Models used for measuring reconstruction
if not train_only_tasknet:
   my_recons_classifier = load_my_recons_classifier(my_recons_classifier_weights, device)
   my_regressor = load_my_regressor(my_regressor_weights, device)
   from lpips.lpips import LPIPS
   lpips_model = LPIPS(net='vgg').to(device)
   lpips_model.eval()
   from pytorch_msssim import ms_ssim, MS_SSIM
   ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], K=(0.01, 0.07))
   #size average is a reduction to average the MS SSIM of images in batch.
   #With k2=0.07 we avoid cases where MS SSIM doesn't perform well (with default 0.03 values are more unstable)

#TRAINING TASKNET BLOCK
if(train_only_tasknet and not train_model_backward):
   if tasknet_get_AP_for_comparison:
      load_checkpoint(tasknet, tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
      val_temp_loss = val_tasknet(val_dataloader, epoch, device, tasknet_save_path, tasknet, tasknet_optimizer,
         tasknet_scheduler, ap_score_threshold, results_dir)
      print("Computed AP for comparison with UNet indoor set")
      sys.exit()
   for epoch in range(starting_epoch, num_epochs+1):
      train_temp_loss = train_tasknet(train_dataloader, epoch, device, tasknet_save_path, tasknet, tasknet_optimizer)
      val_temp_loss = val_tasknet(val_dataloader, epoch, device, tasknet_save_path, tasknet, tasknet_optimizer,
         tasknet_scheduler, ap_score_threshold, results_dir)
      tasknet_scheduler.step()
      #tasknet_scheduler.step(val_temp_loss)
      print(f'EPOCH {epoch} SUMMARY: Train loss {train_temp_loss}, Val loss {val_temp_loss}')
      with open(f'{results_dir}/loss_log.txt', 'a') as file:
         loss_log_append = f"{epoch} {train_temp_loss} {val_temp_loss}\n"
         file.write(loss_log_append)
#TRAINING MODEL BLOCK
elif (not train_model_backward):
   load_checkpoint(tasknet, tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
   for param in tasknet.parameters(): #Freeze layers as Faster R-CNN is not modifiable
      param.requires_grad = False
   for epoch in range(starting_epoch, num_epochs+1):
      train_temp_loss = train_model(train_dataloader, epoch, device, unet, tasknet, unet_optimizer)   		
      val_temp_loss = val_model(val_dataloader, epoch, device, unet, unet_save_path, tasknet, unet_optimizer,
         unet_scheduler, ap_score_threshold, results_dir, my_recons_classifier, my_regressor, 
         lpips_model, ms_ssim_module, example_dataloader)
      #unet_scheduler.step()
      unet_scheduler.step(val_temp_loss)
      print(f'EPOCH {epoch} SUMMARY: Train loss {train_temp_loss}, Val loss {val_temp_loss}')
      with open(f'{results_dir}/loss_log.txt', 'a') as file:
         loss_log_append = f"{epoch} {train_temp_loss} {val_temp_loss}\n"
         file.write(loss_log_append)
#TRAINING BACKWARD BLOCK 			
else:
   loss = torch.nn.MSELoss()
   for epoch in range(starting_epoch, num_epochs+1):
      train_temp_loss = train_model_on_disturbed_images(disturbed_train_dataloader, epoch, device, unet, unet_optimizer, loss)
      val_temp_loss = val_model_on_disturbed_images(disturbed_val_dataloader, epoch, device, unet, unet_save_path,
         unet_optimizer, unet_scheduler, results_dir, lpips_model, ms_ssim_module, example_dataloader, loss)
      #unet_scheduler.step()
      unet_scheduler.step(val_temp_loss)
      print(f'EPOCH {epoch} SUMMARY: Train loss {train_temp_loss}, Val loss {val_temp_loss}')
      with open(f'{results_dir}/loss_log.txt', 'a') as file:
         loss_log_append = f"{epoch} {train_temp_loss} {val_temp_loss}\n"
         file.write(loss_log_append)
   			
print("Done!")
