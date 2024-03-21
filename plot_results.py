import numpy as np 
import torchvision
import torch
import os
import shutil
############ My code imports
from unet_model import UNet
from model_utils_and_functions import *
from plot_utils.loss import plot_model_loss
from plot_utils.ap import plot_ap
from plot_utils.extract_iou50_values import extract_ap
from plot_utils.plot_custom_metric import plot_custom_metric
from plot_utils.compare_ap import plot_compare_between_two_ap
from plot_utils.plot_images import *
from plot_utils.plot_my_recons_classifier_metric import plot_my_recons_classifier_metric, plot_my_recons_classifier_metric_probs
from plot_utils.lpips_score import plot_lpips_score
from plot_utils.ms_ssim import plot_ms_ssim_score
from plot_utils.plot_my_regressor import plot_reconrate
###### CONFIG
seed_everything(0)
torch.use_deterministic_algorithms(mode=True, warn_only=True)
device='cuda'
#Config Plotting and Save files
plt.rcParams['figure.figsize'] = 15, 10
results_dir='results'
save_dir='plotted_results' #Where to plot results
ap_score_threshold=0.75
custom_metric_file_list=[f'{results_dir}/iou0.5_score0.5.json', f'{results_dir}/iou0.5_score0.75.json', f'{results_dir}/iou0.75_score0.5.json', f'{results_dir}/iou0.75_score0.75.json']
custom_metric_file_save_list=[f'{save_dir}/iou0.5_score0.5.png', f'{save_dir}/iou0.5_score0.75.png', f'{save_dir}/iou0.75_score0.5.png', f'{save_dir}/iou0.75_score0.75.png']
#Config Models
unet_save_path = "model_weights/model"
tasknet_save_path = "tasknet_weights/tasknet"
unet_weights_load= "model_weights/model_50.pt"
unet_weights_to_compare= "model_weights/model_100.pt"
tasknet_weights_load= "tasknet_weights/tasknet_10.pt"
plot_backward_images=True #If you want to plot only the backward images
#Config Test images folders:
image_save_prefix='test'
image_list_folder='test_data'
image_name_list=['val', 'cat', 'lenna', 'people']
val_set_list=['000000004395','000000008532','000000019924', '000000020992', '000000070739', '000000117525', '000000131444',
'000000133343', '000000138115', '000000170099', '000000179112', '000000224337', '000000258541', '000000262895', '000000271997']
image_val_folder='from_val_set'
#### Defining models. First UNet, then not modified Faster RCNN to simulate deployment
unet = UNet(3, False)
unet.to(device)
unet_optimizer = torch.optim.SGD(unet.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4, nesterov=True)
unet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(unet_optimizer, mode='min', factor=0.5, patience=2, verbose=True)
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
tasknet = fasterrcnn_resnet50_fpn(weights=weights, progress=False)
num_classes = 2  # 1 class (person) + background
in_features = tasknet.roi_heads.box_predictor.cls_score.in_features
tasknet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
tasknet.to(device)
tasknet_optimizer = torch.optim.SGD(tasknet.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)
tasknet_scheduler = torch.optim.lr_scheduler.StepLR(tasknet_optimizer, step_size=3, gamma=0.1)
#loading weights
load_checkpoint(tasknet, tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)

if not os.path.exists(save_dir):
   os.makedirs(save_dir)
#Remove old results
shutil.rmtree(save_dir)
os.makedirs(save_dir)

#LOSS, MS_SSIM, LPIPS, MY RECONS CLASSIFIER, RECON REGRESSOR, CUSTOM METRIC
plot_model_loss(f'{results_dir}/loss_log.txt', f'{save_dir}/loss.png') #loss
plot_ms_ssim_score(ms_ssim_score_log_path=f"{results_dir}/ms_ssim_score_log.txt", ms_ssim_save_name=f'{save_dir}/ms_ssim_score.png')
plot_lpips_score(lpips_score_log_path=f"{results_dir}/lpips_score_log.txt", lpips_save_name=f"{save_dir}/lpips_score.png")
plot_custom_metric(custom_metric_file_list, custom_metric_file_save_list)
plot_my_recons_classifier_metric(f'{results_dir}/my_recons_classifier_log.json', f'{save_dir}/my_recons_classifier.png')
plot_my_recons_classifier_metric_probs(f'{results_dir}/my_recons_classifier_log.json', f'{save_dir}/my_recons_classifier_probs.png')
plot_reconrate(f'{results_dir}/recon_rate_log.txt', f'{save_dir}/regressor_recon_rate.png')

#STANDARD APs, standard interpolation, no score threshold filter
extract_ap(f'{results_dir}/standard_ap.txt', f'{results_dir}/ext_standard_ap.txt', standard_ap=True, coco_iou_modified=False) #standard AP, all preds, no interp
#Tasknet 65k coco dataset values: AP: 0.818, Recall: 0.608
#Tasknet 10k coco indoor people dataset values: AP: 0.858, Recall: 0.685
plot_ap(f'{results_dir}/ext_standard_ap.txt', f'{save_dir}/standard_ap.png', best_ap_value_for_comparison=0.858, best_recall_value_for_comparison=0.685, model_name='Tasknet', ap_plot_title='(area=all, maxDets=100) AP (IoU=0.50), AR (IoU=0.50:0.95) Over Epochs')

extract_ap(f'{results_dir}/standard_ap_iou50.txt', f'{results_dir}/ext_standard_ap_iou50.txt', standard_ap=True, coco_iou_modified=50)
#Tasknet 65k coco dataset values: AP: 0.818, Recall: 0.914
#Tasknet 10k coco indoor people dataset values: AP: 0.858, Recall: 0.952
plot_ap(f'{results_dir}/ext_standard_ap_iou50.txt', f'{save_dir}/standard_ap_iou50.png', best_ap_value_for_comparison=0.858, best_recall_value_for_comparison=0.952, model_name='Tasknet', ap_plot_title='(IoU=0.50, area=all, maxDets=100) AP, AR Over Epochs')

extract_ap(f'{results_dir}/standard_ap_iou75.txt', f'{results_dir}/ext_standard_ap_iou75.txt', standard_ap=True, coco_iou_modified=75)
#Tasknet 65k coco dataset values: AP: 0.554, Recall: 0.643
#Tasknet 10k coco indoor people dataset values: AP: 0.638, Recall: 0.731
plot_ap(f'{results_dir}/ext_standard_ap_iou75.txt', f'{save_dir}/standard_ap_iou75.png', best_ap_value_for_comparison=0.638, best_recall_value_for_comparison=0.731, model_name='Tasknet', ap_plot_title='(IoU=0.75, area=all, maxDets=100) AP, AR Over Epochs')

#STANDARD APs, standard interpolation, with only preds above score threshold
extract_ap(f'{results_dir}/standard_ap_scoreabovethresh.txt', f'{results_dir}/ext_standard_ap_scoreabovethresh.txt', standard_ap=False, coco_iou_modified=False) #standard AP, all preds, no interp
#Tasknet 65k coco dataset values: AP: 0.710, Recall: 0.520
#Tasknet 10k coco indoor people dataset values: AP: 0.797, Recall: 0.610
plot_ap(f'{results_dir}/ext_standard_ap_scoreabovethresh.txt', f'{save_dir}/standard_ap_scoreabovethresh.png', best_ap_value_for_comparison=0.797, best_recall_value_for_comparison=0.610, model_name='Tasknet', ap_plot_title='(area=all, maxDets=100) AP (IoU=0.50), AR (IoU=0.50:0.95) Over Epochs')

extract_ap(f'{results_dir}/standard_ap_scoreabovethresh_iou50.txt', f'{results_dir}/ext_standard_ap_scoreabovethresh_iou50.txt', standard_ap=False, coco_iou_modified=50)
#Tasknet 65k coco dataset values: AP: 0.710, Recall: 0.736
#Tasknet 10k coco indoor people dataset values: AP: 0.797, Recall: 0.824
plot_ap(f'{results_dir}/ext_standard_ap_scoreabovethresh_iou50.txt', f'{save_dir}/standard_ap_scoreabovethresh_iou50.png', best_ap_value_for_comparison=0.797, best_recall_value_for_comparison=0.824, model_name='Tasknet', ap_plot_title='(IoU=0.50, area=all, maxDets=100) AP, AR Over Epochs')

extract_ap(f'{results_dir}/standard_ap_scoreabovethresh_iou75.txt', f'{results_dir}/ext_standard_ap_scoreabovethresh_iou75.txt', standard_ap=False, coco_iou_modified=75)
#Tasknet 65k coco dataset values: AP: 0.522, Recall: 0.574
#Tasknet 10k coco indoor people dataset values: AP: 0.610, Recall: 0.670
plot_ap(f'{results_dir}/ext_standard_ap_scoreabovethresh_iou75.txt', f'{save_dir}/standard_ap_scoreabovethresh_iou75.png', best_ap_value_for_comparison=0.610, best_recall_value_for_comparison=0.670, model_name='Tasknet', ap_plot_title='(IoU=0.75, area=all, maxDets=100) AP, AR Over Epochs')

#APs with my interpolation, without filtering preds above score thresh
extract_ap(f'{results_dir}/myinterp_ap.txt', f'{results_dir}/ext_myinterp_ap.txt', standard_ap=False, coco_iou_modified=False) #standard AP, all preds, no interp
#Tasknet 65k coco dataset values: AP: 0.997, Recall: 0.608
#Tasknet 10k coco indoor people dataset values: AP: 1.000, Recall: 0.685
plot_ap(f'{results_dir}/ext_myinterp_ap.txt', f'{save_dir}/myinterp_ap.png', best_ap_value_for_comparison=1.000, best_recall_value_for_comparison=0.685, model_name='Tasknet', ap_plot_title='(area=all, maxDets=100) AP (IoU=0.50), AR (IoU=0.50:0.95) Over Epochs')

extract_ap(f'{results_dir}/myinterp_ap_iou50.txt', f'{results_dir}/ext_myinterp_ap_iou50.txt', standard_ap=False, coco_iou_modified=50)
#Tasknet 65k coco dataset values: AP: 0.997, Recall: 0.914
#Tasknet 10k coco indoor people dataset values: AP: 1.000, Recall: 0.952
plot_ap(f'{results_dir}/ext_myinterp_ap_iou50.txt', f'{save_dir}/myinterp_ap_iou50.png', best_ap_value_for_comparison=1.000, best_recall_value_for_comparison=0.952, model_name='Tasknet', ap_plot_title='(IoU=0.50, area=all, maxDets=100) AP, AR Over Epochs')

extract_ap(f'{results_dir}/myinterp_ap_iou75.txt', f'{results_dir}/ext_myinterp_ap_iou75.txt', standard_ap=False, coco_iou_modified=75)
#Tasknet 65k coco dataset values: AP: 0.973, Recall: 0.643
#Tasknet 10k coco indoor people dataset values: AP: 0.969, Recall: 0.731
plot_ap(f'{results_dir}/ext_myinterp_ap_iou75.txt', f'{save_dir}/myinterp_ap_iou75.png', best_ap_value_for_comparison=0.969, best_recall_value_for_comparison=0.731, model_name='Tasknet', ap_plot_title='(IoU=0.75, area=all, maxDets=100) AP, AR Over Epochs')

#APs with my interpolation, with only preds above score thresh
extract_ap(f'{results_dir}/myinterp_ap_scoreabovethresh.txt', f'{results_dir}/ext_myinterp_ap_scoreabovethresh.txt', standard_ap=False, coco_iou_modified=False) #standard AP, all preds, no interp
#Tasknet 65k coco dataset values: AP: 0.997, Recall: 0.520
#Tasknet 10k coco indoor people dataset values: AP: 1.000, Recall: 0.610
plot_ap(f'{results_dir}/ext_myinterp_ap_scoreabovethresh.txt', f'{save_dir}/myinterp_ap_scoreabovethresh.png', best_ap_value_for_comparison=1.000, best_recall_value_for_comparison=0.610, model_name='Tasknet', ap_plot_title='(area=all, maxDets=100) AP (IoU=0.50), AR (IoU=0.50:0.95) Over Epochs')

extract_ap(f'{results_dir}/myinterp_ap_scoreabovethresh_iou50.txt', f'{results_dir}/ext_myinterp_ap_scoreabovethresh_iou50.txt', standard_ap=False, coco_iou_modified=50)
#Tasknet 65k coco dataset values: AP: 0.997, Recall: 0.736
#Tasknet 10k coco indoor people dataset values: AP: 1.000, Recall: 0.824
plot_ap(f'{results_dir}/ext_myinterp_ap_scoreabovethresh_iou50.txt', f'{save_dir}/myinterp_ap_scoreabovethresh_iou50.png', best_ap_value_for_comparison=1.000, best_recall_value_for_comparison=0.824, model_name='Tasknet', ap_plot_title='(IoU=0.50, area=all, maxDets=100) AP, AR Over Epochs')

extract_ap(f'{results_dir}/myinterp_ap_scoreabovethresh_iou75.txt', f'{results_dir}/ext_myinterp_ap_scoreabovethresh_iou75.txt', standard_ap=False, coco_iou_modified=75)
#Tasknet 65k coco dataset values: AP: 0.973, Recall: 0.574
#Tasknet 10k coco indoor people dataset values: AP: 0.969, Recall: 0.670
plot_ap(f'{results_dir}/ext_myinterp_ap_scoreabovethresh_iou75.txt', f'{save_dir}/myinterp_ap_scoreabovethresh_iou75.png', best_ap_value_for_comparison=0.969, best_recall_value_for_comparison=0.670, model_name='Tasknet', ap_plot_title='(IoU=0.75, area=all, maxDets=100) AP, AR Over Epochs')

#COMPARISON BETWEEN APs
plot_compare_between_two_ap(f'{results_dir}/ext_standard_ap2.txt', f'{results_dir}/ext_standard_ap.txt', ap_model_name='Unet2', ap_to_compare_model_name='Unet', plotted_comparison_save_path=f'{save_dir}/compare_AP.png', ap_plot_title='(IoU=0.75, area=all, maxDets=100) AP, AR Over Epochs')

#Image Samples plotting
if not plot_backward_images:
   for img in image_name_list:
      image_path=''
      image_path=f'{image_list_folder}/{img}.jpg'
      image_save_name=f'{save_dir}/{image_save_prefix}_{img}.png'
      compare_two_results_unet(unet, tasknet, device, image_path, image_save_name, unet_weights_load, unet_weights_to_compare, unet_optimizer, unet_scheduler)
      plt.clf()
      
   if not os.path.exists(f'{save_dir}/{image_val_folder}'):
      os.makedirs(f'{save_dir}/{image_val_folder}')
   for img in val_set_list:
      image_path=''
      image_path=f'{image_list_folder}/{image_val_folder}/{img}.jpg'
      image_save_name=f'{save_dir}/{image_val_folder}/{image_save_prefix}_{img}.png'
      compare_two_results_unet(unet, tasknet, device, image_path, image_save_name, unet_weights_load, unet_weights_to_compare, unet_optimizer, unet_scheduler)
      plt.clf()

else:
#show_res_test_unet(unet, tasknet, device, 'plot/val.jpg', True, 'plot/reconstructed_person.png')
#load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
   load_checkpoint(unet, unet_weights_to_compare, unet_optimizer, unet_scheduler)
   for img in image_name_list:
      image_path=f'{image_list_folder}/{img}.jpg'
      image_save_name=f'{save_dir}/disturbed_{img}.png'
      save_disturbed_pred(unet, device, image_path, image_save_name)
      plt.clf()
      
   if not os.path.exists(f'{save_dir}/{image_val_folder}'):
      os.makedirs(f'{save_dir}/{image_val_folder}')
   for img in val_set_list:
      image_path=''
      image_path=f'{image_list_folder}/{image_val_folder}/{img}.jpg'
      image_save_name=f'{save_dir}/{image_val_folder}/disturbed_{img}.png'
      save_disturbed_pred(unet, device, image_path, image_save_name)
      plt.clf()
