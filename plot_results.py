#Importo le librerie necessarie
import numpy as np  #Numpy
import torchvision #libreria principale di Pytorch per riconoscere le immagini
import torch #Pytorch
from torch import nn #Modulo prinicipale di torch che definisce la maggior parte delle cose relative alle reti, come i layer, funzioni di loss...
from torch import optim #Modulo con i vari optimizer

############ QUESTO CODICE MIO
from unet_model import UNet
from model_utils_and_functions import *
from plot_utils.loss import plot_model_loss
from plot_utils.ap import plot_ap
from plot_utils.extract_iou50_values import extract_ap
from plot_utils.plot_michele_metric import plot_michele_metric
from plot_utils.compare_ap import plot_compare_between_two_ap
from plot_utils.plot_images import *
from plot_utils.plot_my_recons_classifier_metric import plot_my_recons_classifier_metric, plot_my_recons_classifier_metric_probs
###### CONFIG
seed_everything(0) #per rendere deterministico l'esperimento
#nota: upsample_bilinear2d_backward_out_cuda della unet non è deterministico
torch.use_deterministic_algorithms(mode=True, warn_only=True)
device='cuda'
#Config Plotting and Save files
plt.rcParams['figure.figsize'] = 15, 10
results_dir='results'
save_dir='plotted_results'
ap_score_threshold=0.75
michele_metric_file_list=[f'{results_dir}/iou0.5_score0.5.json', f'{results_dir}/iou0.5_score0.75.json', f'{results_dir}/iou0.75_score0.5.json', f'{results_dir}/iou0.75_score0.75.json']
michele_metric_file_save_list=[f'{save_dir}/iou0.5_score0.5.png', f'{save_dir}/iou0.5_score0.75.png', f'{save_dir}/iou0.75_score0.5.png', f'{save_dir}/iou0.75_score0.75.png']
#models
unet_save_path = "model_weights/model"
tasknet_save_path = "tasknet_weights/tasknet"
unet_weights_load= "model_weights/model_4_score.pt"
unet_weights_to_compare= "model_weights/model_8.pt"
tasknet_weights_load= "tasknet_weights/tasknet_10.pt"
#Test images:
image_save_prefix='test'
image_list_folder='test_data'
image_name_list=['val', 'cat', 'lenna', 'people']
#image_name_list=['a', 'b']
#image_name_list=['disturbed_val', 'disturbed_cat', 'disturbed_lenna', 'disturbed_people']
####
#unet standard
unet = UNet(3, False)
unet.to(device)
unet_optimizer = torch.optim.SGD(unet.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)
unet_scheduler = torch.optim.lr_scheduler.StepLR(unet_optimizer, step_size=10, gamma=0.5)
#faster rcnn non modificata dato che mi serve per simulare il deployment
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

load_checkpoint(tasknet, tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)


plot_model_loss(f'{results_dir}/loss_log.txt', f'{save_dir}/loss.png') #loss

#STANDARD APs, standard interpolation, no score threshold filter
extract_ap(f'{results_dir}/standard_ap.txt', f'{results_dir}/ext_standard_ap.txt', standard_ap=True, coco_iou_modified=False) #standard AP, all preds, no interp
plot_ap(f'{results_dir}/ext_standard_ap.txt', f'{save_dir}/standard_ap.png', best_ap_value_for_comparison=0.757, best_recall_value_for_comparison=0.585, model_name='Unet with Plain Tasknet', ap_plot_title='(area=all, maxDets=100) AP (IoU=0.50) and Recall (IoU=0.50:0.95) Over Epochs')

extract_ap(f'{results_dir}/standard_ap_iou50.txt', f'{results_dir}/ext_standard_ap_iou50.txt', standard_ap=True, coco_iou_modified=50)
plot_ap(f'{results_dir}/ext_standard_ap_iou50.txt', f'{save_dir}/standard_ap_iou50.png', best_ap_value_for_comparison=None, best_recall_value_for_comparison=None, model_name='Unet with Plain Tasknet', ap_plot_title='(IoU=0.50, area=all, maxDets=100) AP and Recall Over Epochs')

extract_ap(f'{results_dir}/standard_ap_iou75.txt', f'{results_dir}/ext_standard_ap_iou75.txt', standard_ap=True, coco_iou_modified=75)
plot_ap(f'{results_dir}/ext_standard_ap_iou75.txt', f'{save_dir}/standard_ap_iou75.png', best_ap_value_for_comparison=None, best_recall_value_for_comparison=None, model_name='Unet with Plain Tasknet', ap_plot_title='(IoU=0.75, area=all, maxDets=100) AP and Recall Over Epochs')

#STANDARD APs, standard interpolation, with only preds above score threshold
extract_ap(f'{results_dir}/standard_ap_scoreabovethresh.txt', f'{results_dir}/ext_standard_ap_scoreabovethresh.txt', standard_ap=False, coco_iou_modified=False) #standard AP, all preds, no interp
plot_ap(f'{results_dir}/ext_standard_ap_scoreabovethresh.txt', f'{save_dir}/standard_ap_scoreabovethresh.png', best_ap_value_for_comparison=0.757, best_recall_value_for_comparison=0.585, model_name='Unet with Plain Tasknet', ap_plot_title='(area=all, maxDets=100) AP (IoU=0.50) and Recall (IoU=0.50:0.95) Over Epochs')

extract_ap(f'{results_dir}/standard_ap_scoreabovethresh_iou50.txt', f'{results_dir}/ext_standard_ap_scoreabovethresh_iou50.txt', standard_ap=False, coco_iou_modified=50)
plot_ap(f'{results_dir}/ext_standard_ap_scoreabovethresh_iou50.txt', f'{save_dir}/standard_ap_scoreabovethresh_iou50.png', best_ap_value_for_comparison=None, best_recall_value_for_comparison=None, model_name='Unet with Plain Tasknet', ap_plot_title='(IoU=0.50, area=all, maxDets=100) AP and Recall Over Epochs')

extract_ap(f'{results_dir}/standard_ap_scoreabovethresh_iou75.txt', f'{results_dir}/ext_standard_ap_scoreabovethresh_iou75.txt', standard_ap=False, coco_iou_modified=75)
plot_ap(f'{results_dir}/ext_standard_ap_scoreabovethresh_iou75.txt', f'{save_dir}/standard_ap_scoreabovethresh_iou75.png', best_ap_value_for_comparison=None, best_recall_value_for_comparison=None, model_name='Unet with Plain Tasknet', ap_plot_title='(IoU=0.75, area=all, maxDets=100) AP and Recall Over Epochs')

#APs with my interpolation, without filtering preds above score thresh
extract_ap(f'{results_dir}/myinterp_ap.txt', f'{results_dir}/ext_myinterp_ap.txt', standard_ap=False, coco_iou_modified=False) #standard AP, all preds, no interp
plot_ap(f'{results_dir}/ext_myinterp_ap.txt', f'{save_dir}/myinterp_ap.png', best_ap_value_for_comparison=0.757, best_recall_value_for_comparison=0.585, model_name='Unet with Plain Tasknet', ap_plot_title='(area=all, maxDets=100) AP (IoU=0.50) and Recall (IoU=0.50:0.95) Over Epochs')

extract_ap(f'{results_dir}/myinterp_ap_iou50.txt', f'{results_dir}/ext_myinterp_ap_iou50.txt', standard_ap=False, coco_iou_modified=50)
plot_ap(f'{results_dir}/ext_myinterp_ap_iou50.txt', f'{save_dir}/myinterp_ap_iou50.png', best_ap_value_for_comparison=None, best_recall_value_for_comparison=None, model_name='Unet with Plain Tasknet', ap_plot_title='(IoU=0.50, area=all, maxDets=100) AP and Recall Over Epochs')

extract_ap(f'{results_dir}/myinterp_ap_iou75.txt', f'{results_dir}/ext_myinterp_ap_iou75.txt', standard_ap=False, coco_iou_modified=75)
plot_ap(f'{results_dir}/ext_myinterp_ap_iou75.txt', f'{save_dir}/myinterp_ap_iou75.png', best_ap_value_for_comparison=None, best_recall_value_for_comparison=None, model_name='Unet with Plain Tasknet', ap_plot_title='(IoU=0.75, area=all, maxDets=100) AP and Recall Over Epochs')

#APs with my interpolation, with only preds above score thresh
extract_ap(f'{results_dir}/myinterp_ap_scoreabovethresh.txt', f'{results_dir}/ext_myinterp_ap_scoreabovethresh.txt', standard_ap=False, coco_iou_modified=False) #standard AP, all preds, no interp
plot_ap(f'{results_dir}/ext_myinterp_ap_scoreabovethresh.txt', f'{save_dir}/myinterp_ap_scoreabovethresh.png', best_ap_value_for_comparison=0.757, best_recall_value_for_comparison=0.585, model_name='Unet with Plain Tasknet', ap_plot_title='(area=all, maxDets=100) AP (IoU=0.50) and Recall (IoU=0.50:0.95) Over Epochs')

extract_ap(f'{results_dir}/myinterp_ap_scoreabovethresh_iou50.txt', f'{results_dir}/ext_myinterp_ap_scoreabovethresh_iou50.txt', standard_ap=False, coco_iou_modified=50)
plot_ap(f'{results_dir}/ext_myinterp_ap_scoreabovethresh_iou50.txt', f'{save_dir}/myinterp_ap_scoreabovethresh_iou50.png', best_ap_value_for_comparison=None, best_recall_value_for_comparison=None, model_name='Unet with Plain Tasknet', ap_plot_title='(IoU=0.50, area=all, maxDets=100) AP and Recall Over Epochs')

extract_ap(f'{results_dir}/myinterp_ap_scoreabovethresh_iou75.txt', f'{results_dir}/ext_myinterp_ap_scoreabovethresh_iou75.txt', standard_ap=False, coco_iou_modified=75)
plot_ap(f'{results_dir}/ext_myinterp_ap_scoreabovethresh_iou75.txt', f'{save_dir}/myinterp_ap_scoreabovethresh_iou75.png', best_ap_value_for_comparison=None, best_recall_value_for_comparison=None, model_name='Unet with Plain Tasknet', ap_plot_title='(IoU=0.75, area=all, maxDets=100) AP and Recall Over Epochs')

"""
plot_compare_between_two_ap(ap_standard_extracted, f'{results_dir}/ap_overlap.txt', ap_model_name='Unet without overlapping prop, iou=0.6', ap_to_compare_model_name='Unet with overlapping prop (old method)', plotted_comparison_save_path=ap_comparison_save_path, ap_plot_title=ap_plot_title)
"""

"""
from plot_utils.ms_ssim import plot_ms_ssim_score
plot_ms_ssim_score(ms_ssim_score_log_path=f"{results_dir}/ms_ssim_score_log.txt", ms_ssim_save_name=f'{save_dir}/ms_ssim_score.png')
"""

from plot_utils.lpips_score import plot_lpips_score
plot_lpips_score(lpips_score_log_path=f"{results_dir}/lpips_score_log.txt", lpips_save_name=f"{save_dir}/lpips_score.png")

plot_michele_metric(michele_metric_file_list, michele_metric_file_save_list)

plot_my_recons_classifier_metric(f'{results_dir}/my_recons_classifier_log.json', f'{save_dir}/my_recons_classifier.png')
plot_my_recons_classifier_metric_probs(f'{results_dir}/my_recons_classifier_log.json', f'{save_dir}/my_recons_classifier_probs.png')

#Plotting per le img
for img in image_name_list:
	image_path=''
	image_path=f'{image_list_folder}/{img}.jpg'
	image_save_name=f'{save_dir}/{image_save_prefix}_{img}.png'
	compare_two_results_unet(unet, tasknet, device, image_path, image_save_name, unet_weights_load, unet_weights_to_compare, unet_optimizer, unet_scheduler)
	plt.clf()

"""
#show_res_test_unet(unet, tasknet, device, 'plot/val.jpg', True, 'plot/reconstructed_person.png')
load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
for img in image_name_list:
	image_path=f'plot/{img}.jpg'
	image_save_name=f'plot/disturbed_{img}.png'
	save_disturbed_pred(unet, device, image_path, image_save_name)
"""
