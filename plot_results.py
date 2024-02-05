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
from plot_utils.extract_iou50_values import extract_iou50_ap
from plot_utils.plot_michele_metric import plot_michele_metric
from plot_utils.compare_ap import plot_compare_between_two_ap
from plot_utils.plot_images import *

###### CONFIG
seed_everything(0) #per rendere deterministico l'esperimento
#nota: upsample_bilinear2d_backward_out_cuda della unet non è deterministico
torch.use_deterministic_algorithms(mode=True, warn_only=True)
device='cuda'
#Config Plotting and Save files
plt.rcParams['figure.figsize'] = 15, 10
loss_log_path='results/loss_log.txt'
loss_save_name='plotted_results/loss.png'

ap_score_threshold=0.75
ap_plot_title='(IoU=0.50, area=all, maxDets=100) AP and Recall Over Epochs'
ap_log_path="results/ap_log.txt" #standard AP
ap_standard_extracted='results/ap_standard_iou50.txt'
ap_standard_save_name='plotted_results/ap_standard_iou50.png'

my_ap_log_path="results/my_ap_log.txt" #mia ap con interpolation senza filtrare i result
my_ap_standard_extracted='results/my_ap_standard_iou50.txt'
my_ap_standard_save_name='plotted_results/my_ap_standard_iou50.png'

my_ap_nointerp_thresh_path="results/my_ap_nointerp_thresh_log.txt" #mia ap senza interpolation ma filtrando i result
my_ap_nointerp_thresh_extracted='results/my_ap_nointerp_thresh_iou50.txt'
my_ap_nointerp_thresh_save_name='plotted_results/my_ap_nointerp_thresh_iou50.png'

my_ap_interp_thresh_path="results/my_ap_interp_thresh_log.txt" #mia Ap con interpolation ma filtrando i result
my_ap_interp_thresh_extracted="results/my_ap_interp_thresh_iou50.txt"
my_ap_interp_thresh_save_name="plotted_results/my_ap_interp_thresh_iou50.png"

michele_metric_file_list=['results/iou0.5_score0.5.json', 'results/iou0.5_score0.75.json', 'results/iou0.75_score0.5.json', 'results/iou0.75_score0.75.json']
michele_metric_file_save_list=['plotted_results/iou0.5_score0.5.png', 'plotted_results/iou0.5_score0.75.png', 'plotted_results/iou0.75_score0.5.png', 'plotted_results/iou0.75_score0.75.png']

ap_comparison_save_path='plotted_results/comparison_ap.png'
#models
unet_save_path = "model_weights/model"
tasknet_save_path = "tasknet_weights/tasknet"
unet_weights_load= "model_weights/model_25.pt"
unet_weights_to_compare= "model_weights/model_8.pt"
tasknet_weights_load= "tasknet_weights/tasknet_10.pt"
#Test images:
image_save_prefix='test'
image_list_folder='test_data'
image_save_folder='plotted_results'
image_name_list=['val', 'cat', 'lenna', 'people']
#image_name_list=['a', 'b']
#image_name_list=['disturbed_val', 'disturbed_cat', 'disturbed_lenna', 'disturbed_people']
####
#unet standard
unet = UNet(3, 3, False)
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

plot_model_loss(loss_log_path, loss_save_name) #loss

extract_iou50_ap(ap_log_path, ap_standard_extracted, standard_ap=True) #standard AP, all preds, no interp
plot_ap(ap_standard_extracted, ap_standard_save_name, best_ap_value_for_comparison=0.757, best_recall_value_for_comparison=0.585, model_name='Unet with Plain Tasknet', ap_plot_title=ap_plot_title)

extract_iou50_ap(my_ap_log_path, my_ap_standard_extracted, standard_ap=False) #My AP, all preds, no interps
plot_ap(my_ap_standard_extracted, my_ap_standard_save_name, best_ap_value_for_comparison=None, best_recall_value_for_comparison=None, model_name='Unet with Plain Tasknet', ap_plot_title=ap_plot_title)

extract_iou50_ap(my_ap_nointerp_thresh_path, my_ap_nointerp_thresh_extracted, standard_ap=False) #My AP, preds that satisfies threshold score, no interpolation
plot_ap(my_ap_nointerp_thresh_extracted, my_ap_nointerp_thresh_save_name, best_ap_value_for_comparison=None, best_recall_value_for_comparison=None, model_name='Unet with Plain Tasknet', ap_plot_title=ap_plot_title)

extract_iou50_ap(my_ap_interp_thresh_path, my_ap_interp_thresh_extracted, standard_ap=False) #My AP, preds that satisfies threshold score, with interpolation
plot_ap(my_ap_interp_thresh_extracted, my_ap_interp_thresh_save_name, best_ap_value_for_comparison=1, best_recall_value_for_comparison=0.585, model_name='Unet with Plain Tasknet', ap_plot_title=ap_plot_title)

#plot_compare_between_two_ap(ap_standard_extracted, 'results/ap_iou50_plaintasknet.txt', ap_model_name='Unet with IoU method', ap_to_compare_model_name='Unet with plain tasknet', plotted_comparison_save_path=ap_comparison_save_path, ap_plot_title=ap_plot_title)
from plot_utils.ms_ssim import plot_ms_ssim_score
plot_ms_ssim_score(ms_ssim_score_log_path="results/ms_ssim_score_log.txt", ms_ssim_save_name='plotted_results/ms_ssim_score.png')

plot_michele_metric(michele_metric_file_list, michele_metric_file_save_list)

#Plotting per le img
for img in image_name_list:
	image_path=''
	image_path=f'{image_list_folder}/{img}.jpg'
	image_save_name=f'{image_save_folder}/{image_save_prefix}_{img}.png'
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
