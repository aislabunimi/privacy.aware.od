import numpy as np 
import torchvision
import torch
import os
import shutil
import argparse
############ My code imports
from unet_model import UNet
from model_utils_and_functions import *
from plot_utils.loss import plot_model_loss
from plot_utils.ap import plot_ap
from plot_utils.extract_iou50_values import extract_ap
from plot_utils.plot_custom_metric import plot_custom_metric #, plot_custom_metric_allclasses
from plot_utils.compare_ap import plot_compare_between_two_ap
from plot_utils.plot_images import *
###### CONFIG

def get_args_parser():
   parser = argparse.ArgumentParser('Set parameters and config', add_help=False)
   parser.add_argument('--seed', default=0, type=int, help='Seed for experiments')
   parser.add_argument('--device', default='cuda', type=str, help='Device to use for experiments')
   
   #FOLDERS for input files and output filesa
   parser.add_argument('--results_dir', default='results', type=str, help='Directory root for storing the results, logs and so on. WARNING: this folder and its contents will be deleted before starting next experiment, remember to backup the results! Also, be aware of the folder you choose: you may delete your whole system!')
   parser.add_argument('--save_dir', default='plotted_results', type=str, help='Directory root for storing the plotted results, logs and so on. WARNING: this folder and its contents will be deleted before starting next plotting! Also, be aware of the folder you choose: you may delete your whole system!')
   parser.add_argument('--test_data_dir', default='test_data', type=str, help='Directory root for the images to test (including subdirectories)')
   #parser.add_argument('--ap_score_thresh', default=0.75, type=float, help='AP score threshold for computing AP in COCO')
   
   #PATHS to weights
   parser.add_argument('--unet_weights_forward', default='model_weights/model_fw_50.pt', type=str, help='Path to forward UNet weights')
   parser.add_argument('--unet_weights_backward', default='model_weights/model_bw_80.pt', type=str, help='Path to backward UNet weights')
   parser.add_argument('--tasknet_weights_load', default='tasknet_weights/tasknet_10.pt', type=str, help='Path to Tasknet weights to load if resuming training or training the UNet')
   
   #FLAGS FOR CHANGING TRAINING BEHAVIOR
   parser.add_argument('--all_classes', action='store_true', default=False, help='If the experiments were executed with all classes, you need to set to True this flag to compute correct metric values and showing the labels in the plotted images')
   parser.add_argument('--five_classes', action='store_true', default=False, help='If the experiments were executed with all classes, using "cat dog horse sheep cow" as classes')
   parser.add_argument('--plot_only_bw_img', action='store_true', default=False, help='If you want to plot only the backward images')
   parser.add_argument('--plot_fw_along_bw', action='store_false', default=True, help='If you want to plot forward image (defined by unet_weights_forward, with tasknet prediction) with alongside the backward reconstruction model (defined by unet_weights_backward). If this flag is false, you should provide another weight in \'unet_weights_backward\' that is instead a forward weight')
   
   return parser

def main(args):
   seed_everything(args.seed)
   torch.use_deterministic_algorithms(mode=True, warn_only=True)
   plt.rcParams['figure.figsize'] = 15, 10
   #### Defining models. First UNet, then not modified Faster RCNN to simulate deployment
   #optimizer parameters and scheduler are just placeholder, not important as I just put the model in evaluate
   unet = UNet(3, False)
   unet.to(args.device)
   unet_optimizer = torch.optim.SGD(unet.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4, nesterov=True)
   unet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(unet_optimizer, mode='min', factor=0.5, patience=2)
   #Tasknet with same behavior of training. No normalize, no resize
   from faster_custom.faster_rcnn import fasterrcnn_resnet50_fpn_custom, FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor
   weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  
   tasknet = fasterrcnn_resnet50_fpn_custom(weights=weights, progress=False, use_resize=args.all_classes) #Default Tasknet
   if not args.all_classes:
      num_classes = 2  # 1 class (person) + background
      in_features = tasknet.roi_heads.box_predictor.cls_score.in_features
      tasknet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
   tasknet.to(args.device)
   tasknet_optimizer = torch.optim.SGD(tasknet.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)
   tasknet_scheduler = torch.optim.lr_scheduler.StepLR(tasknet_optimizer, step_size=3, gamma=0.1)
   #loading weights
   if not args.all_classes:
      load_checkpoint(tasknet, args.tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
   load_checkpoint(unet, args.unet_weights_forward, unet_optimizer, unet_scheduler)
   
   #Creating some paths and folders used later
   if os.path.exists(args.save_dir):
      shutil.rmtree(args.save_dir)
   os.makedirs(args.save_dir)
   
   custom_metric_file_list=[f'{args.results_dir}/iou0.5_score0.5.json', f'{args.results_dir}/iou0.5_score0.75.json', f'{args.results_dir}/iou0.75_score0.5.json', f'{args.results_dir}/iou0.75_score0.75.json']
   custom_metric_file_save_list=[f'{args.save_dir}/iou0.5_score0.5.png', f'{args.save_dir}/iou0.5_score0.75.png', f'{args.save_dir}/iou0.75_score0.5.png', f'{args.save_dir}/iou0.75_score0.75.png']
   extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
   image_name_list = [os.path.relpath(os.path.join(root, file), args.test_data_dir)
                   for root, dirs, files in os.walk(args.test_data_dir)
                   for file in files 
                   if file.lower().endswith(extensions)]
   
   #LOSS, MS_SSIM, LPIPS, MY RECONS CLASSIFIER, RECON REGRESSOR, CUSTOM METRIC
   from plot_utils.plot_similarity_metric import plot_sim_metric
   plot_model_loss(f'{args.results_dir}/loss_log.txt', f'{args.save_dir}/loss.png') #loss
   plot_sim_metric(f"{args.results_dir}/ms_ssim_score_log.txt", f'{args.save_dir}/ms_ssim_score.png', 'MS_SSIM score', 'MS_SSIM score Over Epochs')
   plot_sim_metric(f"{args.results_dir}/lpips_score_log.txt", f"{args.save_dir}/lpips_score.png", 'LPIPS score', 'LPIPS score Over Epochs')
   
   plot_custom_metric(custom_metric_file_list, custom_metric_file_save_list, args.all_classes, args.five_classes)
   
   #Contains Input AP Files
   ap_input_names = [f'{args.results_dir}/standard_ap.txt', f'{args.results_dir}/standard_ap_iou50.txt', f'{args.results_dir}/standard_ap_iou75.txt', f'{args.results_dir}/standard_ap_scoreabovethresh.txt', f'{args.results_dir}/standard_ap_scoreabovethresh_iou50.txt', f'{args.results_dir}/standard_ap_scoreabovethresh_iou75.txt']
   #Contains Output AP Files
   ap_output_names = [f'{args.save_dir}/standard_ap.png', f'{args.save_dir}/standard_ap_iou50.png', f'{args.save_dir}/standard_ap_iou75.png', f'{args.save_dir}/standard_ap_scoreabovethresh.png', f'{args.save_dir}/standard_ap_scoreabovethresh_iou50.png', f'{args.save_dir}/standard_ap_scoreabovethresh_iou75.png']
   #Contains Flag for extraction, to True if it is standard AP COCO
   std_ap = [True, True, True, False, False], False
   #Contains values for extraction, used to grab the right AP and AR with certain IoU threshold
   coco_iou_modified = [False, 50, 75, False, 50, 75]
   #Contains comparison AP value of Tasknet or any other model (you can compare with another UNet experiment if wanted)
   ap_compare_values = [0.875, 0.875, 0.647, 0.801, 0.801, 0.615]
   #Contains comparison AR value of Tasknet or any other model (you can compare with another UNet experiment if wanted)
   ar_compare_values = [0.682, 0.954, 0.731, 0.610, 0.824, 0.666]
   #Contains name of the model from which we get the above AP and AR values of comparison. Standard are all Tasknet
   compare_name = ['Tasknet', 'Tasknet', 'Tasknet', 'Tasknet', 'Tasknet', 'Tasknet']
   #Contains AP plot title based on the computed AP
   plot_title = ['(area=all, maxDets=100) AP (IoU=0.50), AR (IoU=0.50:0.95) Over Epochs', '(IoU=0.50, area=all, maxDets=100) AP, AR Over Epochs', '(IoU=0.75, area=all, maxDets=100) AP, AR Over Epochs', '(area=all, maxDets=100) AP (IoU=0.50), AR (IoU=0.50:0.95) Over Epochs', '(IoU=0.50, area=all, maxDets=100) AP, AR Over Epochs', '(IoU=0.75, area=all, maxDets=100) AP, AR Over Epochs']
   
   for inp, out, std_ap, coco_iou, ap_comp, ar_comp, comp_name, title in zip(ap_input_names, ap_output_names, std_ap, coco_iou_modified, ap_compare_values, ar_compare_values, compare_name, plot_title):
      extract_ap(inp, f'{args.results_dir}/temp.txt', standard_ap=std_ap, coco_iou_modified=coco_iou)
      plot_ap(f'{args.results_dir}/temp.txt', out, best_ap_value_for_comparison=ap_comp, best_recall_value_for_comparison=ar_comp, model_name=comp_name, ap_plot_title=title)
      if os.path.exists('{args.results_dir}/temp.txt'):
         os.remove('{args.results_dir}/temp.txt')
   
   #COMPARISON BETWEEN APs
   plot_compare_between_two_ap(f'{args.results_dir}/ext_standard_ap2.txt', f'{args.results_dir}/ext_standard_ap.txt', ap_model_name='Unet2', ap_to_compare_model_name='Unet', plotted_comparison_save_path=f'{args.save_dir}/compare_AP.png', ap_plot_title='(IoU=0.75, area=all, maxDets=100) AP, AR Over Epochs')
   if os.path.exists('{args.results_dir}/temp.txt'):
      os.remove('{args.results_dir}/temp.txt')
   
   #Image Samples plotting
   if args.plot_only_bw_img: #ACTUALLY DOESN'T PLOT BW IMAGES BUT RECONS ONE
      #show_res_test_unet(unet, tasknet, device, 'plot/val.jpg', True, 'plot/reconstructed_person.png')
      #load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
      load_checkpoint(unet, args.unet_weights_backward, unet_optimizer, unet_scheduler)
      for img in image_name_list:
         image_path=f'{args.test_data_dir}/{img}'
         image_save_name=f'{args.save_dir}/{img}'
         save_disturbed_pred(unet, args.device, image_path, image_save_name, args.unet_weights_forward, args.unet_weights_backward, unet_optimizer, unet_scheduler)
         plt.clf()

      if os.path.exists('temp_for_backward.jpg'):
         os.remove('temp_for_backward.jpg')

   else:
      for img in image_name_list:
         image_path=f'{args.test_data_dir}/{img}'
         image_save_name=f'{args.save_dir}/{img}'
         compare_two_results_unet(args.plot_fw_along_bw, unet, tasknet, args.device, image_path, image_save_name, args.unet_weights_forward, args.unet_weights_backward, unet_optimizer, unet_scheduler, args.all_classes, args.five_classes)
         plt.clf()
         
      if os.path.exists('temp_for_backward.jpg'):
         os.remove('temp_for_backward.jpg')
      
if __name__ == '__main__':
   parser = argparse.ArgumentParser('Plotting results script', parents=[get_args_parser()])
   args = parser.parse_args()
   main(args)
