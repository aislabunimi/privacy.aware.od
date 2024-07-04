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
   parser.add_argument('--results_dir', default='results', type=str, help='Directory root containing the results')
   parser.add_argument('--save_dir', default='plotted_results', type=str, help='Directory root for storing the plotted results, logs and so on. WARNING: this folder and its contents will be deleted before starting next plotting! Also, be aware of the folder you choose: you may delete your whole system!')
   parser.add_argument('--test_data_dir', default='test_data', type=str, help='Directory root for the images to test (including subdirectories)')
   parser.add_argument('--plot_my_test_data', action='store_true', default=False, help='If you want to test the model on your own data provided in the folder specified by test_data_dir parameter')
   parser.add_argument('--pascal_dataset', default='dataset/pascalVOC2012/images', type=str, help='Directory root for the images from pascal dataset to use as test')
   
   #PATHS to weights
   parser.add_argument('--unet_fw_weights_load', default='model_weights/model_fw_50.pt', type=str, help='Path to forward UNet weights')
   parser.add_argument('--unet_bw_weights_load', default='model_weights/model_bw_80.pt', type=str, help='Path to backward UNet weights')
   parser.add_argument('--tasknet_weights_load', default='tasknet_weights/tasknet_10_myresize.pt', type=str, help='Path to Tasknet weights to load if resuming training or training the UNet')
   
   #FLAGS FOR CHANGING TRAINING BEHAVIOR
   parser.add_argument('--all_classes', action='store_true', default=False, help='If the experiments were executed with all classes, you need to set to True this flag to compute correct metric values and showing the labels in the plotted images')
   parser.add_argument('--five_classes', action='store_true', default=False, help='If the experiments were executed with all classes, using "cat dog horse sheep cow" as classes')
   parser.add_argument('--plot_only_bw_img', action='store_true', default=False, help='If you want to plot only the backward images reconstructed by an attacker')
   parser.add_argument('--plot_two_unet', action='store_true', default=False, help='If you want to plot original image with tasknet preds, forward image (defined by unet_fw_weights_load, with tasknet prediction) with alongside the backward reconstruction model (defined by unet_fw_weights_load)')
   parser.add_argument('--plot_fw_along_bw', action='store_false', default=True, help='If you want to plot forward image (defined by unet_fw_weights_load, with tasknet prediction) with alongside the backward reconstruction model (defined by unet_fw_weights_load). If this flag is false, you should provide another weight in \'unet_bw_weights_load\' that is instead a forward weight')
   parser.add_argument('--plot_single_image', action='store_true', default=False, help='If you want to plot only the image alone')
   parser.add_argument('--plot_tasknet', action='store_true', default=False, help='Parameter used in combinations with plot_single_image. If you want to plot only the image with the prediction of the Tasknet')
   parser.add_argument('--plot_fw', action='store_true', default=False, help='Parameter used in combinations with plot_single_image. If you want to plot only the forward reconstructed image with the prediction of the Tasknet')
   parser.add_argument('--plot_bw', action='store_true', default=False, help='Parameter used in combinations with plot_single_image. If you want to plot only the backward reconstructed image')
   
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
      if args.five_classes:
         num_classes = 6
      else:
         num_classes = 2
      in_features = tasknet.roi_heads.box_predictor.cls_score.in_features
      tasknet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
   tasknet.to(args.device)
   tasknet_optimizer = torch.optim.SGD(tasknet.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005, nesterov=True)
   tasknet_scheduler = torch.optim.lr_scheduler.StepLR(tasknet_optimizer, step_size=3, gamma=0.1)
   #loading weights
   if not args.all_classes:
      load_checkpoint(tasknet, args.tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
   
   #Creating some paths and folders used later
   if os.path.exists(args.save_dir):
      shutil.rmtree(args.save_dir)
   os.makedirs(args.save_dir)
   
   custom_metric_file_list=[f'{args.results_dir}/iou0.5_score0.5.json', f'{args.results_dir}/iou0.5_score0.75.json', f'{args.results_dir}/iou0.75_score0.5.json', f'{args.results_dir}/iou0.75_score0.75.json']
   custom_metric_file_save_list=[f'{args.save_dir}/iou0.5_score0.5.png', f'{args.save_dir}/iou0.5_score0.75.png', f'{args.save_dir}/iou0.75_score0.5.png', f'{args.save_dir}/iou0.75_score0.75.png']
   
   extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
   if args.plot_my_test_data:
      data_dir = args.test_data_dir
      image_name_list = [os.path.relpath(os.path.join(root, file), data_dir)
                   for root, dirs, files in os.walk(data_dir)
                   for file in files 
                   if file.lower().endswith(extensions)]
   else:
      data_dir = args.pascal_dataset
      if args.five_classes:
         image_name_list = ['2007_000027.jpg', '2007_000272.jpg']
      else:
         image_name_list = ['2007_000027.jpg', '2007_000272.jpg', '2007_000323.jpg', '2007_000346.jpg', '2007_000480.jpg', '2007_002120.jpg', '2007_000733.jpg', '2007_000999.jpg', '2007_001185.jpg', '2007_001284.jpg', '2007_001583.jpg', '2007_001717.jpg', '2007_002142.jpg', '2007_002293.jpg', '2007_002624.jpg', '2007_003091.jpg', '2007_003541.jpg', '2007_003580.jpg', '2007_003581.jpg', '2007_003831.jpg', '2007_004000.jpg', '2007_004289.jpg', '2007_004476.jpg', '2008_000734.jpg', '2008_001092.jpg']
   
   #LOSS, MS_SSIM, LPIPS, MY RECONS CLASSIFIER, RECON REGRESSOR, CUSTOM METRIC
   from plot_utils.plot_similarity_metric import plot_sim_metric
   plot_model_loss(f'{args.results_dir}/loss_log.txt', f'{args.save_dir}/loss.png') #loss
   plot_sim_metric(f'{args.results_dir}/only_val_loss_log_batch1.txt', f'{args.save_dir}/loss.png', 'Loss', 'Loss Over Epochs')
   plot_sim_metric(f"{args.results_dir}/ms_ssim_score_log_batch1.txt", f'{args.save_dir}/ms_ssim_score.png', 'MS_SSIM score', 'MS_SSIM score Over Epochs')
   plot_sim_metric(f"{args.results_dir}/lpips_score_log_batch1.txt", f"{args.save_dir}/lpips_score.png", 'LPIPS score', 'LPIPS score Over Epochs')
   
   plot_custom_metric(custom_metric_file_list, custom_metric_file_save_list, args.all_classes, args.five_classes)
   
   #Contains Input AP Files
   ap_input_names = [f'{args.results_dir}/standard_ap.txt', f'{args.results_dir}/standard_ap_iou50.txt', f'{args.results_dir}/standard_ap_iou75.txt', f'{args.results_dir}/standard_ap_scoreabovethresh.txt', f'{args.results_dir}/standard_ap_scoreabovethresh_iou50.txt', f'{args.results_dir}/standard_ap_scoreabovethresh_iou75.txt']
   #Contains Output AP Files
   ap_output_names = [f'{args.save_dir}/standard_ap.png', f'{args.save_dir}/standard_ap_iou50.png', f'{args.save_dir}/standard_ap_iou75.png', f'{args.save_dir}/standard_ap_scoreabovethresh.png', f'{args.save_dir}/standard_ap_scoreabovethresh_iou50.png', f'{args.save_dir}/standard_ap_scoreabovethresh_iou75.png']
   #Contains Flag for extraction, to True if it is standard AP COCO
   std_ap = [True, True, True, False, False, False]
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
   #plot_compare_between_two_ap(f'{args.results_dir}/ext_standard_ap2.txt', f'{args.results_dir}/ext_standard_ap.txt', ap_model_name='Unet2', ap_to_compare_model_name='Unet', plotted_comparison_save_path=f'{args.save_dir}/compare_AP.png', ap_plot_title='(IoU=0.75, area=all, maxDets=100) AP, AR Over Epochs')
   #if os.path.exists('{args.results_dir}/temp.txt'):
   #   os.remove('{args.results_dir}/temp.txt')
   
   #Image Samples plotting
   if args.plot_only_bw_img: #PLOT RECONS IMAGES FROM FW ONE
      for img in image_name_list:
         image_path=f'{data_dir}/{img}'
         image_save_name=f'{args.save_dir}/{img}'
         save_disturbed_pred(unet, args.device, image_path, image_save_name, args.unet_fw_weights_load, args.unet_bw_weights_load, unet_optimizer, unet_scheduler)
         plt.clf()
      if os.path.exists('temp_for_backward.jpg'):
         os.remove('temp_for_backward.jpg')
   
   elif args.plot_single_image:
      for img in image_name_list:
         image_path=f'{data_dir}/{img}'
         image_save_name=f'{args.save_dir}/{img}'
         plot_single_image(args.plot_fw_along_bw, unet, tasknet, args.device, image_path, image_save_name, args.unet_fw_weights_load, args.unet_bw_weights_load, unet_optimizer, unet_scheduler, args.all_classes, args.five_classes, args.plot_tasknet, args.plot_fw, args.plot_bw)
         plt.clf()   
      if os.path.exists('temp_for_backward.jpg'):
         os.remove('temp_for_backward.jpg')
   
   elif args.plot_two_unet:
      for img in image_name_list:
         image_path=f'{data_dir}/{img}'
         image_save_name=f'{args.save_dir}/{img}'
         compare_two_results_unet(args.plot_fw_along_bw, unet, tasknet, args.device, image_path, image_save_name, args.unet_fw_weights_load, args.unet_bw_weights_load, unet_optimizer, unet_scheduler, args.all_classes, args.five_classes)
         plt.clf() 
      if os.path.exists('temp_for_backward.jpg'):
         os.remove('temp_for_backward.jpg')
      
if __name__ == '__main__':
   parser = argparse.ArgumentParser('Plotting results script', parents=[get_args_parser()])
   args = parser.parse_args()
   main(args)
