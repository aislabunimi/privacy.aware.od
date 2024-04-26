import numpy as np
import torchvision
import torch
import os
import shutil
import argparse
############ Import from my code
from unet_model import UNet
from dataset import *
from model_utils_and_functions import *
from train_and_test_loops import *
############
###### CONFIG

def get_args_parser():
   parser = argparse.ArgumentParser('Set parameters and config', add_help=False)
   parser.add_argument('--seed', default=0, type=int, help='Seed for experiments')
   parser.add_argument('--device', default='cuda', type=str, help='Device to use for experiments')
   
   #Config Plotting, Save files, Model Weights.
   parser.add_argument('--results_dir', default='results', type=str, help='Directory root for storing the results, logs and so on. WARNING: this folder and its contents will be deleted before starting next experiment, remember to backup the results! Also, be aware of the folder you choose: you may delete your whole system!')
   parser.add_argument('--ap_score_thresh', default=0.75, type=float, help='AP score threshold for computing AP in COCO')
   parser.add_argument('--unet_save_path', default='model_weights/model', type=str, help='Directory root and base name for storing the UNet weights. Expected to be in the form "folder_name/weights_base_name" without any file extension. WARNING: this folder and its contents will be deleted before starting next UNet training experiment, remember to backup the weights! Also, be aware of the folder you choose: you may delete your whole system!')
   parser.add_argument('--tasknet_save_path', default='tasknet_weights/tasknet', type=str, help='Directory root and base name for storing the Tasknet weights. Expected to be in the form "folder_name/weights_base_name" without any file extension. WARNING: this folder and its contents will be deleted before starting next Tasknet training experiment (the Tasknet weights for training the UNet could be left in this folder as soon as you don\'t retrain the Tasknet), remember to backup the weights! Also, be aware of the folder you choose: you may delete your whole system!')
   parser.add_argument('--unet_weights_load', default='model_weights/model_fw_50.pt', type=str, help='Path to UNet weights to load if resuming training')
   parser.add_argument('--save_all_weights', action='store_true', default=False, help='If you want to save model weights for each epoch. By default, this script saves weights at final epoch and at tot_epochs/2 (halfway checkpoint)')
   parser.add_argument('--tasknet_weights_load', default='tasknet_weights/tasknet_1norm_myresize.pt', type=str, help='Path to Tasknet weights to load if resuming training or training the UNet')
   parser.add_argument('--my_classifier_weights', default='my_recons_classifier/my_recons_classifier_weights.pt', type=str, help='Path to load pretrained reconstruction classifier (useless)')
   parser.add_argument('--my_regressor_weights', default='my_recons_classifier/my_regressor_weights.pt', type=str, help='Path to load pretrained reconstruction regressor (useless)')
   
   #Config Dataset. Paths to image folders and annotations. Paths to disturbed dataset and annotations.
   parser.add_argument('--use_openimages_for_disturbed_set', action='store_true', default=False, help='If you want to use openimages train dataset for generating disturbed training dataset instead of default COCO one')
   parser.add_argument('--coco_allpeople_path', default='/home/alberti/coco_person', type=str, help='Path of the folder containing all the people of the COCO dataset. The folder is expected to contain two subfolders, "train" and "val". Each one of these folders contains an "images" folder (with the images) and an annotation file called "train.json" and "val.json" respectively.')
   parser.add_argument('--coco_indoor_path', default='/home/alberti/coco_people_indoor', type=str, help='Path of the folder containing the COCO indoor dataset. The folder is expected to contain two subfolders, "train" and "val". Each one of these folders contains an "images" folder (with the images) and an annotation file called "train.json" and "val.json" respectively.')
   parser.add_argument('--disturbed_dataset_path', default='disturbed_dataset', type=str, help='Path of the folder used for containing the generated disturbed dataset. WARNING: this folder and its contents will be deleted before starting next backward training experiment, be aware of the folder you choose: you may delete your whole system!')
   parser.add_argument('--openimages_dataset_path', default='/home/alberti/open_images', type=str, help='Path of the folder containing the Open Images training dataset. This folder is expected to contain an "images" folder with the images, and an annotation filed named "open_images_id_list.json"')
   parser.add_argument('--use_dataset_subset', default=0, type=int, help='If you want to use a subset of data (0 is the default, whole dataset; n!=0 means you use n images)')
   
   #HYPERPARAMETERS
   #Num epochs and Batch Size
   parser.add_argument('--batch_size_tasknet', default=8, type=int, help='Batch size for experiments with Tasknet, default value 8')
   parser.add_argument('--batch_size_unet', default=4, type=int, help='Batch size for experiments with UNet, default value 4')
   parser.add_argument('--num_epochs_tasknet', default=10, type=int, help='Number of epochs for fine tuning Tasknet. The default value of 10 is more than enough for fine tuning')
   parser.add_argument('--num_epochs_unet_forward', default=50, type=int, help='Number of epochs for training the UNet forward (default value 50)')
   parser.add_argument('--num_epochs_unet_backward', default=80, type=int, help='Number of epochs for training the UNet backward (default value 100)')
   #Tasknet Hyperparameters
   parser.add_argument('--lr_tasknet', default=1e-3, type=float, help='Learning rate for Tasknet (SGD optimizer). Default 1e-3')
   parser.add_argument('--momentum_tasknet', default=0.9, type=float, help='Momentum for Tasknet (SGD optimizer). Default 0.9')
   parser.add_argument('--weight_decay_tasknet', default=5e-4, type=float, help='Weight decay for Tasknet (SGD optimizer). Default 5e-4')
   parser.add_argument('--no_nesterov_tasknet', default=True, action='store_false', help='By default Nesterov is active (SGD optimizer). If you pass this flag, nesterov will be disabled')
   parser.add_argument('--step_size_tasknet', default=3, type=int, help='Step Size for Tasknet (StepLR scheduler). Default 3')
   parser.add_argument('--gamma_tasknet', default=0.1, type=float, help='Gamma for Tasknet (StepLR scheduler). Default 0.1')
   #Unet Forward and Backward training Hyperparameters
   parser.add_argument('--lr_unet', default=5e-4, type=float, help='Learning rate for UNet (SGD optimizer). Default 5e-4')
   parser.add_argument('--momentum_unet', default=0.9, type=float, help='Momentum for UNet (SGD optimizer). Default 0.9')
   parser.add_argument('--weight_decay_unet', default=5e-4, type=float, help='Weight decay for UNet (SGD optimizer). Default 5e-4')
   parser.add_argument('--no_nesterov_unet', default=True, action='store_false', help='By default Nesterov is active (SGD optimizer). If you pass this flag, nesterov will be disabled')
   parser.add_argument('--factor_unet', default=0.5, type=float, help='Factor for UNet (ReduceLROnPlateau scheduler). Default 0.5')
   parser.add_argument('--patience_unet_forward', default=2, type=int, help='Patience for UNet (ReduceLROnPlateau scheduler). Default 2')
   parser.add_argument('--patience_unet_backward', default=4, type=int, help='Patience for UNet (ReduceLROnPlateau scheduler). Default 4 (should be higher for backward experiments as validation loss is more sensitive to small changes and we should avoid to reduce the lr too quickly)')
   
   #Faster R-CNN Weakining Loss Method Hyperparameters
   #Selecting Method
   parser.add_argument('--not_use_custom_filter_prop', default=False, action='store_true', help='By default, custom filtering proposal is active when training UNet. If you pass this flag, you can train the UNet using a plain Faster R-CNN')
   parser.add_argument('--filter_prop_objectness', default=True, action='store_false', help='By default, the proposal filtering method filters negative proposals based on objectness score obtained from RPN, as it is faster, performs better and don\'t require to forward all proposal for obtaining classification score (time expensive). If you pass this flag, the proposal filtering method will filter negative proposals based on classification score')
   parser.add_argument('--filter_anchors', action='store_true', default=False, help='By default, filtering anchors in similar way to proposal approach is disabled as it reduces performance and not influence much the reconstruction rate. If you pass this flag, anchors filtering will be done')
   #Anchors filtering settings
   parser.add_argument('--anc_pos', default=2, type=int, help='Number of RPN positive anchors to keep if filtering anchors method is used. Positive anchors are anchors with highest IoU with ground truth independently from IoU value or score. Default 2')
   parser.add_argument('--anc_neg', default=2, type=int, help='Number of RPN negative anchors to keep if filtering anchors method is used. Negative anchors are anchors with highest score but low IoU (below default Faster R-CNN threshold of 0.3) with ground truth. Default 2')
   parser.add_argument('--anc_bg', default=0, type=int, help='Number of RPN background anchors to keep if filtering anchors method is used. Background anchors are anchors with highest score but zero IoU with all ground truths. Default 0, as this idea is not used as it leads to much more reconstruction rate')
   parser.add_argument('--anc_bg_thresh', default=0.0, type=float, help='Threshold for objectness score for keeping only high scoring background anchors (for minimizing reconstruction rate). Default 0.0, as these are logits and not a score in the interval 0-1')
   #Proposals filtering settings
   parser.add_argument('--prop_pos', default=4, type=int, help='Number of positive proposals to keep if filtering proposal method is used. Positive proposals are proposals with highest IoU with ground truth independently from IoU value or score. Default 4')
   parser.add_argument('--prop_neg', default=4, type=int, help='Number of negative proposals to keep if filtering proposal method is used. Negative proposals are proposals with highest score but low IoU (below default Faster R-CNN threshold of 0.5) with ground truth. Default 4')
   parser.add_argument('--prop_bg', default=0, type=int, help='Number of background proposals to keep if filtering proposals method is used. Background proposals are proposals with highest score but zero IoU with all ground truths. Default 0, as this idea is not used as it leads to much more reconstruction rate')
   parser.add_argument('--prop_bg_thresh', default=0.9, type=float, help='Threshold for background score for keeping only high scoring background proposals (for minimizing reconstruction rate). Default 0.9: be aware that the value is expected to be between 0-1 if the proposal selection is based on objectness score; otherwise it can be set to higher value, as the proposal selection based on classification score uses logits and compares them to this threshold')
   #Proposals filtering settings if classification score method is used
   parser.add_argument('--n_prop_class_method', default=2000, type=int, help='Number of proposals to be kept after RPN. This parameter is used only if you use the filtering proposals based on classification score. Default is 2000, meaning all standard Faster R-CNN proposals are kept. You may want to decrease this value if you want to tradeoff the number of proposals with speed (e.g. 512 is similar to the standard training time)')
   
   #FLAGS FOR CHANGING TRAINING BEHAVIOR
   parser.add_argument('--train_tasknet', action='store_true', default=False, help='If you want to first train only the Tasknet; if False the code assumes you want to train the UNet (forward or backward) with freezed Tasknet')
   parser.add_argument('--tasknet_get_indoor_AP', action='store_true', default=False, help='Needed for getting the AP for comparison with UNet')
   parser.add_argument('--save_disturbed_dataset', action='store_true', default=False, help='To use for creating the disturbed dataset for Backward training')
   parser.add_argument('--keep_original_size', action='store_true', default=False, help='Must be True if you want to fine tune the Tasknet on the disturbed COCO set; can be false for Backward training. This is done to enforce the disturbed images to have exact same size of original ones')
   parser.add_argument('--train_model_backward', action='store_true', default=False, help='For executing Backward training on disturbed dataset')
   parser.add_argument('--compute_similarity_metrics', action='store_true', default=False, help='Needed for getting the right similarities metrics')
   parser.add_argument('--resume_training', action='store_true', default=False, help='For resuming training by restoring the specified weights from "unet_weights_load" or "tasknet_weights_load" variable')
   
   return parser


def main(args):
   seed_everything(args.seed) #Deterministic experiments
   torch.use_deterministic_algorithms(mode=True, warn_only=True) #To force CUDA deterministic algorithms if possible
   
   if args.use_openimages_for_disturbed_set:
      disturbed_train_img_gen = f'{args.openimages_dataset_path}/images'
      disturbed_train_ann_gen = f'{args.openimages_dataset_path}/open_images_id_list.json'
   else:
      disturbed_train_img_gen = f'{args.coco_indoor_path}/train/images'
      disturbed_train_ann_gen = f'{args.coco_indoor_path}/train/train.json'
   disturbed_train_img_folder = f'{args.disturbed_dataset_path}/train'
   disturbed_train_ann = f'{args.disturbed_dataset_path}/train.json'
   disturbed_val_img_folder = f'{args.disturbed_dataset_path}/val'
   disturbed_val_ann = f'{args.disturbed_dataset_path}/val.json'
   if args.train_tasknet or args.tasknet_get_indoor_AP:
      train_batch_size = val_batch_size = args.batch_size_tasknet
      train_img_folder = f'{args.coco_allpeople_path}/train/images' 
      train_ann_file = f'{args.coco_allpeople_path}/train/train.json'
      val_img_folder = f'{args.coco_allpeople_path}/val/images'
      val_ann_file = f'{args.coco_allpeople_path}/val/val.json'
      #resize_scales_transform = [200, 300, 400, 500, 600]
      resize_scales_transform = [256, 288, 320, 352, 384, 416]
      #For having comparable results with UNet, it's best if you execute a validation epoch with Indoor dataset
      if args.tasknet_get_indoor_AP:
         resize_scales_transform = [256] #same resize used for UNet AP 
         val_img_folder = f'{args.coco_indoor_path}/val/images'
         val_ann_file = f'{args.coco_indoor_path}/val/val.json'
   else:
      train_batch_size = val_batch_size = args.batch_size_unet
      train_img_folder = f'{args.coco_indoor_path}/train/images'
      open_train_img_folder = f'{args.openimages_dataset_path}/images'
      train_ann_file = f'{args.coco_indoor_path}/train/train.json'
      val_img_folder = f'{args.coco_indoor_path}/val/images'
      val_ann_file = f'{args.coco_indoor_path}/val/val.json'
      resize_scales_transform = [256, 288, 320, 352, 384, 416]
      
   if not args.train_tasknet:
      unet = UNet(n_channels=3, bilinear=False) #UNet modified without skip connection
      unet.to(args.device)
      unet_optimizer = torch.optim.SGD(unet.parameters(), lr=args.lr_unet, momentum=args.momentum_unet, weight_decay=args.weight_decay_unet, nesterov=args.no_nesterov_unet)
      #unet_scheduler = torch.optim.lr_scheduler.StepLR(unet_optimizer, step_size=10, gamma=0.5)
      if args.train_model_backward:
         unet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(unet_optimizer, mode='min', factor=args.factor_unet, patience=args.patience_unet_backward)
      else:
         unet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(unet_optimizer, mode='min', factor=args.factor_unet, patience=args.patience_unet_forward)
   
   if args.save_disturbed_dataset: #If it's True, we save the dataset and exit the script.
      train_dataloader_gen_disturbed, val_dataloader_gen_disturbed = load_dataset_for_generating_disturbed_set(
         disturbed_train_img_gen, disturbed_train_ann_gen, val_img_folder, val_ann_file,
         args.use_dataset_subset, args.use_openimages_for_disturbed_set, resize_scales_transform)
      if os.path.exists(args.disturbed_dataset_path):
         shutil.rmtree(args.disturbed_dataset_path)
      os.makedirs(disturbed_train_img_folder)
      os.makedirs(disturbed_val_img_folder)
      load_checkpoint(unet, args.unet_weights_load, unet_optimizer, unet_scheduler)
      generate_disturbed_dataset(train_dataloader_gen_disturbed, val_dataloader_gen_disturbed, args.device, unet,
         disturbed_train_img_folder, disturbed_train_ann, disturbed_val_img_folder, disturbed_val_ann, 
         args.keep_original_size, args.use_openimages_for_disturbed_set)
      print("Generated disturbed dataset")
      sys.exit()
         
   if not args.train_model_backward: #Modified Tasknet with custom proposal method
      from faster_custom.faster_rcnn import fasterrcnn_resnet50_fpn_custom, FasterRCNN_ResNet50_FPN_Weights, FastRCNNPredictor
      weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT  
      if args.train_tasknet or args.tasknet_get_indoor_AP or args.not_use_custom_filter_prop:
         tasknet = fasterrcnn_resnet50_fpn_custom(weights=weights, progress=False) #Default Tasknet
      else:
         if args.filter_prop_objectness:
            tasknet = fasterrcnn_resnet50_fpn_custom(weights=weights, progress=False, 
               rpn_use_custom_filter_anchors=args.filter_anchors, rpn_n_top_pos_to_keep=args.anc_pos, rpn_n_top_neg_to_keep=args.anc_neg,
               rpn_n_top_bg_to_keep=args.anc_bg, rpn_objectness_bg_thresh=args.anc_bg_thresh, box_use_custom_filter_proposals_objectness=True, 
               box_n_top_pos_to_keep=args.prop_pos, box_n_top_neg_to_keep=args.prop_neg, box_n_top_bg_to_keep=args.prop_bg, box_obj_bg_score_thresh=args.prop_bg_thresh, box_batch_size_per_image=100000) #100000 for be sure that sampler keep all proposals, as we want to use all of them in custom filtering selection
               #rpn_objectness_bg_thresh is set to 0, as it's objectness score (so 0 is already a decent high confidence that the anchor contains an object.
         else: #Based on score, slower and not necessarily better resutls
            tasknet = fasterrcnn_resnet50_fpn_custom(weights=weights, progress=False,
               rpn_post_nms_top_n_train=args.n_prop_class_method,
               rpn_use_custom_filter_anchors=args.filter_anchors, rpn_n_top_pos_to_keep=args.anc_pos, rpn_n_top_neg_to_keep=args.anc_neg,
               rpn_n_top_bg_to_keep=args.anc_bg, rpn_objectness_bg_thresh=args.anc_bg_thresh, box_use_custom_filter_proposals_scores=True, 
               box_n_top_pos_to_keep=args.prop_pos, box_n_top_neg_to_keep=args.prop_neg, box_n_top_bg_to_keep=args.prop_bg, box_obj_bg_score_thresh=args.prop_bg_thresh,
               box_batch_size_per_image=100000) #100000 for be sure that sampler keep all proposals, as we want to use all of them in custom filtering selection
      num_classes = 2
      in_features = tasknet.roi_heads.box_predictor.cls_score.in_features
      tasknet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
      tasknet.to(args.device)
      tasknet_optimizer = torch.optim.SGD(tasknet.parameters(), lr=args.lr_tasknet, momentum=args.momentum_tasknet, weight_decay=args.weight_decay_tasknet, nesterov=args.no_nesterov_tasknet)
      tasknet_scheduler = torch.optim.lr_scheduler.StepLR(tasknet_optimizer, step_size=args.step_size_tasknet, gamma=args.gamma_tasknet)
   
   if args.train_model_backward: #Disturbed dataset are different
      if args.use_openimages_for_disturbed_set:
         orig_train_img_folder = open_train_img_folder
      else:
         orig_train_img_folder = train_img_folder
      disturbed_train_dataloader, disturbed_val_dataloader, example_dataloader = load_disturbed_dataset(
         disturbed_train_img_folder, disturbed_train_ann, disturbed_val_img_folder, disturbed_val_ann, 
         orig_train_img_folder, val_img_folder, train_batch_size, val_batch_size, resize_scales_transform, 
         args.use_dataset_subset, val_ann_file)
      _, disturbed_val_batch1, _ = load_disturbed_dataset(
         disturbed_train_img_folder, disturbed_train_ann, disturbed_val_img_folder, disturbed_val_ann, 
         orig_train_img_folder, val_img_folder, 1, 1, resize_scales_transform, 
         args.use_dataset_subset, val_ann_file)
   else: #Same datasets for training Tasknet and training Unet forward
      tasknet_setting = args.train_tasknet or args.tasknet_get_indoor_AP
      train_dataloader, val_dataloader, example_dataloader= load_dataset(
         train_img_folder, train_ann_file, val_img_folder, val_ann_file, train_batch_size, val_batch_size,
         args.save_disturbed_dataset, tasknet_setting, resize_scales_transform, args.use_dataset_subset)
         
   if not os.path.exists(args.results_dir):
      os.makedirs(args.results_dir)
   unet_weights_dir = args.unet_save_path.split('/')[0]
   if not os.path.exists(unet_weights_dir):
      os.makedirs(unet_weights_dir)
   tasknet_weights_dir = args.tasknet_save_path.split('/')[0]
   if not os.path.exists(tasknet_weights_dir):
      os.makedirs(tasknet_weights_dir)
   
   starting_epoch=1 #Used for counting epoch from 1
   if args.resume_training:
      if args.train_tasknet: #+1 to start at next epoch
         starting_epoch = load_checkpoint(tasknet, args.tasknet_weights_load, tasknet_optimizer, tasknet_scheduler) + 1
      else:
         starting_epoch = load_checkpoint(unet, args.unet_weights_load, unet_optimizer, unet_scheduler) + 1
         #load_checkpoint_encoder(unet, args.unet_weights_load, unet_optimizer, unet_scheduler, load_optim_scheduler=False)
         #load_checkpoint_decoder(unet, args.unet_weights_load, unet_optimizer, unet_scheduler, load_optim_scheduler=False)
         #freeze_encoder(unet)
         #freeze_decoder(unet)
   else: #Remove results folder as it's from old experiment
      shutil.rmtree(args.results_dir)
      os.makedirs(args.results_dir)
      shutil.rmtree(unet_weights_dir)
      os.makedirs(unet_weights_dir)
      if args.train_tasknet and not args.tasknet_get_indoor_AP: #otherwise I might remove tasknet weights stored in this folders used for training the UNet
         shutil.rmtree(tasknet_weights_dir)
         os.makedirs(tasknet_weights_dir)
      
   #Models used for measuring reconstruction
   if not args.train_tasknet:
      my_recons_classifier = load_my_recons_classifier(args.my_classifier_weights, args.device)
      my_regressor = load_my_regressor(args.my_regressor_weights, args.device)
      from lpips.lpips import LPIPS
      lpips_model = LPIPS().to(args.device) #LPIPS(net='vgg').to(device) #vgg best if i backpropagate with LPIPS loss; I want to use it only as metric, so Alex is faster and perform better
      lpips_model.eval()
      from pytorch_msssim import ms_ssim, MS_SSIM
      ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3) #for backward standard MS_SSIM works well
      #ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], K=(0.01, 0.07)) #older approach, but discarded for forward
      #size average is a reduction to average the MS SSIM of images in batch.
      #With k2=0.07 we avoid cases where MS SSIM doesn't perform well (with default 0.03 values are more unstable with forward; with backward we don't have this problem as much)
   
   if args.compute_similarity_metrics:
      similarity_gen_dataloader = load_similarity_dataset(
         disturbed_val_img_folder, disturbed_val_ann, None, resize_scales_transform, args.use_dataset_subset, generate_similarity_dataset=True)
      disturbed_val_img_folder = f'temp_dir/val'
      disturbed_val_ann = f'temp_dir/val.json' 
      if os.path.exists('temp_dir'):
         shutil.rmtree('temp_dir')
      os.makedirs(disturbed_val_img_folder)
      load_checkpoint(unet, args.unet_weights_load, unet_optimizer, unet_scheduler) #load unet backward weights
      generate_similarity_dataset(similarity_gen_dataloader, args.device, unet, disturbed_val_img_folder, disturbed_val_ann)
      print("Generated similarity dataset")
      similarity_dataloader = load_similarity_dataset(
         disturbed_val_img_folder, disturbed_val_ann, val_img_folder, resize_scales_transform, args.use_dataset_subset, generate_similarity_dataset=False)     
      val_similarity_disturbed_images(similarity_dataloader, args.device, args.results_dir, lpips_model, ms_ssim_module)
      print("Computed similarity metrics")
      shutil.rmtree('temp_dir')
      sys.exit()
      
   #TRAINING TASKNET BLOCK
   if((args.train_tasknet or args.tasknet_get_indoor_AP) and not args.train_model_backward):
      if args.tasknet_get_indoor_AP:
         load_checkpoint(tasknet, args.tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
         val_temp_loss = val_tasknet(val_dataloader, starting_epoch, args.device, args.tasknet_save_path, tasknet, tasknet_optimizer,
            tasknet_scheduler, args.ap_score_thresh, args.results_dir, args.num_epochs_tasknet, args.save_all_weights, skip_saving_weights=True)
         print("Computed AP for comparison with UNet indoor set")
         sys.exit()
      for epoch in range(starting_epoch, args.num_epochs_tasknet+1):
         train_temp_loss = train_tasknet(train_dataloader, epoch, args.device, tasknet_save_path, tasknet, tasknet_optimizer)
         val_temp_loss = val_tasknet(val_dataloader, epoch, args.device, args.tasknet_save_path, tasknet, tasknet_optimizer,
            tasknet_scheduler, args.ap_score_thresh, args.results_dir, args.num_epochs_tasknet, args.save_all_weights, skip_saving_weights=False)
         tasknet_scheduler.step()
         #tasknet_scheduler.step(val_temp_loss)
         print(f'EPOCH {epoch} SUMMARY: Train loss {train_temp_loss}, Val loss {val_temp_loss}')
         with open(f'{args.results_dir}/loss_log.txt', 'a') as file:
            loss_log_append = f"{epoch} {train_temp_loss} {val_temp_loss}\n"
            file.write(loss_log_append)
   
   #TRAINING MODEL BLOCK
   elif (not args.train_model_backward):
      load_checkpoint(tasknet, args.tasknet_weights_load, tasknet_optimizer, tasknet_scheduler)
      for param in tasknet.parameters(): #Freeze layers as Faster R-CNN is not modifiable
         param.requires_grad = False
      for epoch in range(starting_epoch, args.num_epochs_unet_forward+1):
         train_temp_loss = train_model(train_dataloader, epoch, args.device, unet, tasknet, unet_optimizer)
         val_temp_loss = val_model(val_dataloader, epoch, args.device, unet, args.unet_save_path, tasknet, unet_optimizer,
            unet_scheduler, args.ap_score_thresh, args.results_dir, args.num_epochs_unet_forward, args.save_all_weights, my_recons_classifier, my_regressor, 
            lpips_model, ms_ssim_module, example_dataloader)
         #unet_scheduler.step()
         unet_scheduler.step(val_temp_loss)
         print(f'EPOCH {epoch} SUMMARY: Train loss {train_temp_loss}, Val loss {val_temp_loss}')
         with open(f'{args.results_dir}/loss_log.txt', 'a') as file:
            loss_log_append = f"{epoch} {train_temp_loss} {val_temp_loss}\n"
            file.write(loss_log_append)
   
   #TRAINING BACKWARD BLOCK 			
   else:
      for epoch in range(starting_epoch, args.num_epochs_unet_backward+1):
         train_temp_loss = train_model_on_disturbed_images(disturbed_train_dataloader, epoch, args.device, unet, unet_optimizer, ms_ssim_module)
         val_temp_loss = val_model_on_disturbed_images(disturbed_val_dataloader, epoch, args.device, unet, args.unet_save_path,
            unet_optimizer, unet_scheduler, args.results_dir, args.num_epochs_unet_backward, args.save_all_weights, lpips_model, ms_ssim_module, example_dataloader)
         #unet_scheduler.step()
         unet_scheduler.step(val_temp_loss)
         _ = val_model_on_disturbed_images(disturbed_val_batch1, epoch, args.device, unet, args.unet_save_path,
            unet_optimizer, unet_scheduler, args.results_dir, args.num_epochs_unet_backward, args.save_all_weights, lpips_model, ms_ssim_module, example_dataloader, compute_right_similarity_metrics=True) #This is done to grab the real similarity metrics without the added padding for batch size > 1: otherwise I get higher similarity score just because of the black padding
         print(f'EPOCH {epoch} SUMMARY: Train loss {train_temp_loss}, Val loss {val_temp_loss}')
         with open(f'{args.results_dir}/loss_log.txt', 'a') as file:
            loss_log_append = f"{epoch} {train_temp_loss} {val_temp_loss}\n"
            file.write(loss_log_append)
            
   print("Done!")

if __name__ == '__main__':
   parser = argparse.ArgumentParser('Training and Evaluation script', parents=[get_args_parser()])
   args = parser.parse_args()
   main(args)
