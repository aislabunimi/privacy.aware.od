import torch
from tqdm import tqdm
from model_utils_and_functions import create_checkpoint, compute_ap, apply_nms, compute_custom_metric
import numpy as np
from custom_metric.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
from model_utils_and_functions import compute_my_recons_classifier_pred, save_my_recons_classifier_dict
#import for saving disturbed set
from torchvision.utils import save_image
import json
import torchvision.transforms as transforms
from coco_eval import get_coco_api_from_dataset
import os

def adjust_orig_target_to_reconstructed_imgs(targets, imgs, reconstructed):
   """ The reconstructed image due to the absence of skip connection is not of the same size of original, but lose some pixel. e.g. before unet ,shape [200, 250]. After: [192, 256].
   We need to resize the bbox target to avoid losing AP. I prefer to modify the bbox target instead of image size.
   """
   orig_batch_w, orig_batch_h = imgs.shape[3], imgs.shape[2]
   rec_batch_w, rec_batch_h =  reconstructed.shape[3], reconstructed.shape[2]
   for e in targets:
      for i, box in enumerate(e['boxes']):
         xm, ym, xM, yM = box
         #x_rec : rec_w = x_res : or_w
         rec_xm = (rec_batch_w * xm) / orig_batch_w
         rec_xM = (rec_batch_w * xM) / orig_batch_w
         rec_ym = (rec_batch_h * ym) / orig_batch_h
         rec_yM = (rec_batch_h * yM) / orig_batch_h
         e['boxes'][i] = torch.tensor([rec_xm, rec_ym, rec_xM, rec_yM], device=box.device)
   return targets

def adjust_outputs_to_cocoeval_api(targets, outputs, reconstructed=None):
   for t, pred in zip(targets, outputs):
      orig_h, orig_w = t['orig_size'] #original size from annotation, indipendent on previous modfication
      if reconstructed is not None: #if not None i am using the UNet
         rec_batch_h_pad, rec_batch_w_pad = reconstructed.shape[2], reconstructed.shape[3] #pred on image without some pixel
         rec_batch_h_img, rec_batch_w_img = t['size'] #original size of image without padding
         """ I take the smallest one; it's needed to resize the bbox w.r.t. the img
         e.g: rec_batch_pad = 288 304    rec_batch_img = 297, 256
         Tasknet makes the prediction on rec_batch_padded as size; I resize target w.r.t. to this. 
         I also need to resize without padding to avoid smaller targets with wrong coordiantes when rescaled; hence the use of min rec_batch_img
         """
         rec_batch_h = min(rec_batch_h_pad, rec_batch_h_img)
         rec_batch_w = min(rec_batch_w_pad, rec_batch_w_img)
      else: #pred made only by tasknet
         rec_batch_h, rec_batch_w = t['size']
      for i, box in enumerate(pred['boxes']):
         #x_rec : rec_w = x_res : or_w
         xm, ym, xM, yM = box
         resized_xm = (orig_w * xm) / rec_batch_w
         resized_xM = (orig_w * xM) / rec_batch_w
         resized_ym = (orig_h * ym) / rec_batch_h
         resized_yM = (orig_h * yM) / rec_batch_h
         pred['boxes'][i] = torch.tensor([resized_xm, resized_ym, resized_xM, resized_yM], device=box.device)
   return outputs

def train_model(train_dataloader, epoch, device, model, tasknet, model_optimizer): #Train model
   model.train()
   tasknet.train()
   batch_size = len(train_dataloader)
   running_loss = 0
   for imgs, targets in tqdm(train_dataloader, desc=f'Epoch {epoch} - Train model'):
      imgs, _ = imgs.decompose()
      imgs = imgs.to(device)
      targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
      #print_batch_after_collatefn(imgs, targets, 2) #Debug only, with DECOMPOSE		
      tasknet.train()
      reconstructed = model(imgs)
      
      targets = adjust_orig_target_to_reconstructed_imgs(targets, imgs, reconstructed) #needed as explained before
      """
      import matplotlib.pyplot as plt
      import matplotlib.image as mpimg
      mean = torch.tensor([0.485, 0.456, 0.406])
      std = torch.tensor([0.229, 0.224, 0.225])
      unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
      plt.imshow(unnormalize(reconstructed[0]).detach().cpu().permute(1, 2, 0).numpy())
      ax = plt.gca()
      xmin, ymin, xmax, ymax = targets[0]['boxes'][0]
      xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
      ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
      plt.show()
      plt.clf()
      """
      reconstructed_loss_dict = tasknet(reconstructed, targets) #grab tasknet losses
      reconstructed_losses = sum(loss for loss in reconstructed_loss_dict.values()) 
      
      true_loss=reconstructed_losses
      model_optimizer.zero_grad(set_to_none=True) #backpropagation
      true_loss.backward()
      model_optimizer.step()		
      running_loss += true_loss.item() #storing value
      
   running_loss /= batch_size #average loss
   return running_loss


def val_model(val_dataloader, epoch, device, model, model_save_path, tasknet, model_optimizer, model_scheduler, ap_score_threshold, results_dir, my_recons_classifier, my_regressor, lpips_model, ms_ssim_module, example_dataloader):
   model.eval()
   batch_size = len(val_dataloader)
   res={} #dictionary to store all prediction for AP
   evaluator_complete_metric = MyEvaluatorCompleteMetric() #for custom metric
   my_rec_class_dict = {}
   my_rec_class_dict['epoch']=epoch
   my_rec_class_dict['total']=0
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device) #for ms ssim
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   recon_rate=0 #for regressor
   lpips_score = 0 #for lpips
   ms_ssim_score = 0 #for msssim
   running_loss = 0 
   with torch.no_grad(): #not computing gradient
      for imgs, targets in tqdm(val_dataloader, desc=f'Epoch {epoch} - Validating model'):
         imgs, _ = imgs.decompose()
         imgs = imgs.to(device)
         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
         reconstructed = model(imgs)
         tasknet.train()
         
         targets = adjust_orig_target_to_reconstructed_imgs(targets, imgs, reconstructed)
         
         reconstructed_loss_dict = tasknet(reconstructed, targets)
         reconstructed_losses = sum(loss for loss in reconstructed_loss_dict.values())		
         tasknet.eval()
         outputs = tasknet(reconstructed)
         outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
         outputs_evaluator = [{k: v.clone().to(device) for k, v in t.items()} for t in outputs] #need to clone them as they will be modified after
         
         # in validation targets need to be converted to original size, otherwise coco ap is wrong i target devono poi essere riconvertiti alle dim originali, se no l'AP non va!
         adj_outputs = adjust_outputs_to_cocoeval_api(targets, outputs, reconstructed)
         
         res.update({target["image_id"].item(): output for target, output in zip(targets, adj_outputs)})
         preds = [apply_nms(pred, iou_thresh=0.5, score_thresh=0.01) for pred in outputs_evaluator]
         evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=imgs.size()[2:][::-1])
         """
         import matplotlib.pyplot as plt
         import matplotlib.image as mpimg
         plt.imshow(unnormalize(reconstructed[1]).detach().cpu().permute(1, 2, 0).numpy())
         ax = plt.gca()
         xmin, ymin, xmax, ymax = outputs[1]['boxes'][0]
         xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
         plt.show()
         plt.clf()
         """
         
         trans = transforms.Resize((reconstructed.shape[2], reconstructed.shape[3]), antialias=False)
         orig_imgs = trans(imgs)
         reconstructed_ms=unnormalize(reconstructed)
         orig_imgs_ms = unnormalize(orig_imgs)
         #clamping to avoid small negative values related to not perfect unnormalize
         reconstructed_ms = torch.clamp(reconstructed_ms, min=0, max=1)
         orig_imgs_ms = torch.clamp(orig_imgs_ms, min=0, max=1)
         ms_ssim_score += ms_ssim_module(reconstructed_ms, orig_imgs_ms).item()
         trans_te = transforms.Resize((64, 64), antialias=False)
         reconstructed = trans_te(reconstructed)
         orig_imgs = trans_te(imgs)
         lpips_loss = lpips_model(reconstructed, orig_imgs)
         lpips_score += (torch.mean(lpips_loss)).item()
         
         #Before passing reconstructed, input needs to be normalized as ImageNet and resized to 64x64
         compute_my_recons_classifier_pred(my_recons_classifier, reconstructed, my_rec_class_dict)			
         recon_rate += torch.mean(my_regressor(reconstructed)).item()
         
         true_loss=reconstructed_losses
         running_loss += true_loss.item()	
   running_loss /= batch_size #the val loss if custom proposal is active is computed w.r.t. the custom method, 
   #to remain consistent and show meaningful loss for understanding if training is going in the right way
   compute_ap(val_dataloader, tasknet, epoch, device, ap_score_threshold, results_dir, res)
   compute_custom_metric(evaluator_complete_metric, results_dir, epoch)
   ms_ssim_score /= batch_size
   ms_ssim_path= f"{results_dir}/ms_ssim_score_log.txt"
   with open(ms_ssim_path, 'a') as file:
      file.write(f"{epoch} {ms_ssim_score}\n")
   lpips_score /= batch_size
   lpips_path = f"{results_dir}/lpips_score_log.txt"
   with open(lpips_path, 'a') as file:
      file.write(f"{epoch} {lpips_score}\n")
   recon_rate /= batch_size
   recon_path= f"{results_dir}/recon_rate_log.txt"
   with open(recon_path, 'a') as file:
      file.write(f"{epoch} {recon_rate}\n")
	
   my_recons_classifier_path = f"{results_dir}/my_recons_classifier_log.json"
   save_my_recons_classifier_dict(my_recons_classifier_path, epoch, my_rec_class_dict)
	
   model_save_path = f'{model_save_path}_{epoch}.pt'   
   create_checkpoint(model, model_optimizer, epoch, running_loss, model_scheduler, model_save_path)
   save_image_examples(example_dataloader, results_dir, model, epoch, device)
   return running_loss

def save_image_examples(example_dataloader, results_dir, model, epoch, device):
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   model.eval()
   coco = get_coco_api_from_dataset(example_dataloader.dataset)
   with torch.no_grad(): #non calcolo il gradiente, sto testando e bassa
      for imgs, targets in tqdm(example_dataloader, desc=f'Epoch {epoch} - Saving Validation examples'):
         imgs, _ = imgs.decompose()
         imgs = imgs.to(device)
         reconstructed = model(imgs)
         coco_image_id = targets[0]["image_id"].item()
         path = coco.loadImgs(coco_image_id)[0]["file_name"]
         reconstructed = unnormalize(reconstructed)
         reconstructed = torch.clamp(reconstructed, min=0, max=1)
         save_dir_path = f'{results_dir}/epoch_{epoch}'
         if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
         save_image(reconstructed, os.path.join(save_dir_path, path))

def generate_disturbed_dataset(train_dataloader_gen_disturbed, val_dataloader_gen_disturbed, device, model, train_img_folder, train_ann, val_img_folder, val_ann, keep_original_size, use_coco_train): #for generating disturbed set
   model.eval()
   batch_size = len(train_dataloader_gen_disturbed) #first disturbed trainset
   if use_coco_train:
      coco = get_coco_api_from_dataset(train_dataloader_gen_disturbed.dataset)
   disturbed_list = []
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   with torch.no_grad():
      for imgs, targets in tqdm(train_dataloader_gen_disturbed, desc=f'Generating disturbed train images from dataset'):
         imgs, _ = imgs.decompose()
         imgs = imgs.to(device)
         reconstructed = model(imgs)
         if use_coco_train:
            coco_image_id = targets[0]["image_id"].item()
            path = coco.loadImgs(coco_image_id)[0]["file_name"]
            if keep_original_size: #for fine tune Tasknet on disturbed set; less meaningful for backward
               orig_size = [imgs.shape[2], imgs.shape[3]]
               trans = transforms.Resize(orig_size, antialias=False)
               reconstructed = trans(reconstructed)
            reconstructed = unnormalize(reconstructed) #unnormalized and clamping needed before storing images in [0,1]
            reconstructed = torch.clamp(reconstructed, min=0, max=1)
            save_image(reconstructed, os.path.join(train_img_folder, path))
            disturbed_list.append({"image_path": path,"coco_image_id": coco_image_id})
         else:
            path = ''.join(targets)
            if keep_original_size:
               orig_size = [imgs.shape[2], imgs.shape[3]]
               trans = transforms.Resize(orig_size, antialias=False)
               reconstructed = trans(reconstructed)
            reconstructed = unnormalize(reconstructed)
            reconstructed = torch.clamp(reconstructed, min=0, max=1)
            save_image(reconstructed, os.path.join(train_img_folder, path))
            disturbed_list.append({"image_path": path,})
   with open(train_ann, 'w') as json_file:
      json.dump(disturbed_list, json_file, indent=2)
   batch_size = len(val_dataloader_gen_disturbed) #now disturbed valset
   coco = get_coco_api_from_dataset(val_dataloader_gen_disturbed.dataset)
   disturbed_list = []
   with torch.no_grad():
      for imgs, targets in tqdm(val_dataloader_gen_disturbed, desc=f'Generating disturbed val images from dataset'):
         imgs, _ = imgs.decompose()
         imgs = imgs.to(device)
         reconstructed = model(imgs)
         coco_image_id = targets[0]["image_id"].item()
         path = coco.loadImgs(coco_image_id)[0]["file_name"]
         if keep_original_size:
            orig_size = [imgs.shape[2], imgs.shape[3]]
            trans = transforms.Resize(orig_size, antialias=False)
            reconstructed = trans(reconstructed)
         reconstructed = unnormalize(reconstructed)
         reconstructed = torch.clamp(reconstructed, min=0, max=1)
         save_image(reconstructed, os.path.join(val_img_folder, path))
         disturbed_list.append({"image_path": path, "coco_image_id": coco_image_id})
   with open(val_ann, 'w') as json_file:
      json.dump(disturbed_list, json_file, indent=2)
   model.train()


def train_model_on_disturbed_images(train_dataloader, epoch, device, model, model_optimizer, loss_fn): #train model backward
   model.train()
   batch_size = len(train_dataloader)
   running_loss = 0
   for disturbed_imgs, orig_imgs in tqdm(train_dataloader, desc=f'Epoch {epoch} - Train model'):
      disturbed_imgs, _ = disturbed_imgs.decompose()
      disturbed_imgs = disturbed_imgs.to(device)
      orig_imgs, _ = orig_imgs.decompose()
      orig_imgs = orig_imgs.to(device)		
      
      """ # For testing
      mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
      std = torch.tensor([0.229, 0.224, 0.225]).to(device)
      unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
      import matplotlib.pyplot as plt
      import matplotlib.image as mpimg
      disturbed_imgs_plot = unnormalize(disturbed_imgs[0])
      plt.imshow(disturbed_imgs_plot.cpu().permute(1, 2, 0))
      plt.title(disturbed_imgs)
      plt.axis('off')
      plt.show()
      orig_imgs_plot = unnormalize(orig_imgs[0])
      plt.imshow(orig_imgs_plot.cpu().permute(1, 2, 0))
      plt.title(orig_imgs)
      plt.axis('off')
      plt.show()
      plt.clf()
      """
      reconstructed = model(disturbed_imgs) #from disturbed img, create back original
      trans = transforms.Resize((reconstructed.shape[2], reconstructed.shape[3]), antialias=False)
      orig_imgs = trans(orig_imgs) #resize original to reconstructed one as it will have slighlty less pixels
      true_loss = loss_fn(reconstructed, orig_imgs)
      
      model_optimizer.zero_grad(set_to_none=True)
      true_loss.backward() 
      model_optimizer.step()		
      running_loss += true_loss.item()	
   running_loss /= batch_size
   return running_loss

def val_model_on_disturbed_images(val_dataloader, epoch, device, model, model_save_path, model_optimizer, model_scheduler, results_dir, lpips_model, ms_ssim_module, example_dataloader, loss_fn):
   model.eval()
   batch_size = len(val_dataloader)
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   ms_ssim_score = 0
   lpips_score = 0
   running_loss = 0
   with torch.no_grad():
      for disturbed_imgs, orig_imgs in tqdm(val_dataloader, desc=f'Epoch {epoch} - Validating model'):
         disturbed_imgs, _ = disturbed_imgs.decompose()
         disturbed_imgs = disturbed_imgs.to(device)
         orig_imgs, _ = orig_imgs.decompose()
         orig_imgs = orig_imgs.to(device)
         
         reconstructed = model(disturbed_imgs)
         trans = transforms.Resize((reconstructed.shape[2], reconstructed.shape[3]), antialias=False)
         orig_imgs = trans(orig_imgs)
         true_loss = loss_fn(reconstructed, orig_imgs)
         
         trans_lpips = transforms.Resize((64, 64), antialias=False)
         reconstructed_lpips = trans_lpips(reconstructed)
         orig_imgs_lpips = trans_lpips(orig_imgs)
         lpips_temp = lpips_model(reconstructed_lpips, orig_imgs_lpips)
         lpips_score += (torch.mean(lpips_temp)).item()
         
         reconstructed=unnormalize(reconstructed)
         orig_imgs=unnormalize(orig_imgs)
         reconstructed = torch.clamp(reconstructed, min=0, max=1)
         orig_imgs = torch.clamp(orig_imgs, min=0, max=1)	
         ms_ssim_score += ms_ssim_module(reconstructed, orig_imgs).item()
         running_loss += true_loss.item()		
   running_loss /= batch_size
   lpips_score /= batch_size 
   lpips_path = f"{results_dir}/lpips_score_log.txt"
   with open(lpips_path, 'a') as file:
      file.write(f"{epoch} {lpips_score}\n")
   ms_ssim_score /= batch_size
   ms_ssim_path = f"{results_dir}/ms_ssim_score_log.txt"
   with open(ms_ssim_path, 'a') as file:
      file.write(f"{epoch} {ms_ssim_score}\n")
   model_save_path = f'{model_save_path}_{epoch}.pt'   
   if epoch%5==0:
      create_checkpoint(model, model_optimizer, epoch, running_loss, model_scheduler, model_save_path)
   save_image_examples(example_dataloader, results_dir, model, epoch, device)
   return running_loss


def train_tasknet(train_dataloader, epoch, device, tasknet_save_path, tasknet, tasknet_optimizer): #tasknet training
   tasknet.train()
   for param in tasknet.parameters():
      param.requires_grad = True
   batch_size = len(train_dataloader)
   running_loss = 0
   for imgs, targets in tqdm(train_dataloader, desc=f'Epoch {epoch} - Train model'):
      imgs = list(img.to(device) for img in imgs)
      targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
      loss_dict = tasknet(imgs, targets)
      losses = sum(loss for loss in loss_dict.values())
      tasknet_optimizer.zero_grad(set_to_none=True)
      losses.backward()
      tasknet_optimizer.step()
      running_loss += losses.item()
   running_loss /= batch_size
   return running_loss

def val_tasknet(val_dataloader, epoch, device, tasknet_save_path, tasknet, tasknet_optimizer, tasknet_scheduler, ap_score_threshold, results_dir):
   for param in tasknet.parameters():
      param.requires_grad = False
   res={}
   evaluator_complete_metric = MyEvaluatorCompleteMetric()	
   batch_size = len(val_dataloader)
   running_loss = 0
   with torch.no_grad():
      for imgs, targets in tqdm(val_dataloader, desc=f'Epoch {epoch} - Validating model'):
         imgs = list(img.to(device) for img in imgs)
         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
         tasknet.train()
         loss_dict = tasknet(imgs, targets)
         losses = sum(loss for loss in loss_dict.values())
         
         tasknet.eval()
         outputs = tasknet(imgs)
         outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
         outputs_evaluator = [{k: v.clone().to(device) for k, v in t.items()} for t in outputs] #clone needed
         
         adj_outputs = adjust_outputs_to_cocoeval_api(targets, outputs)
         
         res.update({target["image_id"].item(): output for target, output in zip(targets, adj_outputs)})
         preds = [apply_nms(pred, iou_thresh=0.5, score_thresh=0.01) for pred in outputs_evaluator]
         #with faster i have a list, slightly different code
         evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=imgs[0].size()[1:][::-1])
         running_loss += losses.item()
   running_loss /= batch_size
   compute_ap(val_dataloader, tasknet, epoch, device, ap_score_threshold, results_dir, res)	
   compute_custom_metric(evaluator_complete_metric, results_dir, epoch)	
   tasknet_save_path = f'{tasknet_save_path}_{epoch}.pt' 
   create_checkpoint(tasknet, tasknet_optimizer, epoch, running_loss, tasknet_scheduler, tasknet_save_path)
   return running_loss
