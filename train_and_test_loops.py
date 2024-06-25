import torch
from tqdm import tqdm
from model_utils_and_functions import create_checkpoint, compute_ap, apply_nms, compute_custom_metric
import numpy as np
from custom_metric.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
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
      plt.imshow(unnormalize(imgs[0]).detach().cpu().permute(1, 2, 0).numpy())
      ax = plt.gca()
      for t, l in zip(targets[0]['boxes'], targets[0]['labels']):
         xmin, ymin, xmax, ymax = t
         xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
         ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
         ax.text(xmin, ymin, l, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
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


def val_model(val_dataloader, epoch, device, model, model_save_path, tasknet, model_optimizer, model_scheduler, ap_score_threshold, results_dir, tot_epochs, save_all_weights, lpips_model, ms_ssim_module, example_dataloader):
   model.eval()
   batch_size = len(val_dataloader)
   res={} #dictionary to store all prediction for AP
   evaluator_complete_metric = MyEvaluatorCompleteMetric() #for custom metric
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device) #for ms ssim
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   lpips_score = ms_ssim_score = running_loss = 0
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
         
         # in validation targets need to be converted to original size, otherwise coco ap is wrong
         adj_outputs = adjust_outputs_to_cocoeval_api(targets, outputs, reconstructed)
         
         res.update({target["image_id"].item(): output for target, output in zip(targets, adj_outputs)})
         preds = [apply_nms(pred, iou_thresh=0.5, score_thresh=0.01) for pred in outputs_evaluator]
         evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=imgs.size()[2:][::-1])
         """
         import matplotlib.pyplot as plt
         import matplotlib.image as mpimg
         plt.imshow(unnormalize(imgs[0]).detach().cpu().permute(1, 2, 0).numpy())
         ax = plt.gca()
         for t, l in zip(targets[0]['boxes'], targets[0]['labels']):
            xmin, ymin, xmax, ymax = t
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
            ax.text(xmin, ymin, l, fontsize=15, bbox=dict(facecolor='yellow', alpha=0.5))
         plt.show()
         plt.clf()
         """
         
         trans = transforms.Resize((reconstructed.shape[2], reconstructed.shape[3]), antialias=False)
         orig_imgs = trans(imgs)
         #clamping to avoid small negative values related to not perfect unnormalize
         reconstructed = torch.clamp(unnormalize(reconstructed), min=0, max=1)
         orig_imgs = torch.clamp(unnormalize(orig_imgs), min=0, max=1)
         ms_ssim_score += ms_ssim_module(reconstructed, orig_imgs).item()
         
         true_loss=reconstructed_losses
         
         #LPIPS expects input to be [-1,1]. As they normalize input multiple times, it's best to just give input between [0,1] and let LPIPS code handle the normalization
         lpips_temp = lpips_model(reconstructed, orig_imgs, normalize=True)
         lpips_score += (torch.mean(lpips_temp)).item()
         
         #Before passing reconstructed, input needs to be normalized as ImageNet and resized to 64x64
         trans_te = transforms.Resize((64, 64), antialias=False)
         reconstructed = trans_te(reconstructed)
         nor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         reconstructed = nor(reconstructed)
         
         #true_loss=reconstructed_losses
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

   if save_all_weights:
      model_save_path = f'{model_save_path}_fw_{epoch}.pt' 
      create_checkpoint(model, model_optimizer, epoch, running_loss, model_scheduler, model_save_path)
   elif epoch==tot_epochs or epoch==int(tot_epochs/2):
      model_save_path = f'{model_save_path}_fw_{epoch}.pt'
      create_checkpoint(model, model_optimizer, epoch, running_loss, model_scheduler, model_save_path)
   save_image_examples(example_dataloader, results_dir, model, epoch, device)
   return running_loss

#same code for dissim version, as weight for MAE doesn't influence metrics, is used only for computing loss  
def test_model(test_dataloader, device, tasknet, model, ap_score_threshold, results_dir, lpips_model, ms_ssim_module):
   model.eval()
   tasknet.eval()
   batch_size = len(test_dataloader)
   res={} #dictionary to store all prediction for AP
   evaluator_complete_metric = MyEvaluatorCompleteMetric() #for custom metric
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device) #for ms ssim
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   lpips_score = ms_ssim_score = epoch = 0
   with torch.no_grad(): #not computing gradient
      for imgs, targets in tqdm(test_dataloader, desc=f'Testing forward model'):
         imgs, _ = imgs.decompose()
         imgs = imgs.to(device)
         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
         reconstructed = model(imgs)
         targets = adjust_orig_target_to_reconstructed_imgs(targets, imgs, reconstructed)	
         outputs = tasknet(reconstructed)
         outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
         outputs_evaluator = [{k: v.clone().to(device) for k, v in t.items()} for t in outputs] #need to clone them as they will be modified after
         
         # in validation targets need to be converted to original size, otherwise coco ap is wrong
         adj_outputs = adjust_outputs_to_cocoeval_api(targets, outputs, reconstructed)
         
         res.update({target["image_id"].item(): output for target, output in zip(targets, adj_outputs)})
         preds = [apply_nms(pred, iou_thresh=0.5, score_thresh=0.01) for pred in outputs_evaluator]
         evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=imgs.size()[2:][::-1])
         
         trans = transforms.Resize((reconstructed.shape[2], reconstructed.shape[3]), antialias=False)
         orig_imgs = trans(imgs)
         #clamping to avoid small negative values related to not perfect unnormalize
         reconstructed = torch.clamp(unnormalize(reconstructed), min=0, max=1)
         orig_imgs = torch.clamp(unnormalize(orig_imgs), min=0, max=1)
         ms_ssim_score += ms_ssim_module(reconstructed, orig_imgs).item()
         #LPIPS expects input to be [-1,1]. As they normalize input multiple times, it's best to just give input between [0,1] and let LPIPS code handle the normalization
         lpips_temp = lpips_model(reconstructed, orig_imgs, normalize=True)
         lpips_score += (torch.mean(lpips_temp)).item()
         
         #Before passing reconstructed, input needs to be normalized as ImageNet and resized to 64x64
         trans_te = transforms.Resize((64, 64), antialias=False)
         reconstructed = trans_te(reconstructed)
         nor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         reconstructed = nor(reconstructed)
   compute_ap(test_dataloader, tasknet, epoch, device, ap_score_threshold, results_dir, res)
   compute_custom_metric(evaluator_complete_metric, results_dir, epoch)
   ms_ssim_score /= batch_size
   ms_ssim_path= f"{results_dir}/ms_ssim_score_log.txt"
   with open(ms_ssim_path, 'a') as file:
      file.write(f"{epoch} {ms_ssim_score}\n")
   lpips_score /= batch_size
   lpips_path = f"{results_dir}/lpips_score_log.txt"
   with open(lpips_path, 'a') as file:
      file.write(f"{epoch} {lpips_score}\n")
   return 


def train_model_dissim(train_dataloader, epoch, device, model, tasknet, model_optimizer, weight): #Train model
   model.train()
   tasknet.train()
   batch_size = len(train_dataloader)
   running_loss = 0
   mae_loss = torch.nn.L1Loss()
   for imgs, targets in tqdm(train_dataloader, desc=f'Epoch {epoch} - Train model'):
      imgs, _ = imgs.decompose()
      imgs = imgs.to(device)
      targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]		
      tasknet.train()
      reconstructed = model(imgs)
      targets = adjust_orig_target_to_reconstructed_imgs(targets, imgs, reconstructed) #needed as explained before
      reconstructed_loss_dict = tasknet(reconstructed, targets) #grab tasknet losses
      reconstructed_losses = sum(loss for loss in reconstructed_loss_dict.values()) 
      noise = torch.zeros(reconstructed.size(), dtype=torch.float32, device=device)
      true_loss = (1-weight)*reconstructed_losses + weight*mae_loss(reconstructed, noise)
      model_optimizer.zero_grad(set_to_none=True) #backpropagation
      true_loss.backward()
      model_optimizer.step()		
      running_loss += true_loss.item() #storing value   
   running_loss /= batch_size #average loss
   return running_loss

def val_model_dissim(val_dataloader, epoch, device, model, model_save_path, tasknet, model_optimizer, model_scheduler, ap_score_threshold, results_dir, tot_epochs, save_all_weights, lpips_model, ms_ssim_module, example_dataloader, weight):
   model.eval()
   batch_size = len(val_dataloader)
   res={} #dictionary to store all prediction for AP
   evaluator_complete_metric = MyEvaluatorCompleteMetric() #for custom metric
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device) #for ms ssim
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   lpips_score = ms_ssim_score = running_loss = 0
   mae_loss = torch.nn.L1Loss()
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
         
         # in validation targets need to be converted to original size, otherwise coco ap is wrong
         adj_outputs = adjust_outputs_to_cocoeval_api(targets, outputs, reconstructed)
         
         res.update({target["image_id"].item(): output for target, output in zip(targets, adj_outputs)})
         preds = [apply_nms(pred, iou_thresh=0.5, score_thresh=0.01) for pred in outputs_evaluator]
         evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=imgs.size()[2:][::-1])
         
         trans = transforms.Resize((reconstructed.shape[2], reconstructed.shape[3]), antialias=False)
         orig_imgs = trans(imgs)
         #clamping to avoid small negative values related to not perfect unnormalize
         reconstructed = torch.clamp(unnormalize(reconstructed), min=0, max=1)
         orig_imgs = torch.clamp(unnormalize(orig_imgs), min=0, max=1)
         ms_ssim_score += ms_ssim_module(reconstructed, orig_imgs).item()
         
         noise = torch.zeros(reconstructed.size(), dtype=torch.float32, device=device)
         true_loss = (1-weight)*reconstructed_losses + weight*mae_loss(reconstructed, noise)
         
         #LPIPS expects input to be [-1,1]. As they normalize input multiple times, it's best to just give input between [0,1] and let LPIPS code handle the normalization
         lpips_temp = lpips_model(reconstructed, orig_imgs, normalize=True)
         lpips_score += (torch.mean(lpips_temp)).item()
         
         #Before passing reconstructed, input needs to be normalized as ImageNet and resized to 64x64
         trans_te = transforms.Resize((64, 64), antialias=False)
         reconstructed = trans_te(reconstructed)
         nor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         reconstructed = nor(reconstructed)
         
         #true_loss=reconstructed_losses
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
   if save_all_weights:
      model_save_path = f'{model_save_path}_fw_{epoch}.pt' 
      create_checkpoint(model, model_optimizer, epoch, running_loss, model_scheduler, model_save_path)
   elif epoch==tot_epochs or epoch==int(tot_epochs/2):
      model_save_path = f'{model_save_path}_fw_{epoch}.pt'
      create_checkpoint(model, model_optimizer, epoch, running_loss, model_scheduler, model_save_path)
   save_image_examples(example_dataloader, results_dir, model, epoch, device)
   return running_loss

def save_image_examples(example_dataloader, results_dir, model, epoch, device):
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   model.eval()
   coco = get_coco_api_from_dataset(example_dataloader.dataset)
   with torch.no_grad(): 
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

def generate_disturbed_dataset(train_dataloader_gen_disturbed, val_dataloader_gen_disturbed, device, model, train_img_folder, train_ann, val_img_folder, val_ann, keep_original_size): #for generating disturbed set
   model.eval()
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   #First validation set
   batch_size = len(val_dataloader_gen_disturbed)
   coco = get_coco_api_from_dataset(val_dataloader_gen_disturbed.dataset)
   disturbed_list = []
   with torch.no_grad():
      for imgs, targets in tqdm(val_dataloader_gen_disturbed, desc=f'Generating disturbed val images from dataset'):
         imgs, _ = imgs.decompose()
         imgs = imgs.to(device)
         reconstructed = model(imgs)
         coco_image_id = targets[0]["image_id"].item()
         path = coco.loadImgs(coco_image_id)[0]["file_name"]
         if keep_original_size: #for fine tune Tasknet on disturbed set
            orig_h, orig_w = targets[0]['orig_size']
            #orig_size = [imgs.shape[2], imgs.shape[3]]
            orig_size = [orig_h, orig_w]
            trans = transforms.Resize(orig_size, antialias=False)
            reconstructed = trans(reconstructed)
         reconstructed = unnormalize(reconstructed)
         reconstructed = torch.clamp(reconstructed, min=0, max=1)
         save_image(reconstructed, os.path.join(val_img_folder, path))
         disturbed_list.append({"image_path": path, "coco_image_id": coco_image_id})
   with open(val_ann, 'w') as json_file:
      json.dump(disturbed_list, json_file, indent=2)
   #Then training set
   batch_size = len(train_dataloader_gen_disturbed)
   coco = get_coco_api_from_dataset(train_dataloader_gen_disturbed.dataset)
   disturbed_list = []
   with torch.no_grad():
      for imgs, targets in tqdm(train_dataloader_gen_disturbed, desc=f'Generating disturbed train images from dataset'):
         imgs, _ = imgs.decompose()
         imgs = imgs.to(device)
         reconstructed = model(imgs)
         coco_image_id = targets[0]["image_id"].item()
         path = coco.loadImgs(coco_image_id)[0]["file_name"]
         if keep_original_size: #for fine tune Tasknet on disturbed set; less meaningful for backward
            orig_h, orig_w = targets[0]['orig_size']
            #orig_size = [imgs.shape[2], imgs.shape[3]]
            orig_size = [orig_h, orig_w]
            trans = transforms.Resize(orig_size, antialias=False)
            reconstructed = trans(reconstructed)
         reconstructed = unnormalize(reconstructed) #unnormalized and clamping needed before storing images in [0,1]
         reconstructed = torch.clamp(reconstructed, min=0, max=1)
         save_image(reconstructed, os.path.join(train_img_folder, path))
         disturbed_list.append({"image_path": path,"coco_image_id": coco_image_id})
   with open(train_ann, 'w') as json_file:
      json.dump(disturbed_list, json_file, indent=2)
   return

def train_model_bw(train_dataloader, epoch, device, model, model_optimizer, ms_ssim_module): #train model backward
   model.train()
   batch_size = len(train_dataloader)
   running_loss = 0
   loss = torch.nn.L1Loss()
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
      
      true_loss = loss(reconstructed, orig_imgs)
      
      model_optimizer.zero_grad(set_to_none=True)
      true_loss.backward() 
      model_optimizer.step()		
      running_loss += true_loss.item()	
   running_loss /= batch_size
   return running_loss

def val_model_bw(val_dataloader, epoch, device, model, model_save_path, model_optimizer, model_scheduler, results_dir, tot_epochs, save_all_weights, lpips_model, ms_ssim_module, example_dataloader, compute_right_similarity_metrics=False):
   model.eval()
   batch_size = len(val_dataloader)
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   ms_ssim_score = lpips_score = running_loss = mse_score = 0
   mse_as_metric = torch.nn.MSELoss()
   loss = torch.nn.L1Loss()
   with torch.no_grad():
      for disturbed_imgs, orig_imgs in tqdm(val_dataloader, desc=f'Epoch {epoch} - Validating model'):
         disturbed_imgs, _ = disturbed_imgs.decompose()
         disturbed_imgs = disturbed_imgs.to(device)
         orig_imgs, _ = orig_imgs.decompose()
         orig_imgs = orig_imgs.to(device)
         
         reconstructed = model(disturbed_imgs)
         trans = transforms.Resize((reconstructed.shape[2], reconstructed.shape[3]), antialias=False)
         orig_imgs = trans(orig_imgs)
         
         true_loss = loss(reconstructed, orig_imgs)
         
         mse_score += mse_as_metric(reconstructed, orig_imgs).item()
         
         reconstructed=unnormalize(reconstructed)
         orig_imgs=unnormalize(orig_imgs)
         reconstructed = torch.clamp(reconstructed, min=0, max=1)
         orig_imgs = torch.clamp(orig_imgs, min=0, max=1)
         
         ms_ssim_score += ms_ssim_module(reconstructed, orig_imgs).item() #compute msssim
         
         #LPIPS expects input to be [-1,1]. As they normalize input multiple times, it's best to just give input between [0,1] and let LPIPS code handle the normalization
         lpips_temp = lpips_model(reconstructed, orig_imgs, normalize=True)
         lpips_score += (torch.mean(lpips_temp)).item()
         
         running_loss += true_loss.item() #used for appending in train and val loss file	
   running_loss /= batch_size
   lpips_score /= batch_size
   ms_ssim_score /= batch_size
   mse_score /= batch_size
   if compute_right_similarity_metrics:
      lpips_path = f"{results_dir}/lpips_score_log_batch1.txt"
      ms_ssim_path = f"{results_dir}/ms_ssim_score_log_batch1.txt"
      mse_path = f"{results_dir}/mse_score_log_batch1.txt"
      val_loss_path = f"{results_dir}/only_val_loss_log_batch1.txt"
      with open(val_loss_path, 'a') as file:
         file.write(f"{epoch} {running_loss}\n")
   else:
      lpips_path = f"{results_dir}/lpips_score_log.txt"
      ms_ssim_path = f"{results_dir}/ms_ssim_score_log.txt"
      mse_path = f"{results_dir}/mse_score_log.txt"
   with open(lpips_path, 'a') as file:
      file.write(f"{epoch} {lpips_score}\n")
   with open(ms_ssim_path, 'a') as file:
      file.write(f"{epoch} {ms_ssim_score}\n")
   with open(mse_path, 'a') as file:
      file.write(f"{epoch} {mse_score}\n")
   if compute_right_similarity_metrics:
      return running_loss #don't care of saving weights or image examples, I have done it in the real validation loop
   if save_all_weights:
      model_save_path = f'{model_save_path}_bw_{epoch}.pt' 
      create_checkpoint(model, model_optimizer, epoch, running_loss, model_scheduler, model_save_path)
   elif epoch==tot_epochs or epoch==int(tot_epochs/2):
      model_save_path = f'{model_save_path}_bw_{epoch}.pt' 
      create_checkpoint(model, model_optimizer, epoch, running_loss, model_scheduler, model_save_path)
   save_image_examples(example_dataloader, results_dir, model, epoch, device)
   return running_loss

def test_model_bw(test_dataloader, device, model, results_dir, lpips_model, ms_ssim_module):
   model.eval()
   batch_size = len(test_dataloader)
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   ms_ssim_score = lpips_score = running_loss = mse_score = epoch = 0
   mse_as_metric = torch.nn.MSELoss()
   loss = torch.nn.L1Loss()
   with torch.no_grad():
      for disturbed_imgs, orig_imgs in tqdm(test_dataloader, desc=f'Testing backward model'):
         disturbed_imgs, _ = disturbed_imgs.decompose()
         disturbed_imgs = disturbed_imgs.to(device)
         orig_imgs, _ = orig_imgs.decompose()
         orig_imgs = orig_imgs.to(device)
         
         reconstructed = model(disturbed_imgs)
         trans = transforms.Resize((reconstructed.shape[2], reconstructed.shape[3]), antialias=False)
         orig_imgs = trans(orig_imgs)
         
         true_loss = loss(reconstructed, orig_imgs)
         
         mse_score += mse_as_metric(reconstructed, orig_imgs).item()
         
         reconstructed=unnormalize(reconstructed)
         orig_imgs=unnormalize(orig_imgs)
         reconstructed = torch.clamp(reconstructed, min=0, max=1)
         orig_imgs = torch.clamp(orig_imgs, min=0, max=1)
         
         ms_ssim_score += ms_ssim_module(reconstructed, orig_imgs).item() #compute msssim
         
         #LPIPS expects input to be [-1,1]. As they normalize input multiple times, it's best to just give input between [0,1] and let LPIPS code handle the normalization
         lpips_temp = lpips_model(reconstructed, orig_imgs, normalize=True)
         lpips_score += (torch.mean(lpips_temp)).item()
         
         running_loss += true_loss.item() #used for appending in test loss file	
   running_loss /= batch_size
   lpips_score /= batch_size
   ms_ssim_score /= batch_size
   mse_score /= batch_size
   lpips_path = f"{results_dir}/lpips_score_test_bw.txt"
   ms_ssim_path = f"{results_dir}/ms_ssim_score_test_bw.txt"
   mse_path = f"{results_dir}/mse_score_test_bw.txt"
   test_loss_path = f"{results_dir}/only_test_loss_bw.txt"
   with open(test_loss_path, 'a') as file:
      file.write(f"{epoch} {running_loss}\n")
   with open(lpips_path, 'a') as file:
      file.write(f"{epoch} {lpips_score}\n")
   with open(ms_ssim_path, 'a') as file:
      file.write(f"{epoch} {ms_ssim_score}\n")
   with open(mse_path, 'a') as file:
      file.write(f"{epoch} {mse_score}\n")
   return

def generate_similarity_dataset(similarity_dataloader, device, model, val_img_folder, val_ann):
   model.eval()
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   batch_size = len(similarity_dataloader)
   disturbed_list = []
   with torch.no_grad():
      for imgs, targets in tqdm(similarity_dataloader, desc=f'Generating reconstructed images from disturbed val set'):
         imgs = imgs.to(device)
         reconstructed = model(imgs)
         path = ''.join(targets)
         reconstructed = unnormalize(reconstructed)
         reconstructed = torch.clamp(reconstructed, min=0, max=1)
         save_image(reconstructed, os.path.join(val_img_folder, path))
         disturbed_list.append({"image_path": path,})
   with open(val_ann, 'w') as json_file:
      json.dump(disturbed_list, json_file, indent=2)
   return

def generate_disturbed_testset(test_dataloader_gen_disturbed, device, model, test_img_folder, test_ann, keep_original_size):
   model.eval()
   mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
   std = torch.tensor([0.229, 0.224, 0.225]).to(device)
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   batch_size = len(test_dataloader_gen_disturbed)
   coco = get_coco_api_from_dataset(test_dataloader_gen_disturbed.dataset)
   disturbed_list = []
   with torch.no_grad():
      for imgs, targets in tqdm(test_dataloader_gen_disturbed, desc=f'Generating disturbed test images from dataset'):
         imgs, _ = imgs.decompose()
         imgs = imgs.to(device)
         reconstructed = model(imgs)
         coco_image_id = targets[0]["image_id"].item()
         path = coco.loadImgs(coco_image_id)[0]["file_name"]
         if keep_original_size: #for fine tune Tasknet on disturbed set
            orig_h, orig_w = targets[0]['orig_size']
            #orig_size = [imgs.shape[2], imgs.shape[3]]
            orig_size = [orig_h, orig_w]
            trans = transforms.Resize(orig_size, antialias=False)
            reconstructed = trans(reconstructed)
         reconstructed = unnormalize(reconstructed)
         reconstructed = torch.clamp(reconstructed, min=0, max=1)
         save_image(reconstructed, os.path.join(test_img_folder, path))
         disturbed_list.append({"image_path": path, "coco_image_id": coco_image_id})
   with open(test_ann, 'w') as json_file:
      json.dump(disturbed_list, json_file, indent=2)
   return

def val_similarity_disturbed_images(val_dataloader, device, results_dir, lpips_model, ms_ssim_module):
   batch_size = len(val_dataloader)
   mse_as_metric = torch.nn.MSELoss()
   mae_as_metric = torch.nn.L1Loss()
   mse_score = mae_score = ms_ssim_score = lpips_score = 0
   with torch.no_grad():
      for disturbed_imgs, orig_imgs in tqdm(val_dataloader, desc=f'Computing Similarity Metrics'):
         disturbed_imgs = disturbed_imgs.to(device)
         orig_imgs = orig_imgs.to(device)
         trans = transforms.Resize((disturbed_imgs.shape[2], disturbed_imgs.shape[3]), antialias=False)
         orig_imgs = trans(orig_imgs) #force original image to be scaled down to the smaller reconstructed one
         mse_score += mse_as_metric(disturbed_imgs, orig_imgs).item()
         mae_score += mae_as_metric(disturbed_imgs, orig_imgs).item()
         ms_ssim_score += ms_ssim_module(disturbed_imgs, orig_imgs).item() #compute msssim
         lpips_score += lpips_model(disturbed_imgs, orig_imgs, normalize=True).item() #let LPIPS normalize itself, it wants images in [0,1]
         
   mse_score /= batch_size
   mae_score /= batch_size
   ms_ssim_score /= batch_size
   lpips_score /= batch_size
   similarity_path = f"{results_dir}/similarity_log.txt"
   with open(similarity_path, 'w') as file:
      file.write(f"MSE: {mse_score}\nMAE: {mae_score}\nMS_SSIM: {ms_ssim_score}\nLPIPS: {lpips_score}\n")
   return

def train_tasknet(train_dataloader, epoch, device, tasknet_save_path, tasknet, tasknet_optimizer): #tasknet training
   tasknet.train()
   for param in tasknet.parameters():
      param.requires_grad = True
   batch_size = len(train_dataloader)
   running_loss = 0
   for imgs, targets in tqdm(train_dataloader, desc=f'Epoch {epoch} - Train Tasknet'):
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

def val_tasknet(val_dataloader, epoch, device, tasknet_save_path, tasknet, tasknet_optimizer, tasknet_scheduler, ap_score_threshold, results_dir, tot_epochs, save_all_weights, skip_saving_weights=False):
   for param in tasknet.parameters():
      param.requires_grad = False
   res={}
   evaluator_complete_metric = MyEvaluatorCompleteMetric()	
   batch_size = len(val_dataloader)
   running_loss = 0
   with torch.no_grad():
      for imgs, targets in tqdm(val_dataloader, desc=f'Epoch {epoch} - Validating Tasknet'):
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
   if not skip_saving_weights:
      if save_all_weights:
         tasknet_save_path = f'{tasknet_save_path}_{epoch}.pt'
         create_checkpoint(tasknet, tasknet_optimizer, epoch, running_loss, tasknet_scheduler, tasknet_save_path)
      elif epoch==tot_epochs or epoch==int(tot_epochs/2):
         tasknet_save_path = f'{tasknet_save_path}_{epoch}.pt'
         create_checkpoint(tasknet, tasknet_optimizer, epoch, running_loss, tasknet_scheduler, tasknet_save_path)
   return running_loss
   
def test_tasknet(test_dataloader, device, tasknet, ap_score_threshold, results_dir):
   for param in tasknet.parameters():
      param.requires_grad = False
   tasknet.eval()
   res={}
   evaluator_complete_metric = MyEvaluatorCompleteMetric()	
   batch_size = len(test_dataloader)
   epoch = 0 #set epoch at 0 for compatibility with code
   with torch.no_grad():
      for imgs, targets in tqdm(test_dataloader, desc=f'Testing Tasknet'):
         imgs = list(img.to(device) for img in imgs)
         #n_imgs = []
         #for img in imgs:
         #   sampl_size = [16, 16]
         #   trans_sampl = transforms.Resize(sampl_size, antialias=False)
         #   img_t = trans_sampl(img)
         #   orig_size = [img.shape[1], img.shape[2]]
         #   trans_orig = transforms.Resize(orig_size, antialias=False)
         #   img_n = trans_orig(img_t)
         #   n_imgs.append(img_n)
         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
         #outputs = tasknet(n_imgs)
         outputs = tasknet(imgs)
         outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
         outputs_evaluator = [{k: v.clone().to(device) for k, v in t.items()} for t in outputs] #clone needed   
         adj_outputs = adjust_outputs_to_cocoeval_api(targets, outputs)
         
         res.update({target["image_id"].item(): output for target, output in zip(targets, adj_outputs)})
         preds = [apply_nms(pred, iou_thresh=0.5, score_thresh=0.01) for pred in outputs_evaluator]
         #with faster i have a list, slightly different code
         evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=imgs[0].size()[1:][::-1])
   compute_ap(test_dataloader, tasknet, epoch, device, ap_score_threshold, results_dir, res)	
   compute_custom_metric(evaluator_complete_metric, results_dir, epoch)  
   return
   
def finetune_tasknet_train(train_dataloader, epoch, device, model, tasknet, tasknet_optimizer): #Train model
   model.eval()
   tasknet.train()
   for param in tasknet.parameters():
      param.requires_grad = True
   batch_size = len(train_dataloader)
   running_loss = 0
   for imgs, targets in tqdm(train_dataloader, desc=f'Epoch {epoch} - Train model'):
      imgs, _ = imgs.decompose()
      imgs = imgs.to(device)
      targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]		
      reconstructed = model(imgs)
      targets = adjust_orig_target_to_reconstructed_imgs(targets, imgs, reconstructed) #needed as explained before
      reconstructed_loss_dict = tasknet(reconstructed, targets) #grab tasknet losses
      reconstructed_losses = sum(loss for loss in reconstructed_loss_dict.values()) 
      true_loss = reconstructed_losses
      tasknet_optimizer.zero_grad(set_to_none=True) #backpropagation
      true_loss.backward()
      tasknet_optimizer.step()		
      running_loss += true_loss.item() #storing value   
   running_loss /= batch_size #average loss
   return running_loss
   
def finetune_tasknet_val(val_dataloader, epoch, device, model, tasknet_save_path, tasknet, tasknet_optimizer, tasknet_scheduler, ap_score_threshold, results_dir, tot_epochs, save_all_weights):
   tasknet.eval()
   for param in tasknet.parameters():
      param.requires_grad = False
   model.eval()
   batch_size = len(val_dataloader)
   res={} #dictionary to store all prediction for AP
   evaluator_complete_metric = MyEvaluatorCompleteMetric() #for custom metric
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
         
         # in validation targets need to be converted to original size, otherwise coco ap is wrong
         adj_outputs = adjust_outputs_to_cocoeval_api(targets, outputs, reconstructed)
         
         res.update({target["image_id"].item(): output for target, output in zip(targets, adj_outputs)})
         preds = [apply_nms(pred, iou_thresh=0.5, score_thresh=0.01) for pred in outputs_evaluator]
         evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=imgs.size()[2:][::-1])

         true_loss = reconstructed_losses
         running_loss += true_loss.item()	
   running_loss /= batch_size
   compute_ap(val_dataloader, tasknet, epoch, device, ap_score_threshold, results_dir, res)
   compute_custom_metric(evaluator_complete_metric, results_dir, epoch)
   
   if save_all_weights:
      tasknet_save_path = f'{tasknet_save_path}_finetuned_{epoch}.pt' 
      create_checkpoint(tasknet, tasknet_optimizer, epoch, running_loss, tasknet_scheduler, tasknet_save_path)
   elif epoch==tot_epochs or epoch==int(tot_epochs/2):
      tasknet_save_path = f'{tasknet_save_path}_finetuned_{epoch}.pt'
      create_checkpoint(tasknet, tasknet_optimizer, epoch, running_loss, tasknet_scheduler, tasknet_save_path)
   return running_loss
   
def test_finetuned_tasknet(test_dataloader, device, tasknet, model, ap_score_threshold, results_dir):
   model.eval()
   tasknet.eval()
   batch_size = len(test_dataloader)
   res={} #dictionary to store all prediction for AP
   evaluator_complete_metric = MyEvaluatorCompleteMetric() #for custom metric
   epoch = 0
   with torch.no_grad(): #not computing gradient
      for imgs, targets in tqdm(test_dataloader, desc=f'Testing finetuned tasknet'):
         imgs, _ = imgs.decompose()
         imgs = imgs.to(device)
         targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
         reconstructed = model(imgs)
         targets = adjust_orig_target_to_reconstructed_imgs(targets, imgs, reconstructed)	
         outputs = tasknet(reconstructed)
         outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
         outputs_evaluator = [{k: v.clone().to(device) for k, v in t.items()} for t in outputs] #need to clone them as they will be modified after
         
         # in validation targets need to be converted to original size, otherwise coco ap is wrong
         adj_outputs = adjust_outputs_to_cocoeval_api(targets, outputs, reconstructed)
         
         res.update({target["image_id"].item(): output for target, output in zip(targets, adj_outputs)})
         preds = [apply_nms(pred, iou_thresh=0.5, score_thresh=0.01) for pred in outputs_evaluator]
         evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=imgs.size()[2:][::-1])
         
   compute_ap(test_dataloader, tasknet, epoch, device, ap_score_threshold, results_dir, res)
   compute_custom_metric(evaluator_complete_metric, results_dir, epoch)
   return 
