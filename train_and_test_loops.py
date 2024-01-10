import torch
from tqdm import tqdm
import sys

from model_utils_and_functions import create_checkpoint

#from plot_utils import *

from coco_eval import *
from coco_eval import _get_iou_types

def train_model(train_dataloader, epoch, device, train_loss, model, tasknet, model_optimizer): #funzione che si occupa del training
	model.train()
	tasknet.train()
	batch_size = len(train_dataloader) #recupero la batch size
	running_loss = 0 # Iniziallizzo la variabile per la loss 
	for imgs, targets in tqdm(train_dataloader, desc=f'Epoch {epoch} - Train model'):
		imgs, _ = imgs.decompose()
		imgs = imgs.to(device)
		targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
		#print_batch_after_collatefn(imgs, targets, 2) #QUESTO CON DECOMPOSE
		tasknet.train()
		reconstructed = model(imgs)		
		reconstructed_loss_dict = tasknet(reconstructed, targets)
		reconstructed_losses = sum(loss for loss in reconstructed_loss_dict.values())
		
		true_loss=reconstructed_losses
		model_optimizer.zero_grad(set_to_none=True) #pulisco i gradienti prima del backpropagation step
		true_loss.backward() #computo il gradiente per la total loss con rispetto ai parametri
		model_optimizer.step()		
		running_loss += true_loss.item()
	running_loss /= batch_size #calcolo la loss media
	train_loss.append(running_loss)
	return running_loss

def val_and_ap_model(val_dataloader, epoch, device, val_loss, model, model_save_path, tasknet, model_optimizer, model_scheduler, ap_log_path): #funzione che si occupa del test
	model.eval()
	coco = get_coco_api_from_dataset(val_dataloader.dataset)
	iou_types = _get_iou_types(tasknet)
	coco_evaluator = CocoEvaluator(coco, iou_types)
	#print(coco_evaluator.coco_eval['bbox'].params.recThrs)
	#new_interpolation = np.linspace(.75, 1.00, int(np.round((1.00 - .75) / .01)) + 1, endpoint=True)
	#coco_evaluator.coco_eval['bbox'].params.recThrs = new_interpolation #necessaria visto che tengo solo gli score da 0.75 compreso in poi
	#print(coco_evaluator.coco_eval['bbox'].params.recThrs)
	batch_size = len(val_dataloader) #recupero la batch size
	running_loss = 0 # Initializing variable for storing  loss 
	with torch.no_grad(): #non calcolo il gradiente, sto testando e bassa
		res={}
		filtered_results = {}
		for imgs, targets in tqdm(val_dataloader, desc=f'Epoch {epoch} - Validating model'):
			imgs, _ = imgs.decompose()
			imgs = imgs.to(device)
			new_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
			reconstructed = model(imgs)
			tasknet.train()
			reconstructed_loss_dict = tasknet(reconstructed, new_targets)
			reconstructed_losses = sum(loss for loss in reconstructed_loss_dict.values())
					
			tasknet.eval()
			outputs = tasknet(reconstructed)
			outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
			res.update({target["image_id"].item(): output for target, output in zip(targets, outputs)})
			#res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
			#coco_evaluator.update(res)
			#print(res)
			
			#This is done for evaluating the model only on the 0.75 or greater score predictions
			
			#Dummy input con tutte le predictions esatte perchè le bbox derivano dal GT
			#questo dimostra che in teoria il modo di scrivere il risultato filtrato è giusto
			#for target, output in zip(targets, outputs):
				#Dummy input con tutte le predictions esatte perchè le bbox derivano dal GT
				#questo dimostra il modo di scrivere il risultato filtrato è giusto
				#filtered_results.update({target["image_id"].item():{
				#		'boxes': target['boxes'],
				#		'labels': torch.ones(len(target['boxes'])),
				#		'scores': torch.ones(len(target['boxes']))
				#}})
				
			#	filtered_boxes = output['boxes'][output['scores'] > 0.74]
			#	filtered_labels = output['labels'][output['scores'] > 0.74]
			#	filtered_scores = output['scores'][output['scores'] > 0.74]
			#	if any(filtered_scores):
			#		filtered_results.update({target["image_id"].item(): {
			#			'boxes': filtered_boxes,
			#			'labels': filtered_labels,
			#			'scores': filtered_scores
			#		}})
					#coco_evaluator.update(filtered_results)
					#print("Trovata predizione con score>0.74")
				#else:
				#	filtered_results.update({target["image_id"].item():{
				#		'boxes': target['boxes'],
				#		'labels': torch.ones(len(target['boxes'])),
				#		'scores': torch.full((1,) * len(target['boxes']), 1)
				#	}})
					#coco_evaluator.update(filtered_results)
				#	print("Nessuna predizione trovata")
					#filtered_results={}
					
			#in our context AP is for the robot to recognize the presence of a person, not general AP

			true_loss=reconstructed_losses
			running_loss += true_loss.item()		
	running_loss /= batch_size #calcolo la loss media
	coco_evaluator.update(res)
	#coco_evaluator.update(filtered_results)
	coco_evaluator.synchronize_between_processes()
	coco_evaluator.coco_eval['bbox'].evaluate()
	coco_evaluator.accumulate()
	
	orig_stdout = sys.stdout
	f = open(ap_log_path, 'a')
	sys.stdout = f
	print(f'AP for Epoch {epoch}')
	coco_evaluator.summarize()
	
	sys.stdout = orig_stdout
	f.close()
	
	coco_evaluator.summarize()
	val_loss.append(running_loss)
	model_save_path = f'{model_save_path}_{epoch}.pt'   
	create_checkpoint(model, model_optimizer, epoch, running_loss, model_scheduler, model_save_path)
	return running_loss

def train_tasknet(train_dataloader, epoch, device, train_loss, tasknet_save_path, tasknet, tasknet_optimizer): #funzione che si occupa del training
	tasknet.train()
	for param in tasknet.parameters():
		param.requires_grad = True
	batch_size = len(train_dataloader) #recupero la batch size
	running_loss = 0 # Iniziallizo la variabile per la loss 
	for imgs, targets in tqdm(train_dataloader, desc=f'Epoch {epoch} - Train model'):
		imgs = list(img.to(device) for img in imgs)
		targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
		loss_dict = tasknet(imgs, targets)
		losses = sum(loss for loss in loss_dict.values())
		tasknet_optimizer.zero_grad(set_to_none=True) #pulisco i gradienti prima del backpropagation step
		losses.backward() #computo il gradiente per la total loss con rispetto ai parametri
		tasknet_optimizer.step()
		
		running_loss += losses.item()
	running_loss /= batch_size #calcolo la loss media
	train_loss.append(running_loss)
	return running_loss

def val_and_ap_tasknet(val_dataloader, epoch, device, val_loss, tasknet_save_path, tasknet, tasknet_optimizer, tasknet_scheduler, ap_log_path): #funzione che si occupa del test
	for param in tasknet.parameters():
		param.requires_grad = False
	coco = get_coco_api_from_dataset(val_dataloader.dataset)
	iou_types = _get_iou_types(tasknet)
	
	coco_evaluator = CocoEvaluator(coco, iou_types)
	
	batch_size = len(val_dataloader) #recupero la batch size
	running_loss = 0 # Initializing variable for storing  loss 
	with torch.no_grad(): #non calcolo il gradiente, sto testando e bassa
		for imgs, targets in tqdm(val_dataloader, desc=f'Epoch {epoch} - Validating model'):
			imgs = list(img.to(device) for img in imgs)
			new_targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
			tasknet.train()
			loss_dict = tasknet(imgs, new_targets)
			losses = sum(loss for loss in loss_dict.values())
			running_loss += losses.item()
			tasknet.eval()
			outputs = tasknet(imgs)
			
			#new_output = resize_orig(orig_image_size)
			#losses, outputs = eval_forward(tasknet, imgs, new_targets)
			#PROBLEMA: EVAL FORWARD NON DA VALORI DI AP CORRETTI!
			outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
			res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
			#print(res)
			coco_evaluator.update(res)
			
	running_loss /= batch_size #calcolo la loss media
	coco_evaluator.accumulate()
	
	orig_stdout = sys.stdout
	f = open(ap_log_path, 'a')
	sys.stdout = f
	print(f'AP for Epoch {epoch}')
	coco_evaluator.summarize()
	
	sys.stdout = orig_stdout
	f.close()
	
	val_loss.append(running_loss)
	tasknet_save_path = f'{tasknet_save_path}_{epoch}.pt' 
	create_checkpoint(tasknet, tasknet_optimizer, epoch, running_loss, tasknet_scheduler, tasknet_save_path)
	return running_loss
