import torch
from tqdm import tqdm

from model_utils_and_functions import create_checkpoint, compute_ap, apply_nms, compute_michele_metric

import numpy as np
from michele_metric.my_evaluators_complete_metric import MyEvaluatorCompleteMetric
#import per salvare dataset versione disturbata
from torchvision.utils import save_image
import json
import torchvision.transforms as transforms
from coco_eval import get_coco_api_from_dataset
import os
from pytorch_msssim import ms_ssim, MS_SSIM
#MS SSIM dovrebbe tenere conto di più di come appare l'img a una persona, rispetto alla sola distribuzione dell'EMD

def adjust_orig_target_to_reconstructed_imgs(targets, imgs, reconstructed):
   """
   L'img ricostruita per via della mancanza delle skip connection non ha la stessa dim dell'originale ma perde qualche pixel. 
   Ad esempio, una img prima di darla all'unet ha shape [200, 250]. Dopo ha shape [192, 256].
   Si potrebbe verificare che la bbox target originale sfori la nuova img più piccola, dando origine a minor ap.
   Serve resizare o l'img originale (ma porterebbe a uno stretch), oppure modificare le bbox target e proporzionarle all'img ricostruita.
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
      orig_h, orig_w = t['orig_size']
      if reconstructed is not None: #se non è none la sto usando con l'unet
         rec_batch_h_pad, rec_batch_w_pad = reconstructed.shape[2], reconstructed.shape[3] #qui pred è fatta su img resizata con qualche pixel in meno
         rec_batch_h_img, rec_batch_w_img = t['size'] #queste sono le size dell'img resizata senza padding
         """
         prendo la più piccola delle due, serve per scalare le bbox rispetto all'img
         e.g: rec_batch_pad = 288 304    rec_batch_img = 297, 256
         Mi serve perché la tasknet fa la predizione sul rec_batch_pad come dimensione, quindi i target 
         sono su quella immagine che ha perso qualche pixel (da qui il min con rec_batch_pad);
         devo però anche eliminare il padding di DETR se è stato messo, perché i target li devo poi 
         riscalare alle img originali (se tengo il padding scalerei l'img in modo sbagliato, finirei 
         per fare considerare il target più piccolo e con coordinate sbagliate) (ecco il min rec_batch_img).
         """
         rec_batch_h = min(rec_batch_h_pad, rec_batch_h_img)
         rec_batch_w = min(rec_batch_w_pad, rec_batch_w_img)
      else: #pred fatta da solo la tasknet
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
		
		#motivo per cui è necessario descritto dalla funzione stessa
		targets = adjust_orig_target_to_reconstructed_imgs(targets, imgs, reconstructed)

		#import matplotlib.pyplot as plt
		#import matplotlib.image as mpimg
		#mean = torch.tensor([0.485, 0.456, 0.406])
		#std = torch.tensor([0.229, 0.224, 0.225])
		#unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
		"""
		print(targets)
		plt.imshow(unnormalize(reconstructed[0]).detach().cpu().permute(1, 2, 0).numpy())
		ax = plt.gca()
		xmin, ymin, xmax, ymax = targets[0]['boxes'][0]
		xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
		ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
		plt.show()
		plt.clf()
		"""
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

from lpips.lpips import LPIPS
from model_utils_and_functions import compute_my_recons_classifier_pred, save_my_recons_classifier_dict
def val_model(val_dataloader, epoch, device, val_loss, model, model_save_path, tasknet, model_optimizer, model_scheduler, ap_score_threshold, results_dir, my_recons_classifier, my_regressor, example_dataloader): #funzione che si occupa del test
	model.eval()
	batch_size = len(val_dataloader) #recupero la batch size
	running_loss = 0 # Initializing variable for storing  loss 
	res={}
	#ms_ssim_score = 0
	evaluator_complete_metric = MyEvaluatorCompleteMetric()
	mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
	std = torch.tensor([0.229, 0.224, 0.225]).to(device)
	unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
	#ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], K=(0.01, 0.07)) #size average fa la media fra ms_ssim delle img nel batch. Metto k2=0.07 anzichè il default 0.03 per "normalizzare" i valori ed evitare casi in cui ms_ssim=0
	#lpips_loss_fn = LPIPS(reduction='mean', mean=[0., 0., 0.], std=[1., 1., 1.]) #mean e std settati così, in questo modo la vggnet non rinormalizza visto che i miei input sono già normalizzati
	
	my_rec_class_dict = {}
	my_rec_class_dict['epoch']=epoch
	my_rec_class_dict['total']=0
	recon_rate=0
	
	#lpips_model = LPIPS(net='vgg', model_path='lpips/lpips_my_weights.pth').to(device)
	lpips_model = LPIPS(net='vgg').to(device) #alexnet, pesi di default, alla fine sembrano migliori	
	lpips_score = 0
	with torch.no_grad(): #non calcolo il gradiente, sto testando e bassa
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
			outputs_evaluator = [{k: v.clone().to(device) for k, v in t.items()} for t in outputs] #devo clonarli perché poi gli originali li modifico
			
			# in validation i target devono poi essere riconvertiti alle dim originali, se no l'AP non va!
			# i targets che usa l'AP sono gli originali dall'annotation, quindi anche se sopra li avevo resizati non c'è problema visto che quei target vengono ignorati
			adj_outputs = adjust_outputs_to_cocoeval_api(targets, outputs, reconstructed)
			
			res.update({target["image_id"].item(): output for target, output in zip(targets, adj_outputs)})
			preds = [apply_nms(pred, iou_thresh=0.5, score_thresh=0.01) for pred in outputs_evaluator]
			#for pred in preds: #Questo non devo farlo, ce le ho già giuste
			# 	pred['labels'] = pred['labels'] - 1
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
			
			#trans = transforms.Resize((reconstructed.shape[2], reconstructed.shape[3]), antialias=False)
			#orig_imgs = trans(imgs)
			#reconstructed=unnormalize(reconstructed)
			#orig_imgs=unnormalize(orig_imgs)
			#alcuni valori possono capitare negativi. Il clamping lo faccio per essere sicuro di riportare il tutto alle condizioni necessarie per poter usare l'ms_sssim
			#reconstructed = torch.clamp(reconstructed, min=0, max=1)
			#orig_imgs = torch.clamp(orig_imgs, min=0, max=1)
			#ms_ssim_score += ms_ssim_module(reconstructed, orig_imgs)
			#lpips_score += lpips_loss_fn(reconstructed, orig_imgs) #lpips può essere anche sopra 1 come valore
			trans_te = transforms.Resize((64, 64), antialias=False)
			reconstructed = trans_te(reconstructed)
			orig_imgs = trans_te(imgs)
			lpips_loss = lpips_model(reconstructed, orig_imgs)
			lpips_score += (torch.mean(lpips_loss)).item()
			
			#Prima di passare reconstructed, ricordarsi che:
			#1 deve essere normalizzato come ImageNet
			#2 deve essere resizato a 64x64
			compute_my_recons_classifier_pred(my_recons_classifier, reconstructed, my_rec_class_dict)			
			recon_rate += torch.mean(my_regressor(reconstructed)).item()
			
			true_loss=reconstructed_losses
			running_loss += true_loss.item()	
	
	running_loss /= batch_size #calcolo la loss media
	#la validation loss se è attivo il custom filter proposal è fatta in base a ciò, non è una val
	#loss con i parametri di default della tasknet, ma val loss con i parametri usati per allenamento
	compute_ap(val_dataloader, tasknet, epoch, device, ap_score_threshold, results_dir, res)
	compute_michele_metric(evaluator_complete_metric, results_dir, epoch)
	#ms_ssim_score /= batch_size #0.6 su plain, 0.34 su 10prop, 0.09 su 3prop (completamente irriconoscibile), 0.20 su 1bestiou5neg0bg. Quindi 0.20 potrebbe essere la soglia di privacy?
	#ms_ssim_path="results/ms_ssim_score_log.txt"
	#with open(ms_ssim_path, 'a') as file:
	#	file.write(f"{epoch} {ms_ssim_score}\n")
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
	
	val_loss.append(running_loss)
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
         #path = ''.join(targets)
         reconstructed = unnormalize(reconstructed)
         reconstructed = torch.clamp(reconstructed, min=0, max=1)
         save_dir_path = f'{results_dir}/epoch_{epoch}'
         if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
         save_image(reconstructed, os.path.join(save_dir_path, path))

def generate_disturbed_dataset(train_dataloader_gen_disturbed, val_dataloader_gen_disturbed, device, model, train_img_folder, train_ann, val_img_folder, val_ann, keep_original_size, use_coco_train): #funzione che si occupa di generare il dataset disturbato
	model.eval()
	#Prima genero il disturbed training
	batch_size = len(train_dataloader_gen_disturbed) #recupero la batch size
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
				if keep_original_size:
					orig_size = [imgs.shape[2], imgs.shape[3]]
					trans = transforms.Resize(orig_size, antialias=False)
					reconstructed = trans(reconstructed)
				#train_lpips=True
				#if train_lpips:
				#	trans = transforms.Resize((64, 64), antialias=False)
				#	reconstructed = trans(reconstructed)
				reconstructed = unnormalize(reconstructed)
				#alcuni valori possono capitare negativi. Il clamping lo faccio per essere sicuro di riportare il tutto alle condizioni necessarie
				reconstructed = torch.clamp(reconstructed, min=0, max=1)
				save_image(reconstructed, os.path.join(train_img_folder, path))
				disturbed_list.append({
					"image_path": path,
					"coco_image_id": coco_image_id
				})
			else:
				path = ''.join(targets)
				#le img sono già normalizzate, faccio solo il resize per tenerle della stessa dimensione dell'originale (si perde qualche pixel per via del padding rimosso che veniva fatto con le skip connections dall'unet). Ha senso farlo per il comparison con la tasknet e basta, non nel backward però, dove simulo il fatto che un attaccante colleziona coppie plain-disturbate.
				if keep_original_size:
					orig_size = [imgs.shape[2], imgs.shape[3]]
					trans = transforms.Resize(orig_size, antialias=False)
					reconstructed = trans(reconstructed)
				reconstructed = unnormalize(reconstructed)
				reconstructed = torch.clamp(reconstructed, min=0, max=1)
				save_image(reconstructed, os.path.join(train_img_folder, path))
				disturbed_list.append({
					"image_path": path,
				})
	with open(train_ann, 'w') as json_file:
		json.dump(disturbed_list, json_file, indent=2)
	#Ora genero il disturbed val
	batch_size = len(val_dataloader_gen_disturbed) #recupero la batch size
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
			#train_lpips=True
			#if train_lpips:
			#	trans = transforms.Resize((64, 64), antialias=False)
			#	reconstructed = trans(reconstructed)
			reconstructed = unnormalize(reconstructed)
			reconstructed = torch.clamp(reconstructed, min=0, max=1)
			save_image(reconstructed, os.path.join(val_img_folder, path))
			disturbed_list.append({
				"image_path": path,
				"coco_image_id": coco_image_id
			})
	with open(val_ann, 'w') as json_file:
		json.dump(disturbed_list, json_file, indent=2)
	model.train()


def train_model_on_disturbed_images(train_dataloader, epoch, device, train_loss, model, model_optimizer, loss_fn): 
	model.train()
	batch_size = len(train_dataloader)
	running_loss = 0
	mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
	std = torch.tensor([0.229, 0.224, 0.225]).to(device)
	unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
	#ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], K=(0.01, 0.07))
	#loss_fn = torch.nn.MSELoss()
	#lpips_model = LPIPS(net='vgg').to(device)
	for disturbed_imgs, orig_imgs in tqdm(train_dataloader, desc=f'Epoch {epoch} - Train model'):
		disturbed_imgs, _ = disturbed_imgs.decompose()
		disturbed_imgs = disturbed_imgs.to(device)
		orig_imgs, _ = orig_imgs.decompose()
		orig_imgs = orig_imgs.to(device)
		
		#import random
		#scales = [200, 150, 100, 80, 120, 140]
		#random_size = random.choice(scales)
		#trans = transforms.Resize((random_size, random_size), antialias=False)
		#disturbed_imgs = trans(disturbed_imgs)
		
		"""#For testing
		import matplotlib.pyplot as plt
		import matplotlib.image as mpimg
		plt.imshow(disturbed_imgs[0].cpu().permute(1, 2, 0))
		plt.title(disturbed_imgs)
		plt.axis('off')
		plt.show()
		
		plt.imshow(orig_imgs[0].cpu().permute(1, 2, 0))
		plt.title(orig_imgs)
		plt.axis('off')
		plt.show()
		"""
		#print_batch_after_collatefn(imgs, targets, 2) #QUESTO CON DECOMPOSE
		#reconstructed = model(imgs)
		reconstructed = model(disturbed_imgs)
		trans = transforms.Resize((reconstructed.shape[2], reconstructed.shape[3]), antialias=False)
		#riduco la dimensione dell'originale all'output dell'unet; ha più senso rispetto che riportare all'originale l'ouput dell'unet, che richiederebbe di aggiungere info
		orig_imgs = trans(orig_imgs)
		
		"""
		plt.imshow(orig_imgs[1].cpu().permute(1, 2, 0))
		plt.title('orig_imgs')
		plt.axis('off')
		plt.show()
		"""
		
		#reconstructed=unnormalize(reconstructed)
		#orig_imgs=unnormalize(orig_imgs)
		#alcuni valori possono capitare negativi. Il clamping lo faccio per essere sicuro di riportare il tutto alle condizioni necessarie per poter usare l'ms_sssim
		#reconstructed = torch.clamp(reconstructed, min=0, max=1)
		#orig_imgs = torch.clamp(orig_imgs, min=0, max=1)	
		#true_loss = 1 - ms_ssim_module(reconstructed, orig_imgs)
		#lpips_loss = lpips_model(reconstructed, orig_imgs)
		#true_loss = (torch.mean(lpips_loss))
		true_loss = loss_fn(reconstructed, orig_imgs)

		#plt.imshow(orig_imgs[0].cpu().permute(1, 2, 0))
		#plt.title(orig_imgs)
		#plt.axis('off')
		#plt.show()
		
		model_optimizer.zero_grad(set_to_none=True) #pulisco i gradienti prima del backpropagation step
		
		true_loss.backward() #computo il gradiente per la total loss con rispetto ai parametri
		model_optimizer.step()		
		running_loss += true_loss.item()
	
	running_loss /= batch_size #calcolo la loss media
	train_loss.append(running_loss)
	return running_loss

def val_model_on_disturbed_images(val_dataloader, epoch, device, val_loss, model, model_save_path, model_optimizer, model_scheduler, results_dir, example_dataloader, loss_fn): #funzione che si occupa del test
	model.eval()
	batch_size = len(val_dataloader) #recupero la batch size
	running_loss = 0 # Initializing variable for storing  loss 
	mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
	std = torch.tensor([0.229, 0.224, 0.225]).to(device)
	unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
	#ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], K=(0.01, 0.07))
	#loss_fn = torch.nn.MSELoss()
	#lpips_model = LPIPS(net='vgg').to(device)
	with torch.no_grad(): #non calcolo il gradiente, sto testando e bassa
		for disturbed_imgs, orig_imgs in tqdm(val_dataloader, desc=f'Epoch {epoch} - Validating model'):
			disturbed_imgs, _ = disturbed_imgs.decompose()
			disturbed_imgs = disturbed_imgs.to(device)
			orig_imgs, _ = orig_imgs.decompose()
			orig_imgs = orig_imgs.to(device)
			
			reconstructed = model(disturbed_imgs)
			#print(reconstructed[0].shape[1], reconstructed[0].shape[2])
			trans = transforms.Resize((reconstructed.shape[2], reconstructed.shape[3]), antialias=False)
			orig_imgs = trans(orig_imgs)

			#reconstructed=unnormalize(reconstructed)
			#orig_imgs=unnormalize(orig_imgs)
			#alcuni valori possono capitare negativi. Il clamping lo faccio per essere sicuro di riportare il tutto alle condizioni necessarie per poter usare l'ms_sssim
			#reconstructed = torch.clamp(reconstructed, min=0, max=1)
			#orig_imgs = torch.clamp(orig_imgs, min=0, max=1)	
			#true_loss = 1 - ms_ssim_module(reconstructed, orig_imgs)
			#lpips_loss = lpips_model(reconstructed, orig_imgs)
			#true_loss = (torch.mean(lpips_loss))
			true_loss = loss_fn(reconstructed, orig_imgs)
			running_loss += true_loss.item()		
	running_loss /= batch_size #calcolo la loss media
	val_loss.append(running_loss)
	model_save_path = f'{model_save_path}_{epoch}.pt'   
	create_checkpoint(model, model_optimizer, epoch, running_loss, model_scheduler, model_save_path)
	save_image_examples(example_dataloader, results_dir, model, epoch, device)
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

def val_tasknet(val_dataloader, epoch, device, val_loss, tasknet_save_path, tasknet, tasknet_optimizer, tasknet_scheduler, ap_score_threshold, results_dir): #funzione che si occupa del test
	for param in tasknet.parameters():
		param.requires_grad = False
	res={}	
	batch_size = len(val_dataloader) #recupero la batch size
	running_loss = 0 # Initializing variable for storing  loss 
	evaluator_complete_metric = MyEvaluatorCompleteMetric()
	with torch.no_grad(): #non calcolo il gradiente, sto testando e bassa
		for imgs, targets in tqdm(val_dataloader, desc=f'Epoch {epoch} - Validating model'):
			imgs = list(img.to(device) for img in imgs)
			targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
			tasknet.train()
			loss_dict = tasknet(imgs, targets)
			losses = sum(loss for loss in loss_dict.values())
			running_loss += losses.item()
			tasknet.eval()
			outputs = tasknet(imgs)
			outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
			outputs_evaluator = [{k: v.clone().to(device) for k, v in t.items()} for t in outputs] #devo clonarli perché poi gli originali li modifico
			
			adj_outputs = adjust_outputs_to_cocoeval_api(targets, outputs)

			res.update({target["image_id"].item(): output for target, output in zip(targets, adj_outputs)})
			preds = [apply_nms(pred, iou_thresh=0.5, score_thresh=0.01) for pred in outputs_evaluator]
			#for pred in preds: #Questo non devo farlo, ce le ho già giuste
			# 	pred['labels'] = pred['labels'] - 1
			#con la faster non ho un batch, ho una list
			evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=imgs[0].size()[1:][::-1])
			#evaluator_complete_metric.add_predictions_faster_rcnn(targets=targets, predictions=preds, img_size=imgs.size()[2:][::-1])
			
	running_loss /= batch_size #calcolo la loss media
	compute_ap(val_dataloader, tasknet, epoch, device, ap_score_threshold, results_dir, res)	
	compute_michele_metric(evaluator_complete_metric, results_dir, epoch)	
	val_loss.append(running_loss)
	tasknet_save_path = f'{tasknet_save_path}_{epoch}.pt' 
	create_checkpoint(tasknet, tasknet_optimizer, epoch, running_loss, tasknet_scheduler, tasknet_save_path)
	return running_loss
