import torch
from tqdm import tqdm

from model_utils_and_functions import create_checkpoint, compute_ap

#import per salvare dataset versione disturbata
from torchvision.utils import save_image
import json
import torchvision.transforms as transforms
from coco_eval import get_coco_api_from_dataset
import os

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

def val_model(val_dataloader, epoch, device, val_loss, model, model_save_path, tasknet, model_optimizer, model_scheduler, ap_log_path, ap_score_threshold, my_ap_log_path): #funzione che si occupa del test
	model.eval()
	batch_size = len(val_dataloader) #recupero la batch size
	running_loss = 0 # Initializing variable for storing  loss 
	res={}
	with torch.no_grad(): #non calcolo il gradiente, sto testando e bassa
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
			
			true_loss=reconstructed_losses
			running_loss += true_loss.item()		
	
	running_loss /= batch_size #calcolo la loss media
	
	compute_ap(val_dataloader, tasknet, epoch, device, ap_score_threshold, ap_log_path, my_ap_log_path, res)	
	val_loss.append(running_loss)
	model_save_path = f'{model_save_path}_{epoch}.pt'   
	#create_checkpoint(model, model_optimizer, epoch, running_loss, model_scheduler, model_save_path)
	return running_loss

def generate_disturbed_dataset(train_dataloader_gen_disturbed, val_dataloader_gen_disturbed, epoch, device, model, train_img_folder, train_ann, val_img_folder, val_ann): #funzione che si occupa di generare il dataset disturbato
	model.eval()
	#Prima genero il disturbed training
	batch_size = len(train_dataloader_gen_disturbed) #recupero la batch size
	coco = get_coco_api_from_dataset(train_dataloader_gen_disturbed.dataset)
	disturbed_list = []
	mean = torch.tensor([0.485, 0.456, 0.406])
	std = torch.tensor([0.229, 0.224, 0.225])
	unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
	with torch.no_grad():
		for imgs, targets in tqdm(train_dataloader_gen_disturbed, desc=f'Epoch {epoch} - Generating disturbed train images from dataset'):
			imgs, _ = imgs.decompose()
			imgs = imgs.to(device)
			reconstructed = model(imgs)
			for image, recons in zip(targets, reconstructed):
				coco_image_id = image["image_id"].item()
				path = coco.loadImgs(coco_image_id)[0]["file_name"]
				#le img sono già normalizzate, faccio solo il resize per tenerle della stessa dimensione dell'originale (si perde qualche pixel per via del padding rimosso che veniva fatto con le skip connections dall'unet)
				#orig_size = image['orig_size'].tolist()
				#trans = transforms.Resize(orig_size, antialias=False)
				#recons = trans(recons)
				recons = unnormalize(recons)
				save_image(recons, os.path.join(train_img_folder, path))
				disturbed_list.append({
					"coco_image_path": path,
					"coco_image_id": coco_image_id
				})
	with open(train_ann, 'w') as json_file:
		json.dump(disturbed_list, json_file, indent=2)
	#Ora genero il disturbed val
	batch_size = len(val_dataloader_gen_disturbed) #recupero la batch size
	coco = get_coco_api_from_dataset(val_dataloader_gen_disturbed.dataset)
	disturbed_list = []
	with torch.no_grad():
		for imgs, targets in tqdm(val_dataloader_gen_disturbed, desc=f'Epoch {epoch} - Generating disturbed val images from dataset'):
			imgs, _ = imgs.decompose()
			imgs = imgs.to(device)
			reconstructed = model(imgs)
			for image, recons in zip(targets, reconstructed):
				coco_image_id = image["image_id"].item()
				path = coco.loadImgs(coco_image_id)[0]["file_name"]
				#orig_size = image['orig_size'].tolist()
				#trans = transforms.Resize(orig_size, antialias=False)
				#recons = trans(recons)
				recons = unnormalize(recons)
				save_image(recons, os.path.join(val_img_folder, path))
				disturbed_list.append({
					"coco_image_path": path,
					"coco_image_id": coco_image_id
				})
	with open(val_ann, 'w') as json_file:
		json.dump(disturbed_list, json_file, indent=2)
	model.train()

from pytorch_msssim import ms_ssim, MS_SSIM
def train_model_on_disturbed_images(train_dataloader, epoch, device, train_loss, model, model_optimizer): #funzione che si occupa del training
	model.train()
	#loss_fn = torch.nn.MSELoss() # Mean squared loss which computes difference between two images.
	batch_size = len(train_dataloader) #recupero la batch size
	running_loss = 0 # Iniziallizzo la variabile per la loss
	mean = torch.tensor([0.485, 0.456, 0.406])
	std = torch.tensor([0.229, 0.224, 0.225])
	unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
	ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)
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
		trans = transforms.Resize((reconstructed[0].shape[1], reconstructed[0].shape[2]), antialias=False)
		orig_imgs = trans(orig_imgs)
		
		"""
		plt.imshow(orig_imgs[1].cpu().permute(1, 2, 0))
		plt.title('orig_imgs')
		plt.axis('off')
		plt.show()
		"""
		
		reconstructed=unnormalize(reconstructed)
		orig_imgs=unnormalize(orig_imgs)	
		true_loss = 1 - ms_ssim_module(reconstructed, orig_imgs)
		#true_loss = loss_fn(reconstructed, orig_imgs)
		
		model_optimizer.zero_grad(set_to_none=True) #pulisco i gradienti prima del backpropagation step
		
		true_loss.backward() #computo il gradiente per la total loss con rispetto ai parametri
		model_optimizer.step()		
		running_loss += true_loss.item()
	
	running_loss /= batch_size #calcolo la loss media
	train_loss.append(running_loss)
	return running_loss

def val_model_on_disturbed_images(val_dataloader, epoch, device, val_loss, model, model_save_path, model_optimizer, model_scheduler): #funzione che si occupa del test
	model.eval()
	#loss_fn = torch.nn.MSELoss() # Mean squared loss which computes difference between two images.
	batch_size = len(val_dataloader) #recupero la batch size
	running_loss = 0 # Initializing variable for storing  loss 
	mean = torch.tensor([0.485, 0.456, 0.406])
	std = torch.tensor([0.229, 0.224, 0.225])
	unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
	ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)
	with torch.no_grad(): #non calcolo il gradiente, sto testando e bassa
		for disturbed_imgs, orig_imgs in tqdm(val_dataloader, desc=f'Epoch {epoch} - Validating model'):
			disturbed_imgs, _ = disturbed_imgs.decompose()
			disturbed_imgs = disturbed_imgs.to(device)
			orig_imgs, _ = orig_imgs.decompose()
			orig_imgs = orig_imgs.to(device)
			
			# For testing
			"""
			import matplotlib.pyplot as plt
			import matplotlib.image as mpimg
			plt.imshow(disturbed_imgs[0].cpu().permute(1, 2, 0))
			plt.title('disturbed_imgs')
			plt.axis('off')
			plt.show()
			
			plt.imshow(orig_imgs[0].cpu().permute(1, 2, 0))
			plt.title('orig_imgs')
			plt.axis('off')
			plt.show()
			
			plt.imshow(disturbed_imgs[1].cpu().permute(1, 2, 0))
			plt.title('disturbed_imgs')
			plt.axis('off')
			plt.show()
			
			plt.imshow(orig_imgs[1].cpu().permute(1, 2, 0))
			plt.title('orig_imgs')
			plt.axis('off')
			plt.show()
			"""
			
			reconstructed = model(disturbed_imgs)
			#print(reconstructed[0].shape[1], reconstructed[0].shape[2])
			trans = transforms.Resize((reconstructed[0].shape[1], reconstructed[0].shape[2]), antialias=False)
			orig_imgs = trans(orig_imgs)
			"""
			plt.imshow(orig_imgs[1].cpu().permute(1, 2, 0))
			plt.title('orig_imgs')
			plt.axis('off')
			plt.show()
			"""
			reconstructed=unnormalize(reconstructed)
			orig_imgs=unnormalize(orig_imgs)	
			true_loss = 1 - ms_ssim_module(reconstructed, orig_imgs)
			#true_loss = loss_fn(reconstructed, orig_imgs)
			running_loss += true_loss.item()		
	running_loss /= batch_size #calcolo la loss media
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

def val_tasknet(val_dataloader, epoch, device, val_loss, tasknet_save_path, tasknet, tasknet_optimizer, tasknet_scheduler, ap_log_path, ap_score_threshold, my_ap_log_path): #funzione che si occupa del test
	for param in tasknet.parameters():
		param.requires_grad = False
	res={}	
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
			outputs = [{k: v.to(device) for k, v in t.items()} for t in outputs]
			
			res.update({target["image_id"].item(): output for target, output in zip(targets, outputs)})
			
	running_loss /= batch_size #calcolo la loss media
	compute_ap(val_dataloader, tasknet, epoch, device, ap_score_threshold, ap_log_path, my_ap_log_path, res)	
	val_loss.append(running_loss)
	tasknet_save_path = f'{tasknet_save_path}_{epoch}.pt' 
	create_checkpoint(tasknet, tasknet_optimizer, epoch, running_loss, tasknet_scheduler, tasknet_save_path)
	return running_loss
