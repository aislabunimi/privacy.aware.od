import matplotlib.pyplot as plt
#import torch
from PIL import Image

#faccio con * direttamente che cosi import pure torch, PIL e transforms che sono dipendenze
from dataset import * #get_transform

import torchvision.transforms as transforms
from model_utils_and_functions import apply_nms
from model_utils_and_functions import load_checkpoint
			
def plot_img_after_transform(img_tensor, boxes):
    plt.figure(figsize=(16,10))
    img_tensor = img_tensor.permute(1, 2, 0)
    plt.imshow(img_tensor)
    ax = plt.gca()
    for data in boxes:
    	#xmin, ymin, xmax, ymax, prob, class_id = data #formato box xyxy, xmax-xminn è width, ymax-ymin è height
    	xmin, ymin, xmax, ymax = data
    	color = 'red'
    	ax.add_patch(plt.Rectangle((xmin, ymin), (xmax-xmin), (ymax-ymin), fill=False, color=color, linewidth=3))
    #plt.axis('off')
    #plt.tight_layout()
    plt.show()
    
def show_res_test_unet(model, tasknet, device, img_file_path, not_reconstructed, name_path_save): #funzione che si occupa del test
	model.eval() #metto il modello in evaluate
	tasknet.eval()
	image = Image.open(img_file_path).convert("RGB")
	eval_transform = get_transform()
	img = eval_transform(image)
	img = img.unsqueeze(0)
	img = img.to(device)
	
	mean = torch.tensor([0.485, 0.456, 0.406])
	std = torch.tensor([0.229, 0.224, 0.225])
	unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
	
	res_tasknet = tasknet(img)
	#print(res_tasknet[0])
	#print(res_tasknet[0]['boxes'])
	
	#pred_tasknet = res_tasknet[0]
	img_primo_plot = img.squeeze(0)
	img_primo_plot = unnormalize(img_primo_plot)

	nms_pred = apply_nms(res_tasknet[0], iou_thresh=0.1)
	
	plt.subplot(1, 3, 1)
	
	print("Original Labels and scores: ", nms_pred["labels"], nms_pred["scores"])
	plt.title('Original Image, bbox by only tasknet')
	plt.axis('off')
	#plt.imshow(img_to_plot.cpu().permute(1, 2, 0))
	plot_results(img_primo_plot, nms_pred['scores'], nms_pred['boxes'])
	
	if not_reconstructed: #cioe se l'immagine non è una ricostruita
		#out = model(img)
		out = model(img)

		out_to_plot = unnormalize(out)

		#La transform qui sotto fixxa il mismatch visivo fra ouput e ricostruzione
		#out_to_plot = transforms.functional.convert_image_dtype(out[0], torch.uint8)
		#out_to_plot = out_to_plot.squeeze(0) #necessario per lvpga
		#recons_out = out_to_plot #serve dopo per vedere di fare le bbox sopra la ricostruita
		#out_only_recon = out_to_plot.detach().cpu().numpy().transpose(1,2,0)
	else:
		out = img
		out_to_plot = img
	plt.subplot(1, 3, 2)
	#plt.subplot(2, 2, 2)
	#plt.imshow(out_only_recon)
	plt.imshow(out_to_plot.squeeze(0).detach().cpu().numpy().transpose(1,2,0))
	plt.title('Reconstructed Image')
	plt.axis('off')
	plt.tight_layout()
	plt.subplot(1, 3, 3)
	#plt.subplot(2, 2, 3)
	#reconstructed = tasknet(out_to_plot)
	reconstructed = tasknet(out)
	pred_recon = reconstructed[0]
	#img = img.squeeze(0)
	
	nms_pred_recon = apply_nms(pred_recon, iou_thresh=0.1) #per applicare nms e salvare l'ultima box
	
	print("Reconstructed Labels and Scores: ", nms_pred_recon["labels"], nms_pred_recon["scores"])
	
	plot_results(out_to_plot, nms_pred_recon['scores'], nms_pred_recon['boxes'])
	plt.title('Reconstructed Image, bbox by full model')
	plt.axis('off')
	plt.savefig(name_path_save, format='png', bbox_inches='tight')


  
def get_transform():
    transform = []
    transform.append(transforms.PILToTensor())
    transform.append(transforms.ConvertImageDtype(torch.float))
    transform.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(transform)   

def plot_results(img, prob, boxes):
    #plt.figure(figsize=(16,10))
    plt.axis('off')
    plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0).detach().numpy())
    ax = plt.gca()
    prob = prob.cpu()
    boxes = boxes.cpu()
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color='red', linewidth=3))
        text = f'{p:0.3f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

from pytorch_msssim import ms_ssim, MS_SSIM	
def compare_two_results_unet(unet, tasknet, device, img_file_path, name_path_save, unet_weights_load, unet_weights_to_compare, unet_optimizer, unet_scheduler): 

	unet.eval() #metto il modello in evaluate
	tasknet.eval()
	plt.figure(figsize=(15, 10))
	mean = torch.tensor([0.485, 0.456, 0.406])
	std = torch.tensor([0.229, 0.224, 0.225])
	unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
	
	image = Image.open(img_file_path).convert("RGB")
	eval_transform = get_transform()
	img = eval_transform(image)
	img = img.unsqueeze(0)
	img = img.to(device)
	
	res_tasknet = tasknet(img)
	img_primo_plot = img.squeeze(0)
	img_primo_plot = unnormalize(img_primo_plot)
	nms_pred = apply_nms(res_tasknet[0], iou_thresh=0.1)
		
	load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
	out = unet(img)
	#preparo poi per calcolo msssim
	ms_ssim_module = MS_SSIM(data_range=1, size_average=True, channel=3)
	trans_r = transforms.Resize((out.shape[2], out.shape[3]), antialias=False)
	orig_img = trans_r(img)
	orig_img = unnormalize(orig_img)
	
	img_primo_plot = trans_r(img_primo_plot) #per fare img uguale a quelle ricostruite
	plt.subplot(1, 3, 1)
	plt.title('Original Image', fontsize=16)	
	plot_results(img_primo_plot, nms_pred['scores'], nms_pred['boxes'])
	
	plt.subplot(1, 3, 2)
	out_to_plot = unnormalize(out)
	reconstructed = tasknet(out)
	pred_recon = reconstructed[0]
	nms_pred_recon = apply_nms(pred_recon, iou_thresh=0.1) #per applicare nms e salvare l'ultima box
	filename = os.path.basename(unet_weights_load)
	name = os.path.splitext(filename)[0]	
	ms_ssim_score = ms_ssim_module(out_to_plot, orig_img)
	plt.title(f'{name}, MS_SSIM: {ms_ssim_score:0.3f}', fontsize=16)
	plot_results(out_to_plot, nms_pred_recon['scores'], nms_pred_recon['boxes'])
	plt.subplot(1, 3, 3)
	load_checkpoint(unet, unet_weights_to_compare, unet_optimizer, unet_scheduler)
	out = unet(img)
	#out_to_plot = transforms.functional.convert_image_dtype(out[0], torch.uint8)
	out_to_plot = unnormalize(out)
	reconstructed = tasknet(out)
	pred_recon = reconstructed[0]
	nms_pred_recon = apply_nms(pred_recon, iou_thresh=0.1) #per applicare nms e salvare l'ultima box
	filename = os.path.basename(unet_weights_to_compare)
	name = os.path.splitext(filename)[0]
	ms_ssim_score = ms_ssim_module(out_to_plot, orig_img)
	plt.title(f'{name}, MS_SSIM: {ms_ssim_score:0.3f}', fontsize=16)
	plot_results(out_to_plot, nms_pred_recon['scores'], nms_pred_recon['boxes'])
	
	plt.subplots_adjust(wspace=0.05)
	plt.savefig(name_path_save, format='png', bbox_inches='tight')
	plt.clf()

from torchvision.utils import save_image
def save_disturbed_pred(unet, device, img_file_path, name_path_save):
	image = Image.open(img_file_path).convert("RGB")
	eval_transform = transforms.Compose([ transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float)])
	img = eval_transform(image)
	img = img.to(device)
	img = img.unsqueeze(0)
	recons = unet(img)
	mean = torch.tensor([0.485, 0.456, 0.406])
	std = torch.tensor([0.229, 0.224, 0.225])
	unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
	recons = unnormalize(recons)
	save_image(recons, name_path_save)
