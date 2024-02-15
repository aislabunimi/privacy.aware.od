import torch
import torchvision
import random
import os
import numpy as np

#import per la funzione compute_ap
from coco_eval import *
from coco_eval import _get_iou_types
import sys
#import per la funzione di compute_michele_metric
import json
from michele_metric.my_evaluators_complete_metric import MyEvaluatorCompleteMetric

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False

def deterministic_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)

def deterministic_generator():
	g = torch.Generator()
	g.manual_seed(0)
	return g

def compute_ap(val_dataloader, tasknet, epoch, device, ap_score_threshold, results_dir, res):

	coco = get_coco_api_from_dataset(val_dataloader.dataset)

	iou_types = _get_iou_types(tasknet)
	coco_evaluator = CocoEvaluator(coco, iou_types) #AP con tutte le pred, curva non interpolata
	my_ap_evaluator = CocoEvaluator(coco, iou_types) #AP con tutte le pred, curva interpolata
	my_ap_nointerp_thresh = CocoEvaluator(coco, iou_types) #AP con solo pred superiori score thresh, curva non interpolata
	my_ap_interp_thresh = CocoEvaluator(coco, iou_types) #AP con solo pred superiori score thresh, curva interpolata
	
	coco_evaluator_iou50 = CocoEvaluator(coco, iou_types) #come coco_evaluator solo limitando iou thresh
	my_ap_evaluator_iou50 = CocoEvaluator(coco, iou_types)
	my_ap_nointerp_thresh_iou50 = CocoEvaluator(coco, iou_types)
	my_ap_interp_thresh_iou50 = CocoEvaluator(coco, iou_types)
	
	coco_evaluator_iou75 = CocoEvaluator(coco, iou_types) #come coco_evaluator solo limitando iou thresh
	my_ap_evaluator_iou75 = CocoEvaluator(coco, iou_types)
	my_ap_nointerp_thresh_iou75 = CocoEvaluator(coco, iou_types)
	my_ap_interp_thresh_iou75 = CocoEvaluator(coco, iou_types)
	
	#new_interpolation = np.linspace(.0, (1-ap_score_threshold), int(np.round(((1-ap_score_threshold) - .0) / .01)) + 1, endpoint=True)
	new_interpolation = np.linspace(.0, (1-ap_score_threshold), int(np.round(((1-ap_score_threshold) - .0) / (1-ap_score_threshold)/.01)) + 1, endpoint=True) #questa fa 101 point sulla nuova curva, quella sopra faceva solo 26 punti. La differenza in AP è però trascurabile, ma mi sembra più corretto i 101 punti piuttosto che 26
	iou50_thresh=[0.5] #se voglio guardare solo quella
	iou75_thresh=[0.75]
	my_ap_evaluator.coco_eval['bbox'].params.recThrs = new_interpolation #necessaria visto che devo prendere i punti solo fino allo ap_score_threshold
	my_ap_interp_thresh.coco_eval['bbox'].params.recThrs = new_interpolation
	
	coco_evaluator_iou50.coco_eval['bbox'].params.iouThrs= iou50_thresh
	my_ap_evaluator_iou50.coco_eval['bbox'].params.iouThrs= iou50_thresh
	my_ap_nointerp_thresh_iou50.coco_eval['bbox'].params.iouThrs= iou50_thresh
	my_ap_interp_thresh_iou50.coco_eval['bbox'].params.iouThrs= iou50_thresh
	my_ap_evaluator_iou50.coco_eval['bbox'].params.recThrs = new_interpolation
	my_ap_interp_thresh_iou50.coco_eval['bbox'].params.recThrs = new_interpolation
	
	coco_evaluator_iou75.coco_eval['bbox'].params.iouThrs= iou75_thresh
	my_ap_evaluator_iou75.coco_eval['bbox'].params.iouThrs= iou75_thresh
	my_ap_nointerp_thresh_iou75.coco_eval['bbox'].params.iouThrs= iou75_thresh
	my_ap_interp_thresh_iou75.coco_eval['bbox'].params.iouThrs= iou75_thresh
	my_ap_evaluator_iou75.coco_eval['bbox'].params.recThrs = new_interpolation
	my_ap_interp_thresh_iou75.coco_eval['bbox'].params.recThrs = new_interpolation
	
	
	execute_coco_eval(f'{results_dir}/standard_ap.txt', f'AP for Epoch {epoch}', coco_evaluator, res)
	execute_coco_eval(f'{results_dir}/myinterp_ap.txt', f'AP for Epoch {epoch} with all predictions (not filtered by score thresh) but with my curve interpolation, score thresh: {ap_score_threshold}', my_ap_evaluator, res)
	
	execute_coco_eval(f'{results_dir}/standard_ap_iou50.txt', f'AP for Epoch {epoch}', coco_evaluator_iou50, res)
	execute_coco_eval(f'{results_dir}/myinterp_ap_iou50.txt', f'AP for Epoch {epoch} with all predictions (not filtered by score thresh) but with my curve interpolation, score thresh: {ap_score_threshold}', my_ap_evaluator_iou50, res)
	
	execute_coco_eval(f'{results_dir}/standard_ap_iou75.txt', f'AP for Epoch {epoch}', coco_evaluator_iou75, res)
	execute_coco_eval(f'{results_dir}/myinterp_ap_iou75.txt', f'AP for Epoch {epoch} with all predictions (not filtered by score thresh) but with my curve interpolation, score thresh: {ap_score_threshold}', my_ap_evaluator_iou75, res)
	
	#per filtrare solo le pred con score pari a ap_score_threshold o superiore
	for image_id, pred in res.items():
		filtered_boxes = []
		filtered_scores = []
		filtered_labels = []
		for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
			if score >= ap_score_threshold:
				filtered_boxes.append(box)
				filtered_scores.append(score)
				filtered_labels.append(label)
		# Se ci sono box filtrate aggiorna res dell'immagine con quelle, se no tensore vuoto
		if filtered_boxes:
			res[image_id] = {'boxes': torch.stack(filtered_boxes),
                                         'labels': torch.tensor(filtered_labels),
                                         'scores': torch.tensor(filtered_scores)}
		else:
                	res[image_id] = {'boxes': torch.empty((0,4)).to(device),
                			 'labels': torch.empty((0,)).to(device),
                			 'scores': torch.empty((0,)).to(device)}
	execute_coco_eval(f'{results_dir}/standard_ap_scoreabovethresh.txt', f'AP for Epoch {epoch} with only pred above score thresh: {ap_score_threshold}', my_ap_nointerp_thresh, res)
	execute_coco_eval(f'{results_dir}/myinterp_ap_scoreabovethresh.txt', f'AP for Epoch {epoch} with only pred above score thresh and my curve interpolation, score thresh: {ap_score_threshold}', my_ap_interp_thresh, res)
	
	execute_coco_eval(f'{results_dir}/standard_ap_scoreabovethresh_iou50.txt', f'AP for Epoch {epoch} with only pred above score thresh: {ap_score_threshold}', my_ap_nointerp_thresh_iou50, res)
	execute_coco_eval(f'{results_dir}/myinterp_ap_scoreabovethresh_iou50.txt', f'AP for Epoch {epoch} with only pred above score thresh and my curve interpolation, score thresh: {ap_score_threshold}', my_ap_interp_thresh_iou50, res)
	
	execute_coco_eval(f'{results_dir}/standard_ap_scoreabovethresh_iou75.txt', f'AP for Epoch {epoch} with only pred above score thresh: {ap_score_threshold}', my_ap_nointerp_thresh_iou75, res)
	execute_coco_eval(f'{results_dir}/myinterp_ap_scoreabovethresh_iou75.txt', f'AP for Epoch {epoch} with only pred above score thresh and my curve interpolation, score thresh: {ap_score_threshold}', my_ap_interp_thresh_iou75, res)
	
def execute_coco_eval(ap_log_path, print_msg, coco_evaluator, res):
	coco_evaluator.update(res)
	coco_evaluator.synchronize_between_processes()
	coco_evaluator.coco_eval['bbox'].evaluate()
	coco_evaluator.accumulate()	
	orig_stdout = sys.stdout
	f = open(ap_log_path, 'a')
	sys.stdout = f
	print(print_msg)
	coco_evaluator.summarize()	
	sys.stdout = orig_stdout
	f.close()

def compute_michele_metric(evaluator_complete_metric, michele_metric_folder, epoch):
	#iou_threshold=0.5
	#confidence_threshold=0.5
	complete_metrics = {}
	for iou_threshold in np.arange(0.5, 0.76, 0.25): #np.arange(0.5, 0.96, 0.05):
		for confidence_threshold in np.arange(0.5, 0.76, 0.25): #np.arange(0.5, 0.96, 0.05):
			iou_threshold = round(iou_threshold, 2)
			confidence_threshold = round(confidence_threshold, 2)
			complete_metrics[(iou_threshold, confidence_threshold)] = evaluator_complete_metric.get_metrics(iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, door_no_door_task=False, plot_curves=False)
	#complete_metrics[(iou_threshold, confidence_threshold)] = evaluator_complete_metric.get_metrics(iou_threshold=iou_threshold, confidence_threshold=confidence_threshold, door_no_door_task=False, plot_curves=False)
	#print(complete_metrics)
	filtered_metrics = {key: value['1'] for key, value in complete_metrics.items()} #prendo solo label persone
	for m in filtered_metrics.items():
		save_path=f'{michele_metric_folder}/iou{m[0][0]}_score{m[0][1]}.json'
		m[1]['epoch'] = epoch #aggiungo il campo epoca cosi recupero più facilmente
		if epoch>1:
			with open(save_path, 'r') as json_file:
				data = json.load(json_file)
			data.append(m[1])
			with open(save_path, 'w') as json_file:
				json.dump(data, json_file, indent=2)
		else:
			temp_list=[m[1]]
			with open(save_path, 'w') as json_file:
				json.dump(temp_list, json_file, indent=2)

def load_my_recons_classifier(my_recons_classifier_weights, device):
   vgg16 = torchvision.models.vgg16()
   num_classes=4
   vgg16.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes)
   vgg16_optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
   vgg16_scheduler = torch.optim.lr_scheduler.StepLR(vgg16_optimizer, step_size=10, gamma=0.5)
   load_checkpoint(vgg16, my_recons_classifier_weights, vgg16_optimizer, vgg16_scheduler)
   vgg16.to(device)
   vgg16.eval()
   return vgg16
   
def save_my_recons_classifier_dict(classifier_path, epoch, my_rec_class_dict):
   for value in range(0,4):
      if value not in my_rec_class_dict:
         my_rec_class_dict[value] = 0
   if epoch>1:
      with open(classifier_path, 'r') as json_file:
         data = json.load(json_file)
      data.append(my_rec_class_dict)
      with open(classifier_path, 'w') as json_file:
         json.dump(data, json_file, indent=2)
   else:
      temp_list=[my_rec_class_dict]
      with open(classifier_path, 'w') as json_file:
         json.dump(temp_list, json_file, indent=2)

def compute_my_recons_classifier_pred(my_recons_classifier, reconstructed, my_rec_class_dict):
   outputs = my_recons_classifier(reconstructed)
   _, predicted = torch.max(outputs.data, 1)
   values, occurrences = torch.unique(predicted, return_counts=True)
   my_rec_class_dict['total']+=len(reconstructed)
   for value, occurrence in zip(values, occurrences):
      value_item = value.item()
      occurrence_item = occurrence.item()
      if value_item in my_rec_class_dict:
         my_rec_class_dict[value_item] += occurrence_item
      else:
         my_rec_class_dict[value_item] = occurrence_item
   outputs = torch.nn.functional.softmax(outputs) #ottengo le prob per ogni classe
   if 'prob0tot' not in my_rec_class_dict:
      my_rec_class_dict['prob0tot']=0
   if 'prob1tot' not in my_rec_class_dict:
      my_rec_class_dict['prob1tot']=0
   if 'prob2tot' not in my_rec_class_dict:
      my_rec_class_dict['prob2tot']=0
   if 'prob3tot' not in my_rec_class_dict:
      my_rec_class_dict['prob3tot']=0
   for prob0, prob1, prob2, prob3 in outputs:
      my_rec_class_dict['prob0tot'] += prob0.item()
      my_rec_class_dict['prob1tot'] += prob1.item()
      my_rec_class_dict['prob2tot'] += prob2.item()
      my_rec_class_dict['prob3tot'] += prob3.item()

import torch.nn as nn
class CNNRegression(nn.Module):
    def __init__(self):
        super(CNNRegression, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 1)  

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = torch.relu(self.conv3(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) 
        return x
     
def load_my_regressor(my_regressor_weights, device):
   model = CNNRegression()  
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
   load_checkpoint(model, my_regressor_weights, optimizer, scheduler)
   model.to(device)
   model.eval()
   return model

def create_checkpoint(model, optimizer, current_epoch, current_loss, scheduler, model_save_path):
	torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': current_loss,
            'lr_scheduler': scheduler.state_dict(),
            #'train_loss': train_loss,
            #'val_loss': val_loss,
            }, model_save_path)
	print(f'Checkpoint created at epoch {current_epoch}')
            
def load_checkpoint(model, model_save_path, optimizer, scheduler):
	checkpoint = torch.load(model_save_path)
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
	scheduler.load_state_dict(checkpoint['lr_scheduler'])
	#train_loss += checkpoint['train_loss']
	#val_loss += checkpoint['val_loss']
	print('Model Loaded')
	return epoch

def load_checkpoint_encoder(model, model_save_path, optimizer, scheduler):
	checkpoint = torch.load(model_save_path)
	#model.load_state_dict(checkpoint['model_state_dict'])
	mod = checkpoint['model_state_dict']
	#for key, value in mod.items() :
    	#	print (key)
	with torch.no_grad():
		model.inc.double_conv[0].weight.copy_(mod['inc.double_conv.0.weight'])
		model.inc.double_conv[1].weight.copy_(mod['inc.double_conv.1.weight'])
		model.inc.double_conv[1].bias.copy_(mod['inc.double_conv.1.bias'])
		model.inc.double_conv[1].running_mean.copy_(mod['inc.double_conv.1.running_mean'])
		model.inc.double_conv[1].running_var.copy_(mod['inc.double_conv.1.running_var'])
		model.inc.double_conv[1].num_batches_tracked.copy_(mod['inc.double_conv.1.num_batches_tracked'])
		model.inc.double_conv[3].weight.copy_(mod['inc.double_conv.3.weight'])
		model.inc.double_conv[4].weight.copy_(mod['inc.double_conv.4.weight'])
		model.inc.double_conv[4].bias.copy_(mod['inc.double_conv.4.bias'])
		model.inc.double_conv[4].running_mean.copy_(mod['inc.double_conv.4.running_mean'])
		model.inc.double_conv[4].running_var.copy_(mod['inc.double_conv.4.running_var'])
		model.inc.double_conv[4].num_batches_tracked.copy_(mod['inc.double_conv.4.num_batches_tracked'])
		model.down1.maxpool_conv[1].double_conv[0].weight.copy_(mod['down1.maxpool_conv.1.double_conv.0.weight'])
		model.down1.maxpool_conv[1].double_conv[1].weight.copy_(mod['down1.maxpool_conv.1.double_conv.1.weight'])
		model.down1.maxpool_conv[1].double_conv[1].bias.copy_(mod['down1.maxpool_conv.1.double_conv.1.bias'])
		model.down1.maxpool_conv[1].double_conv[1].running_mean.copy_(mod['down1.maxpool_conv.1.double_conv.1.running_mean'])
		model.down1.maxpool_conv[1].double_conv[1].running_var.copy_(mod['down1.maxpool_conv.1.double_conv.1.running_var'])
		model.down1.maxpool_conv[1].double_conv[1].num_batches_tracked.copy_(mod['down1.maxpool_conv.1.double_conv.1.num_batches_tracked'])
		model.down1.maxpool_conv[1].double_conv[3].weight.copy_(mod['down1.maxpool_conv.1.double_conv.3.weight'])
		model.down1.maxpool_conv[1].double_conv[4].weight.copy_(mod['down1.maxpool_conv.1.double_conv.4.weight'])
		model.down1.maxpool_conv[1].double_conv[4].bias.copy_(mod['down1.maxpool_conv.1.double_conv.4.bias'])
		model.down1.maxpool_conv[1].double_conv[4].running_mean.copy_(mod['down1.maxpool_conv.1.double_conv.4.running_mean'])
		model.down1.maxpool_conv[1].double_conv[4].running_var.copy_(mod['down1.maxpool_conv.1.double_conv.4.running_var'])
		model.down1.maxpool_conv[1].double_conv[4].num_batches_tracked.copy_(mod['down1.maxpool_conv.1.double_conv.4.num_batches_tracked'])
		model.down2.maxpool_conv[1].double_conv[0].weight.copy_(mod['down2.maxpool_conv.1.double_conv.0.weight'])
		model.down2.maxpool_conv[1].double_conv[1].weight.copy_(mod['down2.maxpool_conv.1.double_conv.1.weight'])
		model.down2.maxpool_conv[1].double_conv[1].bias.copy_(mod['down2.maxpool_conv.1.double_conv.1.bias'])
		model.down2.maxpool_conv[1].double_conv[1].running_mean.copy_(mod['down2.maxpool_conv.1.double_conv.1.running_mean'])
		model.down2.maxpool_conv[1].double_conv[1].running_var.copy_(mod['down2.maxpool_conv.1.double_conv.1.running_var'])
		model.down2.maxpool_conv[1].double_conv[1].num_batches_tracked.copy_(mod['down2.maxpool_conv.1.double_conv.1.num_batches_tracked'])
		model.down2.maxpool_conv[1].double_conv[3].weight.copy_(mod['down2.maxpool_conv.1.double_conv.3.weight'])
		model.down2.maxpool_conv[1].double_conv[4].weight.copy_(mod['down2.maxpool_conv.1.double_conv.4.weight'])
		model.down2.maxpool_conv[1].double_conv[4].bias.copy_(mod['down2.maxpool_conv.1.double_conv.4.bias'])
		model.down2.maxpool_conv[1].double_conv[4].running_mean.copy_(mod['down2.maxpool_conv.1.double_conv.4.running_mean'])
		model.down2.maxpool_conv[1].double_conv[4].running_var.copy_(mod['down2.maxpool_conv.1.double_conv.4.running_var'])
		model.down2.maxpool_conv[1].double_conv[4].num_batches_tracked.copy_(mod['down2.maxpool_conv.1.double_conv.4.num_batches_tracked'])
		model.down3.maxpool_conv[1].double_conv[0].weight.copy_(mod['down3.maxpool_conv.1.double_conv.0.weight'])
		model.down3.maxpool_conv[1].double_conv[1].weight.copy_(mod['down3.maxpool_conv.1.double_conv.1.weight'])
		model.down3.maxpool_conv[1].double_conv[1].bias.copy_(mod['down3.maxpool_conv.1.double_conv.1.bias'])
		model.down3.maxpool_conv[1].double_conv[1].running_mean.copy_(mod['down3.maxpool_conv.1.double_conv.1.running_mean'])
		model.down3.maxpool_conv[1].double_conv[1].running_var.copy_(mod['down3.maxpool_conv.1.double_conv.1.running_var'])
		model.down3.maxpool_conv[1].double_conv[1].num_batches_tracked.copy_(mod['down3.maxpool_conv.1.double_conv.1.num_batches_tracked'])
		model.down3.maxpool_conv[1].double_conv[3].weight.copy_(mod['down3.maxpool_conv.1.double_conv.3.weight'])
		model.down3.maxpool_conv[1].double_conv[4].weight.copy_(mod['down3.maxpool_conv.1.double_conv.4.weight'])
		model.down3.maxpool_conv[1].double_conv[4].bias.copy_(mod['down3.maxpool_conv.1.double_conv.4.bias'])
		model.down3.maxpool_conv[1].double_conv[4].running_mean.copy_(mod['down3.maxpool_conv.1.double_conv.4.running_mean'])
		model.down3.maxpool_conv[1].double_conv[4].running_var.copy_(mod['down3.maxpool_conv.1.double_conv.4.running_var'])
		model.down3.maxpool_conv[1].double_conv[4].num_batches_tracked.copy_(mod['down3.maxpool_conv.1.double_conv.4.num_batches_tracked'])
		model.down4.maxpool_conv[1].double_conv[0].weight.copy_(mod['down4.maxpool_conv.1.double_conv.0.weight'])
		model.down4.maxpool_conv[1].double_conv[1].weight.copy_(mod['down4.maxpool_conv.1.double_conv.1.weight'])
		model.down4.maxpool_conv[1].double_conv[1].bias.copy_(mod['down4.maxpool_conv.1.double_conv.1.bias'])
		model.down4.maxpool_conv[1].double_conv[1].running_mean.copy_(mod['down4.maxpool_conv.1.double_conv.1.running_mean'])
		model.down4.maxpool_conv[1].double_conv[1].running_var.copy_(mod['down4.maxpool_conv.1.double_conv.1.running_var'])
		model.down4.maxpool_conv[1].double_conv[1].num_batches_tracked.copy_(mod['down4.maxpool_conv.1.double_conv.1.num_batches_tracked'])
		model.down4.maxpool_conv[1].double_conv[3].weight.copy_(mod['down4.maxpool_conv.1.double_conv.3.weight'])
		model.down4.maxpool_conv[1].double_conv[4].weight.copy_(mod['down4.maxpool_conv.1.double_conv.4.weight'])
		model.down4.maxpool_conv[1].double_conv[4].bias.copy_(mod['down4.maxpool_conv.1.double_conv.4.bias'])
		model.down4.maxpool_conv[1].double_conv[4].running_mean.copy_(mod['down4.maxpool_conv.1.double_conv.4.running_mean'])
		model.down4.maxpool_conv[1].double_conv[4].running_var.copy_(mod['down4.maxpool_conv.1.double_conv.4.running_var'])
		model.down4.maxpool_conv[1].double_conv[4].num_batches_tracked.copy_(mod['down4.maxpool_conv.1.double_conv.4.num_batches_tracked'])

	
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
	scheduler.load_state_dict(checkpoint['lr_scheduler'])
	#train_loss += checkpoint['train_loss']
	#val_loss += checkpoint['val_loss']
	load_checkpoint_decoder(model, model_save_path, optimizer, scheduler)
	print('Model Loaded')

def freeze_encoder(model):
	model.inc.double_conv[0].weight.requires_grad=False
	model.inc.double_conv[1].weight.requires_grad=False
	model.inc.double_conv[1].bias.requires_grad=False
	model.inc.double_conv[1].running_mean.requires_grad=False
	model.inc.double_conv[1].running_var.requires_grad=False
	model.inc.double_conv[1].num_batches_tracked.requires_grad=False
	model.inc.double_conv[3].weight.requires_grad=False
	model.inc.double_conv[4].weight.requires_grad=False
	model.inc.double_conv[4].bias.requires_grad=False
	model.inc.double_conv[4].running_mean.requires_grad=False
	model.inc.double_conv[4].running_var.requires_grad=False
	model.inc.double_conv[4].num_batches_tracked.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[0].weight.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[1].weight.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[1].bias.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[1].running_mean.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[1].running_var.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[1].num_batches_tracked.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[3].weight.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[4].weight.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[4].bias.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[4].running_mean.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[4].running_var.requires_grad=False
	model.down1.maxpool_conv[1].double_conv[4].num_batches_tracked.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[0].weight.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[1].weight.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[1].bias.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[1].running_mean.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[1].running_var.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[1].num_batches_tracked.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[3].weight.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[4].weight.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[4].bias.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[4].running_mean.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[4].running_var.requires_grad=False
	model.down2.maxpool_conv[1].double_conv[4].num_batches_tracked.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[0].weight.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[1].weight.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[1].bias.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[1].running_mean.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[1].running_var.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[1].num_batches_tracked.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[3].weight.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[4].weight.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[4].bias.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[4].running_mean.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[4].running_var.requires_grad=False
	model.down3.maxpool_conv[1].double_conv[4].num_batches_tracked.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[0].weight.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[1].weight.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[1].bias.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[1].running_mean.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[1].running_var.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[1].num_batches_tracked.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[3].weight.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[4].weight.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[4].bias.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[4].running_mean.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[4].running_var.requires_grad=False
	model.down4.maxpool_conv[1].double_conv[4].num_batches_tracked.requires_grad=False
	print('Encoder Freezed')

def load_checkpoint_decoder(model, model_save_path, optimizer, scheduler):
	checkpoint = torch.load(model_save_path)
	#model.load_state_dict(checkpoint['model_state_dict'])
	mod = checkpoint['model_state_dict']
	#for key, value in mod.items() :
    	#	print (key)
	with torch.no_grad():
		model.up1.up.weight.copy_(mod['up1.up.weight'])
		model.up1.up.bias.copy_(mod['up1.up.bias'])
		model.up1.conv.double_conv[0].weight.copy_(mod['up1.conv.double_conv.0.weight'])
		model.up1.conv.double_conv[1].weight.copy_(mod['up1.conv.double_conv.1.weight'])
		model.up1.conv.double_conv[1].bias.copy_(mod['up1.conv.double_conv.1.bias'])
		model.up1.conv.double_conv[1].running_mean.copy_(mod['up1.conv.double_conv.1.running_mean'])
		model.up1.conv.double_conv[1].running_var.copy_(mod['up1.conv.double_conv.1.running_var'])
		model.up1.conv.double_conv[1].num_batches_tracked.copy_(mod['up1.conv.double_conv.1.num_batches_tracked'])
		model.up1.conv.double_conv[3].weight.copy_(mod['up1.conv.double_conv.3.weight'])
		model.up1.conv.double_conv[4].weight.copy_(mod['up1.conv.double_conv.4.weight'])
		model.up1.conv.double_conv[4].bias.copy_(mod['up1.conv.double_conv.4.bias'])
		model.up1.conv.double_conv[4].running_mean.copy_(mod['up1.conv.double_conv.4.running_mean'])
		model.up1.conv.double_conv[4].running_var.copy_(mod['up1.conv.double_conv.4.running_var'])
		model.up1.conv.double_conv[4].num_batches_tracked.copy_(mod['up1.conv.double_conv.4.num_batches_tracked'])
		model.up2.up.weight.copy_(mod['up2.up.weight'])
		model.up2.up.bias.copy_(mod['up2.up.bias'])
		model.up2.conv.double_conv[0].weight.copy_(mod['up2.conv.double_conv.0.weight'])
		model.up2.conv.double_conv[1].weight.copy_(mod['up2.conv.double_conv.1.weight'])
		model.up2.conv.double_conv[1].bias.copy_(mod['up2.conv.double_conv.1.bias'])
		model.up2.conv.double_conv[1].running_mean.copy_(mod['up2.conv.double_conv.1.running_mean'])
		model.up2.conv.double_conv[1].running_var.copy_(mod['up2.conv.double_conv.1.running_var'])
		model.up2.conv.double_conv[1].num_batches_tracked.copy_(mod['up2.conv.double_conv.1.num_batches_tracked'])
		model.up2.conv.double_conv[3].weight.copy_(mod['up2.conv.double_conv.3.weight'])
		model.up2.conv.double_conv[4].weight.copy_(mod['up2.conv.double_conv.4.weight'])
		model.up2.conv.double_conv[4].bias.copy_(mod['up2.conv.double_conv.4.bias'])
		model.up2.conv.double_conv[4].running_mean.copy_(mod['up2.conv.double_conv.4.running_mean'])
		model.up2.conv.double_conv[4].running_var.copy_(mod['up2.conv.double_conv.4.running_var'])
		model.up2.conv.double_conv[4].num_batches_tracked.copy_(mod['up2.conv.double_conv.4.num_batches_tracked'])
		model.up3.up.weight.copy_(mod['up3.up.weight'])
		model.up3.up.bias.copy_(mod['up3.up.bias'])
		model.up3.conv.double_conv[0].weight.copy_(mod['up3.conv.double_conv.0.weight'])
		model.up3.conv.double_conv[1].weight.copy_(mod['up3.conv.double_conv.1.weight'])
		model.up3.conv.double_conv[1].bias.copy_(mod['up3.conv.double_conv.1.bias'])
		model.up3.conv.double_conv[1].running_mean.copy_(mod['up3.conv.double_conv.1.running_mean'])
		model.up3.conv.double_conv[1].running_var.copy_(mod['up3.conv.double_conv.1.running_var'])
		model.up3.conv.double_conv[1].num_batches_tracked.copy_(mod['up3.conv.double_conv.1.num_batches_tracked'])
		model.up3.conv.double_conv[3].weight.copy_(mod['up3.conv.double_conv.3.weight'])
		model.up3.conv.double_conv[4].weight.copy_(mod['up3.conv.double_conv.4.weight'])
		model.up3.conv.double_conv[4].bias.copy_(mod['up3.conv.double_conv.4.bias'])
		model.up3.conv.double_conv[4].running_mean.copy_(mod['up3.conv.double_conv.4.running_mean'])
		model.up3.conv.double_conv[4].running_var.copy_(mod['up3.conv.double_conv.4.running_var'])
		model.up3.conv.double_conv[4].num_batches_tracked.copy_(mod['up3.conv.double_conv.4.num_batches_tracked'])
		model.up4.up.weight.copy_(mod['up4.up.weight'])
		model.up4.up.bias.copy_(mod['up4.up.bias'])
		model.up4.conv.double_conv[0].weight.copy_(mod['up4.conv.double_conv.0.weight'])
		model.up4.conv.double_conv[1].weight.copy_(mod['up4.conv.double_conv.1.weight'])
		model.up4.conv.double_conv[1].bias.copy_(mod['up4.conv.double_conv.1.bias'])
		model.up4.conv.double_conv[1].running_mean.copy_(mod['up4.conv.double_conv.1.running_mean'])
		model.up4.conv.double_conv[1].running_var.copy_(mod['up4.conv.double_conv.1.running_var'])
		model.up4.conv.double_conv[1].num_batches_tracked.copy_(mod['up4.conv.double_conv.1.num_batches_tracked'])
		model.up4.conv.double_conv[3].weight.copy_(mod['up4.conv.double_conv.3.weight'])
		model.up4.conv.double_conv[4].weight.copy_(mod['up4.conv.double_conv.4.weight'])
		model.up4.conv.double_conv[4].bias.copy_(mod['up4.conv.double_conv.4.bias'])
		model.up4.conv.double_conv[4].running_mean.copy_(mod['up4.conv.double_conv.4.running_mean'])
		model.up4.conv.double_conv[4].running_var.copy_(mod['up4.conv.double_conv.4.running_var'])
		model.up4.conv.double_conv[4].num_batches_tracked.copy_(mod['up4.conv.double_conv.4.num_batches_tracked'])
		model.outc.conv.weight.copy_(mod['outc.conv.weight'])
		model.outc.conv.bias.copy_(mod['outc.conv.bias'])

	
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']
	loss = checkpoint['loss']
	scheduler.load_state_dict(checkpoint['lr_scheduler'])
	#train_loss += checkpoint['train_loss']
	#val_loss += checkpoint['val_loss']
	print('Model Loaded')

def freeze_decoder(model):
	model.up1.up.weight.requires_grad=False
	model.up1.up.bias.requires_grad=False
	model.up1.conv.double_conv[0].weight.requires_grad=False
	model.up1.conv.double_conv[1].weight.requires_grad=False
	model.up1.conv.double_conv[1].bias.requires_grad=False
	model.up1.conv.double_conv[1].running_mean.requires_grad=False
	model.up1.conv.double_conv[1].running_var.requires_grad=False
	model.up1.conv.double_conv[1].num_batches_tracked.requires_grad=False
	model.up1.conv.double_conv[3].weight.requires_grad=False
	model.up1.conv.double_conv[4].weight.requires_grad=False
	model.up1.conv.double_conv[4].bias.requires_grad=False
	model.up1.conv.double_conv[4].running_mean.requires_grad=False
	model.up1.conv.double_conv[4].running_var.requires_grad=False
	model.up1.conv.double_conv[4].num_batches_tracked.requires_grad=False
	model.up2.up.weight.requires_grad=False
	model.up2.up.bias.requires_grad=False
	model.up2.conv.double_conv[0].weight.requires_grad=False
	model.up2.conv.double_conv[1].weight.requires_grad=False
	model.up2.conv.double_conv[1].bias.requires_grad=False
	model.up2.conv.double_conv[1].running_mean.requires_grad=False
	model.up2.conv.double_conv[1].running_var.requires_grad=False
	model.up2.conv.double_conv[1].num_batches_tracked.requires_grad=False
	model.up2.conv.double_conv[3].weight.requires_grad=False
	model.up2.conv.double_conv[4].weight.requires_grad=False
	model.up2.conv.double_conv[4].bias.requires_grad=False
	model.up2.conv.double_conv[4].running_mean.requires_grad=False
	model.up2.conv.double_conv[4].running_var.requires_grad=False
	model.up2.conv.double_conv[4].num_batches_tracked.requires_grad=False
	model.up3.up.weight.requires_grad=False
	model.up3.up.bias.requires_grad=False
	model.up3.conv.double_conv[0].weight.requires_grad=False
	model.up3.conv.double_conv[1].weight.requires_grad=False
	model.up3.conv.double_conv[1].bias.requires_grad=False
	model.up3.conv.double_conv[1].running_mean.requires_grad=False
	model.up3.conv.double_conv[1].running_var.requires_grad=False
	model.up3.conv.double_conv[1].num_batches_tracked.requires_grad=False
	model.up3.conv.double_conv[3].weight.requires_grad=False
	model.up3.conv.double_conv[4].weight.requires_grad=False
	model.up3.conv.double_conv[4].bias.requires_grad=False
	model.up3.conv.double_conv[4].running_mean.requires_grad=False
	model.up3.conv.double_conv[4].running_var.requires_grad=False
	model.up3.conv.double_conv[4].num_batches_tracked.requires_grad=False
	model.up4.up.weight.requires_grad=False
	model.up4.up.bias.requires_grad=False
	model.up4.conv.double_conv[0].weight.requires_grad=False
	model.up4.conv.double_conv[1].weight.requires_grad=False
	model.up4.conv.double_conv[1].bias.requires_grad=False
	model.up4.conv.double_conv[1].running_mean.requires_grad=False
	model.up4.conv.double_conv[1].running_var.requires_grad=False
	model.up4.conv.double_conv[1].num_batches_tracked.requires_grad=False
	model.up4.conv.double_conv[3].weight.requires_grad=False
	model.up4.conv.double_conv[4].weight.requires_grad=False
	model.up4.conv.double_conv[4].bias.requires_grad=False
	model.up4.conv.double_conv[4].running_mean.requires_grad=False
	model.up4.conv.double_conv[4].running_var.requires_grad=False
	model.up4.conv.double_conv[4].num_batches_tracked.requires_grad=False
	model.outc.conv.weight.requires_grad=False
	model.outc.conv.bias.requires_grad=False
	print('Decoder Freezed')

"""
def custom_loss_fn(orig_loss_d, rec_loss_d): #dizionari con dentro loss su img originale e reconstructed
	print(loss) 
	Come è composto il dizionario della loss:
	{'loss_classifier': tensor(1.2024, device='cuda:0'), 'loss_box_reg': tensor(0.0861, device='cuda:0'), 'loss_objectness': tensor(3.5456, device='cuda:0'), 'loss_rpn_box_reg': tensor(1.4881, device='cuda:0')}
"""

# the function takes the original prediction and the iou threshold.
def apply_nms(orig_prediction, iou_thresh=0.3, score_thresh=0.75):
	
	# Filter predictions based on score threshold
	keep_score_indices = orig_prediction['scores'] > score_thresh
	filtered_boxes = orig_prediction['boxes'][keep_score_indices]
	filtered_scores = orig_prediction['scores'][keep_score_indices]
	filtered_labels = orig_prediction['labels'][keep_score_indices]
	
	# torchvision returns the indices of the bboxes to keep
	keep = torchvision.ops.nms(filtered_boxes.cpu(), filtered_scores.cpu(), iou_thresh)
	
	# NMS and score thresh
	final_prediction = orig_prediction
	final_prediction['boxes'] = filtered_boxes.cpu()[keep]
	final_prediction['scores'] = filtered_scores.cpu()[keep]
	final_prediction['labels'] = filtered_labels.cpu()[keep]
	
	return final_prediction
