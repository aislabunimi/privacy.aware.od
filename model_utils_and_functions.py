import torch
import torchvision
import random
import os
import numpy as np

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  
   # torch.backends.cudnn.benchmark = True

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
