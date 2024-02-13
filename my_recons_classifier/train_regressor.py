import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import random
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

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

def default_loader(path):
    return Image.open(path).convert('RGB')

class ImageFolder(Dataset):
    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        if os.path.basename(path).startswith('0'): #gli 0 rappresentano img con solo rumore
           label=torch.tensor(0.0)
        elif os.path.basename(path).startswith('1'): #gli 1 img con contorni persone, non identificabili
           label=torch.tensor(0.25)
        elif os.path.basename(path).startswith('2'): #gli 2 img con presenza persone riconoscibile facilmente ma difficilmente identificabili
           label=torch.tensor(0.55)
        elif os.path.basename(path).startswith('3'): #gli 3 img con persone riconoscibili e identificabili
           label=torch.tensor(1.0)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, label, path
        else:
            return img, label

    def __len__(self):
        return len(self.imgs)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NP_EXTENSIONS = ['.npy',]

def is_image_file(filename, mode='img'):
    if(mode=='img'):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    elif(mode=='np'):
        return any(filename.endswith(extension) for extension in NP_EXTENSIONS)

def make_dataset(dirs, mode='img'):
    if(not isinstance(dirs,list)):
        dirs = [dirs,]

    images = []
    for dir in dirs:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname, mode=mode):
                    path = os.path.join(root, fname)
                    images.append(path)

    # print("Found %i images in %s"%(len(images),root))
    return images

def get_transform_train():
    transform = []
    transform.append(transforms.PILToTensor())
    transform.append(transforms.ConvertImageDtype(torch.float))
    transform.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    transform.append(transforms.RandomVerticalFlip(p=0.5))
    transform.append(transforms.RandomHorizontalFlip(p=0.5))
    transform.append(transforms.RandomRotation(degrees=(0, 180)))
    transform.append(transforms.RandomGrayscale(p=0.1))
    return transforms.Compose(transform)
    
def get_transform_eval():
    transform = []
    transform.append(transforms.PILToTensor())
    transform.append(transforms.ConvertImageDtype(torch.float))
    transform.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(transform) 

#def collate_fn(batch):
#    return (zip(*batch))

def train_vgg(train_dataloader, epoch, device, train_loss, vgg, optimizer): #funzione che si occupa del training
	vgg.train()
	batch_size = len(train_dataloader) #recupero la batch size
	running_loss = 0 # Iniziallizo la variabile per la loss 
	for imgs, targets in tqdm(train_dataloader, desc=f'Epoch {epoch} - Train model'):
		#imgs = list(img.to(device) for img in imgs)
		#targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
		imgs = imgs.to(device)
		labels = targets.to(device)
		optimizer.zero_grad(set_to_none=True) #pulisco i gradienti prima del backpropagation step
		
		outputs = vgg(imgs)
		#loss = criterion(outputs, labels)
		loss = criterion(outputs, labels.unsqueeze(1))  # Labels need to be unsqueezed to match the shape of outputs
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
	running_loss /= batch_size #calcolo la loss media
	train_loss.append(running_loss)
	return running_loss

def val_vgg(val_dataloader, epoch, device, val_loss, vgg_save_path, vgg, optimizer, scheduler): #funzione che si occupa del test
	vgg.eval()
	batch_size = len(val_dataloader) #recupero la batch size
	correct = 0
	total = 0
	running_loss=0
	result_dict = {}
	result_dict['epoch']=epoch
	result_dict['total']=0	
	with torch.no_grad(): #non calcolo il gradiente, sto testando e bassa
		for imgs, targets in tqdm(val_dataloader, desc=f'Epoch {epoch} - Validating model'):
			imgs = imgs.to(device)
			labels = targets.to(device)
					
			outputs = vgg(imgs)
			
			#loss = criterion(outputs, labels)
			loss = criterion(outputs, labels.unsqueeze(1))  # Labels need to be unsqueezed to match the shape of outputs
			running_loss += loss.item()
			"""
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			values, occurrences = torch.unique(predicted, return_counts=True)
			result_dict['total']+=len(imgs)
			for value, occurrence in zip(values, occurrences):
			   value_item = value.item()
			   occurrence_item = occurrence.item()
			   if value_item in result_dict:
			      result_dict[value_item] += occurrence_item
			   else:
			      result_dict[value_item] = occurrence_item
			correct += (predicted == labels).sum().item()
			"""
			print(outputs, labels)
	running_loss /= batch_size #calcolo la loss media
	#print(f'Accuracy of the network on the test images: {100 * correct // total} %')
	#print(result_dict)
	vgg_save_path = f'{vgg_save_path}_{epoch}.pt' 
	create_checkpoint(vgg, optimizer, epoch, running_loss, scheduler, vgg_save_path)
	return running_loss

def test_vgg(val_dataloader, epoch, device, val_loss, vgg_save_path, vgg, optimizer, scheduler): #funzione che si occupa del test
	vgg.eval()
	batch_size = len(val_dataloader) #recupero la batch size
	correct = 0
	total = 0
	running_loss=0
	result_dict = {}
	result_dict['epoch']=epoch
	result_dict['total']=0	
	with torch.no_grad(): #non calcolo il gradiente, sto testando e bassa
		for imgs, targets in tqdm(val_dataloader, desc=f'Epoch {epoch} - Testing model'):
			imgs = imgs.to(device)
			labels = targets.to(device)
					
			outputs = vgg(imgs)
			
			loss = criterion(outputs, labels)
			running_loss += loss.item()
			
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			values, occurrences = torch.unique(predicted, return_counts=True)
			result_dict['total']+=len(imgs)
			for value, occurrence in zip(values, occurrences):
			   value_item = value.item()
			   occurrence_item = occurrence.item()
			   if value_item in result_dict:
			      result_dict[value_item] += occurrence_item
			   else:
			      result_dict[value_item] = occurrence_item
			correct += (predicted == labels).sum().item()
			
	running_loss /= batch_size #calcolo la loss media
	print(f'Accuracy of the network on the test images: {100 * correct // total} %')
	print(result_dict)
	print(f'Test running loss: {running_loss}')
	
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

device='cuda'
seed_everything(0)
#vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights)
#vgg16 = torchvision.models.vgg16()
#num_classes=4
#vgg16.classifier[-1] = torch.nn.Linear(in_features=4096, out_features=num_classes)
#vgg16.to(device)

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

model = CNNRegression()
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_dataset = ImageFolder('/home/math0012/Tesi_magistrale/classifier/dataset/train', get_transform_train())
val_dataset = ImageFolder('/home/math0012/Tesi_magistrale/classifier/dataset/val', get_transform_eval())
#test_dataset = ImageFolder('/home/math0012/Tesi_magistrale/classifier/dataset/test', get_transform_eval())
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=deterministic_worker, generator=deterministic_generator())
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=deterministic_worker, generator=deterministic_generator())
#test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=deterministic_worker, generator=deterministic_generator())

#criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005, nesterov=True)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

train_loss = [] # Lista che conserva la training loss. Mi serve se voglio vedere l'andamento della loss
val_loss = [] #Lista che conversa la test loss
log = {'TRAIN_LOSS': [], 'VAL_LOSS': []}
num_epochs=10
model_save_path='/home/math0012/Tesi_magistrale/regressor/regressor'

for epoch in range(1, num_epochs+1): #itero ora facendo un train e un test per ogni epoca
    log['TRAIN_LOSS'].append(train_vgg(train_dataloader, epoch, device, train_loss, model, optimizer))
    #scheduler.step()
    log['VAL_LOSS'].append(val_vgg(val_dataloader, epoch, device, val_loss, model_save_path, model, optimizer, scheduler))
    #test_vgg(test_dataloader, epoch, device, val_loss, vgg16_save_path, vgg16, optimizer, scheduler)
    print(f'EPOCH {epoch} SUMMARY: ' + ', '.join([f'{k}: {v[epoch-1]}' for k, v in log.items()]))
