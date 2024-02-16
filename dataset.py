import torchvision
import os
import json
from PIL import Image
from typing import Optional, List
from torch import Tensor
from model_utils_and_functions import deterministic_worker, deterministic_generator
#from plot_utils.plot_images import *
from detr_transforms import * 

######## Definisco i dataset e i dataloader
#da DETR qui
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx) #qui uso funzione padre, equivalente a:
    
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        
        #target = [obj for obj in target if obj["category_id"] ==1]
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
            
        #plot_img_after_transform(img, target["boxes"])    #DEBUG ONLY, serve per verificare che le img transformate hanno le bbox a posto. Le bbox qui sono nel formato xmin, ymin, xmax, ymax
        return img, target
     
    ###__len__ aggiunta io, da DETR non c'era   
    def __len__(self) -> int:
    	return len(self.ids)

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]
        
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
           
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

def make_coco_transforms(image_set, scales=None):

    normalize = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if image_set == 'train':
        return Compose([
            RandomHorizontalFlip(),
            RandomResize(scales, max_size=None),
            normalize,
        ])

    if image_set == 'val':
        return Compose([
            normalize,
        ])
    
    if image_set == 'val_resize':
        first_size = [scales[0]]
        return Compose([
            RandomResize(first_size, max_size=None),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        #if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
        #    return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)

def collate_fn_nested_tensors(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)
    
def collate_fn_disturbed(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0]) #0 rappresenta le img disturbate
    batch[1] = nested_tensor_from_tensor_list(batch[1]) #1 rappresenta le img originali
    return tuple(batch)

def collate_fn_tasknet(batch):
    return list(zip(*batch))


class DisturbedDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, disturbed_path, orig_path=None, transform=None, is_training=False, resize_scales=None, generate_disturbed_dataset=False):
        self.data = json.load(open(json_file))
        self.transform = transform
        self.disturbed_path = disturbed_path
        self.orig_path = orig_path
        self.is_training=is_training
        self.resize_scales=resize_scales
        self.generate_disturbed_dataset = generate_disturbed_dataset

    def __getitem__(self, index):
    	x = self.data[index]['image_path']
    	disturbed_img_path = os.path.join(self.disturbed_path, x)
    	disturbed_image= Image.open(disturbed_img_path).convert("RGB")
    	if self.orig_path is not None: #serve solo per riusare la classe quando genero il dataset
    		orig_img_path = os.path.join(self.orig_path, x)
    		orig_image= Image.open(orig_img_path).convert("RGB")

    	if self.transform:
        	
        	#Faccio il horizontal split qui così sia disturbed che il target originale sono specchiati
        	if self.is_training and not self.generate_disturbed_dataset:
        		
        		p_flip = random.random()
        		flip_trans = RandomHorizontalFlip(p=1) #inizializzo oggetto flip in modo da farlo sempre
        		if(p_flip>0.5):
        			disturbed_image, _ = flip_trans(disturbed_image, target=None)
        			orig_image, _ = flip_trans(orig_image, target=None)
        		#se non supero quella soglia, non eseguo il flip
        		
        		#Faccio il resize qui così viene scelto lo stesso valore per entrambi e il resize è fatto allo stesso modo
        		#evito il 200 perché potrebbe dare problemi com msssim
        		#scales = [scale for scale in self.resize_scales if scale >= 200]
        		random_size = random.choice(self.resize_scales)
        		random_size = [random_size]
        		resize = RandomResize(random_size, max_size=None)
        		disturbed_image, _ = resize(disturbed_image, target=None)
        		orig_image, _ = resize(orig_image, target=None)
        	
        		#Per ultime le transform "normali", cioè a tensore e poi normalize per il target (le orig image)
        	elif not self.is_training and not self.generate_disturbed_dataset:
        	   first_size = [self.resize_scales[0]]
        	   resize = RandomResize(first_size, max_size=None)
        	disturbed_image, _ = self.transform(disturbed_image, target=None)
        	
        	if self.orig_path is not None:
        		orig_image, _ = self.transform(orig_image, target=None )
        
    	if self.orig_path is None: #solo se lo uso per generare, ritorno il path dell'img così lo posso usare per salvare la nuova anno
        	orig_image=x
    	return disturbed_image, orig_image

    def __len__(self):
        return len(self.data)

def load_dataset(train_img_folder, train_ann_file, val_img_folder, val_ann_file, train_batch_size, val_batch_size, save_disturbed, train_only_tasknet, resize_scales_transform, use_dataset_subset): #split_size_train_set):
	train_coco_dataset = CocoDetection(train_img_folder, train_ann_file, transforms=make_coco_transforms('train', resize_scales_transform), return_masks=None)
	val_coco_dataset = CocoDetection(val_img_folder, val_ann_file, transforms=make_coco_transforms('val_resize', resize_scales_transform), return_masks=False)
	example_coco_dataset = CocoDetection(val_img_folder, val_ann_file, transforms=make_coco_transforms('val_resize', resize_scales_transform), return_masks=False)
	
	train_indices = list(range(0, len(train_coco_dataset), 1)) 
	val_indices = list(range(0, len(val_coco_dataset), 1))
	if use_dataset_subset>0: #and split_size_train_set==0:
		train_coco_dataset = torch.utils.data.Subset(train_coco_dataset, train_indices[:use_dataset_subset])
		val_coco_dataset = torch.utils.data.Subset(val_coco_dataset, val_indices[:use_dataset_subset])
	#elif split_size_train_set>0:
	#	train_coco_dataset = torch.utils.data.Subset(train_coco_dataset, train_indices[:split_size_train_set]) #prendo i primi n elementi
	#seleziono arbitrariamente alcuni esempi che sono interessanti	
	example_indices = [1, 3, 13, 15, 60, 92, 97, 99, 101, 128, 134, 176, 208, 209, 214]
	example_dataset = torch.utils.data.Subset(example_coco_dataset, example_indices)
	
	if train_only_tasknet:
		collate = collate_fn_tasknet
	else:
		collate = collate_fn_nested_tensors
	
	train_dataloader = torch.utils.data.DataLoader(train_coco_dataset, batch_size=train_batch_size, 
		shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate, 
		worker_init_fn=deterministic_worker, generator=deterministic_generator())
	val_dataloader = torch.utils.data.DataLoader(val_coco_dataset, batch_size=val_batch_size, 
		shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate, 
		worker_init_fn=deterministic_worker, generator=deterministic_generator())
	example_dataloader = torch.utils.data.DataLoader(example_dataset, batch_size=1, 
		shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate, 
		worker_init_fn=deterministic_worker, generator=deterministic_generator())
	
	return train_dataloader, val_dataloader, example_dataloader
	
def load_dataset_for_generating_disturbed_set(train_img_folder, train_ann_file, val_img_folder, val_ann_file, use_dataset_subset, use_coco_train): #, split_size_train_set):
	val_coco_dataset = CocoDetection(val_img_folder, val_ann_file, transforms=make_coco_transforms('val'), return_masks=False)
	if use_coco_train:
		disturbed_train_dataset_gen = CocoDetection(train_img_folder, train_ann_file, transforms=make_coco_transforms('val'), return_masks=None)
	else:
		disturbed_train_dataset_gen = DisturbedDataset(train_ann_file, train_img_folder, transform=make_coco_transforms('val'), is_training=False, generate_disturbed_dataset=True)
	
	train_indices = list(range(0, len(disturbed_train_dataset_gen), 1))
	val_indices = list(range(0, len(val_coco_dataset), 1))
	if use_dataset_subset>0: # and split_size_train_set==0:
		val_coco_dataset = torch.utils.data.Subset(val_coco_dataset, val_indices[:use_dataset_subset])
		disturbed_train_dataset_gen = torch.utils.data.Subset(disturbed_train_dataset_gen, train_indices[:use_dataset_subset])
	#elif split_size_train_set>0:
	#	train_coco_dataset_without_resize = torch.utils.data.Subset(train_coco_dataset_without_resize, train_indices[split_size_train_set:]) #prendo gli ultimi n elementi. Il val lo posso usare così come è
	
	disturbed_train_dataloader_gen = torch.utils.data.DataLoader(disturbed_train_dataset_gen, 
		batch_size=1, shuffle=True, num_workers=4, pin_memory=True, 
		collate_fn=collate_fn_nested_tensors, worker_init_fn=deterministic_worker, 
		generator=deterministic_generator())
	val_dataloader_gen_disturbed = torch.utils.data.DataLoader(val_coco_dataset, batch_size=1,
		shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_nested_tensors, 
		worker_init_fn=deterministic_worker, generator=deterministic_generator())
	
	return disturbed_train_dataloader_gen, val_dataloader_gen_disturbed
	
def load_disturbed_dataset(disturbed_train_img_folder, disturbed_train_ann, disturbed_val_img_folder, disturbed_val_ann, orig_train_folder, orig_val_folder, train_batch_size, val_batch_size, resize_scales_transform, use_dataset_subset, orig_val_ann_file):
	disturbed_train_dataset = DisturbedDataset(disturbed_train_ann, disturbed_train_img_folder, orig_train_folder, transform=make_coco_transforms('val'), is_training=True, resize_scales=resize_scales_transform)
	disturbed_val_dataset = DisturbedDataset(disturbed_val_ann, disturbed_val_img_folder, orig_val_folder, transform=make_coco_transforms('val'), is_training=False, resize_scales=resize_scales_transform)
	example_dataset = CocoDetection(disturbed_val_img_folder, orig_val_ann_file, transforms=make_coco_transforms('val_resize', resize_scales_transform), return_masks=False)
	
	train_indices = list(range(0, len(disturbed_train_dataset), 1))
	val_indices = list(range(0, len(disturbed_val_dataset), 1)) 
	if use_dataset_subset>0:
		disturbed_train_dataset = torch.utils.data.Subset(disturbed_train_dataset, train_indices[:use_dataset_subset])
		disturbed_val_dataset = torch.utils.data.Subset(disturbed_val_dataset, val_indices[:use_dataset_subset])
	
	example_indices = [1, 3, 13, 15, 60, 92, 97, 99, 101, 128, 134, 176, 208, 209, 214]
	example_dataset = torch.utils.data.Subset(example_dataset, example_indices)
	
	disturbed_train_dataloader = torch.utils.data.DataLoader(disturbed_train_dataset, 
		batch_size=train_batch_size, shuffle=True, num_workers=4, pin_memory=True, 
		collate_fn=collate_fn_disturbed, worker_init_fn=deterministic_worker, 
		generator=deterministic_generator())
	disturbed_val_dataloader = torch.utils.data.DataLoader(disturbed_val_dataset, 
		batch_size=val_batch_size, shuffle=False, num_workers=4, pin_memory=True, 
		collate_fn=collate_fn_disturbed, worker_init_fn=deterministic_worker, 
		generator=deterministic_generator())	
	example_dataloader = torch.utils.data.DataLoader(example_dataset, batch_size=1, 
		shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn_nested_tensors, 
		worker_init_fn=deterministic_worker, generator=deterministic_generator())
		
	return disturbed_train_dataloader, disturbed_val_dataloader, example_dataloader
