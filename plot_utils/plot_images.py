import matplotlib.pyplot as plt
from PIL import Image
from dataset import *

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
    plt.show()
    
#from DETR, slightly modified
def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size[3], image_size[2] #modified here with shape
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.shape, size, max_size) #modified here with shape
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target
 
def get_transform():
    transform = []
    transform.append(transforms.PILToTensor())
    transform.append(transforms.ConvertImageDtype(torch.float))
    transform.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    return transforms.Compose(transform)   

def plot_results(img, labels, prob, boxes, all_classes, five_classes):
    #plt.figure(figsize=(16,10))
    #cat, dog, horse, sheep, cow
    COCO_5_CLASSES=['__background__', 'bicycle', 'airplane', 'bus', 'train', 'boat']
    COCO_91_CLASSES=[
    '__background__', 
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    plt.axis('off')
    plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0).detach().numpy())
    ax = plt.gca()
    if (prob is not None and boxes is not None):
       prob = prob.cpu()
       boxes = boxes.cpu()
       labels = labels.cpu()
       for p, label, (xmin, ymin, xmax, ymax) in zip(prob, labels, boxes.tolist()):
           ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='red', linewidth=3))
           if all_classes:
              label_name=COCO_91_CLASSES[label]
              text = f'{label_name} {p:0.3f}'
           elif five_classes:
              label_name=COCO_5_CLASSES[label]
              text = f'{label_name} {p:0.3f}'
           else:
              text = f'{p:0.3f}'
           font_kwargs = dict(fontfamily="monospace", fontweight="bold", fontsize=20)
           ax.text(xmin, ymin, text, **font_kwargs, bbox=dict(facecolor='yellow', alpha=0.5))
    return


def compare_two_results_unet(print_forward_along_backward, unet, tasknet, device, img_file_path, name_path_save, unet_weights_load, unet_weights_to_compare, unet_optimizer, unet_scheduler, all_classes, five_classes): 
   unet.eval()
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
   img, _ = resize(img, None, 256) #simulating validation first size
   
   res_tasknet = tasknet(img)
   img_primo_plot = img.squeeze(0)
   img_primo_plot = unnormalize(img_primo_plot)
   nms_pred = apply_nms(res_tasknet[0], iou_thresh=0.5) #saving last bboxes
   
   load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
   out = unet(img)
   trans_r = transforms.Resize((out.shape[2], out.shape[3]), antialias=False)
   orig_img_r = trans_r(img)
   orig_img = unnormalize(orig_img_r)
   
   img_primo_plot = trans_r(img_primo_plot) #for making first image equal in size to the reconstructed ones
   plt.subplot(1, 3, 1)
   plt.title('Original Image', fontsize=20)	
   plot_results(img_primo_plot, nms_pred['labels'], nms_pred['scores'], nms_pred['boxes'], all_classes, five_classes)
   
   plt.subplot(1, 3, 2)
   out_to_plot = unnormalize(out)
   out_to_plot = torch.clamp(out_to_plot, min=0, max=1)
   
   reconstructed = tasknet(out)
   pred_recon = reconstructed[0]
   nms_pred_recon = apply_nms(pred_recon, iou_thresh=0.5) 
   filename = os.path.basename(unet_weights_load)
   if print_forward_along_backward:
      name = 'Forward'
      save_image(out_to_plot, 'temp_for_backward.jpg')
   else:
      name = os.path.splitext(filename)[0]	
   
   plt.title(f'{name}', fontsize=20)
   plot_results(out_to_plot, nms_pred_recon['labels'], nms_pred_recon['scores'], nms_pred_recon['boxes'], all_classes, five_classes)
   plt.subplot(1, 3, 3)
   load_checkpoint(unet, unet_weights_to_compare, unet_optimizer, unet_scheduler)

   if print_forward_along_backward:
      image = Image.open('temp_for_backward.jpg').convert("RGB")
      img = eval_transform(image)
      img = img.unsqueeze(0)
      img = img.to(device)
      img, _ = resize(img, None, 256)
      out = unet(img)
      nms_pred_recon['labels'] = None
      nms_pred_recon['scores'] = None
      nms_pred_recon['boxes'] = None
      name = 'Backward'
   else:
      out = unet(img)
      reconstructed = tasknet(out)
      pred_recon = reconstructed[0]
      nms_pred_recon = apply_nms(pred_recon, iou_thresh=0.5)
      filename = os.path.basename(unet_weights_to_compare)
      name = os.path.splitext(filename)[0]
   out_to_plot = unnormalize(out)
   out_to_plot = torch.clamp(out_to_plot, min=0, max=1)
   plt.title(f'{name}', fontsize=20)
   plot_results(out_to_plot, nms_pred_recon['labels'], nms_pred_recon['scores'], nms_pred_recon['boxes'], all_classes, five_classes)
   plt.subplots_adjust(wspace=0.05)
   directory = os.path.dirname(name_path_save)
   if not os.path.exists(directory):
      os.makedirs(directory)
   new_ext="png"
   root, ext = os.path.splitext(name_path_save)
   path_save = f"{root}.{new_ext}"
   plt.savefig(path_save, format=new_ext, bbox_inches='tight')
   plt.close()
   
def plot_single_image(print_forward_along_backward, unet, tasknet, device, img_file_path, name_path_save, unet_weights_load, unet_weights_to_compare, unet_optimizer, unet_scheduler, all_classes, five_classes, plot_tasknet, plot_fw, plot_bw): 
   unet.eval()
   tasknet.eval()
   mean = torch.tensor([0.485, 0.456, 0.406])
   std = torch.tensor([0.229, 0.224, 0.225])
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())   
   image = Image.open(img_file_path).convert("RGB")
   eval_transform = get_transform()
   img = eval_transform(image)
   img = img.unsqueeze(0)
   img = img.to(device)
   img, _ = resize(img, None, 256) #simulating validation first size 
   plt.figure(figsize=(8, 6))
   
   if plot_tasknet:
      res_tasknet = tasknet(img)
      img_primo_plot = img.squeeze(0)
      img_primo_plot = unnormalize(img_primo_plot)
      nms_pred = apply_nms(res_tasknet[0], iou_thresh=0.5) #saving last bboxes
      plot_results(img_primo_plot, nms_pred['labels'], nms_pred['scores'], nms_pred['boxes'], all_classes, five_classes)
      directory = os.path.dirname(name_path_save)
      if not os.path.exists(directory):
         os.makedirs(directory)
      new_ext="png"
      root, ext = os.path.splitext(name_path_save)
      path_save = f"{root}.{new_ext}"
      plt.savefig(path_save, format=new_ext, bbox_inches='tight')
      plt.close()
      return
   
   load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
   out = unet(img)
   trans_r = transforms.Resize((out.shape[2], out.shape[3]), antialias=False)
   orig_img_r = trans_r(img)
   orig_img = unnormalize(orig_img_r)
   
   out_to_plot = unnormalize(out)
   out_to_plot = torch.clamp(out_to_plot, min=0, max=1)
   if plot_fw:
      reconstructed = tasknet(out)
      pred_recon = reconstructed[0]
      nms_pred_recon = apply_nms(pred_recon, iou_thresh=0.5, score_thresh=0.75) 
      filename = os.path.basename(unet_weights_load)
      name = os.path.splitext(filename)[0]
      plot_results(out_to_plot, nms_pred_recon['labels'], nms_pred_recon['scores'], nms_pred_recon['boxes'], all_classes, five_classes)
      directory = os.path.dirname(name_path_save)
      if not os.path.exists(directory):
         os.makedirs(directory)
      new_ext="png"
      root, ext = os.path.splitext(name_path_save)
      path_save = f"{root}.{new_ext}"
      plt.savefig(path_save, format=new_ext, bbox_inches='tight')
      plt.close()
      return
      
   if plot_bw:
      load_checkpoint(unet, unet_weights_to_compare, unet_optimizer, unet_scheduler)
      filename = os.path.basename(unet_weights_load)
      save_image(out_to_plot, 'temp_for_backward.jpg')
      image = Image.open('temp_for_backward.jpg').convert("RGB")
      img = eval_transform(image)
      img = img.unsqueeze(0)
      img = img.to(device)
      img, _ = resize(img, None, 256)
      out = unet(img)
      out_to_plot = unnormalize(out)
      out_to_plot = torch.clamp(out_to_plot, min=0, max=1)
      plot_results(out_to_plot, None, None, None, all_classes, five_classes)
      directory = os.path.dirname(name_path_save)
      if not os.path.exists(directory):
         os.makedirs(directory)
      new_ext="png"
      root, ext = os.path.splitext(name_path_save)
      path_save = f"{root}.{new_ext}"
      plt.savefig(path_save, format=new_ext, bbox_inches='tight')
      plt.close()
   return

from torchvision.utils import save_image
def save_disturbed_pred(unet, device, img_file_path, name_path_save, unet_weights_load, unet_weights_to_compare, unet_optimizer, unet_scheduler):
   unet.eval()
   image = Image.open(img_file_path).convert("RGB")
   eval_transform = get_transform()
   mean = torch.tensor([0.485, 0.456, 0.406])
   std = torch.tensor([0.229, 0.224, 0.225])
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   img = eval_transform(image)
   img = img.unsqueeze(0)
   img = img.to(device)
   img, _ = resize(img, None, 256)
   load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
   out = unet(img)
   out = unnormalize(out)
   out = torch.clamp(out, min=0, max=1)
   save_image(out, 'temp_for_backward.jpg')   
   load_checkpoint(unet, unet_weights_to_compare, unet_optimizer, unet_scheduler)
   image = Image.open('temp_for_backward.jpg').convert("RGB")
   img = eval_transform(image)
   img = img.unsqueeze(0)
   img = img.to(device)
   img, _ = resize(img, None, 256)
   recons = unet(img)
   recons = unnormalize(recons)
   recons = torch.clamp(recons, min=0, max=1)
   trans_r = transforms.Resize((img.shape[2], img.shape[3]), antialias=False)
   recons = trans_r(recons)
   directory = os.path.dirname(name_path_save)
   if not os.path.exists(directory):
      os.makedirs(directory)
   save_image(recons, name_path_save)
