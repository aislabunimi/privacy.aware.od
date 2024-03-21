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
    #plt.axis('off')
    #plt.tight_layout()
    plt.show()
    
def show_res_test_unet(model, tasknet, device, img_file_path, not_reconstructed, name_path_save):
   model.eval()
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
   img_primo_plot = img.squeeze(0)
   img_primo_plot = unnormalize(img_primo_plot)
   nms_pred = apply_nms(res_tasknet[0], iou_thresh=0.1)
   plt.subplot(1, 3, 1)
   
   print("Original Labels and scores: ", nms_pred["labels"], nms_pred["scores"])
   plt.title('Original Image, bbox by only tasknet')
   plt.axis('off')
   plot_results(img_primo_plot, nms_pred['scores'], nms_pred['boxes'])
   if not_reconstructed: #cioe se l'immagine non è una ricostruita
      out = model(img)
      out_to_plot = unnormalize(out)
   else:
      out = img
      out_to_plot = img
   plt.subplot(1, 3, 2)
   plt.imshow(out_to_plot.squeeze(0).detach().cpu().numpy().transpose(1,2,0))
   plt.title('Reconstructed Image')
   plt.axis('off')
   plt.tight_layout()
   plt.subplot(1, 3, 3)
   reconstructed = tasknet(out)
   pred_recon = reconstructed[0]
   nms_pred_recon = apply_nms(pred_recon, iou_thresh=0.1) #per applicare nms e salvare l'ultima box
   
   print("Reconstructed Labels and Scores: ", nms_pred_recon["labels"], nms_pred_recon["scores"])
   plot_results(out_to_plot, nms_pred_recon['scores'], nms_pred_recon['boxes'])
   plt.title('Reconstructed Image, bbox by full model')
   plt.axis('off')
   plt.savefig(name_path_save, format='png', bbox_inches='tight')

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
from .lpips import LPIPS
def compare_two_results_unet(unet, tasknet, device, img_file_path, name_path_save, unet_weights_load, unet_weights_to_compare, unet_optimizer, unet_scheduler): 
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
   nms_pred = apply_nms(res_tasknet[0], iou_thresh=0.1)
   
   load_checkpoint(unet, unet_weights_load, unet_optimizer, unet_scheduler)
   out = unet(img)
   trans_r = transforms.Resize((out.shape[2], out.shape[3]), antialias=False)
   orig_img_r = trans_r(img)
   orig_img = unnormalize(orig_img_r)
   
   img_primo_plot = trans_r(img_primo_plot) #for making first image equal in size to the reconstructed ones
   plt.subplot(1, 3, 1)
   plt.title('Original Image', fontsize=20)	
   plot_results(img_primo_plot, nms_pred['scores'], nms_pred['boxes'])
   
   plt.subplot(1, 3, 2)
   out_to_plot = unnormalize(out)
   
   reconstructed = tasknet(out)
   pred_recon = reconstructed[0]
   nms_pred_recon = apply_nms(pred_recon, iou_thresh=0.1) #per applicare nms e salvare l'ultima box
   filename = os.path.basename(unet_weights_load)
   name = os.path.splitext(filename)[0]	
   
   out_to_plot = torch.clamp(out_to_plot, min=0, max=1)
   orig_img = torch.clamp(orig_img, min=0, max=1)
   trans_te = transforms.Resize((64, 64), antialias=False)
   orig_img_r = trans_te(orig_img_r)
   out = trans_te(out)
   
   lpips_model = LPIPS(net='vgg').to(device)
   lpips_score = lpips_model(out, orig_img_r).item()
   plt.title(f'{name}, LPIPS: {lpips_score:0.3f}', fontsize=20)
   plot_results(out_to_plot, nms_pred_recon['scores'], nms_pred_recon['boxes'])
   plt.subplot(1, 3, 3)
   load_checkpoint(unet, unet_weights_to_compare, unet_optimizer, unet_scheduler)
   out = unet(img)
   out_to_plot = unnormalize(out)
   reconstructed = tasknet(out)
   pred_recon = reconstructed[0]
   nms_pred_recon = apply_nms(pred_recon, iou_thresh=0.1)
   filename = os.path.basename(unet_weights_to_compare)
   name = os.path.splitext(filename)[0]
   
   out_to_plot = torch.clamp(out_to_plot, min=0, max=1)
   trans_te = transforms.Resize((64, 64), antialias=False)
   out = trans_te(out)
   lpips_score = lpips_model(out, orig_img_r).item()
   plt.title(f'{name}, LPIPS: {lpips_score:0.3f}', fontsize=20)
   plot_results(out_to_plot, nms_pred_recon['scores'], nms_pred_recon['boxes'])
   plt.subplots_adjust(wspace=0.05)
   plt.savefig(name_path_save, format='png', bbox_inches='tight')
   plt.clf()

from torchvision.utils import save_image
def save_disturbed_pred(unet, device, img_file_path, name_path_save):
   unet.eval()
   image = Image.open(img_file_path).convert("RGB")
   eval_transform = get_transform()
   img = eval_transform(image)
   img = img.unsqueeze(0)
   img = img.to(device)
   img, _ = resize(img, None, 256) #in validation ora è così
   recons = unet(img)
   mean = torch.tensor([0.485, 0.456, 0.406])
   std = torch.tensor([0.229, 0.224, 0.225])
   unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
   recons = unnormalize(recons)
   recons = torch.clamp(recons, min=0, max=1)
   trans_r = transforms.Resize((img.shape[2], img.shape[3]), antialias=False)
   recons = trans_r(recons)
   save_image(recons, name_path_save)
