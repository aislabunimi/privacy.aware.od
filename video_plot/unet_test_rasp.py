from torchvision.models import detection
#from imutils.video import VideoStream
#from imutils.video import FPS
import numpy as np
#import imutils
import torch
import torch.nn as nn
import time
import cv2
from torchvision import transforms

def load_checkpoint_encoder(model, model_save_path, optimizer, scheduler, load_optim_scheduler=False):
   checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'))
   mod = checkpoint['model_state_dict']
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

   if load_optim_scheduler:
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      epoch = checkpoint['epoch']
      loss = checkpoint['loss']
      scheduler.load_state_dict(checkpoint['lr_scheduler'])
   print('Encoder Loaded')

class Encoder(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(Encoder, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))

        #Removed skip connections in forward
    def forward(self, x):
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

#config
conf_threshold = 0.75
unet_weights_load= "model_weights/model_50.pt"
video_path='test_data/resize_video/parasite256x192.mp4'
unet_enc = Encoder(n_channels=3, bilinear=False)
unet_optimizer = torch.optim.SGD(unet_enc.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4, nesterov=True)
unet_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(unet_optimizer, mode='min', factor=0.5, patience=2, verbose=True)
load_checkpoint_encoder(unet_enc, unet_weights_load, unet_optimizer, unet_scheduler, load_optim_scheduler=False)
device='cpu'
unet_enc.to(device)
unet_enc.eval()
#norm and unnorm
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
   print("Error: Could not open the video file.")
   exit()
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = cap.get(cv2.CAP_PROP_FPS)

#On raspberry pi 3 model B (not B+)
#4:3			#removing last layer
#256x192 -> 0.20 fps 	0.26
#200x150 -> 0.33 fps	0.43	
#160x120 -> 0.50 fps	0.68

#16:9			#removing last layer
#256x144 -> 0.27 fps	0.35
#192x108 -> 0.48 fps	0.64
#160x90 -> 0.70 fps	0.95 fps

fps_start_time = time.time()
fps_counter = 0
tot_fps_measurements=0
fps_sum=0
while True:
   ret, frame = cap.read()
   if not ret:
      break
   orig = frame.copy()
   frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   frame = frame.transpose((2, 0, 1)) #inverting channels
   frame = torch.from_numpy(frame).float().to(device)
   frame = frame.unsqueeze(0) #add first dim to simulate batchaggiungo prima dim per simulare il batch
   frame = frame / 255.0 #first normalize between 0 and 1, then w.r.t. the ImageNet
   frame = norm(frame)
   reconstructed = unet_enc(frame)
   fps_counter += 1
   if time.time() - fps_start_time >= 1:
      fps = fps_counter / (time.time() - fps_start_time)
      fps_start_time = time.time()
      fps_counter = 0
      print("Average frame per second:", round(fps, 4))
      tot_fps_measurements+=1
      fps_sum+=fps
final_fps=fps_sum/tot_fps_measurements
print("Final average frame per second:", round(fps, 4))
cap.release()
out.release()
cv2.destroyAllWindows()
