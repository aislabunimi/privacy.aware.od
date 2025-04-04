import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
import numpy as np

# columns extraction
def extract_numbers(row, key_col, metric):
    name = row[key_col] #get setting
    if name[0].isdigit(): #if not tasknet or allprop
        parts = name.split('pos')
        x_part = parts[0]
        y_part = parts[1].split('neg')[0]
        if BFD and row[metric]<1:
           return int(x_part), int(y_part), 1 # to avoid problem with logs
        return int(x_part), int(y_part), row[metric]
    return None

BFD = False
inverted_view = False

file_path = 'testing_ap.csv'
key_col = 'Setting'
metric = 'AP$_{50}$'
label = 'AP$_{50}$'

"""
file_path = 'testing_backward.csv'
key_col = 'Setting'
metric = 'MS-SSIM'
label = 'MS-SSIM'
"""
"""
file_path = 'testing_iou50_custom_metric.csv'
key_col = 'Setting'
metric = 'FPiou_1'
label = 'ln(BFD%)'
BFD = True
#metric = 'TP_1'
#label = 'TP%'
inverted_view = True
"""
"""
file_path = 'testing_iou75_custom_metric.csv'
key_col = 'Setting'
#metric = 'FPiou_1'
label = 'ln(BFD%)'
#BFD = True
metric = 'TP_1'
label = 'TP%'
inverted_view = True
"""
df = pd.read_csv(file_path)

# grab only n_pos n_neg results
extracted = df.apply(extract_numbers, axis=1, args=(key_col, metric))
filtered = extracted.dropna()
# save them for bars
x, y, z = zip(*filtered)
if BFD:
   z = np.log(z)
# here for 3d bar plot
#fig = plt.figure()
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111, projection='3d')
# here colors
cmap = plt.get_cmap('autumn')
max_height = np.max(z)   # get range of colorbars so we can normalize
min_height = np.min(z)
min_height = min_height   #5 bias for avoiding that the min bar is black with inferno
max_height = max_height #+ 5
# scale each z to [0,1], and get their rgb values
rgba = [cmap((k-min_height)/(max_height-min_height)) for k in z] 

z_norm = (z - np.min(z)) / (np.max(z) - np.min(z)) #reduce max z for better scale
#z_norm = [((val - min(dz)) / (max(z) - min(dz)))*max(z) for val in dz]

if inverted_view:
   light = LightSource(azdeg=180, altdeg=45)
else:
   light = LightSource(azdeg=0, altdeg=45)

width = 0.75 # width of bar
depth = 0.75 # depth of bar (y axis)
#ax.bar3d(x, y, [0]*len(z_norm), width, depth, z_norm, shade=True, color=rgba)
ax.bar3d(x, y, [0]*len(z), width, depth, z, shade=True, color=rgba, lightsource=light)

#grab tasknet if exist and allprop
tasknet_df = df[df[key_col] == 'tasknet']
if not tasknet_df.empty:
   max_z_tasknet = tasknet_df[metric].values[0]
   if BFD:
      max_z_tasknet = np.log(max_z_tasknet)
   #if BFD:
   #   max_z_tasknet = (tasknet_df[metric].values[0] - np.min(z)) / (np.max(z) - np.min(z))
   #max_z_tasknet = 1
   
allprop_df = df[df[key_col] == 'all_proposals']
if not allprop_df.empty:
   max_z_allprop = allprop_df[metric].values[0]
   if BFD:
      max_z_allprop = np.log(max_z_allprop)
   #if BFD:
   #   max_z_allprop = (allprop_df[metric].values[0] - np.min(z)) / (np.max(z) - np.min(z))

#set z axis limit for displaying theoretical values
if not tasknet_df.empty:
   ax.set_zlim([0, max_z_tasknet])
   if BFD:
      ax.set_zlim([0, 6])
else:
   ax.set_zlim([0, max_z_allprop])

# labels, ticks and fontsize
ax.set_xlabel('Pos', fontsize=12)
ax.set_ylabel('Neg', fontsize=12)
ax.zaxis.set_rotate_label(False) #removing default orientation
ax.set_zlabel(label, fontsize=12, rotation=90)
ax.set_xticks([1, 2, 3, 4])
ax.set_yticks([0, 1, 2, 3, 4])
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.tick_params(axis='z', labelsize=12)

#old config with width=0.5
#verts = [[[0.5, -0.5, max_z_tasknet],  # bottom left
   # [4.5, -0.5, max_z_tasknet],  # bottom right
   # [4.5, 4.5, max_z_tasknet],   # top right
   # [0.5, 4.5, max_z_tasknet]]]    # top left

# surface for allprop results
verts = [[[0.75, -0.25, max_z_allprop],  # bottom left
    [5, -0.25, max_z_allprop],  # bottom right
    [5, 5, max_z_allprop],   # top right
    [0.75, 5, max_z_allprop]]]    # top left
surface = Poly3DCollection(verts, color='yellow', alpha=0)
ax.add_collection3d(surface)

if not tasknet_df.empty: # surface for tasknet results 
   verts = [[[0.75, -0.25, max_z_tasknet],  # bottom left
    [5, -0.25, max_z_tasknet],  # bottom right
    [5, 5, max_z_tasknet],   # top right
    [0.75, 5, max_z_tasknet]]]    # top left
   surface = Poly3DCollection(verts, color='red', alpha=0)
   ax.add_collection3d(surface)

if inverted_view:
   ax.view_init(elev=20, azim=135)  # used for BFD
else:
   ax.view_init(elev=20, azim=225)  # set elevation and azimuth angles


fig.tight_layout()
if inverted_view:
   fig.subplots_adjust(left=-0.11)

#format for paper:pdf
fig.savefig(f'{label}.pdf', format='pdf', bbox_inches='tight', pad_inches=0.4)
plt.show()
