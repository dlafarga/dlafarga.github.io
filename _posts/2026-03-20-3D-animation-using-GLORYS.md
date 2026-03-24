---
layout: post
title: "3D animation of regional cross-sections for high-resolution climate data"
author: "Dani Lafarga"
categories: journal
tags: [documentation,sample]
image: /3D_viz/EOF 1_zonal_animation.gif
---

# Motivation
In the [previous tutorial](https://dlafarga.github.io/journal/3D-Visualization-Tutorial-using-GLORYS.html) I taught you how to create 3D visualizations for zonal, meridional, and depth cross-sections. Though the cross-sections are great, sometimes they are hard to understand and intuiatively place their location. For example, the meridional cut is tough to understand on its own:
![Meridonal Cross]({{ site.url }}/assets/img/post4/merd_cross.png){: .center-image }

<center>Meridional cross-section at 160E in the North Pacific. This is harder to place intuatively than the cube.</center>

But if you visualize this cut with other cross-sections we can better interpret the figure

![Cube]({{ site.url }}/assets/img/post4/3D_cube.png){: .center-image }

<center>North Pacific with a depth, zonal, and meridional cross-section.</center>

You can also plot multiple of the same cross-sections at once. This is great for publications to show how the climate changes with depth, latitude, or longitude.

![multi zonal]({{ site.url }}/assets/img/3D_viz/multi_zonal_EOF1.png){: .center-image }

<center>Multiple zonal cross-sections for the Cali coast.</center>

I've found animating the cross-section has more impact, especially for presentations. 

In this tutorial we will go over how to create an animation of cross-sections by using functions created using the [previous tutorial](https://dlafarga.github.io/journal/3D-Visualization-Tutorial-using-GLORYS.html).

# Background and set up
For information on the data please see the data and methods section of [this post](https://dlafarga.github.io/journal/3D-Visualization-Tutorial-using-GLORYS.html). The main thing you need to keep in mind is we are working with high-resolution data ($1/12^{\circ}$ latitude by $1/12^{\circ}$ longitude with 50 depth layers). This means that regional animations are going to look great! We will be taking advantage of this and mostly animating within the Pacific region. 

Here are the libraries you will need:
```python
# all visualization libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# to get path
import os

# libraries to read data
import netCDF4 as nc
from netCDF4 import Dataset as ds
import numpy as np

# libraries used for some math
from numpy import linspace
from numpy import meshgrid
import math
```
For the climate data I build a custom colormap

```python
# Creating the custom colorbar
top2 = cm.get_cmap('GnBu_r')   # get green blue colormap
bottom2 = cm.get_cmap('hot_r') # get hot colormap
top_array = top2(np.linspace(0, 1, 128))        # create array with colorvalues
bottom_array = bottom2(np.linspace(0, .9, 128)) # create array with colorvalues
# edit array with color values to have better transition shades
top_array[-2:,:] = bottom_array[0,:]
top_array[-3,:] = np.array([1., 0.98823529, 1., 1.])
top_array[-4,:] = np.array([0.96862745, 0.98823529, 1., 1.])
top_array[-5,:] = np.array([0.96862745, 0.98823529, 0.94117647, 1.])

newcolors2 = np.vstack((top_array, bottom_array))         # stacking color arrays on top of each other
newcmp2 = ListedColormap(newcolors2, name='OrangeBlue')   # creating new colormap
```

To format the latitude and longitude axis I use two main functions that will add N (north) or S (south) to the latitudes and W (west) or E (east) to the correspoding longitudes. 

```python
#################################################################################################################
#################################################################################################################
# Function formats longitude to get rid of degree symbols
# Input:
#         - longitude: int with longitude from 0 to 360
# Ouput:
#         - string with longitude value and W (west) or E (east)
def format_longitude(longitude):
    if not 0 <= longitude <= 360:
        return "Invalid longitude. Must be between 0 and 360 degrees."

    if longitude == 0:
        hemisphere = ''
        degrees = longitude
    elif longitude < 180:
        hemisphere = 'E'
        degrees = longitude
    elif longitude == 180:
        hemisphere = ''
        degrees = longitude
    else:
        hemisphere = 'W'
        degrees = 360 - longitude

    return f"{degrees:.0f}{hemisphere}"

#################################################################################################################
#################################################################################################################

# Function formats latitude to get rid of degree symbols
# Input:
#         - latitude: int with latitude in degrees. Positive values are N and negative are S.
# Output:
#         - string with absolute value of latitude and S or N
def format_latitude(latitude):
    if not -90 <= latitude <= 90:
        return "Invalid latitude. Must be between -90 and 90 degrees."
# adding S or N based on negative or positive value
    if latitude > 0:
        hemisphere = "N"
    elif latitude == 0:
        hemisphere = ""
    else:
        hemisphere = 'S'

    degrees = abs(latitude)

    return f"{degrees:.0f}{hemisphere}"
```

There are three functions used to process the data
- get_var: Used to read in the latitude, longitude, and depth variables of the data we are plotting. This is heavily dependent on the names of the variables in the data file.
- vol_weight: used to calculate the volume at each grid point. This is important to represent the dimensions right in the data, but is not necessary for plotting. **The EOF is weighted meaning the volume is multiplied in already,** this means we need to divide the volume out for visualization.
- read_EOFs: this reads in the data file and will divide out the volume at the end.

```python
#################################################################################################################
#################################################################################################################
# Function get_var() will get variables that will be required for EOFs
# Input:
#         - fn: a string with the complete path of the data
# Output:
#         - lat: 1d array with all latitude values
#         - lon: 1d array with all longitude values
#         - depth: 1d array with all depth values
#         - years: 1d array with all year values
# NOTE: Change variable names according to your file
def get_var(fn):
    fn     =  ds(fn,'r')
    lat    = fn.variables['lat'][:].data    # read in latitude
    lon    = fn.variables['lon'][:].data    # read in longitude
    depths = fn.variables['depth'][:].data  # read in depth
    fn.close()
    return lat, lon, depths
#################################################################################################################
#################################################################################################################
# Function compute volume weights based on latitude and depthe values. Although longitude values are not
# in the equation the length of the longitude array is necessary for building 3D volume weight array
# Input:
#         - lat: 1d array with all latitude values
#         - lon: 1d array with all longitude values
#         - depth: 1d array with all depth values
# Output:
#         - area_weight: 3D array with volume weight
def vol_weight(depths, lon, lat):
    xx, yy = meshgrid(lon, lat)
    tot_depth = len(depths)
    # area weight for lattitude values
    area_w = np.cos(yy*math.pi/180)
    if lat[-1] == 90.0:
        area_w[-1,:] = 0.0
    # volume weights for depth
    volume_weight = []
    for i in range(tot_depth):
        if i == 0:
            volume_weight.append(np.sqrt(depths[0] * area_w)) # first depth thickness
        else:
            volume_weight.append( np.sqrt((depths[i] - depths[i - 1]) * area_w))
    # Turning weights into one array
    volume_weight = np.array(volume_weight)
    return volume_weight
#################################################################################################################
#################################################################################################################
# Function will read one EOF mode
# Input:
#         - mode: (int) describing which mode to read
# Output:
#         - EOF: 3D float array with the EOF at a defined cut
def read_EOFs(fn):
  get_var(fn)
  EOF_ncfile = ds(fn, 'r')
  EOF = EOF_ncfile.variables['EOF']
  EOF = EOF[:].filled()
  EOF_ncfile.close()
  volume_weight = vol_weight(depths, lon, lat)
  EOF = EOF/volume_weight  # remember to div by volume weight
  return EOF
  ```

  To read in the EOFs just use **os** to change to the appropriate directory and then run:
  ```python
# define the complete path with the file name
data_directory = os.getcwd()
fn     = 'EOF_1.nc'
fn     = os.path.join(data_directory, fn)

lat, lon, depths = get_var(fn) # read variables
EOF1 = read_EOFs(fn) # read EOF 1
  ```
## Some notes before we get into plotting each cross-section
There are is one main function that does most of the plotting. You can find the documentation of the function at https://matplotlib.org/stable/gallery/mplot3d/box3d.html.

For each of these plots assume:
- X-axis is longitude
- Y-axis is latitude
- Z-axis is depth

We define the variables **X, Y,** and **Z** as 3D arrays with their respective values that are cut down to a specific range. To do this we use the function meshgrid with the order longitude, latitude, and depth. The meshgrid is built from cut dimensions to focus on a specific region.

## The overview
For each cross-section we create a function that will produce a figure object with one cut plotted. We will then save this figure as a PNG and then put all the PNG files together into a GIF. 

There are a few things that we keep seperate from the function so that it is easier to customize the GIF without having to change the function:
- The set up for the region
- Defining the ticks for each axis
  - this makes sure your labels aren't to close or far apart
- Defining the cross-sections you want to plot based on the indices
- Setting up the bounds and total color bins for the colorbar
- Creating the title



# Zonal cross-section animation
Lets start with one of my favorite cross-sections, the zonal cross-section. This cut will tell us how the climate changes with latitude. 

We will use the California coast as an example. We define the indices, tick labels, and the aforementioned meshgrid variables for this region:

  ```python
# Set up cube for the Cali coast
lat_cut_start = 1320   # index for 30 N
lat_cut_end = 1537     # index for 48 N
lon_cut_start = 2760   # index for 130 W
lon_cut_end = 3013     # index for 109 W
depth_cut_end = 30     # index for 400

# Defining the range and position of each tick for labeling
# last number changes the interval
lon_ticks   = np.arange(lon[lon_cut_start], lon[lon_cut_end], 3)
lat_ticks   = np.arange(lat[lat_cut_start], lat[lat_cut_end], 3)
depth_ticks = np.arange(0, depths[depth_cut_end], 100) # if plotting the first 100 meters change the last number to something =<25

# create grid for each lat, lon, and depth variable
X, Y, Z = np.meshgrid(lon[lon_cut_start:lon_cut_end], lat[lat_cut_start:lat_cut_end], -depths[0:depth_cut_end])
```

We create a function that will take in the arguments:
    - title: string with tite for each figure
    - EOF: 3D array with data to be visualized
    - clip: float clip value that defines maximum and minimum for the colorbar
    - levels: int that sets how many colors you want to plot
    - lat_ind: int with the latitude index that defines the cross-section

The function can be broken down into 6 parts:
- figure object creation and set up
- land plotting
- cross-section plotting
- axis labeling and formatting
- 3D view setting
- colorbar labeling and formatting


```python
def plot_zonal_3D(title, EOF, clip, levels, lat_ind):
    # --- Setup Figure ---
    fig = plt.figure(figsize=(12, 13))
    fig.subplots_adjust(right = .95)  # Add this line
    
    title_sz = 20
    label_sz = title_sz-3
    
    ax = fig.add_subplot(111, projection='3d')  # This is what defines the plot as 3D
    levels = np.linspace(-clip, clip, levels + 1) # sets ticks on colorbar
    
    # Contour Norms
    norm = mpl.colors.Normalize(vmin= -clip, vmax=clip)
    #############################################################################
    # plotting land
    surface3D = EOF[0, lat_cut_start:lat_cut_end,lon_cut_start:lon_cut_end]
    
    mask = np.isnan(surface3D) # create a mask for the NaN values
    masked_array = np.where(mask, surface3D, np.nan)          # change points with values to NaN
    masked_array = np.where(~mask, masked_array, -clip) # change NaN points to values
    _ = ax.contourf(X[:, :, 0], Y[:, :, 0], masked_array, zdir='z', offset=-depths[0], cmap = mpl.colors.ListedColormap(['black']) ) # plotting the land
    # Contours
    #############################################################################
    # plotting the cross-section
    lat_depth3D = EOF[:depth_cut_end , lat_ind, lon_cut_start:lon_cut_end] # define the cross-section from the cut and index
    lat_depth3D = np.clip(lat_depth3D, -clip, clip) # clip the max and min
    # plot the cross-section contour
    C = ax.contourf(X[0, :, :], lat_depth3D.T, Z[0,:,:], zdir='y', levels=levels, cmap=newcmp2, offset= lat[lat_ind],
              norm = norm)
    #############################################################################
    # axis labeling and formatting
    ax.grid(True)
    ax.set_xticks(lon_ticks, labels=[format_longitude(int(l)) for l in lon_ticks], fontsize=label_sz, rotation = -65, ha = 'left')  # Requires format_longitude function to remove degree symbol
    ax.set_yticks(lat_ticks, labels=[format_latitude(int(l)) for l in lat_ticks], fontsize=label_sz, rotation = 45, va = 'center')  # Requires format_latitude function to remove degree symbol
    ax.set_zticks(-depth_ticks, labels=[f"{t:.0f}" for t in depth_ticks], fontsize=label_sz)
    ax.tick_params(axis='x', pad=0, labelsize=label_sz)
    ax.tick_params(axis='y', pad=6,  labelsize=label_sz)
    ax.tick_params(axis='z', pad=7,  labelsize=label_sz)
    
    ax.set_xlabel('Longitude', fontsize=label_sz, labelpad=47)
    ax.set_ylabel('Latitude', fontsize=label_sz, labelpad=16)
    ax.set_zlabel('Depth [m]', fontsize=label_sz, labelpad=14, rotation=0)
    ax.set_title(title, fontsize=title_sz)
    
    # Set limits
    ax.set_xlim(lon_ticks[0], lon_ticks[-1])
    ax.set_ylim(lat_ticks[0], lat_ticks[-1])
    ax.set_zlim(-depth_ticks[-1], 0)
    
    #############################################################################
    # 3D view setting
    ax.set_box_aspect((1, 1, 1))
    
    # view from above to make sure plot matches
    ax.view_init(elev=40, azim=-150, vertical_axis='z')
    #############################################################################
    # colorbar labeling and formatting
    cbar = fig.colorbar(C, format = mpl.ticker.ScalarFormatter(useMathText=True), fraction=0.03, pad = .05)
    cbar.ax.yaxis.get_offset_text().set_fontsize(title_sz) # change exp size
    cbar.ax.yaxis.OFFSETTEXTPAD = 11           # moving exponent so it doesnt overlap with top of colorbar
    cbar.ax.yaxis.set_offset_position('left')  # setexponent so it is more left
    cbar.ax.tick_params(labelsize=label_sz)    # set label size of ticks
    cbar.formatter.set_powerlimits((0, 0))     # formatting scientific notation
    cbar.update_ticks()
    #############################################################################
    
    return fig
  ```

Each time we call the function it will plot the cut based on a latitude index and output the figure object that we can then use to save as a PNG.  


```python
# create cross-section figures and save as PNG
clip = 0.00009
title = 'EOF 1'
data  = EOF1
lat_indices = np.arange(lat_cut_start, lat_cut_end, 6)
pic_directory = data_directory
for i, lat_ind in enumerate(lat_indices):
    fig = plot_zonal_3D(title, data, clip, lat_ind)
    fn     = 'EOF1_Zonal_Cross_Section' + str(i) + '.png'
    fn     = os.path.join(pic_directory, fn)
    plt.savefig(fn, dpi=300, bbox_inches='tight')
    plt.close(fig)
  ```

I then take the PNGs
```python
import imageio
from PIL import Image
import glob

gif_path = data_directory # set GIF path
frame_files = []
# call all cross-section figures saved
for i in range(len(lat_indices)):
  fn     = 'EOF1_Zonal_Cross_Section' + str(i) + '.png'
  fn     = os.path.join(data_directory, fn)
  frame_files.append(fn)
```
and create a a GIF.
```python
output_path = os.path.join(gif_path, f'{title}_zonal_animation.gif') # give gif a name based on title

frames = [Image.open(frame).convert('RGB') for frame in frame_files] # put all figs together

# save figs into frames
frames[0].save(
    output_path,
    save_all=True,
    append_images=frames[1:],
    duration=200,
    loop=0,
    optimize=False,  # Don't compress
    quality=500  # Maximum quality
)
```
Lastly, don't forget to delete your PNG files!

```python
# Delete the PNG files
for file in frame_files:
    os.remove(file)

print(f"{output_path} created!")
```
This animation shows multiple zonal cross-sections going up the California coast for EOF 1. Using this you can observe  how ENSO cases warm anomalies on the California coast and how changes as it moves north.

<img src="https://raw.githubusercontent.com/dlafarga/Modern-technology-for-climate-data-and-analysis/main/GLORYS%20figures/EOF%201_zonal_animation.gif" width="80%">

# Depth cross-section animation
