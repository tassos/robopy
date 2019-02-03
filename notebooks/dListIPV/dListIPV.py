#!/usr/bin/env python
# coding: utf-8

# **Demonstration of RoboPy plot(DisplayList) rendering capability using IPV (ipyvolume).**

# In[1]:


import os  # for checking values of environment variables.

""" Matplotlib imports
"""
import matplotlib
matplotlib.use('Qt4Agg')
get_ipython().run_line_magic('matplotlib', 'notebook')

""" RoboPy imports
"""
# Import requisite robopy modules.
import _robopy                             # locates desired robopy module
import robopy.base.graphics as graphics    # to perform graphics
import robopy.base.graphics_mpl as gipv    # for rgb_named_colors
import robopy.base.display_list as dList   # to use display lists
import robopy.base.transforms as tr        # to apply transforms
from robopy.base.mesh_geoms import *       # to use mesh geometric shapes
import numpy as np                         # to use NumPy ndarray type


# In[2]:


# Define some GraphicsIPV parameters which will be used in the
# Ipv3dVisual.plot() method calls in following cells.

dMode = 'IPY'
limits = [-0.5, 0.5, -0.5, 0.5, -0.5, 0.5]


# In[3]:


# Define a mesh grid geometric object.

cyl = cylinder(0.2,0.2,0.05,0.1)  # create a cylinder


# In[4]:


# Create two transformation matrices.

Rx = np.array( [[1,0,0,0], [0,0,-1,0], [0,1,0,0],[0,0,0,1]])
Ry = np.array( [[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]] )


# In[5]:


# Create a DisplayList with 3 instances of the cylinder,
# keep the DisplayListItem references.

dl = dList.DisplayList()
dl1 = dl.add('surface', 'cyl1', cyl, color='blue')
dl2 = dl.add('surface', 'cyl2', cyl, color='red')
dl3 = dl.add('surface', 'cyl3', cyl, color='green')


# In[6]:


# Transform two of the cylinders by setting the transform field.

dl2.transform = Rx
dl3.transform = Ry


# In[7]:


# Obtain a robopy graphics renderer which utilizes ipyvolume.

gIpv = graphics.GraphicsRenderer('IPV')  # sets graphics.gRenderer; returns Ipv3dVisual object


# In[8]:


# Display a mesh grid geometric object to show the default figure properties.

rgb = graphics.rgb_named_colors(['purple'])
gIpv.plot_parametric_shape('sphere', solid=True, c=rgb[0])
gIpv.show()

# The purple sphere should be displayed in a figure below.


# In[9]:


# Give a graphics renderer the DisplayList to plot (uses gRenderer.plot()).

gIpv.clear()                             # clears previous figure
graphics.plot(dl, key=1, limits=limits)  # create plot on new figure (key=1)


# In[10]:


# Define transform function to animate DisplayListItems 
#
# Note: Utilizes new animation_timer() class decorator defined 
#       in the graphics module.

timer_rate = 60
frame_rate = 30
@graphics.animation_timer(timer_rate, frame_rate, real_time=False)
def transFunc(n, tstep, *args, **kwargs):
    """
    Sample transformation function to rotate display list
    'surface' items about their x-axis.
    :param n : number of steps (starts at 0)
    :param tstep: step time (sec)
    :return: a homogeneous transform matrix
    """
    t = n*tstep
    return tr.trotx(2.0*t, unit="deg")


# In[11]:


# Give a graphics renderer the DisplayList to animate for all time steps (anim_incr=False).
#
# Note: There may be a considerable delay while all figures are generated 
#       before being passed to animation_control (noticeable in poseIPV).

gIpv.clear()                             # clears previous figure
fps = frame_rate                         # display frame rate
tstep = 1.0/float(fps)                   # time step size for transform function
graphics.animate(dl, transFunc, key=2,   # create plot on new figure (key=2)
                 func_args=[tstep], anim_incr=False, duration=5.0, frame_rate=fps,
                 limits=limits)


# In[12]:


# Give graphics renderer the DisplayList to animate at each time step (anim_incr=True).
#
# Note: There is no delay as each figure is generated and passed to
#       animation control. However, issues with retention of previous 
#       images is still being worked out.

gIpv.clear()                             # clears previous figure
fps = frame_rate                         # display frame rate
tstep = 1.0/float(fps)                   # time step size for transform function
graphics.animate(dl, transFunc, key=3,   # create plot on new figure (key=3)
                 func_args=[tstep], anim_incr=True, duration=5.0, frame_rate=fps, 
                 limits=limits)


# In[ ]:




