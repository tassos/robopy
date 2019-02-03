#!/usr/bin/env python
# coding: utf-8

# **Demonstration of RoboPy plot(DisplayList) rendering capability using MPL (MatplotLib).**

# In[1]:


import os  # for checking values of environment variables.

""" Matplotlib imports
"""
import matplotlib
matplotlib.use('Qt4Agg')
get_ipython().run_line_magic('matplotlib', 'notebook')
from matplotlib import rc         # needed to set Matplotlib animation._repr_html_()

rc('animation', html='jshtml')    # Matplotlib animations will be HTML wrapped JavaScript

""" RoboPy imports
"""
# Import requisite robopy modules.
import _robopy                                     # locates desired robopy module
from robopy.base.graphics import GraphicsRenderer  # to select graphics renderer
import robopy.base.graphics_mpl as gmpl            # for rgb_named_colors
import robopy.base.display_list as dList           # to use display lists
import robopy.base.transforms as tr                # to apply transforms
from robopy.base.mesh_geoms import *               # to use mesh geometric shapes
import numpy as np                                 # to use NumPy ndarray type


# In[2]:


# Define some GraphicsMPL parameters which will be used in the
# Mpl3dArtist.plot() method calls in following cells.

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


# Obtain a robopy graphics renderer which utilizes matplotlib.

gMpl = GraphicsRenderer('MPL')  # sets graphics.gRenderer; returns Mpl3dArtist object

# Figure 0 with an empty XYZ axes system will be displayed below.


# In[9]:


# Display a mesh grid geometric object to show the default figure properties.

rgb = gmpl.rgb_named_colors(['purple'])
gMpl.plot_parametric_shape('sphere', solid=True, c=rgb[0])
gMpl.show()

# The purple sphere should be displayed in Figure 0 above.


# In[10]:


# Give a graphics renderer the DisplayList to plot (gRenderer.plot()).

gMpl.close()                    # close current figure
gMpl = GraphicsRenderer('MPL')  # sets graphics.gRenderer (creates new Figure 0)
gMpl.plot(dl, limits=limits)    # plot display list 'dl' created above.


# In[11]:


# Define transform function to animate DisplayListItems.

def transFunc(t):
    """
    Sample transformation function to rotate display list
    'surface' items about their x-axis.
    :param t: time (sec)
    :return: a homogeneous transform matrix
    """
    return tr.trotx(2.0*t, unit="deg")


# In[12]:


# Give the graphics renderer the DisplayList to animate.

gMpl.close()                    # closes current figure
gMpl = GraphicsRenderer('MPL')  # sets graphics.gRenderer (creates new Figure 0)
gMpl.animate(dl, transFunc, duration=5.0, frame_rate=30, limits=limits)


# In[13]:


# Wait until the animation completes before running this cell to create a controlled animation.

gMpl.getAnimDL()


# In[ ]:




