#!/usr/bin/env python
# coding: utf-8

# **Demonstration of RoboPy Pose.plot() and SerialLink.plot() rendering capability using IPV (ipyvolume).**

# In[1]:


import os  # for checking values of environment variables.

""" Matplotlib imports
"""
import matplotlib
matplotlib.use('Qt4Agg')
get_ipython().run_line_magic('matplotlib', 'notebook')

""" Numerical imports
"""
import numpy as np

""" RoboPy imports
"""
import _robopy
from robopy.base.graphics import GraphicsRenderer, trplot
import robopy.base.transforms as tr
import robopy.base.pose as pose
import robopy.base.model as model


# In[2]:


# Select a Graphics Rendering package to use.
gobj = GraphicsRenderer('IPV')  # this sets graphics.gRenderer

# Display a blue open-ended box to show default figure properties.
gobj.draw_cube()
gobj.show()


# In[3]:


# Define some GraphicsIPV parameters which will be used in plot()
# method calls in following cells.

dMode = 'IPY'
limits = [-1.5, 1.5, -1.5, 1.5, -1.5, 1.5]


# In[4]:


# Plot SE3 pose using IPV (ipyvolume) and display below.
pose.SE3.Rx(theta=[45, 90], unit='deg').plot(dispMode=dMode, key=1, z_up=True, limits=limits)


# In[5]:


# Plot same SE3 transforms as previous cell, but use the trplot() function.

T = tr.rotx([45, 90], unit='deg')
trplot(T, key=2)


# In[6]:


# Define a Puma560 robot model.
robot = model.Puma560()
    
# Puma560 manipulator arm pose plot using MPL and displayed below.
robot.plot(robot.qn, dispMode=dMode, key=3, z_up=False, limits=None)


# In[7]:


# Puma560 animation

a = np.transpose(np.asmatrix(np.linspace(1, -180, 500)))
b = np.transpose(np.asmatrix(np.linspace(1, 180, 500)))
c = np.transpose(np.asmatrix(np.linspace(1, 90, 500)))
d = np.transpose(np.asmatrix(np.linspace(1, 450, 500)))
e = np.asmatrix(np.zeros((500, 1)))
f = np.concatenate((d, b, a, e, c, d), axis=1)

# Give graphics renderer pose DisplayList to animate for all 500 poses.
#
# Note: There may be a considerable delay while all figures are generated 
#       before being passed to animation_control (noticeable here).

gIpv = GraphicsRenderer('IPV')  # sets graphics.gRenderer (to clear previous figure)
robot.animate(stances=f, unit='deg', key=4, timer_rate=60, gif="Puma560", 
                         frame_rate=30, dispMode='IPY', limits=None)


# In[ ]:




