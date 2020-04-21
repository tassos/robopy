#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: graphics_ipv.py
DATE: Tue Jan 26 08:23:00 2019

@author: garyd
"""

import sys
import pkg_resources
import copy
from types import *
from abc import abstractmethod

# RoboPy modules
from robopy.base.tb_parseopts import *
from robopy.base.graphics import Graphics, animation_timer
from robopy.base.display_list import *
from robopy.base.param_geoms import *

from . import transforms
from . import graphics as rtbG
from . import pose as Pose
from . import serial_link as SerialLink

# To load and handle STL mesh data
try:
    import trimesh
except ImportError:
    print("* Warning: trimesh package required for SerialLink")
    print("  plotting and animation. Attempts to use robot.plot()")
    print("  or robot.animate() will fail.")

# Graphics rendering package
try:
    import ipyvolume.pylab as p3
    import ipyvolume as ipv
except ImportError:
    print("* Error: ipyvolume package required.")
    sys.exit()

import matplotlib as mpl  # for color map
from IPython.display import display, clear_output
from ipywidgets import Play, FloatSlider, jslink, HBox, VBox, Output

'''
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
'''
# To produce animated GIFs
###import imageio
from . import images2gif as imgs2gif

# Support for IPython use in Jupyter Notebooks
# and Spyder IDE
import IPython.display
import PIL.Image

# Numerical packages
import math
import numpy as np

__all__ = ('GraphicsIPV', 'Ipv3dVisual',  # classes
           'rgb_named_colors',)           # functions

###
### Utility routines which do not require Ipv3dVisual class instances.
###

def rgb_named_colors(colors):
        """
        Returns a list of Matplotlib colors.
        :param colors: list of color names supported by Matplotlib
        :return rgb_colors: list of corresponding rgb color values
        """        
        if type(colors) is not list:
            colors = [colors]
            
        rgb_colors = [0] * len(colors)
        for i in range(len(colors)):
            rgb_colors[i] = mpl.colors.to_rgb(colors[i])
            
        return rgb_colors


###
### MODULE CLASS DEFINITIONS
###
    
class GraphicsIPV(Graphics):
    """ 
    Graphics rendering interface for the IPV rendering package.
      
    This class acts as the interface between RTB drawing and plotting
    routines and the ipyvolume (IPV) graphics library. Its attributes
    are private and its methods do not require access to RTB manipulator
    modeling object methods and data structures.
    """
    def __init__(self, fig):

        ##super(Graphics, self).__init__()

        # Graphics environment properties
        
        self._dispMode = 'IPY'
        self._fig = fig
        
        # Instantiate a ipyvolume.pylab figure.
        
        #self.setFigure()
        #self.setFigureAxes(self.getFigure())
        
        # plotting (graphing) properties

        self.setAxesLimits(-1.5, 1.5, -1.5, 1.5, -1.5, 1.5)

        p3.xlabel('X')
        p3.ylabel('Y')
        p3.zlabel('Z')
        
        # rendered artist properties
        
        self.mesh_list = []    # list of trimesh meshes from STL files
        self.item_list = []    # list of items plotted
        self.disp_list = None  # display list of graphics entities to render
        self.HBox = None
        self.VBox = None       # ipyvolume widgets virtual box?
    
    def setGraphicsRenderer(self, gRenderer):
        """ Sets graphic renderer for this IPV 3dVisuals.
        """
        super(GraphicsIPV, self).setGraphicsRenderer(gRenderer)
        
    def setGtransform(self, *args, **kwargs):
        """ Set graphics transform.
        """
        super(GraphicsIPV, self).setGtransform()
              
    ### Class properties and methods
       
    theDispModes = ['VTK','IPY','PIL']
          
    @classmethod
    def isDispMode(cls, dmode):
        return dmode in cls.theDispModes
       
    @staticmethod  # simply call a module scope routine
    def rgb_named_colors(cls, colors):           
        return rgb_named_colors
    
    ### Instance property setters/getters

    """
      _dispMode = property(fset=setDispMode, fget=getDispMode)
      _fig = property(fset=setFigure, fget=getFigure)
      _figkey = property(fget=getFigureKey)
      _ax = property(fset=setFigureAxes, fget=getFiguresAxis)
      mesh_list = property(fset=setMeshes, fget=getMeshes, fdel=delMeshes)
    """
    
    def setDispMode(self, dispmode):
        self._dispMode = dispmode
        
    def getDispMode(self):
        return self._dispMode
    
    def setFigure(self, key):
        if key is not None:
           self._fig = p3.figure(key=key, width=480, height=480)
           p3.style.background_color('black')
           p3.style.box_off()
           p3.style.use("dark")
        else:
           self._fig = p3.figure(key=key)
        
    def getFigure(self):
        return p3.gcf()

    def setFigureAxes(self, fig):
        '''
        ##self._ax = fig.add_subplot(111, projection='3d')
        ##self._ax = mplot3d.Axes3D(fig)
        ##self._ax.view_init(elev=2.0, azim=-35.0)  # a hardcoded dev convenience
        ##self._ax.set_frame_on(b=True)
        '''
        p3.view(0.0, -120.0)
        self.setFigure(p3.gcf())
        self._ax = fig

    def getFigureAxes(self):
        return self._ax
    
    def setAxesLimits(self, xlim, *args):
        if (type(xlim) is list) and len(xlim) == 6:
            [xmin, xmax, ymin, ymax, zmin, zmax] = xlim[:]
        elif len(args) == 5:
            xmin = xlim
            [xmax, ymin, ymax, zmin, zmax] = args[:]
        else:
            return  # no warning given

        p3.xlim(xmin, xmax)
        p3.ylim(ymin, ymax)
        p3.zlim(zmin, zmax)
        self._axesLimits = [xmin, xmax, ymin, ymax, zmin, zmax]
    
    def getAxesLimits(self):
        return self._axesLimits
    
    def setMeshes(self, meshes):
        self.mesh_list = meshes
        
    def setMeshI(self, i, mesh):
        if i in range(len(self.mesh_list)):
            self.mesh_list[i] = mesh
            
    def getMeshes(self):
        return self.mesh_list
    
    def getMeshI(self, i):
        if i in range(len(self.mesh_list)):
            return self.mesh_list[i]
        
    def delMeshes(self):
        self.mesh_list = []
        
    def addMeshes(self, meshes):
        self.mesh_list.append(meshes)

    ### Class methods (presented by functional group)

    ### ...

    ## Plot elements methods

    def plot_parametric_shape(self, shape, solid=False, Tr=None, **opts):
        """
        Plot specified parametric shape.
        :param shape: geometric shape name ('box', beam', 'sphere', etc.)
        :param solid: True for plot surface, False for plot wire mesh
        :param Tr: list of homogeneous or rotation transform matrices
        :param opts: plot options name/value pairs
        :return:
        """
        # create parametric shape xyz coordinate arrays
        (x, y, z) = param_xyz_coord_arrays(shape, **opts)

        # check if Tr is ...
        if Tr is None or (type(Tr) is list and len(Tr) == 0):
            # undefined or an empty list,
            Tr = [np.identity(4)]
        elif type(Tr) in (np.ndarray, transforms.RTBMatrix) and len(Tr.shape) < 3:
            # or a single transform
            Tr = [Tr]

        # check if color option specified
        c = 'black'
        if 'c' in opts:
           c = opts['c']

        # pack the xyz coordinate arrays based on shape of transform
        (xyz, dim0, dim1) = param_xyz_coord_arrays_packed(x ,y, z, Tr=Tr[0])

        # apply transform to packed xyz coordinate arrays
        n = len(Tr)
        xr = np.ndarray((n, x.shape[0], x.shape[1]));
        yr = np.ndarray((n, y.shape[0], y.shape[1]));
        zr = np.ndarray((n, z.shape[0], z.shape[1]));
        for k in range(n):
            (xr[k,:,:], yr[k,:,:], zr[k,:,:]) = param_xyz_coord_arrays_packed_xform(xyz, dim0, dim1, Tr[k])

        # plot the transformed xyz coordinate arrays as meshes

        if shape == 'frame':
            # coordinate frame +axes vectors

            ### Implementation Note:
            ###
            ### arrowcols presumes colors of Lx, Ly, Lz and quiver are
            ### to be red, green, blue for all frames rendered:

            colors_rgb = rgb_named_colors(['red', 'green', 'blue'])
            arrowcols = np.zeros((1, 3, 3))
            arrowcols[0, 0, 0] = colors_rgb[0][0]
            arrowcols[0, 1, 1] = colors_rgb[1][1]
            arrowcols[0, 2, 2] = colors_rgb[2][2]

            for k in range(n):
                # draw +x, +y, +z axes line segments
                Lx = p3.plot(xr[k, 0:2, 0], yr[k, 0:2, 0], zr[k, 0:2, 0], color=colors_rgb[0])
                Lx.material.visible = False
                Lx.line_material.visible = True
                Ly = p3.plot(xr[k, 0:2, 1], yr[k, 0:2, 1], zr[k, 0:2, 1], color=colors_rgb[1])
                Ly.material.visible = False
                Ly.line_material.visible = True
                Lz = p3.plot(xr[k, 0:2, 2], yr[k, 0:2, 2], zr[k, 0:2, 2], color=colors_rgb[2])
                Lz.material.visible = False
                Lz.line_material.visible = True
                # calculate and draw +x, +y, +z axes line segment tip quiver vectors
                tail = 0
                head = 1
                vxr = xr[k, head, :] - xr[k, tail, :]
                vyr = yr[k, head, :] - yr[k, tail, :]
                vzr = zr[k, head, :] - zr[k, tail, :]
                p3.quiver(xr[k, head, :], yr[k, head, :], zr[k, head, :], vxr[:], vyr[:], vzr[:],
                         size=10, size_selected=5, color=arrowcols, color_selected='gray')
                '''
                ### for whenever ipyvolume can display text
                p3.plot_text(xr[:, head, 0], yr[:, head, 0], zr[:, head, 0], 'X', ha='left', va='center', color='r')
                p3.plot_text(xr[:, head, 1], yr[:, head, 1], zr[:, head, 1], 'Y', ha='left', va='center', color='g')
                p3.plot_text(xr[:, head, 2], yr[:, head, 2], zr[:, head, 2], 'Z', ha='left', va='center', color='b')
                '''
        else:
            wire = not solid
            surf = solid

            ### Implementation Note:
            ###
            ### colors_rgb presumes colors of all rendered meshes are c

            colors_rgb = rgb_named_colors([c])

            ### Implementation Note:
            ###
            ### Passing [:,dim0,dim1] ndarrays to plot_mesh probably introduces
            ### implicit streaming; not sure ipyvolume expects or can properly
            ### handle this as it can with scatter and quiver plots.
            ###
            ### p3.plot_mesh(xr[:,0:dim0,0:dim1],
            ###              yr[:,0:dim0,0:dim1],
            ###              zr[:,0:dim0,0:dim1], wireframe=wire, surface=surf, color=colors_rgb[0])

            for k in range(n):
                p3.plot_mesh(xr[k], yr[k], zr[k], wireframe=wire, surface=surf, color=colors_rgb[0])

    def draw_axes2(self, *args, **kwargs):
        """ Graphics package draw plot axes for 2D space.
        """ 
        print("* Not yet implemented.")
        return None
        
    def draw_axes3(self, *args, **kwargs):
        """ Graphics package draw plot axes for 3D space.
        """ 
        print("* Not yet implemented.")
        return None
        
    def draw_cube(self):
        (x, y, z) = parametric_box(1.0)
        p3.plot_surface(x, y, z, color='blue')
    
    def draw_sphere(self):
        (x, y, z) = parametric_sphere(1.0, 32)
        p3.plot_surface(x, y, z, color='red')
    
    ### Rendering viewpoint methods

    ### ...
    
    ### Animation display methods
    
    def render(self, ui=True):
        """
        Renderers current artists in ready render window.
        """
        print("* Not yet implemented.")
        return None
    
    def mpl_animate(self, *args, **kwargs):
        """
        Creates animation of current actors in ready render window.
        """
        raise NotImplementedError('Need to define animate method.')

    def show(self, *args):
        if len(args) > 0:
            p3.show(extra_widgets=args)
        else:
            p3.show()

    def clear(self):
        p3.clear()

    def close(self):
        p3.close()

    ### Abstract methods for RTB interface
    
    @abstractmethod
    def view(self, *args, **kwargs):
        raise NotImplementedError('Need to define view method.')
        
    @abstractmethod
    def animate(self, *args, **kwargs):
        raise NotImplementedError('Need to define animate method.')

    @abstractmethod
    def fkine(self, *args, **kwargs):
        raise NotImplementedError('Need to define fkine method.')

    @abstractmethod
    def plot(self, *args, **kwargs):
        raise NotImplementedError('Need to define plot method!')

    @abstractmethod
    def qplot(self, *args, **kwargs):
        raise NotImplementedError('Need to define qplot method.')
    
    @abstractmethod
    def trplot(self, *args, **kwargs):
        raise NotImplementedError('Need to define trplot method.')
    
    @abstractmethod    
    def trplot2(self, *args, **kwargs):
        raise NotImplementedError('Need to define trplot2 method.')
    
    @abstractmethod
    def tranimate(self, *args, **kwargs):
        raise NotImplementedError('Need to define tranimate method.')
        
    @abstractmethod
    def tranimate2(self, *args, **kwargs):
        raise NotImplementedError('Need to define tranimate2 method.')

    # DisplayList rendering, plotting and animation methods

    ### Implementation Note:
    ###
    ### Although the following two routines are identified as renderers,
    ### they do not actually interface with a graphical renderer. They
    ### both assemble mesh data structures which will be passed to the
    ### graphical renderer with an appropriate ipyvolume plot routine.

    def renderDisplayListItem(self, item):
        if item.type == 'surface':
            data = item.xform()  # get the transformed coordinates
            x = data[0].reshape((1, data[0].shape[0], data[0].shape[1]))
            y = data[1].reshape((1, data[1].shape[0], data[1].shape[1]))
            z = data[2].reshape((1, data[2].shape[0], data[2].shape[1]))
            if item.gentity is None:
                item.gentity =  [np.ndarray((1, data[0].shape[0], data[0].shape[1])),
                                 np.ndarray((1, data[1].shape[0], data[1].shape[1])),
                                 np.ndarray((1, data[2].shape[0], data[2].shape[1]))]
                item.gentity[0] = x
                item.gentity[1] = y
                item.gentity[2] = z
            else:
                item.gentity = (np.vstack((item.gentity[0], x)),
                                np.vstack((item.gentity[1], y)),
                                np.vstack((item.gentity[2], z)))
            self.item_list.append(item)  # save each item plotted
        elif item.type == 'stl_mesh':
            a_mesh = item.data
            a_mesh.apply_transform(item.transform)
            xyz = np.asarray(a_mesh.vertices).T
            x = np.asarray(xyz[0,:]).flatten().T
            y = np.asarray(xyz[1,:]).flatten().T
            z = np.asarray(xyz[2,:]).flatten().T
            f = a_mesh.faces
            c = np.zeros((1, x.shape[0], 3))
            c[0,:,:] = np.asarray(rgb_named_colors(item.args['color']))
            if item.gentity is None:
                item.gentity =  [np.ndarray((1, x.shape[0])),
                                 np.ndarray((1, y.shape[0])),
                                 np.ndarray((1, z.shape[0])),
                                 np.ndarray((1, f.shape[0], f.shape[1])),
                                 np.ndarray((1, c.shape[1], c.shape[2]))]
                item.gentity[0] = x
                item.gentity[1] = y
                item.gentity[2] = z
                item.gentity[3] = f
                item.gentity[4] = c
            else:
                item.gentity = (np.vstack((item.gentity[0], x)),
                                np.vstack((item.gentity[1], y)),
                                np.vstack((item.gentity[2], z)),
                                np.vstack((item.gentity[3], f)),
                                np.vstack((item.gentity[4], c)))
            self.item_list.append(item)  # save each item plotted
        elif item.type == 'command':
            eval(item.command, globals())  # eval the command in global context
        elif item.type == 'line':
            # TODO
            pass

    def renderDisplayList(self, displayList):
        for item in displayList:
            self.renderDisplayListItem(item)

    def plotDisplayList(self, displayList, dispMode='IPY', **kwargs):
        """
        Plots the DisplayList graphic entities.
        :param displayList: a DisplayList object.
        :param: dispMode: display mode, one of ['VTK', 'IPY', 'PIL'].
        :return: None.
        """
        # parse argument list options
        opts = {'dispMode': dispMode,  # holdover from GraphicsVTK
                'z_up': True,          # holdover from GraphicsVTK
                'limits': self.getAxesLimits(),
                }

        opt = asSimpleNs(opts)

        (opt, args) = tb_parseopts(opt, **kwargs)

        self.setAxesLimits(opt.limits)
        self.renderDisplayList(displayList)

        # plot items on the same figure axes as in MATLAB with 'hold on'
        S = []
        for item in self.item_list:
            if item.type == "surface":
                (x, y, z) = item.gentity
                S.append(ipv.plot_surface(x, y, z, **item.args))
            elif item.type == "stl_mesh":
                (x, y, z, f, c) = item.gentity
                S.append(ipv.plot_trisurf(x, y, z, triangles=f, color=c))

        self.show()

    def animateDisplayList(self, displayList, anim_func, func_args=[],
                                 anim_incr=True, unit='rad', gif=None,
                                 duration=5.0, frame_rate=30, **kwargs):
        """
        Animates DisplayList items thru transformations as function of frame number.
        May be specified as a function of the form Tr = anim_func(nf, *func_args)
        where Tr may be:

          1) a single homogeneous matrix applied to all DisplayList items, or
          2) a list of homogeneous matrices with each applied to the item
             in DisplayList based on its order in the list.

        or as an np.matrix array of the form Tr[nf, n] where nf is frame count and
        n is corresponding item's order in the DisplayList.

        :param displayList: a DisplayList object.
        :param anim_func: homogeneous transformation function or array (see above).
        :param func_args: list of argument values passed to the anim_func
        :param anim_incr: set to True to render incrementally with animation_control
        :param unit: unit of input angles. Allowed values: 'rad' or 'deg'.
        :param gif: name for the written animated GIF image file (not used yet).
        :param duration: duration of animation in seconds.
        :param frame_rate: frame_rate for animation.
        :param kwargs: dictionary of display list options keyword/value pairs.
        :return: None
        """
        # parse argument list options
        opts = {'func_args' : func_args,
                'anim_incr': anim_incr,
                'unit'      : unit,                  # holdover from animateSerialLink
                'gif'       : gif,                   # holdover from GraphicsVTK
                'duration'  : duration,
                'frame_rate': frame_rate,
                'dispMode'  : self.getDispMode(),    # holdover from GraphicsVTK
                'z_up'      : False,                 # holdover from GraphicsVTK
                'limits'    : self.getAxesLimits(),
                }

        opt = asSimpleNs(opts)

        (opt, args) = tb_parseopts(opt, **kwargs)

        # calculate framing timing parameters

        frame_step_msec = int(1000.0 / opt.frame_rate)

        # get number of frames; verify anim_func as Tr function or stances

        nframes = 0
        if anim_func is not None:
            if type(anim_func) is FunctionType:
                if opt.func_args is not None:
                    assert type(opt.func_args) is list
                else:
                    opt.func_args = []
                if type(opt.duration) is float:
                    nframes = int(opt.duration * opt.frame_rate) + 1
                else:
                    nframes = opt.duration
            else:
                # anim_func of the form q[k,n], where k
                # is frame count and n is joint #.
                assert type(anim_func) is np.matrix
                nframes = min(np.matrix.shape[0], int(opt.duration))

        # set figure axes limits
        self.setAxesLimits(opt.limits)

        # reset display list item's gentity property
        displayList.reset()

        ### vvv Development Notes:
        ###
        ### See below.
        if opt.anim_incr:
            out = Output()
            fig = p3.gcf()
            fig.animation = frame_step_msec
            fig.animation_exponent = 1.0
            self.VBox = VBox([p3.gcc()])
        ## ^^^

        # render frames
        for n in range(0, nframes):
            # update display list item's transforms
            if type(anim_func) is FunctionType:
                 Tr = anim_func(n, *opt.func_args)
            else:
                 Tr = anim_func(n)
            if type(Tr) is list:
                # SerialLink
                k = 0
                for item in displayList:
                    if k > 1:
                        a_mesh = copy.deepcopy(self.getMeshI(k-2))
                        a_mesh.apply_transform(Tr[k])
                        # to use STL meshes defined wrt VTK reference frame
                        a_mesh.apply_transform(transforms.trotx(90.0, unit="deg"))
                        item.data =  a_mesh
                        item.transform = np.identity(4)
                    else:
                        item.transform = np.asarray(Tr[k])
                    k += 1
            else:
                for item in displayList:
                    item.transform = np.dot(item.transform, np.asarray(Tr))

            # clear plotted items list
            self.item_list.clear()
            # render display list graphics entities
            self.renderDisplayList(displayList)

            ### vvv Development Notes:
            ###
            ### Attempts to show rendering frame by frame before presenting
            ### completed rendered sequence to animation control.
            if opt.anim_incr:
                S = []
                for item in self.item_list:
                    if item.type == "surface":
                        (x, y, z) = item.gentity
                        S.append(p3.plot_surface(x, y, z, **item.args))
                    elif item.type == "stl_mesh":
                        (x, y, z, f, c) = item.gentity
                        S.append(p3.plot_trisurf(x, y, z, triangles=f, color=c))
                if n == 0:
                    # create initial play and slider widgets, put both in ipywidget HBox
                    play = Play(min=0, max=1, interval=frame_step_msec, value=0, step=1)
                    slider = FloatSlider(min=0, max=play.max, step=1)
                    self.HBox = HBox([play, slider])
                    # put p3 current content children in ipywidget VBox
                    self.VBox = VBox([p3.gcc()])
                    display(self.VBox)
                    display(out)
                else:
                    # remove previous HBox play and slider from p3 current content children
                    p3.gcc().children = [x for x in p3.gcc().children if x != self.HBox]
                    ### remove last p3 plot objects S from p3 current content children
                    ##for s in last_S:
                    ##    p3.gcc().children = [x for x in p3.gcc().children if x != s]
                    out.clear_output()
                    # create updated play and slider widgets; put both in HBox
                    play = Play(min=0, max=n+1, interval=frame_step_msec, value=n, step=1)
                    slider = FloatSlider(min=0, max=play.max, step=1)
                    self.HBox = HBox([play, slider])
                # use ipywidgets jslink to link: play -> slider -> p3 plot objects S
                for s in S:
                    jslink((slider, 'value'), (s, 'sequence_index'))
                jslink((play, 'value'), (slider, 'value'))
                # add play and slider control to p3 current content children
                control = self.HBox
                p3.gcc().children += (control,)
                ##print(p3.gcc().children)
                ##p3.show()
                last_S = S
            ### ^^^

        # plot items on the same figure axes as in MATLAB with 'hold on'

        if not opt.anim_incr:
            # render all display list items over animation durations
            S = []
            for item in self.item_list:
                if item.type == "surface":
                    (x, y, z) = item.gentity
                    S.append(ipv.plot_surface(x, y, z, **item.args))
                elif item.type == "stl_mesh":
                    (x, y, z, f, c) = item.gentity
                    S.append(ipv.plot_trisurf(x, y, z, triangles=f, color=c))

            # assemble iteractive controlled animation
            p3.animation_control(S, sequence_length=nframes,
                                    interval=frame_step_msec)

            # initiate display list animation
            self.show()


### Implementation Note:
###
### Some components of this class still exhibit non-traditional coupling
### between graphics providers and clients due to preservation of existing
### RoboPy code base. This coupling can be mitigated or eliminated in some
### instances by utilizing callback mechanisms as done in most graphical
### rendering and user interface toolkits. A specific example would be the
### RTB fkine function that should be passed to animateSerialLink() method
### as a callback routine as one would pass animation update functions to
### an animator, or keypress and mouse event handlers to a GUI manager.

class Ipv3dVisual(GraphicsIPV):
    """
    ipyvolume (IPV) rendering visuals for RTB plotting and animation.
     
    This class acts as the interface between RTB plotting and animation
    routines and the RTB manipulator modeling and simulation code.
    Its methods must have access to the RTB manipulator model object
    methods and data structures.
    """
    def __init__(self, *args):
        
        if len(args) < 1 :  # crude method for handling type tests
            return None

        super(Ipv3dVisual, self).__init__(self.getFigure())

        self.setFigure(args[0])

        self.setFigureAxes(self.getFigure())
        
        super(Ipv3dVisual, self).__init__(self.getFigure())

        return None
    
    def getAxesLimits(self):
        return super(Ipv3dVisual, self).getAxesLimits()
    
    ### Class methods
    
    def draw_cube(self):
        super(Ipv3dVisual, self).draw_cube()
        
    def draw_sphere(self):
        super(Ipv3dVisual, self).draw_sphere()
    
    ### Plotting methods
    
    def view(self, *args, **kwargs):
        print("* Not yet implemented.")
        return None
    
    def pose_plot2(self, pose, **kwargs):
        print("* Not yet implemented.")
        return None
    
    def pose_plot3(self, pose, **kwargs):
        """
        Plot pose SO3 and SE3 transform coordinate frames.
        :param pose: a Pose object
        :param kwargs: plot properties keyword name/value pairs
        :return: None
        """
        opts = {'dispMode': self.getDispMode(),
                'z_up': True,
                'limits': self.getAxesLimits(),
                }

        opt = asSimpleNs(opts)

        (opt, args) = tb_parseopts(opt, **kwargs)

        pose_se3 = pose

        if type(pose) is type(Pose.SO3()):
            pose_se3 = pose.to_se3()

        T = []
        for each in pose_se3:
           T.append(each)

        super(Ipv3dVisual, self).plot_parametric_shape('frame', Tr=T, **args)

        self.show()

    def plot(self, obj, **kwargs):
        if type(obj) in [type(Pose.SO2()), type(Pose.SE2())]:
            self.pose_plot2(obj, **kwargs)
        elif type(obj) in [type(Pose.SO3()), type(Pose.SE3())]:
            self.pose_plot3(obj, **kwargs)
        elif type(obj) is type(DisplayList()):
            self.plotDisplayList(obj, **kwargs)
        else:
            pass

    def animate(self, obj, stances, **kwargs):
        if isinstance(obj, SerialLink.SerialLink):
            self.animateSerialLink(obj, stances, **kwargs)
        elif isinstance(obj, DisplayList):
            self.animateDisplayList(obj, stances, **kwargs)
        else:
            pass
        
    @staticmethod
    def fkine(obj, stances, unit='rad', apply_stance=False, mesh_list=None, timer=None):
        """
        Calculates forward kinematics for array of joint angles.
        :param stances: stances is a mxn array of joint angles.
        :param unit: unit of input angles (rad)
        :param apply_stance: If True, then applied to actor_list.
        :param mesh_list: list of meshes for given SerialLink object
        :param timer: used only (for animation).
        :return T: list of n+1 homogeneous transformation matrices.
        """
            
        T = obj.fkine(stances, unit=unit, timer=timer)
        
        if apply_stance and mesh_list is not None \
                        and len(mesh_list) >= len(T):
            for i in range(0,len(T)):
                mesh_list[i].apply_transform(T[i])
                
        return T
    
    def _setup_mesh_objs(self, obj):   
        """
        Internal function to initialise mesh objects.
        :return: mesh_list
        """
        ### Plotting STL meshes requires trimesh
        mesh_list = [0] * len(obj.stl_files)
        for i in range(len(obj.stl_files)):
            loc = pkg_resources.resource_filename("robopy", '/'.join(('media', obj.name, obj.stl_files[i])))
            a_mesh = trimesh.load_mesh(loc)
            mesh_list[i] = a_mesh

        loc = pkg_resources.resource_filename("robopy", "/media/stl/floor/white_tiles.stl")
        white_tiles = trimesh.load_mesh(loc)
        loc = pkg_resources.resource_filename("robopy", "/media/stl/floor/green_tiles.stl")
        green_tiles = trimesh.load_mesh(loc)
        
        return (mesh_list, white_tiles, green_tiles)
    
    ### Stub for pose rendered as stick figure.
    def _render_stick_pose(self, obj, stance, unit, **kwargs):
        """
        Renders given SerialLink object as stick figure desired in stance.
        :param obj: a SerialLink object.
        :param stance: list of joint angles for SerialLink object.
        :param unit: unit of input angles.
        :return: None.
        """ 
        pass
    
    ### Stub for pose rendered as notional body solids. 
    def _render_body_pose(self, obj, stance, unit, **kwargs):
        """
        Renders given SerialLink object as notional body solids in desired stance.
        :param obj: a SerialLink object.
        :param stance: list of joint angles for SerialLink object.
        :param unit: unit of input angles.
        :return: None.
        """ 
        pass
    
    ### Implementation Note:
    ###
    ### Rendering environment objects should be decoupled from rendering
    ### pose objects.
    
    def _render_body_floor(self, obj, limits):
        """ Render floor as paramatric plane surface
        """
        ### NOTE: cannot do hidden surface plots with parametric shapes
        
        # get floor's position
        position = np.asarray(obj.param.get("floor_position")).flatten()
        
        # plot floor plane
        s = math.fabs(limits[1] - limits[0])  # assumes square plane
        h = position[1]
        params = {'s':s, 'h':h, 'c':'lightgrey'}
        self.plot_parametric_shape('plane', solid=True, Tr=np.eye(4), **params)
        params = {'s':s, 'h':h, 'c':'black'}
        self.plot_parametric_shape('plane', solid=False, Tr=np.eye(4), **params)

    def _render_stl_floor(self, obj, white_tiles, green_tiles):
        """ Render floor as white and green STL mesh tiles.
        """
        # get floor's position
        position = np.asarray(obj.param.get("floor_position")).flatten()
        
        # render white floor tiles
        white_tiles.apply_transform(transforms.transl([0.0,0.0,position[1]]))
        self.disp_list.add('stl_mesh', 'white_tiles', white_tiles, color='white')

        # render green floor tiles
        green_tiles.apply_transform(transforms.transl([0.0,0.0,position[1]]))
        self.disp_list.add('stl_mesh', 'green_tiles', green_tiles, color='green')

    def _render_stl_pose(self, obj, stance, unit, limits=None):
        """
        Renders given SerialLink object defined as STL meshes in desired stance.
        :param obj: a SerialLink object.
        :param stance: list of joint angles for SerialLink object.
        :param unit: unit of input angles.
        :param limits; plot x, y, z limits
        :return: tuple = (limits, mesh_list)  # used for animation
        """                
        # load SerialLink mesh definition from STL files.
        (mesh_list, white_tiles, green_tiles) = self._setup_mesh_objs(obj)

        self.setMeshes(copy.deepcopy(mesh_list))
        
        # if necessary, apply plot axes limits
        if limits is None:
            # NOTE: adjust for VTK world coordinate system with Y-axis up
            xlims = np.asarray(obj.param.get("cube_axes_x_bounds")).flatten()
            ylims = np.asarray(obj.param.get("cube_axes_y_bounds")).flatten()
            zlims = np.asarray(obj.param.get("cube_axes_z_bounds")).flatten()
            limits = [zlims[0], zlims[1], xlims[0], xlims[1], ylims[0], ylims[1]]
            
        self.setAxesLimits(limits)

        # render floor
        self._render_stl_floor(obj, white_tiles, green_tiles)
            
        # apply stance to reference pose
        self.fkine(obj, stance, unit=unit, apply_stance=True, 
                        mesh_list=self.getMeshes())
        
        # render SerialLink object; save and return mesh_list
        for i in range(0, len(self.getMeshes())):
            a_mesh = self.getMeshI(i)
            a_mesh.apply_transform(transforms.trotx(90.0, unit="deg"))
            self.disp_list.add('stl_mesh', obj.stl_files[i], a_mesh, color=obj.colors[i])

        # preserve reference pose
        self.setMeshes(copy.deepcopy(mesh_list))
        
        return(limits, mesh_list)

    def qplot(self, obj, stance, unit='rad', dispMode='IPY', **kwargs):
        """
        Plots the SerialLink object in a desired stance.
        :param stance: list of joint angles for SerialLink object.
        :param unit: unit of input angles.
        :param: dispMode: display mode, one of ['VTK', 'IPY', 'PIL'].
        :return: None.
        """
        # parse argument list options
        opts = { 'unit'     : unit,
                 'dispMode' : dispMode,  # holdover from GraphicsVTK
                 'z_up'     : False,     # holdover from GraphicsVTK
                 'limits'   : self.getAxesLimits(),
               }
        
        opt = asSimpleNs(opts)
        
        (opt, args) = tb_parseopts(opt, **kwargs)
        
        # verify stance type
        assert type(stance) is np.matrix

        # check for stance angle unit conversion
        if opt.unit == 'deg':
            stance = stance * (np.pi / 180)
            opt.unit = 'rad'

        # create a display list

        self.disp_list = DisplayList()

        self._render_stl_pose(obj, stance, opt.unit, limits=opt.limits)

        self. plotDisplayList(self.disp_list, dispMode='IPY', **args)
    
    def trplot(self, tqr, **kwargs):
        """
        Plot transform coordinate frame.
        :param tqr: homogeneous transform matrix, quaternion or rotation matrix
        :param kwargs: plot properties keyword name/value pairs
        :return: None
        """
        opts = {'dispMode': self.getDispMode(),
                'z_up': True,
                'limits': self.getAxesLimits(),
                }

        opt = asSimpleNs(opts)

        (opt, args) = tb_parseopts(opt, **kwargs)

        T = rtbG.coerce_to_array_list(tqr)

        super(Ipv3dVisual, self).plot_parametric_shape('frame', Tr=T, **args)

        self.show()

    def trplot2(self, *args, **kwargs):
        print("* Not yet implemented.")
        return None
    
    def animateSerialLink(self, obj, stances, unit='rad', anim_incr=False, timer_rate=60, gif=None, frame_rate=25, **kwargs):
        """
        Animates SerialLink object over mx6 dimensional input matrix, with each row representing list of 6 joint angles.
        :param stances: mx6 dimensional input matrix.
        :param unit: unit of input angles. Allowed values: 'rad' or 'deg'
        :param anim_incr: set to True to render incrementally with animation_control
        :param timer_rate: simulation timer rate (steps per second)
        :param gif: name for the written animated GIF image file (not used yet)
        :param frame_rate: frame_rate for animation.
        :return: None
        """       
        # parse argument list options
        opts = { 'unit'       : unit,
                 'anim_incr'  : anim_incr,
                 'timer_rate' : timer_rate,
                 'gif'        : gif,                 # holdover from GraphicsVTK
                 'frame_rate' : frame_rate,
                 'dispMode'   : self.getDispMode(),  # holdover from GraphicsVTK
                 'z_up'       : False,               # holdover from GraphicsVTK
                 'limits'     : self.getAxesLimits(),
               }
        
        opt = asSimpleNs(opts)
        
        (opt, args) = tb_parseopts(opt, **kwargs)
        
        # verify stance type
        assert type(stances) is np.matrix
        
        # check for stance angle unit conversion
        if opt.unit == 'deg':
            stances = stances * (np.pi / 180)
            opt.unit = 'rad'

        # create a display list
        self.delMeshes()
        self.disp_list = DisplayList()

        # render initial pose
        (limits, mesh_list) = self._render_stl_pose(obj, stances[0,:], opt.unit, limits=opt.limits)

        # define pose transform function
        @animation_timer(opt.timer_rate, opt.frame_rate)
        def transFunc(n, tstep, self, obj, stances, unit):

            T = self.fkine(obj, stances, unit=unit, apply_stance=False, mesh_list=None, timer=n)

            return [np.asmatrix(np.identity(4)), np.asmatrix(np.identity(4))] + T

        tstep  = 1.0 / float(opt.timer_rate)
        funcArgs = [tstep, self, obj, stances, opt.unit]
        duration = stances.shape[0]

        # perform display list animation
        self.animateDisplayList(self.disp_list, transFunc, func_args=funcArgs,
                                anim_incr=opt.anim_incr, unit=opt.unit, duration=duration,
                                frame_rate=int(opt.frame_rate), limits=limits)

        return None
    
    def panimate(self, obj, stances, unit='rad', frame_rate=25, gif=None, **kwargs):
        print("* Not yet implemented.")
        return None
    
    def qanimate(self, obj, stances, unit='rad', frame_rate=25, gif=None, **kwargs):
        print("* Not yet implemented.")
        return None
    
    def tranimate(self, T, **kwargs):
        print("* Not yet implemented.")
        return None
    
    def tranimate2(self, R, **kwargs):
        print("* Not yet implemented.")
        return None