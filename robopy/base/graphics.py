#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: graphics.py
DATE: Tue Jan  8 17:34:00 2019

@author: garyd
"""

import sys
import functools
import time
from math import floor
from abc import ABCMeta, abstractmethod

__all__ = ('Graphics', 'Gtransform', 'GraphicsRenderer', 'animation_timer',
           'rgb_named_colors', 'coerce_to_array_list',
           'plot', 'qplot', 'trplot', 'trplot2',
           'animate', 'panimate', 'qanimate', 'tranimate', 'tranimate2')


###
### MODULE CLASS DEFINITIONS
###

### Implementation Note:
###
### The animation_timer class timer_wrapped_func() is a blocking routine.
### As currently coded, it emulates quasi-realtime animation with images
### captured at a frame rate less than the animation timer rate. If the
### animation rendering and image capture cannot be performed within a
### frame time step, animated images will not be produced at the desired
### frame rate.

class animation_timer(object):
    """
    Animation timer class to provide timed wrapper for transform functions.

    This wrapper class may be used to wrap transformation functions passed
    to the animateDisplayList() method of the GraphicsMPL and GraphicsIPV
    classes, and to wrap the transformation functions defined in the
    animateSerialLink() method of the Mpl3dArtist and Ipv3dVisual classes.
    """

    def __init__(self, timer_rate, frame_rate, real_time=False):

        assert(timer_rate > 0)
        assert(frame_rate > 0)
        assert(timer_rate >= frame_rate)

        # instantiation arguments
        self.timer_rate = timer_rate  # cycles per second
        self.frame_rate = frame_rate  # frames per second
        self.real_time  = real_time   # real-time emulation flag

        # internal static properties
        self.timer_step_sec  = 1.0 / self.timer_rate
        self.timer_step_msec = 1000.0 / self.timer_rate
        self.frame_step_msec = 1000.0 / self.frame_rate

        # internal dynamic properties
        self.last_time   = 0.0    # time at last call to time.time()
        self.timer_count = 0      # number of timer cycles
        self.frame_count = 0      # number of frames captured
        self.timer_elap_msec = 0  # timer elapsed time in msec
        self.frame_save_msec = 0  # frame capture time in msec

    def __call__(self, func):
        @functools.wraps(func)
        def timer_wrapped_func(*args, **kwargs):
            """
            Wrapped transform function must have as its first two args,
            n for step number and tstep for step time (sec or None).
            :param args:    n, [tstep|None], ...
            :param kwargs:  a dictionary of keyword name/value pairs
            :return: that which is returned by the transform function.
            """
            # ensure there are at least two argument
            assert(len(args) >= 2)
            # check for tstep value in wrapped function args list
            if args[1] is not None:
                tstep = args[1]
                if type(tstep) is [int, str]:
                    tstep = float(tstep)
                if type(tstep) is float:
                    if tstep > 0.0:
                       if tstep != self.timer_step_sec:
                           # TODO: adjust current timer and frame counts?
                           pass
                       self.timer_step_sec = tstep
                       self.timer_step_msec = 1000.0 * self.timer_step_sec
            # if real-time, call wrapped function after specified delay
            if self.real_time:
                t_wait = self.timer_step_sec - (time.time() - self.last_time)
                if (t_wait > 0.0):
                    time.sleep(t_wait)
            value = func(self.timer_count, self.timer_step_sec, *args[2:], **kwargs)
            # increment timer step counter
            self.timer_count += 1
            # update animation elapsed time (msec)
            self.timer_elap_msec = self.timer_count*self.timer_step_msec
            # check if sufficient time elapsed since last image capture event
            if (self.timer_elap_msec - self.frame_save_msec) >= self.frame_step_msec:
                 # TODO: capture image for recording media
                self.frame_count += 1
                self.frame_save_msec = self.timer_elap_msec
            self.last_time = time.time()
            return value

        return timer_wrapped_func


class Graphics(metaclass=ABCMeta):
    """ 
    Graphics interface for RTB.
    
    This Abstract Base Class (ABC) presents the graphical interface 
    between Robotics Toolbox (RTB) and graphics package(s) which 
    provide needed plotting, animation and rendering functionality.
    
    Implementation Notes:
        
      1) Graphics modules must provide, as a minimum, methods 
         denoted in this class.
         
      2) This class object should act as a factory for specific
         graphics rendering, animation and transform objects.
         
      3) Must be able to keep track of multiple rendering windows
         and in the case of Matplotlib, not block on show() and
         permit multiple poses rendered to a given figure.
    
    """
    def __init__(self):
        self._gRenderer = None
        self._gTransform = self._Gtransform()
    
    def _Gtransform(self):
        """ Instantiates and returns a graphics transform.
        """
        gxobj = Gtransform()
        return gxobj
    
    ### Instance property setters/getters
    
    """    
      _gRenderer = property(fset=setGraphicsRenderer, fget=getGraphicsRenderer)
      _gTransform = property(fset=None, fget=getGtransform)
    """
    
    def setGraphicsRenderer(self, gRenderer):
        self._gRenderer = gRenderer
        
    def getGraphicsRenderer(self):
        return self._gRenderer
        
    def getGtransform(self):
        return self._gTransform
    
    ### Graphics package interface methods (presented in alphabetical order)
           
    @abstractmethod
    def draw_axes2(self, *args, **kwargs):
        """ Graphics package draw plot axes for 2D space.
        """ 
        raise NotImplementedError('Need to define draw_axes2 emethod.')
        
    @abstractmethod
    def draw_axes3(self, *args, **kwargs):
        """ Graphics package draw plot axes for 3D space.
        """ 
        raise NotImplementedError('Need to define draw_axes3 emethod.')
        
    @abstractmethod
    def draw_cube(self):
        """ Graphics package draw a blue cube method.
        """
        raise NotImplementedError('Need to define draw_cube emethod.')
        
    @abstractmethod
    def draw_sphere(self):
        """ Graphics package draw a red sphere method.
        """
        raise NotImplementedError('Need to define draw_sphere emethod.')
        
    @abstractmethod  # forces definition of a graphics module scope routine
    def rgb_named_colors(cls, *args, **kwarg):
        """ Graphics package returns RGB values for named colors.
        """ 
        raise NotImplementedError('Need to define rgb_named_colors method.')
        
    @abstractmethod
    def setGtransform(self, *args, **kwargs):
        """ Graphics package set graphics transform method.
        """
        raise NotImplementedError('Need to define setGtransform emethod.')
        
    @abstractmethod
    def view(self, *args, **kwargs):
        """ Graphics package set view space method.
        """
        raise NotImplementedError('Need to define view emethod.')
        
    ## RTB interface methods (presented in alphabetical order)
    
    @abstractmethod
    def animate(self, *args, **kwargs):
        """ RTB interface method.
        """
        raise NotImplementedError('Need to define animate method.')
        
    @abstractmethod
    def plot(self, *args, **kwargs):
        """ RTB interface method.
        """
        raise NotImplementedError('Need to define plot method.')
        
    @abstractmethod
    def qplot(self, *args, **kwargs):
        """ RTB interface method. 
        """ 
        raise NotImplementedError('Need to define qplot method.')

    @abstractmethod
    def render(self, *args, **kwargs):
        """ VTk interface method
        """
        raise NotImplementedError('Need to define render method.')
    
    @abstractmethod
    def show(self, *args, **kwargs):
        """ Matplotlib interface method
        """
        raise NotImplementedError('Need to define show method.')
        
    @abstractmethod
    def tranimate(self, *args, **kwargs):
        """ RTB interface method.
        """
        raise NotImplementedError('Need to define tranimate method.')
        
    @abstractmethod
    def tranimate2(self, *args, **kwargs):
        """ RTB interface method.
        """
        raise NotImplementedError('Need to define tranimate2 method.')
        
    @abstractmethod
    def trplot(self, *args, **kwargs):
        """ RTB interface method.
        """
        raise NotImplementedError('Need to define trplot method.')
        
    @abstractmethod        
    def trplot2(self, *args, **kwargs):
        """ RTB interface method.
        """ 
        raise NotImplementedError('Need to define trplot2 method.')

    # Display List Interface - These methods must be defined in Graphics Rendering Classes

    @abstractmethod
    def renderDisplayListItem(self, *args, **kwargs):
        """ Graphics package renderDisplayListItem
        """
        raise NotImplementedError('Need to define renderDisplayListItem method.')

    @abstractmethod
    def renderDisplayList(self, *args, **kwargs):
        """ Graphics package renderDisplayList
        """
        raise NotImplementedError('Need to define renderDisplayList method.')

    @abstractmethod
    def plotDisplayList(self, *args, **kwargs):
        """ Graphics package plotDisplayList
        """
        raise NotImplementedError('Need to define plotDisplayList method.')

    @abstractmethod
    def animateDisplayList(self, *args, **kwargs):
        """ Graphics package animateDisplayList
        """
        raise NotImplementedError('Need to define animateDisplayList method.')


class Gtransform(Graphics):
    """
    Graphics transform for RTB.
    
    This Abstract Base Class (ABC) presents the graphical interface 
    between Robotics Toolbox (RTB) and graphics package(s) which 
    provide needed graphics transform functionality.
    
    Graphics modules must provide, as a minimum, methods denoted 
    in this class
    """
    def __init__(self):
        super(Graphics,self).__init__()
    
    def setGtransform(self, *args, **kwargs):
        """ Set graphics transform.
        """
        raise NotImplementedError('Need to define setGtransform emethod.')
        
###
### MODULE PUBLIC INTERFACE ROUTINES
###

""" These routines comprise the external interface to RTB plotting and 
    animation functions. They preclude the need to instantiate graphics 
    objects in the RTB manipulator and math modules such as pose, 
    serial_link, quaternion and transforms.

    Implementation Notes:
        
      1) These functions are generally not invoked by the user, but 
         by the corresponding calling function in RTB modules which 
         are exposed to the user.

      2) The header documentation for called functions herein should
         match that of corresponding calling functions in RTB modules. 
         See graphics_vtk.VtkPipeline.animate(), animate() below and 
         serial_link.SerialLink.animate() as a pertinent example.
         
      3) The function argument list keyword/value validity checking
         of RTB specific parameters should be done in the RTB modules
         before these functions are called. While graphing, rendering
         and plotting parameters validity should be checked in this
         graphics module, or dedicated submodules, with exceptions
         returned to RTB for resolution. It is possible that all data
         unit conversion could be done in RTB before the values are
         passed here (i.e., use only 'rad' herein for numerical 
         computation, but allow for user's desire for data values to
         be graphically displayed in 'deg' units).
         
      4) Though there may be historical and mathematical precedence
         involved in the preservation of RTB for MATLAB matrix data
         structures, there is no meaningful reason to force the NumPy
         matrix type class on graphics processing. The more appropriate
         data type class commonly used is the NumPy ndarray. Note the 
         effort to convert x, y & z bounds from matrices to arrays which 
         can then be accessed using just one index as in VtkPipeline
         qplot() and animate() methods in the graphics_vtk module.
"""

from . import tb_parseopts as tbpo
from . import graphics_vtk as gVtk
from . import graphics_mpl as gMpl
from . import graphics_ipv as gIpv
from . import common
from . import transforms
from . import quaternion

import numpy as np

gRenderer = None  # the instantiated graphics rendering object

def GraphicsRenderer(renderer):
    """ 
    Instantiates and returns a graphics renderer.
    
    :param renderer: renderer descriptor string
    :return gRenderer: graphics rendering object      
    """
    global gRenderer
    
    if renderer == 'VTK':
        gRenderer = gVtk.VtkPipeline()
    elif renderer == 'MPL':
        gRenderer = gMpl.Mpl3dArtist(0)  # default figure number is 0
    elif renderer == 'IPV':
        gRenderer = gIpv.Ipv3dVisual(0)  # default figure key is 0
    else:
        print("The %s renderer is not supported." % renderer)
        print("Renderer must be VTK, MPL (Matplotlib) or IPV (ipyvolume).")
        sys.exit()
    return gRenderer
    
def rgb_named_colors(colors):
    global gRenderer
    if type(gRenderer) is type(gVtk.VtkPipeline()):
        return gVtk.rgb_named_colors(colors)
    elif type(gRenderer) is type(gMpl.Mpl3dArtist()):
        return gMpl.rgb_named_colors(colors)
    elif type(gRenderer) is type(gIpv.Ipv3dVisual()):
        return gIpv.rgb_named_colors(colors)

def coerce_to_array_list(tqr):
    if type(tqr) in (list, tuple, transforms.RTBMatrix):
        if isinstance(tqr[0], transforms.RTBMatrix):
            T = tqr.to_ndarray()
        elif isinstance(tqr[0], quaternion.Quaternion):
            T = np.asarray(tqr.tr())
        elif common.isrot(tqr[0]):
            T = np.asarray(transforms.r2t(tqr))
        elif common.ishomog(tqr[0], (4, 4)):
            T = np.asarray(tqr)
        else:
            msg = "list of unknown transforms of type {0}".format(type(tqr[0]))
            ValueError(msg)
    else:
        if isinstance(tqr, quaternion.Quaternion):
            T = [np.asarray(tqr.tr())]
        elif common.isrot(tqr):
            T = [np.asarray(transforms.r2t(tqr))]
        elif common.ishomog(tqr, (4, 4)):
            T = [np.asarray(tqr)]
        else:
            msg = "unknown transform of type {0}".format(type(tqr))
            ValueError(msg)
    return T

def plot(obj, **kwargs):
    """
    Displays a pose plot for the given Pose object.
    :param obj: a Pose object
    :param kwargs: graphics renderer and plot properties keyword name/value pairs
    :return:
    """
    global gRenderer
    if type(gRenderer) is type(gVtk.VtkPipeline()):
        opts = { 'dispMode' : 'VTK',
               }
        opt = tbpo.asSimpleNs(opts)
        (opt, args) = tbpo.tb_parseopts(opt, **kwargs)
        pobj = gVtk.VtkPipeline(dispMode=opt.dispMode)
        pobj.plot(obj, **args)
    elif type(gRenderer) is type(gMpl.Mpl3dArtist()):
        opts = { 'dispMode' : 'IPY',
                 'fign'     : 1,
               }
        opt = tbpo.asSimpleNs(opts)
        (opt, args) = tbpo.tb_parseopts(opt, **kwargs)
        pobj = gMpl.Mpl3dArtist(opt.fign)
        pobj.plot(obj, **args)
    elif type(gRenderer) is type(gIpv.Ipv3dVisual()):
        opts = { 'dispMode' : 'IPY',
                 'key'      : 1,
               }
        opt = tbpo.asSimpleNs(opts)
        (opt, args) = tbpo.tb_parseopts(opt, **kwargs)
        pobj = gIpv.Ipv3dVisual(opt.key)
        pobj.plot(obj, **args)

def qplot(obj, stance, unit='rad', dispMode='VTK', **kwargs):
    """
    Displays a SerialLink pose plot for the given SerialLink object.
    :param obj: a SerialLink object
    :param stance; a stance data structure
    :param unit: 'rad' or 'deg'
    :param dispMode: for VtkPipeline ['VTK', 'IPY', 'PIL'], for others just 'IPY'
    :param kwargs: graphics renderer and plot properties keyword name/value pairs
    :return:
    """
    global gRenderer
    if type(gRenderer) is type(gVtk.VtkPipeline()):
        opts = { 'unit'     : unit,
                 'dispMode' : 'VTK',
               }
        opt = tbpo.asSimpleNs(opts)
        (opt, args) = tbpo.tb_parseopts(opt, **kwargs)
        gobj = gVtk.VtkPipeline(dispMode=dispMode)
        gobj.qplot(obj, stance, unit='rad', dispMode=dispMode, **kwargs)
    elif type(gRenderer) is type(gMpl.Mpl3dArtist()):
        opts = { 'unit'     : unit,
                 'dispMode' : 'IPY',
                 'fign'     : 1,
               }
        opt = tbpo.asSimpleNs(opts)
        (opt, args) = tbpo.tb_parseopts(opt, **kwargs)
        gobj = gMpl.Mpl3dArtist(opt.fign)
        gobj.qplot(obj, stance, unit=opt.unit, **args)
    elif type(gRenderer) is type(gIpv.Ipv3dVisual()):
        opts = { 'unit'     : unit,
                 'dispMode' : 'IPY',
                 'key'      : 1,
               }
        opt = tbpo.asSimpleNs(opts)
        (opt, args) = tbpo.tb_parseopts(opt, **kwargs)
        gobj = gIpv.Ipv3dVisual(opt.key)
        gobj.qplot(obj, stance, unit=opt.unit, **args)
    
def tranimate(T):
    global gRenderer
    pass 

def tranimate2(R):
    global gRenderer
    pass
    
def trplot(T, handle=None, dispMode='VTK', **kwargs):
    """
    Plots a Cartesian coordinate axes frame for the given transform T.
    :param T: a RTBMatrix, quaternion, homogeneous transform matrix or rotation matrix
    :param handle; a Gtransform handle (tentatively)
    :param dispMode: for VtkPipeline ['VTK', 'IPY', 'PIL'], for others just 'IPY'
    :param kwargs: graphics renderer and plot properties keyword name/value pairs
    :return:
    """
    global gRenderer
    if type(gRenderer) is type(gVtk.VtkPipeline()):
        if handle is not None:
            if type(handle) is type(gVtk.VtkPipeline()):
                handle.trplot(T)
            elif type(handle) is type(super(gVtk).Hgtransform()):
                pass  # do Hgtransform stuff
            else:
                pass  # do error stuff
        opts = { 'dispMode' : 'VTK',
               }
        opt = tbpo.asSimpleNs(opts)
        (opt, args) = tbpo.tb_parseopts(opt, **kwargs)
        gobj = gVtk.VtkPipeline(dispMode=dispMode)
        gobj.trplot(T, unit='rad', dispMode=opt.dispMode, **args)
    elif type(gRenderer) is type(gMpl.Mpl3dArtist()):
        opts = { 'dispMode' : 'IPY',
                 'fign'     : 1,
               }
        opt = tbpo.asSimpleNs(opts)
        (opt, args) = tbpo.tb_parseopts(opt, **kwargs)
        gobj = gMpl.Mpl3dArtist(opt.fign)
        gobj.trplot(T, unit='rad', dispMode=opt.dispMode, **args)
    elif type(gRenderer) is type(gIpv.Ipv3dVisual()):
        opts = { 'dispMode' : 'IPY',
                 'key'      : 1,
               }
        opt = tbpo.asSimpleNs(opts)
        (opt, args) = tbpo.tb_parseopts(opt, **kwargs)
        gobj = gIpv.Ipv3dVisual(opt.key)
        gobj.trplot(T, unit='rad', dispMode=opt.dispMode, **args)

def trplot2(T, handle=None, dispMode='VTk'):
    global gRenderer
    if handle is not None:
        if type(handle) is type(gVtk.VtkPipeline()):
            handle.trplot2(T)
        elif type(handle) is type(super(gVtk).Hgtransform()):
            pass  # do Hgtransform stuff
        else:
            pass  # do error stuff
    gobj = gVtk.VtkPipeline(dispMode=dispMode)
    gobj.trplot2(T)

### Implementation Note:
###
### To much detail about a SerialLink or DisplayList has been brought into
### this interface. The Graphics class should just call the appropriate
### Graphics Renderer animate() method and let that class determine how to
### handle the details. See how plot() is handled in Mpl3dArtist.

def animate(obj, stances, 
                 unit='rad', timer_rate=60, gif=None, frame_rate=30, 
                 dispMode='VTK', **kwargs):
    """
    Animates SerialLink object over nx6 dimensional input matrix, with each row representing list of 6 joint angles.
    :param obj: a SerialLink object.
    :param stances: nx6 dimensional input matrix.
    :param unit: unit of input angles. Allowed values: 'rad' or 'deg'
    :param timer_rate: time_rate for motion. Could be any integer more than 1. Higher value runs through stances faster.
    :param gif: name for the written animated GIF image file.
    :param frame_rate: frame_rate for animation.
    :dispMode: display mode; one of ['VTK', 'IPY', 'PIL'].
    :return: gobj - graphics object
    """
    global gRenderer
    
    opts = { 'unit'       : unit,
             'fign'       : 1,
             'key'        : 1,
             'timer_rate' : timer_rate,
             'gif'        : gif,
             'frame_rate' : frame_rate,
             'dispMode'   : dispMode,
           }
    
    opt = tbpo.asSimpleNs(opts)
    
    (opt, args) = tbpo.tb_parseopts(opt, **kwargs)
    
    if type(gRenderer) is type(gVtk.VtkPipeline()):
        gobj = gVtk.VtkPipeline(dispMode=opt.dispMode,
                                total_time_steps= stances.shape[0] - 1,
                                timer_rate=opt.timer_rate,
                                gif_file=opt.gif,
                                frame_rate=opt.frame_rate)
        gobj.animate(obj, stances, 
                          unit=opt.unit, frame_rate=opt.frame_rate, gif=opt.gif, 
                          dispMode=opt.dispMode, **args)
        return gobj
    elif type(gRenderer) is type(gMpl.Mpl3dArtist()):
        gobj = gMpl.Mpl3dArtist(opt.fign)
        gobj.animate(obj, stances, 
                          unit=opt.unit, timer_rate=opt.timer_rate, frame_rate=opt.frame_rate, gif=opt.gif,
                          dispMode='IPY', **args)
        return gobj
    elif type(gRenderer) is type(gIpv.Ipv3dVisual()):
        gobj = gIpv.Ipv3dVisual(opt.key)
        gobj.animate(obj, stances,
                          unit=opt.unit, timer_rate=opt.timer_rate, frame_rate=opt.frame_rate, gif=opt.gif,
                          dispMode='IPY', **args)
        return gobj
    return None

def panimate(pose, other=None, duration=5, timer_rate=60, 
                   gif=None, frame_rate=10, **kwargs):
    """
    Animates pose object over nx6 dimensional input matrix, with each row representing list of 6 joint angles.
    :param pose: a Pose object.
    :param other: other Pose object to transition towards.
    :param timer_rate: time_rate for motion. Could be any integer more than 1. Higher value runs through stances faster.
    :param gif: name for the written animated GIF image file.
    :param frame_rate: frame_rate for animation.
    :dispMode: display mode; one of ['VTK', 'IPY', 'PIL'].
    :return: gobj - graphics object
    """
    global gRenderer
    
    opts = { 'other'      : other,
             'duration'   : duration,
             'fign'       : 1,
             'key'        : 1,
             'timer_rate' : timer_rate,
             'gif'        : gif,
             'frame_rate' : frame_rate,
             'dispMode'   : 'VTK',
           }
    
    opt = tbpo.asSimpleNs(opts)
    
    (opt, args) = tbpo.tb_parseopts(opt, **kwargs)
    
    if type(gRenderer) is type(gVtk.VtkPipeline()):
        gobj = gVtk.VtkPipeline(dispMode=opt.dispMode,
                  total_time_steps=opt.duration*opt.timer_rate,
                  timer_rate = opt.timer_rate,
                  gif_file=opt.gif, 
                  frame_rate=opt.frame_rate)
        gobj.panimate(pose, other=opt.other, duration=opt.duration, **args)
        return gobj
    elif type(gRenderer) is type(gMpl.Mpl3dArtist()):
        gobj = gMpl.Mpl3dArtist(opt.fign)
        gobj.panimate(pose, other=opt.other, duration=opt.duration, **args)
        return gobj
    elif type(gRenderer) is type(gIpv.Ipv3dVisual()):
        gobj = gIpv.Ipv3dVisual(opt.key)
        gobj.panimate(pose, other=opt.other, duration=opt.duration, **args)
        return gobj
    return None

def qanimate(obj, stances, unit='rad', dispMode='VTK', frame_rate=25, gif=None, **kwargs):
    global gRenderer
    gobj = gVtk.VtkPipeline(dispMode=dispMode)
    gobj.qanimate(obj, stances, unit=unit, frame_rate=frame_rate, gif=gif, **kwargs)
    
def tranimate(obj, stances, unit='rad', dispMode='VTK', frame_rate=25, gif=None, **anim_params):
    global gRenderer
    gobj = gVtk.VtkPipeline(dispMode=dispMode)
    gobj.tranimate(obj, stances, unit=unit, frame_rate=frame_rate, gif=gif, **anim_params)
    
def tranimate2(obj, stances, unit='rad', dispMode='VTK', frame_rate=25, gif=None, **anim_params):
    global gRenderer
    gobj = gVtk.VtkPipeline(dispMode=dispMode)
    gobj.tranimate2(obj, stances, unit=unit, frame_rate=frame_rate, gif=gif, **anim_params)
