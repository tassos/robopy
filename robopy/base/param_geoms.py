#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FILE: param_geoms.py
DATE: Tue Jan 27 21:05:00 2019

@author: garyd
"""

### Implementation Note:
###
### This module was designed to be used independent of the official
### RoboPy base modules.

"""
Module provides routines to generate and transform representative
geometric solid primitives defined by parametric shape functions.
"""
import functools
import numpy as np

__all__ = ('apply_defaults_opts',
           'parametric_frame',
           'parametric_box',
           'parametric_beam',
           'parametric_sphere',
           'parametric_cylinder',
           'parametric_cone',
           'parametric_disk',
           'parametric_plane',
           'param_xyz_coord_arrays',
           'param_xyz_coord_arrays_packed',
           'param_xyz_coord_arrays_xform',
           'param_xyz_coord_arrays_packed_xform',)


def _zero_dot():
    """
    Returns a zero point.
    :return: tuple of (x, y, z) 2D arrays of zero
    """
    x = np.zeros((1, 1))
    y = np.zeros((1, 1))
    z = np.zeros((1, 1))
    return (x, y, z)

# Wrapper for parametric shape functions to allow defaults and
# optional values to be passed to functions as its arguments.

def apply_defaults_opts(func, defaults, opts):
    """
    Apply defaults or opts values as arguments to given function.
    :param func: a function of the form f(*args)
    :param defaults: default argument keyword:value pairs dictionary
    :param opts: given optional argument keyword:value pairs dictionary
    :return: tuple of (x, y, z) square mesh arrays
    """
    if type(defaults) is dict and type(opts) is dict:
        arg_vals = []
        for key, val in defaults.items():
            if key in opts:
                val = opts[key]
            arg_vals.append(val)
        if not arg_vals:
            return func()
        else:
            return func(*arg_vals)
    else:
        print("*** Error: expected defaults and opts as dict for apply_default_opts.")
        return _zero_dot()


def coerce_xyz_to_Narrays(func):
    @functools.wraps(func)
    def coerce_wrapped_xyz_func(*args, **kwargs):
        """
        Coerces x, y, and z mesh grids into N arrays if Tr is a list.
        :param args: (x, y, z, Tr)
        :param kwargs:
        :return: func(xa, ya, za, tr)
        """
        if type(args[3]) is list:
            k = len(args[3])
            tr = args[3]
        else:
            k = 1
            tr = [args[3]]
        xa = args[0].reshape(k, args[0].shape[0]*args[0].shape[1])
        ya = args[1].reshape(k, args[1].shape[0]*args[1].shape[1])
        za = args[2].reshape(k, args[2].shape[0]*args[2].shape[1])
        xyz = func(xa, ya, za, tr)
        return xyz

    return coerce_wrapped_xyz_func


# Geometric object defined as parametric meshes.

### Implementation Note:
###
### See mesh_geoms and display_list modules for more efficient implementation
### of parametric shape functions utilizing the RoboPy base transforms module.


def parametric_frame(s):
    """
    Parametric Cartesian coordinate frame.
    :param s: size (length) of each coordinate frame +axis.
    :return: tuple of (x, y, z) square mesh arrays.
    """
    x = s * np.asmatrix([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
    y = s * np.asmatrix([[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    z = s * np.asmatrix([[0.0, 0.0, 0.0],[0.0, 0.0, 1.0],[0.0, 0.0, 0.0]])
    return (x, y, z)


def parametric_box(s):
    """
    Parametric box shape.
    :param s: size of each side (x, y and z dimensions).
    :return: tuple of (x, y, z) square mesh arrays.
    """
    r = s*np.sqrt(2.0)/2.
    h = s/2.
    u = np.linspace(0.25*np.pi, 2.25*np.pi, 5)
    v = np.linspace(-1.0, 1.0, 5)
    x = r * np.outer(np.cos(u), np.ones(np.size(v)))
    y = r * np.outer(np.sin(u), np.ones(np.size(v)))
    z = h * np.outer(np.ones(np.size(u)), v)
    return (x, y, z)


def parametric_beam(d, l):
    """
    Parametric beam shape.
    :param d: size in x and y dimensions.
    :param l: length (height) in z dimension.
    :return: tuple of (x, y, z) square mesh arrays.
    """
    r = d*np.sqrt(2.0)/2.
    h = l/2.
    u = np.linspace(0.25*np.pi, 2.25*np.pi, 5)
    v = np.linspace(-1.0, 1.0, 5)
    x = r * np.outer(np.cos(u), np.ones(np.size(v)))
    y = r * np.outer(np.sin(u), np.ones(np.size(v)))
    z = h * np.outer(np.ones(np.size(u)), v)
    return (x, y, z)


def parametric_sphere(d, dim):
    """
    Parametric sphere shape.
    :param s: diameter (x, y and z dimensions).
    :return: tuple of (x, y, z) square mesh arrays.
    """
    r = d/2.
    u = np.linspace(0.0, 2*np.pi, dim)
    v = np.linspace(0.0, np.pi, dim)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return (x, y, z)


def parametric_cylinder(d, l, dim):
    """
    Parametric cylinder shape.
    :param d: diameter in x and y dimensions.
    :param l: length (height) in z dimension.
    :return: tuple of (x, y, z) square mesh arrays.
    """
    r = d/2.
    h = l/2.
    u = np.linspace(0.0, 2*np.pi, dim)
    v = np.linspace(-1.0, 1.0, dim)
    x = r * np.outer(np.cos(u), np.ones(np.size(v)))
    y = r * np.outer(np.sin(u), np.ones(np.size(v)))
    z = h * np.outer(np.ones(np.size(u)), v)
    return (x, y, z)


def parametric_cone(d0, d1, l, dim):
    """
    Parametric cone shape
    :param d0: diameter in x and y dimensions at zmin.
    :param d1: diameter in x and y dimensions at zmax.
    :param l: length (height) in z dimension.
    :return: tuple of (x, y, z) square mesh arrays.
    """
    r0 = d0/2.
    r1 = d1/2.
    f  = (r1-r0)/2.
    h  = l/2
    u  = np.linspace(0.0, 2*np.pi, dim)
    v  = np.linspace(-1.0, 1.0, dim)
    s  = r0 + f*(v+1.0)
    x  = s * np.outer(np.cos(u), np.ones(np.size(v)))
    y  = s * np.outer(np.sin(u), np.ones(np.size(v)))
    z  = h * np.outer(np.ones(np.size(u)), v)
    return (x, y, z)


def parametric_disk(d, h, dim):
    """
    Parametric disk shape
    :param d: diameter in x and y dimensions.
    :param h: height of center in z dimension.
    :return: tuple of (x, y, z) square mesh arrays.
    """
    r = d/2.
    u = np.linspace(0.0, 2*np.pi, dim)
    v = np.linspace(0.0, 1.0, dim)
    x = r * np.outer(np.cos(u), v)
    y = r * np.outer(np.sin(u), v)
    z = h * np.outer(np.ones(np.size(u)), np.ones(np.size(v)))
    return (x, y, z)


def parametric_plane(s, h, dim):
    """
    Parametric plane shape
    :param s: size in x and y dimensions.
    :param h: height of center in z dimension.
    :return: tuple of (x, y, z) square mesh arrays.
    """
    r = s/2.
    u = np.linspace(-1.0, 1.0, dim)
    v = np.linspace(-1.0, 1.0, dim)
    x = r * np.outer(u, np.ones(np.size(v)))
    y = r * np.outer(np.ones(np.size(u)), v)
    z = h * np.outer(np.ones(np.size(u)), np.ones(np.size(v)))
    return (x, y, z)


def param_xyz_coord_arrays(shape, **opts):
    """
    Returns parametric xyx coordinate arrays for named shape.
    :param shape: shape name
    :return: (x, y, z) tuple where each are NxN arrays.
    """
    shapes_list = ('frame', 'box', 'beam', 'sphere',
                   'cylinder', 'cone', 'disk', 'plane',)

    ### Implementation Note:
    ###
    ### The dim values for frame, box and beam must not be changed. The dim
    ### values for other shapes are based on a value of 1 for rstride and
    ### cstride in Matplotlib ax.plot_surface() and ax.plot_wireframe()
    ### functions called in Mpl3dArtist.plot_parametric_shape() method.

    if shape == 'frame':
        dim = 3  # not used here, but implicit in parametric_frame
    elif shape in ['box', 'beam']:
        dim = 5  # not used here, but explicit in parametric_[box|beam]
    elif shape in ['sphere', 'cylinder', 'cone', 'disk', 'plane']:
        dim = 16
    else:
        msg = "shape must be in {0}".format(list(shapes_list))
        raise ValueError(msg)

    if shape == 'frame':
        defaults = {'s': 1.0}
        (x, y, z) = apply_defaults_opts(parametric_frame, defaults, opts)
    elif shape == 'box':
        defaults = {'s': 1.0}
        (x, y, z) = apply_defaults_opts(parametric_box, defaults, opts)
    elif shape == 'beam':
        defaults = {'d': 1.0, 'l': 1.0}
        (x, y, z) = apply_defaults_opts(parametric_beam, defaults, opts)
    elif shape == 'sphere':
        defaults = {'d': 1.0, 'dim': dim}
        (x, y, z) = apply_defaults_opts(parametric_sphere, defaults, opts)
    elif shape == 'cylinder':
        defaults = {'d': 1.0, 'l': 1.0, 'dim': dim}
        (x, y, z) = apply_defaults_opts(parametric_cylinder, defaults, opts)
    elif shape == 'cone':
        defaults = {'d0': 1.0, 'dl': 0.5, 'l': 1.0, 'dim': dim}
        (x, y, z) = apply_defaults_opts(parametric_cone, defaults, opts)
    elif shape == 'disk':
        defaults = {'d': 1.0, 'h': 0.0, 'dim': dim}
        (x, y, z) = apply_defaults_opts(parametric_disk, defaults, opts)
    elif shape == 'plane':
        defaults = {'s': 1.0, 'h': 0.0, 'dim': dim}
        (x, y, z) = apply_defaults_opts(parametric_plane, defaults, opts)
    else:
        # processing should never get here.
        msg = "shape must be in {0}".format(shapes_list)
        raise ValueError(msg)

    return (x, y, z)


def param_xyz_coord_arrays_packed(x, y, z, Tr=np.identity(4)):
    """
    Parametric shape coordinates transformation.
    :param x: dim0xdim1 array of x coordinate values
    :param y: dim0xdim1 array of y coordinate values
    :param z: dim0xdim1 array of z coordinate values
    :param Tr: homogeneous or rotation transformation matrix (np array type)
    :return: (xyz, dim0, dim1) tuple where xzy is a vtack of flattened dim0xdim1 arrays
    """
    dim0 = x.shape[0]
    dim1 = x.shape[1]
    dim = dim0*dim1
    if Tr.shape[0] == 4 and Tr.shape[1] == 4:
        xyz1 = np.vstack([x.reshape((1, dim)),
                          y.reshape((1, dim)),
                          z.reshape((1, dim)),
                          np.ones((1, dim))])
        return (xyz1, dim0, dim1)
    elif Tr.shape[0] == 3 and Tr.shape[1] == 3:
        xyz = np.vstack([x.reshape((1, dim)),
                         y.reshape((1, dim)),
                         z.reshape((1, dim))])
        return (xyz, dim0, dim1)
    else:
        msg = "unknown transform type {}".format(type(Tr))
        raise ValueError(msg)


def param_xyz_coord_arrays_xform(x, y, z, Tr):
    """
    Parametric shape coordinates transformation.
    :param x: dim0xdim1 array of x coordinate values
    :param y: dim0xdim1 array of y coordinate values
    :param z: dim0xdim1 array of z coordinate values
    :param Tr: homogeneous transformation matrix (np array type)
    :return: (xr, yr, zr) tuple where each are MxN arrays.
    """

    if Tr.shape in [(4,4)]:
        # pack shape homogeneous coordinates
        (xyz1, dim0, dim1) = param_xyz_coord_arrays_packed(x, y, z, Tr=Tr)
        # apply homogeneous matrix (array) transform
        Vtr = np.dot(Tr, xyz1)
    elif Tr.shape in [(3,3)]:
        # pack shape Cartesian coordinates
        (xyz, dim0, dim1) = param_xyz_coord_arrays_packed(x, y, z, Tr=Tr)
        # apply rotation matrix (array) transform
        Vtr = np.dot(Tr, xyz)
    else:
        msg = "Tr shape must be in {0}".format(list['4x4','3x3'])
        raise ValueError(msg)

    # unpack transformed shape coordinates
    xr = np.asarray(Vtr[0,:].reshape((dim0, dim1)))
    yr = np.asarray(Vtr[1,:].reshape((dim0, dim1)))
    zr = np.asarray(Vtr[2,:].reshape((dim0, dim1)))

    return (xr, yr, zr)


def param_xyz_coord_arrays_packed_xform(xyz, dim0, dim1, Tr):
    """
    Parametric shape coordinates transformation.
    :param xyz: 4 or 3 vstack arrays of dim0*dim1 flattened homogeneous or Cartesian coordinates
    :param dim0: x, y and z mesh grids shape 0 dimension
    :param dim1: x, y and z mesh grids shape 1 dimension
    :param Tr: homogeneous or rotation transformation matrix (np array type)
    :return: (xr, yr, zr) tuple where each are dim0xdim1 arrays.
    """
    if xyz.shape[0] == Tr.shape[1]:
        # apply matrix (array) transform
        Vtr = np.dot(Tr, xyz)
    else:
        msg = "xyz shape[0] must be sames a Tr.shape[1]"
        raise ValueError(msg)

    # unpack transformed shape coordinates
    xr = np.asarray(Vtr[0, :].reshape((dim0, dim1)))
    yr = np.asarray(Vtr[1, :].reshape((dim0, dim1)))
    zr = np.asarray(Vtr[2, :].reshape((dim0, dim1)))

    return (xr, yr, zr)

