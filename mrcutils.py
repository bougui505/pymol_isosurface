#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-02-14 11:42:10 (UTC+0100)

import mrcfile
import numpy
import scipy.spatial.distance


def read_mrc(mrcfilename):
    """
    Read a mrc file and return the xyz and density values at the given level
    if given
    """
    xyz = []
    with mrcfile.open(mrcfilename) as emd:
        nx, ny, nz = emd.header['nx'], emd.header['ny'], emd.header['nz']
        x0, y0, z0 = emd.header['origin']['x'], emd.header['origin']['y'],\
            emd.header['origin']['z']
        dx, dy, dz = emd.voxel_size['x'], emd.voxel_size['y'], emd.voxel_size['z']
        xyz = numpy.meshgrid(numpy.arange(x0, x0 + nx * dx, dx),
                             numpy.arange(y0, y0 + ny * dy, dy),
                             numpy.arange(z0, z0 + nz * dz, dz),
                             indexing='ij')
        xyz = numpy.asarray(xyz)
        xyz = xyz.reshape(3, nx * ny * nz)
        xyz = xyz.T
        return xyz.reshape(nx, ny, nz, 3),
        emd.data.flatten(order='F').reshape(nx, ny, nz)




def first_to_last(arr):
    """
    Put first axis of an array to last
    """
    return numpy.squeeze(arr[..., None].swapaxes(0, -1))


def zone(mrc, coords, radius, mrcfilename):
    """
    Get a new mrc object with a zone selection around the given coordinates at
    a given radius
    """
    mrc_crop = crop(mrc=mrc, coords=coords, padding=radius,
                    mrcfilename='crop.mrc')
    ni, nj, nk, _ = mrc_crop.grid_coords.shape
    grid_coords = mrc_crop.grid_coords.reshape((-1, 3))
    # grid_indices = mrc_crop.grid_indices.reshape((-1, 3))
    kdtree = scipy.spatial.KDTree(grid_coords, leafsize=10000)
    selection = kdtree.query_ball_point(x=coords, r=radius)
    mask = numpy.zeros(ni * nj * nk)
    numpy.put(mask, ind=selection[0], v=1.)
    mask = mask.reshape((ni, nj, nk))
    grid_zone = mrc_crop.grid * mask
    save_density(density=grid_zone, outfilename=mrcfilename,
                 spacing=mrc_crop.step, origin=mrc_crop.origin)
    mrc_zone = MRC(mrcfilename)
    return mrc_zone


def crop(mrc, coords, padding, mrcfilename):
    """
    Get a new mrc object cropping around the given coordinates with the given
    padding
    """
    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)
    grid_coords = mrc.grid_coords.reshape((-1, 3))
    grid_indices = mrc.grid_indices.reshape((-1, 3))
    ind_min = numpy.linalg.norm(grid_coords - xyz_min, axis=1).argmin()
    ind_max = numpy.linalg.norm(grid_coords - xyz_max, axis=1).argmin()
    ind_min = grid_indices[ind_min]
    ind_max = grid_indices[ind_max]
    grid_crop = mrc.grid[ind_min[0]:ind_max[0],
                         ind_min[1]:ind_max[1],
                         ind_min[2]:ind_max[2]]
    origin_crop = mrc.grid_coords[tuple(ind_min)]
    save_density(density=grid_crop, outfilename=mrcfilename, spacing=mrc.step,
                 origin=origin_crop)
    mrc_crop = MRC(mrcfilename)
    return mrc_crop


class MRC(object):
    def __init__(self, mrcfilename):
        self.mrcfilename = mrcfilename
        self.read_mrc(self.mrcfilename)

    def read_mrc(self, mrcfilename):
        with mrcfile.open(mrcfilename) as emd:
            nx, ny, nz = emd.header['nx'], emd.header['ny'], emd.header['nz']
            # print('Reading density:')
            # print(tuple(numpy.int_([nx, ny, nz])))
            x0, y0, z0 = emd.header['origin']['x'], emd.header['origin']['y'],\
                emd.header['origin']['z']
            dx, dy, dz = emd.voxel_size['x'], emd.voxel_size['y'], emd.voxel_size['z']
            xyz = numpy.meshgrid(numpy.arange(x0, x0 + nx * dx, dx),
                                 numpy.arange(y0, y0 + ny * dy, dy),
                                 numpy.arange(z0, z0 + nz * dz, dz),
                                 indexing='ij')
            xyz = numpy.asarray(xyz)
            xyz = xyz.reshape(3, nx * ny * nz)
            xyz = xyz.T
            self.grid = emd.data.flatten(order='F').reshape(nx, ny, nz)
            print('Reading density:')
            print(self.grid.shape)
            self.grid_coords = xyz.reshape(nx, ny, nz, 3)
            self.grid_indices = numpy.indices(self.grid.shape)
            # First to last
            self.grid_indices = numpy.squeeze(self.grid_indices[..., None].swapaxes(0, -1))
            assert dy == dx and dz == dx
            self.step = dx
            self.origin = numpy.asarray([x0, y0, z0])

    def write_mrc(self, mrcfilename=None):
        """
        Save the density file as a mrc file
        """
        density = self.grid.astype('float32')
        if mrcfilename is not None:
            self.mrcfilename = mrcfilename
        with mrcfile.new(self.mrcfilename, overwrite=True) as mrc:
            nx, ny, nz = density.shape
            data = density.flatten().reshape((nx, ny, nz)).T
            mrc.set_data(data)
            mrc.voxel_size = self.step
            mrc.header['origin']['x'] = self.origin[0]
            mrc.header['origin']['y'] = self.origin[1]
            mrc.header['origin']['z'] = self.origin[2]
            mrc.update_header_from_data()
            mrc.update_header_stats()
            nx, ny, nz = mrc.header['nx'], mrc.header['ny'], mrc.header['nz']
            print('Writing density:')
            print(tuple(numpy.int_([nx, ny, nz])))
