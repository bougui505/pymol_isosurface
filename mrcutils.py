#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-02-14 11:42:10 (UTC+0100)

import mrcfile
import numpy


def save_density(density, outfilename, spacing, origin):
    """
    Save the density file as mrc for the given atomname
    """
    density = density.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(density.T)
        mrc.voxel_size = spacing
        mrc.header['origin']['x'] = origin[0]
        mrc.header['origin']['y'] = origin[1]
        mrc.header['origin']['z'] = origin[2]
        mrc.update_header_from_data()
        mrc.update_header_stats()


def first_to_last(arr):
    """
    Put first axis of an array to last
    """
    return numpy.squeeze(arr[..., None].swapaxes(0, -1))


class MRC(object):
    def __init__(self, mrcfilename):
        self.mrc = mrcfile.open(mrcfilename)
        self.grid = self.mrc.data
        self.nx = self.mrc.header.nx
        self.ny = self.mrc.header.ny
        self.nz = self.mrc.header.nz
        self.dimx = self.mrc.header.cella['x']
        self.dimy = self.mrc.header.cella['y']
        self.dimz = self.mrc.header.cella['z']
        self.step = numpy.mean([self.dimx / self.nx,
                                self.dimy / self.ny,
                                self.dimz / self.nz])
        self.origin = self.mrc.header.origin
        self.origin = numpy.asarray([self.origin['x'],
                                     self.origin['y'],
                                     self.origin['z']])
        self.grid_indices = numpy.indices(self.grid.shape)
        self.grid_coords = (self.grid_indices * self.step) + self.origin[:, None, None, None]
        self.grid_indices = first_to_last(self.grid_indices)
        self.grid_coords = first_to_last(self.grid_coords)
