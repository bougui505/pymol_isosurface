#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-02-14 11:42:10 (UTC+0100)

import collections
import mrcfile
import numpy
import scipy.spatial.distance
import scipy.sparse.csgraph
import sklearn.feature_extraction


def mask_graph(adjmat, mask):
    """
    mask a graph
    - adjmat: coo sparse matrix
    - mask: list of node index to keep
    """
    row_mask = numpy.isin(adjmat.row, mask)
    col_mask = numpy.isin(adjmat.col, mask)
    mask = numpy.logical_or(row_mask, col_mask)
    adjmat.row = adjmat.row[mask]
    adjmat.col = adjmat.col[mask]
    adjmat.data = adjmat.data[mask]
    return adjmat


class MRC(object):
    def __init__(self, mrcfilename):
        self.mrcfilename = mrcfilename
        self.read_mrc(self.mrcfilename)

    def read_mrc(self, mrcfilename=None):
        if mrcfilename is not None:
            self.mrcfilename = mrcfilename
        with mrcfile.open(self.mrcfilename) as emd:
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

    def crop(self, coords, padding, mrcfilename=None):
        """
        Get a new mrc object cropping around the given coordinates with the given
        padding
        """
        xyz_min = coords.min(axis=0)
        xyz_max = coords.max(axis=0)
        grid_coords = self.grid_coords.reshape((-1, 3))
        grid_indices = self.grid_indices.reshape((-1, 3))
        ind_min = numpy.linalg.norm(grid_coords - xyz_min, axis=1).argmin()
        ind_max = numpy.linalg.norm(grid_coords - xyz_max, axis=1).argmin()
        ind_min = grid_indices[ind_min]
        ind_max = grid_indices[ind_max]
        self.grid = self.grid[ind_min[0]:ind_max[0],
                              ind_min[1]:ind_max[1],
                              ind_min[2]:ind_max[2]]
        self.origin = self.grid_coords[tuple(ind_min)]
        self.write_mrc(mrcfilename=mrcfilename)
        self.read_mrc(mrcfilename=mrcfilename)

    def zone(self, coords, radius, mrcfilename=None):
        """
        Get a new mrc object with a zone selection around the given coordinates at
        a given radius
        """
        self.crop(coords, padding=radius, mrcfilename=mrcfilename)
        ni, nj, nk, _ = self.grid_coords.shape
        grid_coords = self.grid_coords.reshape((-1, 3))
        # grid_indices = mrc_crop.grid_indices.reshape((-1, 3))
        kdtree = scipy.spatial.KDTree(grid_coords, leafsize=10000)
        selections = kdtree.query_ball_point(x=coords, r=radius)
        selection = []
        for sel in selections:
            selection.extend(sel)
        selection = numpy.unique(selection)
        del selections
        mask = numpy.zeros(ni * nj * nk)
        numpy.put(mask, ind=selection, v=1.)
        mask = mask.reshape((ni, nj, nk))
        self.grid = self.grid * mask
        self.write_mrc(mrcfilename=mrcfilename)
        self.read_mrc(mrcfilename=mrcfilename)

    def index_to_coords(self, index):
        """
        Convert a list of grid indices to coordinates
        """
        index = [numpy.ravel_multi_index(ind, dims=self.grid.shape) for ind in index]
        coords = self.grid_coords.reshape((-1, 3))
        coords = numpy.asarray([coords[ind] for ind in index])
        return coords

    def minimum_spanning_tree(self, isolevel):
        """
        Compute the minimum spanning tree of the density map
        - isolevel: Isolevel to mask voxels
        """
        MStree = collections.namedtuple('MStree', ['adjmat', 'ind', 'coords',
                                                   'degrees'])
        adjmat = sklearn.feature_extraction.image.img_to_graph(self.grid)
        mask = self.grid > 0.099
        mask = numpy.where(mask.flatten())[0]
        adjmat = mask_graph(adjmat, mask)
        mstree = scipy.sparse.csgraph.minimum_spanning_tree(adjmat)
        mstree = mstree.tocoo()
        ind_unique = numpy.unique(numpy.concatenate((mstree.row, mstree.col)))
        ind = numpy.asarray([numpy.unravel_index(i, self.grid.shape)
                             for i in ind_unique])
        mstree_coords = self.index_to_coords(ind)
        del ind
# ---------------------- Compute the degree of the nodes -----------------------
        mstree_bin = mstree.copy()
        mstree_bin.data = numpy.ones_like(mstree_bin.data)
        _, degrees = scipy.sparse.csgraph.laplacian(mstree_bin, return_diag=True)
        del mstree_bin
        degrees_unique = degrees.take(ind_unique)
        del degrees
# ----------------- Remove coordinates and index with degree 0 -----------------
        sel = degrees_unique > 0
        mstree_coords = mstree_coords[sel]
        ind_unique = ind_unique[sel]
        degrees_unique = degrees_unique[sel]
# ------------------------------------- - --------------------------------------
        self.mstree = MStree(adjmat=mstree, ind=ind_unique,
                             coords=mstree_coords, degrees=degrees_unique)
        return self.mstree
