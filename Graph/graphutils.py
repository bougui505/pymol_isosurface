#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-03-12 17:28:01 (UTC+0100)

import numpy
import scipy.sparse.csgraph


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


def get_degree(adjmat):
    adjmat_bin = adjmat.copy()
    adjmat_bin.data = numpy.ones_like(adjmat_bin.data)
    _, degrees = scipy.sparse.csgraph.laplacian(adjmat_bin, return_diag=True)
    return degrees


def clean_degree_2(adjmat, degrees):
    """
    Remove nodes of degree 2 preserving the topology
    """
