#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-03-12 17:28:01 (UTC+0100)

import numpy
import scipy.sparse.csgraph
from collections import defaultdict


def sparse_to_dict(adjmat, symmetrize=False):
    """
    Convert a sparse adjmat matrix to a dictionnary
    """
    graph_dict = defaultdict(dict)
    for r, c, d in zip(adjmat.row, adjmat.col, adjmat.data):
        graph_dict[r][c] = d
        if symmetrize:
            graph_dict[c][r] = d
    return graph_dict


def dict_to_sparse(graph_dict):
    """
    Convert the graph_dict object to a sparse coo_matrix
    """
    row, col, data = [], [], []
    for r in graph_dict:
        for c in graph_dict[r]:
            d = graph_dict[r][c]
            row.append(r)
            col.append(c)
            data.append(d)
    adjmat = scipy.sparse.coo_matrix((data, (row, col)))
    return adjmat


def symmetrize_adjmat(adjmat):
    graph_dict = sparse_to_dict(adjmat, symmetrize=True)
    return dict_to_sparse(graph_dict)


def mask_graph(adjmat, tokeep):
    """
    mask a graph
    - adjmat: coo sparse matrix
    - tokeep: list of node index to keep
    """
    row_tokeep = numpy.isin(adjmat.row, tokeep)
    col_tokeep = numpy.isin(adjmat.col, tokeep)
    tokeep = numpy.logical_or(row_tokeep, col_tokeep)
    adjmat.row = adjmat.row[tokeep]
    adjmat.col = adjmat.col[tokeep]
    adjmat.data = adjmat.data[tokeep]
    return adjmat


def get_degree(adjmat):
    adjmat_bin = adjmat.copy()
    adjmat_bin.data = numpy.ones_like(adjmat_bin.data)
    _, degrees = scipy.sparse.csgraph.laplacian(adjmat_bin, return_diag=True)
    return degrees


def clean_degree_2(adjmat):
    """
    Remove nodes of degree 2 preserving the topology
    """
    degrees = get_degree(adjmat)
    tokeep = numpy.where((degrees != 2))[0]
    mask_graph(adjmat, tokeep)
