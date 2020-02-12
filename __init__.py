#!/usr/bin/env python
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-02-11 17:13:17 (UTC+0100)

from __future__ import absolute_import
from __future__ import print_function

# Avoid importing "expensive" modules here (e.g. scipy), since this code is
# executed on PyMOL's startup. Only import such modules inside functions.

import os


def __init_plugin__(app=None):
    '''
    Add an entry to the PyMOL "Plugin" menu
    '''
    from pymol.plugins import addmenuitemqt
    addmenuitemqt('Isosurface', run_plugin_gui)


# global reference to avoid garbage collection of our dialog
dialog = None


def run_plugin_gui():
    '''
    Open our custom dialog
    '''
    global dialog

    if dialog is None:
        dialog = make_dialog()

    dialog.show()


def make_dialog():
    # entry point to PyMOL's API
    from pymol import cmd

    # pymol.Qt provides the PyQt5 interface, but may support PyQt4
    # and/or PySide as well
    from pymol.Qt import QtWidgets
    from pymol.Qt.utils import loadUi
    from pymol.Qt.utils import getSaveFileNameWithExt

    # create a new Window
    dialog = QtWidgets.QDialog()

    # populate the Window from our *.ui file which was created with the Qt Designer
    uifile = os.path.join(os.path.dirname(__file__), 'gui.ui')
    form = loadUi(uifile, dialog)

    def load_isosurface(mrc):
        cmd.isosurface('isosurf', mrc, level=0.1)
        grid = cmd.get_volume_field(mrc)
        isomin = grid.min()
        isomax = grid.max()
        form.label_isomin.setText('%.4f' % isomin)
        form.label_isomax.setText('%.4f' % isomax)

    def set_isovalue(isovalue):
        pass

    objects_list = cmd.get_names('objects')
    objects_list = [e for e in objects_list
                    if cmd.get_type(e) == 'object:map']
    form.mapselector.addItems(objects_list)
    form.mapselector.activated.connect(lambda: load_isosurface(form.mapselector.currentText()))
    return dialog
