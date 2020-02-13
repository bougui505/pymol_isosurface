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
# entry point to PyMOL's API
from pymol import cmd

# pymol.Qt provides the PyQt5 interface, but may support PyQt4
# and/or PySide as well
from pymol.Qt import QtWidgets
from pymol.Qt.utils import loadUi
from pymol.Qt.utils import getSaveFileNameWithExt


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
        dialog = QtWidgets.QDialog()
        uifile = os.path.join(os.path.dirname(__file__), 'gui.ui')
        form = loadUi(uifile, dialog)
        Isosurface(form)
    dialog.show()


class Isosurface:
    def __init__(self, form):
        self.form = form
        self.objects_list = cmd.get_names('objects')
        self.maps_list = [e for e in self.objects_list
                          if cmd.get_type(e) == 'object:map']
        self.slider_precision = 1000
        self.bindings()

    def iso_to_slider(self, isovalue):
        return int(isovalue * self.slider_precision)

    def slider_to_iso(self, val):
        return val / float(self.slider_precision)

    @property
    def isoslider_value(self):
        sliderval = self.form.isoslider.value()
        isoval = self.slider_to_iso(sliderval)
        return isoval

    @property
    def isotext_value(self):
        return float(self.form.isoval_edit.text())

    @property
    def current_mrc(self):
        return self.form.mapselector.currentText()

    def load_isosurface(self, mrc):
        grid = cmd.get_volume_field(mrc)
        isomin = grid.min()
        isomax = grid.max()
        self.form.label_isomin.setText('%.4f' % isomin)
        self.form.label_isomax.setText('%.4f' % isomax)
        self.form.isoslider.setMinimum(self.iso_to_slider(isomin))
        self.form.isoslider.setMaximum(self.iso_to_slider(isomax))
        self.form.isoslider.setTickInterval(1)
        self.form.isoslider.setValue(self.iso_to_slider(isomin + (isomax - isomin) / 2.))
        self.form.isoval_edit.setText(str(self.isoslider_value))
        cmd.isosurface('isosurf', mrc, level=self.isoslider_value)

    def set_isovalue(self):
        cmd.isosurface('isosurf', self.current_mrc, level=self.isoslider_value)
        self.form.isoval_edit.setText(str(self.isoslider_value))

    def bindings(self):
        self.form.mapselector.addItems(self.maps_list)
        if len(self.maps_list) == 1:
            self.load_isosurface(self.current_mrc)
        self.form.mapselector.activated.connect(lambda: self.load_isosurface(self.current_mrc))
        self.form.isoslider.valueChanged.connect(lambda: self.set_isovalue())
        self.form.isoval_edit.editingFinished.connect(lambda: self.form.isoslider.setValue(self.iso_to_slider(self.isotext_value)))
