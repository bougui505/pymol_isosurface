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
import urllib.request


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
        self.fill_map_list()
        self.slider_precision = 1000
        self.transparency_slider_precision = 100.
        self.form.transparency_slider.setMinimum(0)
        self.form.transparency_slider.setMaximum(100)
        self.grid = None
        self.isosurfname = None
        for i, mapname in enumerate(self.maps_list):
            self.form.mapselector.setCurrentIndex(i)
            self.load_isosurface(self.current_mrc)
        self.bindings()

    def fill_map_list(self):
        self.objects_list = cmd.get_names('objects')
        self.maps_list = [e for e in self.objects_list
                          if cmd.get_type(e) == 'object:map']
        self.form.mapselector.addItems(self.maps_list)

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

    @property
    def transparency_value(self):
        return self.form.transparency_slider.value() / self.transparency_slider_precision

    @property
    def is_zone(self):
        return self.form.zone_checkBox.isChecked()

    @property
    def zone_selection(self):
        rawtext = self.form.selectionbox.text()
        if rawtext != "":
            return self.form.selectionbox.text()
        else:
            return None

    @property
    def zone_radius(self):
        try:
            return float(self.form.radiusbox.text())
        except ValueError:
            return None

    @property
    def emd_id(self):
        try:
            return self.form.emd_id.text()
        except ValueError:
            return None

    def set_transparency(self):
        cmd.set('transparency', value=self.transparency_value,
                selection=self.isosurfname)

    def set_isoslider(self, mrc, setvalue=True):
        self.grid = cmd.get_volume_field(mrc)
        isomin = self.grid.min()
        isomax = self.grid.max()
        self.form.label_isomin.setText('%.4f' % isomin)
        self.form.label_isomax.setText('%.4f' % isomax)
        self.form.isoslider.setMinimum(self.iso_to_slider(isomin))
        self.form.isoslider.setMaximum(self.iso_to_slider(isomax))
        self.form.isoslider.setTickInterval(1)
        if setvalue:
            self.form.isoslider.setValue(self.iso_to_slider(isomin + (isomax - isomin) / 2.))
        self.form.isoval_edit.setText(str(self.isoslider_value))

    def load_isosurface(self, mrc, setvalue=True):
        self.set_isoslider(mrc, setvalue=setvalue)
        self.isosurfname = '%s_isosurf' % self.current_mrc
        cmd.isosurface(self.isosurfname, mrc, level=self.isoslider_value)

    def set_isovalue(self):
        if self.zone_selection is not None and self.zone_radius is not None and self.is_zone:
            cmd.isosurface(self.isosurfname, self.current_mrc, level=self.isoslider_value,
                           selection=self.zone_selection, carve=self.zone_radius)
        else:
            cmd.isosurface(self.isosurfname, self.current_mrc, level=self.isoslider_value)
        self.form.isoval_edit.setText(str(self.isoslider_value))

    def get_zone_selection(self):
        cmd.select('zone_selection', selection=self.zone_selection, enable=0)

    def zone_map(self):
        cmd.isosurface(self.isosurfname, self.current_mrc, level=self.isoslider_value,
                       selection='zone_selection', carve=self.zone_radius)
        self.form.zone_checkBox.setChecked(True)

    def toggle_zone_map(self):
        if self.is_zone:
            self.zone_map()
        else:
            cmd.isosurface(self.isosurfname, self.current_mrc,
                           level=self.isoslider_value)

    def fetch_emd(self):
        if self.emd_id is not None:
            fetch_path = cmd.get('fetch_path')
            urlstr = 'ftp://ftp.wwpdb.org/pub/emdb/structures/EMD-%s/map/emd_%s.map.gz' % (self.emd_id, self.emd_id)
            emd_filename = '%s/emd_%s.map.gz' % (fetch_path, self.emd_id)
            if not os.path.exists(emd_filename):
                urllib.request.urlretrieve(urlstr, emd_filename)
            cmd.load(emd_filename)
            self.fill_map_list()
            self.load_isosurface(self.current_mrc)

    def bindings(self):
        self.form.mapselector.activated.connect(lambda: self.load_isosurface(self.current_mrc, setvalue=False))
        self.form.isoslider.valueChanged.connect(lambda: self.set_isovalue())
        self.form.isoval_edit.editingFinished.connect(lambda: self.form.isoslider.setValue(self.iso_to_slider(self.isotext_value)))
        self.form.selectionbox.editingFinished.connect(self.get_zone_selection)
        self.form.radiusbox.editingFinished.connect(self.zone_map)
        self.form.transparency_slider.valueChanged.connect(self.set_transparency)
        self.form.emd_id.returnPressed.connect(self.fetch_emd)
        self.form.zone_checkBox.stateChanged.connect(self.toggle_zone_map)
