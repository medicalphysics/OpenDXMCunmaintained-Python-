# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 14:23:31 2015

@author: erlean
"""
import numpy as np
from opendxmc.database.h5database import SIMULATION_DICT_TEMPLATE, ARRAY_TEMPLATES
import logging
logger = logging.getLogger('OpenDXMC')

class Validator(object):
    def __init__(self):
        self._pt = SIMULATION_DICT_TEMPLATE
        self._at = ARRAY_TEMPLATES

        self._props = {key: value[0] for key, value in self._pt.items()}
        self._arrays = {key: None for key in self._at.keys()}

    def reset(self):
        self.set_data(None, True)

    def set_data(self, props=None, reset=True):
        if reset:
            self._props = {key: value[0] for key, value in self._pt.items()}
            self._arrays = {key: None for key in self._at.keys()}
        valid_attrs = self._pt.keys() + self._at.keys()
        if props is not None:
            if reset:
                assert 'name' in props
            prop_list = [(key, value) for key, value in props.items()]
            prop_list.sort(key=lambda x, y: self._pt[key][5] if x in self._pt else 0)
            for key, value in prop_list:
                if key in valid_attrs:
                    setattr(self, key, value)
                else:
                    logger.debug('Error in propety 0{} for simulation {1}'.format(key, props['name']))

    def get_data(self):
        return self._props, self._arrays

    def string_validator(self, value, strict=False):
        if isinstance(value, bytes):
            value = str(value, encoding='utf-8')
        else:
            value = str(value)
        if strict:
            name = "".join([l for l in value.split() if len(l) > 0])
        assert len(name) > 0
        return name.lower()

    def float_validator(self, value, gt_0=False, cut=0.):
        val = float(value)
        if gt_0:
            assert val >= cut
        return val
    def int_validator(self, value, gt_0=False):
        val = int(value)
        if gt_0:
            assert val >= 0
        return val
    def bool_validator(self, value):
        assert isinstance(value, bool)
        return value

    def validate_structured_array(self, value, name):
        if value is None:
            return None

        if isinstance(value, dict):
            value_rec = np.recarray((len(value),),
                                    dtype=self._at[name][0])
            for ind, item in enumerate(value.items()):
                value_rec[ind] = item
            return
        elif isinstance(value, np.ndarray):
            value_rec = np.recarray((len(value),),
                                    dtype=self._at[name][0])
            assert value.dtype.names is not None
            c_title = self._at[name][0].names
            for var_name in c_title:
                assert var_name in value.dtype.names
            for i in range(len(value)):
                value_rec[i] = (value[c_title[0]][i], value[c_title[1]][i])
        else:
            raise ValueError('material_map must be a structured array with correct keys or a dict')
        return value_rec

    @property
    def name(self):
        return self._props['name']

    @name.setter
    def name(self, value):
        self._props['name'] = self.string_validator(value, True)

    @property
    def scan_fov(self):
        return self._props['scan_fov']
    @scan_fov.setter
    def scan_fov(self, value):
        self._props['scan_fov'] = self.float_validator(value, True)

    @property
    def sdd(self):
        return self._props['sdd']
    @sdd.setter
    def sdd(self, value):
        self._props['sdd'] = self.float_validator(value, True)

    @property
    def detector_width(self):
        return self._props['detector_width']
    @detector_width.setter
    def detector_width(self, value):
        self._props['detector_width'] = self.float_validator(value, True)

    @property
    def detector_rows(self):
        return self._props['detector_rows']
    @detector_rows.setter
    def detector_rows(self, value):
        self._props['detector_rows'] = self.int_validator(value, True)

    @property
    def collimation_width(self):
        return self._props['collimation_width']
    @collimation_width.setter
    def collimation_width(self, value):
        self._props['collimation_width'] = self.float_validator(value, True)

    @property
    def al_filtration(self):
        return self._props['al_filtration']
    @al_filtration.setter
    def al_filtration(self, value):
        self._props['al_filtration'] = self.float_validator(value, True)

    @property
    def xcare(self):
        return self._props['xcare']
    @xcare.setter
    def xcare(self, value):
        self._props['xcare'] = self.bool_validator(value)

    @property
    def ctdi_air100(self):
        return self._props['ctdi_air100']
    @ctdi_air100.setter
    def ctdi_air100(self, value):
        self._props['ctdi_air100'] = self.float_validator(value, True)

    @property
    def ctdi_vol100(self):
        return self._props['ctdi_vol100']
    @ctdi_vol100.setter
    def ctdi_vol100(self, value):
        self._props['ctdi_vol100'] = self.float_validator(value, True)
        self._props['ctdi_w100'] = self._props['ctdi_vol100'] * self.pitch

    @property
    def ctdi_w100(self):
        return self._props['ctdi_w100']
    @ctdi_w100.setter
    def ctdi_w100(self, value):
        self._props['ctdi_w100'] = self.float_validator(value, True)
        self._props['ctdi_vol100'] = self._props['ctdi_w100'] / self.pitch

    @property
    def aquired_kV(self):
        return self._props['aquired_kV']
    @aquired_kV.setter
    def aquired_kV(self, value):
        self._props['aquired_kV'] = self.float_validator(value, True)

    @property
    def kV(self):
        return self._props['kV']
    @kV.setter
    def kV(self, value):
        self._props['kV'] = self.float_validator(value, True, 40)

    @property
    def region(self):
        return self._props['region']
    @region.setter
    def region(self, value):
        self._props['region'] = self.string_validator(value)

    @property
    def conversion_factor_ctdiair(self):
        return self._props['conversion_factor_ctdiair']
    @conversion_factor_ctdiair.setter
    def conversion_factor_ctdiair(self, value):
        self._props['conversion_factor_ctdiair'] = self.float_validator(value, True)

    @property
    def conversion_factor_ctdiw(self):
        return self._props['conversion_factor_ctdiw']
    @conversion_factor_ctdiw.setter
    def conversion_factor_ctdiw(self, value):
        self._props['conversion_factor_ctdiw'] = self.float_validator(value, True)


    @property
    def is_spiral(self):
        return self._props['is_spiral']
    @is_spiral.setter
    def is_spiral(self, value):
        self._props['is_spiral'] = self.bool_validator(value)

    @property
    def pitch(self):
        return self._props['pitch']
    @pitch.setter
    def pitch(self, value):
        self._props['pitch'] = self.float_validator(value, True)
        self.ctdi_w100 = self._props['ctdi_w100']

    @property
    def exposures(self):
        return self._props['exposures']
    @exposures.setter
    def exposures(self, value):
        self._props['exposures'] = self.int_validator(value, True)

    @property
    def histories(self):
        return self._props['histories']
    @histories.setter
    def histories(self, value):
        self._props['histories'] = self.int_validator(value, True)

    @property
    def batch_size(self):
        return self._props['batch_size']
    @batch_size.setter
    def batch_size(self, value):
        assert int(value) > 0
        self._props['batch_size'] = self.int_validator(value, True)

    @property
    def start_scan(self):
        return self._props['start_scan']
    @start_scan.setter
    def start_scan(self, value):
        self._props['start_scan'] = self.float_validator(value)

    @property
    def stop_scan(self):
        return self._props['stop_scan']
    @stop_scan.setter
    def stop_scan(self, value):
        self._props['stop_scan'] = self.float_validator(value)

    @property
    def start(self):
        return self._props['start']
    @start.setter
    def start(self, value):
        self._props['start'] = self.float_validator(value)
        rng = [self.start_scan, self.stop_scan]
        assert min(rng) <= self._props['start'] <= max(rng)


    @property
    def stop(self):
        return self._props['stop']
    @stop.setter
    def stop(self, value):
        self._props['stop'] = self.float_validator(value)
        rng = [self.start_scan, self.stop_scan]
        assert min(rng) <= self._props['stop'] <= max(rng)

    @property
    def step(self):
        return self._props['step']
    @step.setter
    def step(self, value):
        self._props['step'] = self.float_validator(value, True)

    @property
    def start_at_exposure_no(self):
        return self._props['start_at_exposure_no']
    @start_at_exposure_no.setter
    def start_at_exposure_no(self, value):
        self._props['start_at_exposure_no'] = self.int_validator(value, True)

    @property
    def MC_finished(self):
        return self._props['MC_finished']
    @MC_finished.setter
    def MC_finished(self, value):
        self._props['MC_finished'] = self.bool_validator(value)

    @property
    def MC_running(self):
        return self._props['MC_running']
    @MC_running.setter
    def MC_running(self, value):
        self._props['MC_running'] = self.bool_validator(value)

    @property
    def MC_ready(self):
        return self._props['MC_ready']
    @MC_ready.setter
    def MC_ready(self, value):
        self._props['MC_ready'] = self.bool_validator(value)

    @property
    def spacing(self):
        return self._props['spacing']
    @spacing.setter
    def spacing(self, value):
        if isinstance(value, np.ndarray):
            self._props['spacing'] = value.astype(np.double)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert len(value) == 3
            self._props['spacing'] = value.astype(np.double)
            assert np.all(self._props['spacing'] > 0)

    @property
    def shape(self):
        return self._props['shape']
    @shape.setter
    def shape(self, value):
        if isinstance(value, np.ndarray):
            self._props['shape'] = value.astype(np.int)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert len(value) == 3
            self._props['shape'] = value.astype(np.int)
        assert np.all(self._props['shape'] > 0)


    @property
    def import_scaling(self):
        return self._props['import_scaling']
    @import_scaling.setter
    def import_scaling(self, value):
        if isinstance(value, np.ndarray):
            assert len(value.shape) == 1
            assert value.shape[0] == 3
            self._props['import_scaling'] = value.astype(np.double)
        elif isinstance(value, str):
            value = np.array([float(s) for s in value.split()], dtype=np.double)
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == 3
            assert len(value) == 3
            self._props['import_scaling'] = value.astype(np.double)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == 3
            assert len(value) == 3
            self._props['import_scaling'] = value.astype(np.double)
        assert np.all(self._props['import_scaling'] > 0)

    @property
    def scaling(self):
        return self._props['scaling']
    @scaling.setter
    def scaling(self, value):
        if isinstance(value, np.ndarray):
            assert len(value.shape) == 1
            assert value.shape[0] == 3
            self._props['scaling'] = value.astype(np.double)
        elif isinstance(value, str):
            value = np.array([float(s) for s in value.split()], dtype=np.double)
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == 3
            assert len(value) == 3
            self._props['scaling'] = value.astype(np.double)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert value.shape[0] == 3
            assert len(value) == 3
            self._props['scaling'] = value.astype(np.double)
        assert np.all(self._props['scaling'] > 0)


    @property
    def image_orientation(self):
        return self._props['image_orientation']
    @image_orientation.setter
    def image_orientation(self, value):
        if isinstance(value, np.ndarray):
            self._props['image_orientation'] = value.astype(np.double)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert len(value) == 3
            self._props['image_orientation'] = value.astype(np.double)
    @property
    def image_position(self):
        return self._props['image_position']
    @image_position.setter
    def image_position(self, value):
        if isinstance(value, np.ndarray):
            self._props['image_position'] = value.astype(np.double)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert len(value) == 3
            self._props['image_position'] = value.astype(np.double)

    @property
    def data_center(self):
        return self._props['data_center']
    @data_center.setter
    def data_center(self, value):
        if isinstance(value, np.ndarray):
            self._props['data_center'] = value.astype(np.double)
        else:
            value=np.array(value)
            assert isinstance(value, np.ndarray)
            assert len(value) == 3
            self._props['data_center'] = value.astype(np.double)

    @property
    def ignore_air(self):
        return self._props['ignore_air']
    @ignore_air.setter
    def ignore_air(self, value):
        self._props['ignore_air'] = self.bool_validator(value)

    @property
    def is_phantom(self):
        return self._props['is_phantom']
    @is_phantom.setter
    def is_phantom(self, value):
        self._props['is_phantom'] = self.bool_validator(value)

    @property
    def material(self):
        return self._arrays['material']
    @material.setter
    def material(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self._arrays['material'] = value.astype(self._at['material'][0])

    @property
    def density(self):
        return self._arrays['density']
    @density.setter
    def density(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        if self._arrays['density'] is not None:
            del self._arrays['density']
        self._arrays['density'] = value.astype(self._at['density'][0])

    @property
    def organ(self):
        return self._arrays['organ']
    @organ.setter
    def organ(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self._arrays['organ'] = value.astype(self._at['organ'][0])

    @property
    def ctarray(self):
        return self._arrays['ctarray']
    @ctarray.setter
    def ctarray(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self._arrays['ctarray'] = value.astype(self._at['ctarray'][0])

    @property
    def exposure_modulation(self):
        return self._arrays['exposure_modulation']
    @exposure_modulation.setter
    def exposure_modulation(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 2
        self._arrays['exposure_modulation'] = value.astype(self._at['exposure_modulation'][0])

    @property
    def energy_imparted(self):
        return self._arrays['energy_imparted']
    @energy_imparted.setter
    def energy_imparted(self, value):
        if value is None:
            del self._arrays['energy_imparted']
            self._arrays['energy_imparted'] = None
            return
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self._arrays['energy_imparted'] = value.astype(self._at['energy_imparted'][0])

    @property
    def material_map(self):
        return self._arrays['material_map']
    @material_map.setter
    def material_map(self, value):
        self._arrays['material_map'] = self.validate_structured_array(value, 'material_map')

    @property
    def organ_map(self):
        return self._arrays['organ_map']
    @organ_map.setter
    def organ_map(self, value):
        self._arrays['organ_map'] = self.validate_structured_array(value, 'organ_map')

    @property
    def organ_material_map(self):
        return self._arrays['organ_material_map']
    @organ_material_map.setter
    def organ_material_map(self, value):
        self._arrays['organ_material_map'] = self.validate_structured_array(value, 'organ_material_map')

