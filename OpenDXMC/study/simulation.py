# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:03:15 2015

@author: erlean
"""
import numpy as np



def generate_attinuation_lut(materials, material_map, min_eV=None,
                             max_eV=None):
    if min_eV is None:
        min_eV = 0.
    if max_eV is None:
        max_eV = 100.e6

    names = [m.name for m in materials]
    atts = {}

    for key, value in list(material_map.items()):
        key = int(key)
        try:
            ind = names.index(value)
        except ValueError:
            raise ValueError('No material named '
                             '{0} in first argument. '
                             'The material_map requires {0}'.format(value))
        atts[key] = materials[ind].attinuation


    energies = np.unique(np.hstack([a['energy'] for a in list(atts.values())]))
    e_ind = (energies <= max_eV) * (energies >= min_eV)
    if not any(e_ind):
        raise ValueError('Supplied minimum or maximum energies '
                         'are out of range')
    energies = energies[e_ind]
    lut = np.empty((len(atts), 5, len(energies)), dtype=np.double)
    for i, a in list(atts.items()):
        lut[i, 0, :] = energies
        for j, key in enumerate(['total', 'rayleigh', 'photoelectric',
                                 'compton']):
            lut[i, j+1, :] = np.interp(energies, a['energy'], a[key])
    return lut


def prepare_geometry_from_ct_array(ctarray, specter, materials):
        """genereate material and density arrays and material map from
           a list of materials to use
           INPUT:
               specter for this study
               list of materials
           OUTPUT :
               material_map, material_array, density_array
        """
        if ctarray is None:
            return
        specter = (specter[0]/1000., specter[1]/specter[1].sum())

        water_key = None
        material_map = {}
        material_att = {}
        material_dens = {}
        materials.sort(key=lambda x: x.density)
        for i, mat in enumerate(materials):
            material_map[i] = mat.name
            material_dens[i] = mat.density
            #interpolationg and integrating attinuation coefficient
            material_att[i] = np.trapz(np.interp(specter[0],
                                                 mat.attinuation['energy'],
                                                 mat.attinuation['total']),
                                       specter[0])
            material_att[i] *= mat.density
            if mat.name == 'water':
                water_key = i
        assert water_key is not None  # we need to include water in materials

        # getting a list of attinuation
        material_HU_list = [(key, (att / material_att[water_key] -1.)*1000.)
                            for key, att in material_att.items()]
        material_HU_list.sort(key=lambda x: x[1])

        material_array = np.zeros_like(ctarray, dtype=np.int)
        density_array = np.zeros_like(ctarray, dtype=np.float)
        llim = -np.inf
        for i in range(len(material_HU_list)):
            if i == len(material_HU_list) -1:
                ulim = np.inf
            else:
                ulim = 0.5 *(material_HU_list[i][1] + material_HU_list[i+1][1])
            ind = np.nonzero((ctarray > llim) * (ctarray <= ulim))
            material_array[ind] = material_HU_list[i][0]
            density_array[ind] = material_dens[material_HU_list[i][0]]
            llim = ulim

        return material_map, material_array, density_array



class Simulation(object):

    __description = { 'name': '',
                      'scan_fov': 50.,
                      'sdd': 100.,
                      'detector_width': 0.6,
                      'detector_rows': 64,
                      'collimation_width': 0.6 * 64,
                      'al_filtration': 7.,
                      'xcare': False,
                      'ctdi_air100': 0.,
                      'ctdi_w100': 0.,
                      'kV': 120.,
                      'region': 'abdomen',
                      # per 1000000 histories to dose
                      'conversion_factor_ctdiair': 0,
                      # per 1000000 histories to dose
                      'conversion_factor_ctdiw': 0,
                      'is_spiral': False,
                      'al_filtration': 7.,
                      'pitch': 0,
                      'exposures': 1200.,
                      'histories': 1000,
                      'batch_size': 1,
                      'start': 0.,
                      'stop': 0.,
                      'step': 0,
                      'start_at_exposure_no': 0,
                      'finish': False,
                      }
    __dtype = { 'name': 'a64',
                'scan_fov': np.float,
                'sdd': np.float,
                'detector_width': np.float,
                'detector_rows': np.int,
                'collimation_width': np.float,
                'al_filtration': np.float,
                'xcare': np.bool,
                'ctdi_air100': np.float,
                'ctdi_w100': np.float,
                'kV': np.float,
                'region': 'a64',
                # per 1000000 histories to dose
                'conversion_factor_ctdiair': np.float,
                # per 1000000 histories to dose
                'conversion_factor_ctdiw': np.float,
                'is_spiral': np.bool,
                'al_filtration': np.float,
                'pitch': np.float,
                'exposures': np.int,
                'histories': np.int,
                'batch_size': np.int,
                'start': np.float,
                'stop': np.float,
                'step': np.float,
                'start_at_exposure_no': np.int,
                'finish': np.bool
                }
    __arrays = { 'material': None,
                 'density': None,
                 'organ': None,
                 'spacing':None,
                 'ctarray': None,
                 'exposure_modulation': None,
                 'energi_imparted': None
                 }
    __tables = { 'material_map': None,
                 'organ_map': None
                 }

    def __init__(self, name, description=None):
        self.name = name

    def numpy_dtype(self):
        d = {'names': [], 'formats': []}
        for key, value in list(self.__dtype.items()):
            d['names'].append(key)
            d['formats'].append(value)
        return np.dtype(d)

    @property
    def description(self):
        return self.__description

    @property
    def arrays(self):
        return self.__arrays

    @property
    def tables(self):
        return self.__tables

    @property
    def name(self):
        return self.__description['name']

    @name.setter
    def name(self, value):
        if isinstance(value, bytes):
            value = str(value, encoding='utf-8')
        else:
            value = str(value)
        name = "".join([l for l in value.split() if len(l) > 0])
        assert len(name) > 0
        self.__description['name'] = name.lower()

    @property
    def scan_fov(self):
        return self.__description['scan_fov']
    @scan_fov.setter
    def scan_fov(self, value):
        assert value > 0.
        self.__description['scan_fov'] = float(value)

    @property
    def sdd(self):
        return self.__description['sdd']
    @sdd.setter
    def sdd(self, value):
        assert value > 0.
        self.__description['sdd'] = float(value)

    @property
    def detector_width(self):
        return self.__description['detector_width']
    @detector_width.setter
    def detector_width(self, value):
        assert value > 0.
        self.__description['detector_width'] = float(value)

    @property
    def detector_rows(self):
        return self.__description['detector_rows']
    @detector_rows.setter
    def detector_rows(self, value):
        assert value > 0
        self.__description['detector_rows'] = int(value)

    @property
    def total_collimation(self):
        return self.__description['detector_rows'] * self.__description['detector_width']

    @property
    def collimation_width(self):
        return self.__description['collimation_width']
    @collimation_width.setter
    def collimation_width(self, value):
        assert value > 0.
        self.__description['collimation_width'] = float(value)

    @property
    def al_filtration(self):
        return self.__description['al_filtration']
    @al_filtration.setter
    def al_filtration(self, value):
        self.__description['al_filtration'] = float(value)

    @property
    def xcare(self):
        return self.__description['xcare']
    @xcare.setter
    def xcare(self, value):
        self.__description['xcare'] = bool(value)

    @property
    def ctdi_air100(self):
        return self.__description['ctdi_air100']
    @ctdi_air100.setter
    def ctdi_air100(self, value):
        self.__description['ctdi_air100'] = float(value)

    @property
    def ctdi_w100(self):
        return self.__description['ctdi_w100']
    @ctdi_w100.setter
    def ctdi_w100(self, value):
        self.__description['ctdi_w100'] = float(value)

    @property
    def kV(self):
        return self.__description['kV']
    @kV.setter
    def kV(self, value):
        assert value >= 40.
        self.__description['kV'] = float(value)

    @property
    def region(self):
        return self.__description['region']
    @region.setter
    def region(self, value):
        self.__description['region'] = value

    @property
    def conversion_factor_ctdiair(self):
        return self.__description['conversion_factor_ctdiair']
    @conversion_factor_ctdiair.setter
    def conversion_factor_ctdiair(self, value):
        assert float(value) > 0
        self.__description['conversion_factor_ctdiair'] = float(value)

    @property
    def conversion_factor_ctdiw(self):
        return self.__description['conversion_factor_ctdiw']
    @conversion_factor_ctdiw.setter
    def conversion_factor_ctdiw(self, value):
        assert float(value) > 0
        self.__description['conversion_factor_ctdiw'] = float(value)


    @property
    def is_spiral(self):
        return self.__description['is_spiral']
    @is_spiral.setter
    def is_spiral(self, value):
        self.__description['is_spiral'] = bool(value)

    @property
    def pitch(self):
        return self.__description['pitch']
    @pitch.setter
    def pitch(self, value):
        if float(value) > 0:
            self.is_spiral = True
        self.__description['pitch'] = float(value)

    @property
    def exposures(self):
        return self.__description['exposures']
    @exposures.setter
    def exposures(self, value):
        assert int(value) > 0
        self.__description['exposures'] = int(value)

    @property
    def histories(self):
        return self.__description['histories']
    @histories.setter
    def histories(self, value):
        assert int(value) > 0
        self.__description['histories'] = int(value)

    @property
    def batch_size(self):
        return self.__description['batch_size']
    @batch_size.setter
    def batch_size(self, value):
        assert int(value) > 0
        self.__description['batch_size'] = int(value)

    @property
    def start(self):
        return self.__description['start']
    @start.setter
    def start(self, value):
        self.__description['start'] = float(value)

    @property
    def stop(self):
        return self.__description['stop']
    @stop.setter
    def stop(self, value):
        self.__description['stop'] = float(value)

    @property
    def step(self):
        return self.__description['step']
    @step.setter
    def step(self, value):
        self.__description['step'] = float(value)

    @property
    def start_at_exposure_no(self):
        return self.__description['start_at_exposure_no']
    @start_at_exposure_no.setter
    def start_at_exposure_no(self, value):
        self.__description['start_at_exposure_no'] = int(value)

    @property
    def finish(self):
        return self.__description['finish']
    @finish.setter
    def finish(self, value):
        self.__description['finish'] = bool(value)

    @property
    def material(self):
        return self.__arrays['material']
    @material.setter
    def material(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self.__arrays['material'] = value
    @property
    def density(self):
        return self.__arrays['density']
    @density.setter
    def density(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self.__arrays['density'] = value
    @property
    def organ(self):
        return self.__arrays['organ']
    @organ.setter
    def organ(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self.__arrays['organ'] = value
    @property
    def spacing(self):
        return self.__arrays['spacing']
    @spacing.setter
    def spacing(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 1
        self.__arrays['spacing'] = value
    @property
    def ctarray(self):
        return self.__arrays['ctarray']
    @ctarray.setter
    def ctarray(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self.__arrays['ctarray'] = value
    @property
    def exposure_modulation(self):
        return self.__arrays['exposure_modulation']
    @exposure_modulation.setter
    def exposure_modulation(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 2
        self.__arrays['exposure_modulation'] = value
    @property
    def energy_imparted(self):
        return self.__arrays['energy_imparted']
    @energy_imparted.setter
    def energy_imparted(self, value):
        assert isinstance(value, np.ndarray)
        assert len(value.shape) == 3
        self.__arrays['energy_imparted'] = value
    @property
    def material_map(self):
        return self.__tables['material_map']
    @material_map.setter
    def material_map(self, value):
        assert isinstance(value, np.recarray)
        self.__tables['material_map'] = value
    @property
    def organ_map(self):
        return self.__tables['organ_map']
    @organ_map.setter
    def organ_map(self, value):
        assert isinstance(value, np.recarray)
        self.__tables['organ_map'] = value

#
#    def obtain_ctdiair_conversion_factor(self, material, callback=None):
#
#        spacing = np.array((1, 1, 10), dtype=np.double)
#        N = np.rint(np.array((self.sdd / spacing[0], self.sdd / spacing[1], 1),
#                             dtype=np.double))
#
#        offset = -N * spacing / 2.
#        material_array = np.zeros(N, dtype=np.intc)
#        material_map = {0: material.name}
#        density_array = np.zeros(N, dtype=np.double) + material.density
#        lut = generate_attinuation_lut([material], material_map, max_eV=0.5e6)
#        dose = np.zeros_like(density_array, dtype=np.double)
#
#        en_specter = specter(self.kV, angle_deg=10., filtration_materials='Al',
#                             filtration_mm=6.)
#        norm_specter = (en_specter[0], en_specter[1]/en_specter[1].sum())
#        particles = phase_space.ct_seq(self.scan_fov, self.sdd,
#                                       self.total_collimation,
#                                       histories=1000, exposures=1200,
#                                       batch_size=100000,
#                                       energy_specter=norm_specter)
##        pdb.set_trace()
#        t0 = time.clock()
#        for batch, i, tot in particles:
#            score_energy(batch, N, spacing, offset, material_array,
#                         density_array, lut, dose)
#            p = round(i * 100 / float(tot), 1)
#            t1 = (time.clock() - t0) / float(i) * (tot - i)
#            print(('{0}% {1}, ETA in {2}'.format(p, time.ctime(),
#                                                utils.human_time(t1))))
#
#        center = np.floor(N / 2).astype(np.int)
#        d = dose[center[0], center[1],
#                 center[2]] / material.density * np.prod(spacing)
#        d /= float(tot) / 1000000.
#        print(d)
#        self.__description['conversion_factor_ctdiair'] = self.ctdi_air100 / d
#
#    def generate_ctdi_phantom(self, pmma, air, size=32.):
#        spacing = np.array((1, 1, 10), dtype=np.double)
#        N = np.rint(np.array((self.sdd / spacing[0], self.sdd / spacing[1], 1),
#                             dtype=np.double))
#
#        offset = -N * spacing / 2.
#        material_array = np.zeros(N, dtype=np.intc)
#        radii_phantom = size * spacing[0]
#        radii_meas = 2. * spacing[0]
#        center = (N * spacing / 2.)[:2]
#        radii_pos = 28*spacing[0]
#        pos = [(center[0], center[1])]
#        for ang in [0, 90, 180, 270]:
#            dx = radii_pos * np.sin(np.deg2rad(ang))
#            dy = radii_pos * np.cos(np.deg2rad(ang))
#            pos.append((center[0] + dx, center[1] + dy))
#
#        for i in range(int(N[2])):
#            material_array[:, :, i] += utils.circle_mask((N[0], N[1]),
#                                                         radii_phantom)
#            for p in pos:
#                material_array[:, :, i] += utils.circle_mask((N[0], N[1]),
#                                                             radii_meas,
#                                                             center=p)
#
#        material_map = {0: air.name, 1: pmma.name, 2: air.name}
#        density_array = np.zeros_like(material_array, dtype=np.double)
#        density_array[material_array == 0] = air.density
#        density_array[material_array == 1] = pmma.density
#        density_array[material_array == 2] = air.density
#
##        density_array = np.zeros(N, dtype=np.double) + material.density
#        lut = generate_attinuation_lut([air, pmma], material_map, max_eV=0.5e6)
#        return N, spacing, offset, material_array, density_array, lut, pos
##        dose = np.zeros_like(density_array, dtype=np.double)
#
#    def obtain_ctdiw_conversion_factor(self, pmma, air,
#                                       callback=None, phantom_size=32.):
#
#        args = self.generate_ctdi_phantom(pmma, air)
#        N, spacing, offset, material_array, density_array, lut, meas_pos = args
#
#        # removing outside air
#        lut[0, 1:, :] = 0
#
#        dose = np.zeros_like(density_array)
#
#        en_specter = specter(self.kV, angle_deg=10., filtration_materials='Al',
#                             filtration_mm=6.)
#        norm_specter = (en_specter[0], en_specter[1]/en_specter[1].sum())
#
#        particles = phase_space.ct_seq(self.scan_fov, self.sdd,
#                                       self.total_collimation,
#                                       histories=1000, exposures=1200,
#                                       batch_size=100000,
#                                       energy_specter=norm_specter)
##        pdb.set_trace()
#        t0 = time.clock()
#        for batch, i, tot in particles:
#            score_energy(batch, N, spacing, offset, material_array,
#                         density_array, lut, dose)
#            p = round(i * 100 / float(tot), 1)
#            t1 = (time.clock() - t0) / float(i) * (tot - i)
#            print(('{0}% {1}, ETA in {2}'.format(p, time.ctime(),
#                                                utils.human_time(t1))))
#
#        d = []
#        for p in meas_pos:
#            x, y = int(p[0]), int(p[1])
#            d.append(dose[x, y, 0] / air.density * np.prod(spacing))
#            d[-1] /= (float(tot) / 1000000.)
#
#        ctdiv = d.pop(0) / 3.
#        ctdiv += 2. * sum(d) / 3. / 4.
#        print(ctdiv)
#        self.conversion_factor_ctdiw = self.ctdi_w100 / ctdiv

