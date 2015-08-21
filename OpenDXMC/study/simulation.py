# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 11:03:15 2015

@author: erlean
"""
import numpy as np



class Simulation(object):
    
    __description = { 'name': '',
                      'scan_fov': 50.,
                      'sdd': 100.,
                      'detector_width': 0.6,
                      'detector_rows': 64,
                      'collimation_width': 0.6 * 64,
                      'modulation_xy': False,
                      'modulation_z': False,
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
                      'pitch': 0,
                      'exposures': 1200.,
                      'histories': 1000,
                      'batch_size': 0,
                      'start': 0.,
                      'stop': 0.,
                      'start_at_exposure_no': 0,
                      'finish': False,
                      }
    __dtype = { 'name': 'a64',
                'scan_fov': np.float,
                'sdd': np.float,
                'detector_width': np.float,
                'detector_rows': np.int,
                'collimation_width': np.float,
                'modulation_xy': np.bool,
                'modulation_z': np.bool,
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
                'pitch': np.float,
                'exposures': np.int,
                'histories': np.int,
                'batch_size': np.int,
                'start': np.float,
                'stop': np.float,
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
                 'density_map': None,
                 'organ_map': None
                 }
    def __init__(self, name, description=None):

        if description is not None:
            for key, value in list(description.items()):
                if key in self.__description:
                    self.__description[key] = value
        self.name = name

    def __setattr__(self, name, value):
        print('setting {0} to {1}'.format(name, value))      
        if name in Simulation.__description:
            if name == 'name':
                print('editing name')
                value = self.validate_name(value)
            Simulation.__description[name] = value
        elif name in Simulation.__arrays:
            Simulation.__arrays[name] = value
        elif name in Simulation.__tables:
            Simulation.__tables[name] = value
        else:
            self.__dict__[name] = value

    def __getattr__(self, name):
        if name in Simulation.__description:
            return Simulation.__description[name]
        elif name in Simulation.__arrays:
            return Simulation.__arrays[name]
        elif name in Simulation.__tables:
            return Simulation.__tables[name]
        return Simulation.__getattr__(self, name)

    def numpy_dtype(self):
        d = {'names': [], 'formats': []}
        for key, value in list(self.__dtype.items()):
            d['names'].append(key)
            d['formats'].append(value)
        return np.dtype(d)
        
    def decription(self):
        return self.__description
   
    def arrays(self):
        return self.__arrays

    def tables(self):
        return self.__tables

    def validate_name(self, value):
        value = str(value)
        name = "".join([l for l in value.split() if len(l) > 0])
        assert len(name) > 0
        return name.lower()

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



def test_simulation():
    s = Simulation('eple')
    s.spacing = np.ones(3)
    s.name='Eple'
    s.eple = 5
    
    print(s.spacing) 
    print(s.name, s.eple)

    
if __name__ == '__main__':
    test_simulation()
    
    
    
