import numpy as np
from .CPF100 import CPF100_M, CPF100_F, CPF100_u, CPF100_M_px, CPF100_F_px
from . import elements
import scipy.interpolate as inter

###############################################################################
# Calculates an emmission spectra from Bhremsstralong on electron interaction
# in tungsten. I.E spectra from an x-ray tube with tungsten anode.
# This code is based on:
#
# PAPER 1
# Calculation of x-ray spectra emerging from an x-ray tube. Part I.
# Electron penetration characteristics in x-ray targets
# Gavin G. Poludniowskia and Philip M. Evans
# Joint Department of Physics, Institute of Cancer Research and
# The Royal Marsden NHS Foundation Trust,
# Downs Road, Sutton, Surrey SM2 5PT, United Kingdom
#
# and
#
# PAPER 2
# Calculation of x-ray spectra emerging from an x-ray tube. Part II.
# X-ray production and filtration in x-ray targets
# Gavin G. Poludniowskia
# Joint Department of Physics, Institute of Cancer Research and
# The Royal Marsden NHS Foundation Trust,
# Downs Road, Sutton, Surrey SM2 5PT, United Kingdom
###############################################################################

# some constants
TUNGSTEN_ATOMIC_NUMBER = 74
TUNGSTEN_DENSITY = 19.25  # g / cm3
AVOGADOS_NUMBER = 6.02214129e23
TUNGSTEN_MOLAR_MASS = 0.005439512619669277  # [mol / g]
TUNGSTEN_NUMBER_DENSITY = AVOGADOS_NUMBER * TUNGSTEN_ATOMIC_NUMBER * TUNGSTEN_MOLAR_MASS
ELECTRON_REST_MASS = 510.9989  # keV/c^2
CLASSIC_ELEC_RADII = 2.81794033E-13  # cm
FINE_STRUCTURE = 0.00729735257   # fine structure constant
THETA_BAR = (TUNGSTEN_ATOMIC_NUMBER * CLASSIC_ELEC_RADII)**2 * FINE_STRUCTURE



#interpolation of attinuation coeffisients obtained from XCOM NIST database
ATTINUATION_W = inter.interp1d(np.squeeze(elements.W_attinuation[0, :]),
                               np.squeeze(elements.W_attinuation[1, :]),
                               'linear',  bounds_error=False, fill_value=0)
ATTINUATION_Al = inter.interp1d(np.squeeze(elements.Al_attinuation[0, :]),
                                np.squeeze(elements.Al_attinuation[1, :]),
                                'linear',  bounds_error=False, fill_value=0)
ATTINUATION_Cu = inter.interp1d(np.squeeze(elements.Cu_attinuation[0, :]),
                                np.squeeze(elements.Cu_attinuation[1, :]),
                                'linear',  bounds_error=False, fill_value=0)
DENSITY = {'copper': 8.960,
           'tungsten': 19.25,
           'aluminum': 2.699}


def attinuation(hv, name='tungsten', density=False):
    """Attinuation (NIST data) for tungsten, aluminum and copper
        hv: photon energy
        name: material to calculate attinuation coefficient
        density: if True multiply attinuation coefficient by the
            material density
    """
    if name.lower() == 'tungsten':
        a = ATTINUATION_W
    elif name.lower() == 'aluminum':
        a = ATTINUATION_Al
    elif name.lower() == 'copper':
        a = ATTINUATION_Cu
    else:
        return np.zeros_like(hv)
    if density:
        return a(hv) * DENSITY[name]
    return a(hv)


def __R_TW(T0):
    """ Thomson-Whiddington electron range in tungsten.
        T0: Initial electron energy
    """
    return 0.0119 * T0**1.513


def __PDF_int_F(u, T0, px):
    """Forward probabilitydistribution of electrons in tungsten with energy
    u * T0 and depth px [mg/cm2]
    u: fraction of initial electron energy (tube potensial)
    T0: initial energy (tube potensial)
    px: depth in tungsten
    """
    # interpolating Monte-Carlo data obtained from ref1 (paper1)
    f = inter.interp2d(CPF100_F_px, CPF100_u, CPF100_F, fill_value=None)
    return f(px, u) * __R_TW(T0) / __R_TW(100.)


def __PDF_int_M(u, T0, px):
    """Backward (scattered) probability distribution of electrons in tungsten
    with energy u * T0 and depth px [mg/cm2]
    u: fraction of initial electron energy (tube potensial)
    T0: initial energy (tube potensial)
    px: depth in tungsten
    """
    # interpolating Monte-Carlo data obtained from ref1 (paper1)
    f = inter.interp2d(CPF100_M_px, CPF100_u, CPF100_M, fill_value=None)
    return f(px, u) * __R_TW(T0) / __R_TW(100.)


def __f(u, T0, px):
    """Combined probability of finding a electron at depth px with energy u*T0
    u: fraction of initial electron energy (tube potensial)
    T0: initial energy (tube potensial)
    px: depth in tungsten
    """
    # based directly on paper 1 eq 3
    rtw = 0.0119 * T0**1.513
    ppx = np.repeat(np.reshape(px, (1, -1)), u.shape[0], axis=0)

    n_f = (1. - ppx / rtw)**(1.753)
    exp_dum = 1. - np.exp(- 18. * ppx / rtw)
    B = 0.5 + .084 * exp_dum
    F = 0.584 * exp_dum
    n_m = n_f * B * (F+1.) / (1.-F*B)
    return n_f * __PDF_int_F(u, T0, px) + n_m * __PDF_int_M(u, T0, px)


def __cross_section_SRMEBH(hv, u, T0):
    """Modified  Bethe Heitler Cross section for emmission of bhremsstralung
    from electron->tungsten interaction.
    hv: photon energy emmitted
    u: fraction of initial energy for electron before interaction
    T0: Tube potensial(initial energy)
    """
    Ti = u * T0
    Ei = Ti + ELECTRON_REST_MASS
    Ef = Ei - hv
    pic = (Ei**2 - ELECTRON_REST_MASS**2)**.5
    pfc = (Ef**2 - ELECTRON_REST_MASS**2)**.5
    L = (2 * np.log((Ei * Ef + pic * pfc - ELECTRON_REST_MASS**2)
         / (ELECTRON_REST_MASS * hv)))
    a = THETA_BAR * 2*(4*Ei*Ef*L-7*pic*pfc) / (3*hv*pic*pic)*(pic/pfc)
    a[Ti <= (hv + 0.001)] = 0  # removing electrons with initial energy < hv
    return a


def __F(hv, px, theta):
    """Anode attinuation correction
    hv: photon energy [kev]
    px: emmission depth in tungsten [mg/cm2]
    theta: anode angle"""

    return np.exp(-attinuation(hv) * px / np.sin(theta) / 1000.)


def __cross_section_char(hv, T0):
    """ Cross section for characteristic radiation emmission
    hv: photon energy [ndarray]
    T0: tube potential
    """
    hvi = np.array([59.3, 58.0, 67.2, 69.1])
    fi = np.array([.505, .291, .162, .042])
    Carr = np.zeros_like(hv)

    if not any(hv >= hvi[0]):
        return Carr
    for i in range(4):
        ind = np.argmin(np.abs(hv-hvi[i]))
        Carr[ind] = (4.4*fi[i] * .94 / 2.) * __Nobserved_char(hvi[i], T0)
    return Carr


def __Nobserved_char(hv, T0):
    """Number of caracteristic photons emmitted for one electron interacting
        with one tungsten atom, se paper 2
    """
    u = np.linspace(.1, 1, 200).astype(np.double)
    px = np.linspace(0, 16, 100).astype(np.double)
    du = u[1] - u[0]
    dpx = (px[1] - px[0]) / 1000  # converting from mg/cm to g/cm
    a = np.repeat(__cross_section_SRMEBH(hv, u, T0).reshape((-1, 1)),
                  px.shape[0], axis=-1)
    corr_a = np.nan_to_num(a * __f(u, T0, px)).sum(axis=0)
    return np.nan_to_num(corr_a).sum()*du*dpx


def __Nobserved_emit(hv, T0, theta):
    """Number of bremssthralung photons emmitted for one electron interacting
        with one tungsten atom, se paper 2
    """
    u = np.linspace(.1, 1, 200).astype(np.double)
    px = np.linspace(0, 16, 100).astype(np.double)
    du = u[1] - u[0]
    dpx = (px[1] - px[0]) / 1000  # converting from mg/cm to g/cm
    a = np.repeat(__cross_section_SRMEBH(hv, u, T0).reshape((-1, 1)),
                  px.shape[0], axis=-1)

    corr_a = np.nan_to_num(a * __f(u, T0, px)).sum(axis=0) * __F(hv, px, theta)

    return np.nan_to_num(corr_a).sum() * du * dpx


def __raw_specter(T0, angle_deg=None, angle_rad=None):
    """Unfiltered specter from anode per electron
    T0: tube potensial
    angle_deg/angle_rad: anode angle in degrees or radians"""

    if angle_deg is not None:
        angle = np.deg2rad(angle_deg)
    elif angle_rad is not None:
        angle = angle_rad
    else:
        angle = np.deg2rad(12.)

    hv = np.linspace(T0 * .1, T0, 200).astype(np.double)
    Nemit = np.empty_like(hv, dtype=np.double)
    Nchar = __cross_section_char(hv, T0)
    for i, e in enumerate(hv):
        Nemit[i] = __Nobserved_emit(e, T0, angle)

    # n is the NA (Avogados constant) * tungsten molar number per gram
    # / by tube potential / tungsten density
#    n = 6.02214129e23*0.005439512619669277  / T0 / 19.3   *10
#    print n
    n = TUNGSTEN_NUMBER_DENSITY
#    print n
    dx = 2.
    Nf = 0.68
    P = 0.33 * 100
    return hv, (Nemit + Nchar * (1 + P)) * n * dx * Nf


def specter(T0, angle_deg=None, angle_rad=None,
            filtration_materials=None, filtration_mm=None,
            mAs=1., sdd=100., detector_area=None):
    """
    Generate x-ray specter from tungsten anode by the methods of ref. 1 and 2.

    INPUT:
        T0 : float
            Tube potential in keV should be between 30 and 150 keV but may
            provide acceptable results for higher potentials.
        angle_deg : float [degrees]
            Anode angle in degrees, typicalvalue is 12 degrees.
        angle_rad : float [radians]
            Anode angle in radians.
        filtration_materials : iterable or string
            Additional Aluminum or Copper filtration in the tube
            Iterable may contain the following values 'al', 'aluminum' or 13
            for aluminum and similar for copper. One of the values may also be
            supplied as a string instead of iterator.
        filtration_mm : iterable or float
            mm of Copper or Aluminum filtration, must be provide with the
            filtration_materials keyword and of same type.
        mAs : float
            Scaling of specter to the mAs setting of the tube.
        sdd : float > 0.0
            Source detector distance in cm, or simply the distance to calculate
            the spectrum intensities.
        detector_area : float
            Area to calculate spectrum intensities at sdd.
    OUTPUT:
        energies, intensity : {float ndarray, float ndarray}
            The spectrum energies and intensity, energy in units of electron
            volts eV, intensity are gives as N_photons / [cm^2 mAs]

    Reference 1:
        Calculation of x-ray spectra emerging from an x-ray tube. Part I.
        Electron penetration characteristics in x-ray targets.
        Medical Physics, Vol. 34(6), 2164-2174

        Gavin G. Poludniowskia and Philip M. Evans
        Joint Department of Physics, Institute of Cancer Research and
        The Royal Marsden NHS Foundation Trust,
        Downs Road, Sutton, Surrey SM2 5PT, United Kingdom

    Reference 2:
        Calculation of x-ray spectra emerging from an x-ray tube. Part II.
        X-ray production and filtration in x-ray targets.
        Medical Physics, Vol. 34(6), 2175-2186

        Gavin G. Poludniowskia
        Joint Department of Physics, Institute of Cancer Research and
        The Royal Marsden NHS Foundation Trust,
        Downs Road, Sutton, Surrey SM2 5PT, United Kingdom
    """
    hv, N_obs = __raw_specter(T0, angle_deg, angle_rad)
    if isinstance(filtration_materials, str):
        filtration_materials = [filtration_materials]
    if not hasattr(filtration_mm, '__iter__'):
        filtration_mm = [filtration_mm]
    try:
        for material, mm in zip(filtration_materials, filtration_mm):
            if str(material).strip().lower() in ['al', 'aluminum', '13']:
                name = 'aluminum'
            elif str(material).strip().lower() in ['cu', 'copper', '29']:
                name = 'copper'
            elif str(material).strip().lower() in ['w', 'tungsten', '74',
                                                   'wolfram']:
                name = 'tungsten'
            else:
                continue
            N_obs *= np.exp(-attinuation(hv, name, True) * mm / 10.)
    except Exception as e:
        raise e
    electrons_per_mas = 1. / 1.60217657e-16
    if detector_area is None:
        red_factor = 1.
    else:
        red_factor = detector_area / (4 * np.pi * sdd**2)
    return (hv * 1000, N_obs * mAs * electrons_per_mas * red_factor)
