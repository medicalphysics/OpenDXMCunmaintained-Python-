#OpenDXMC
**A Monte Carlo dose simulator for diagnostic x-ray (ALPHA).**

OpenDXMC is a computer package to calculate and visualize dose distributions in diagnostic x-ray examinations. This package consists of a small library for Monte Carlo dose calculations for diagnostic x-ray energies in a voxelized geomtry. In addidtion the package includes a GUI for calculation of radiation dose from CT examinations. The application is currently in early development and a alpha release is available. However, the Monte Carlo code is not yet validated and use of this application should be for educational purposes only. 

Dose calculations are done by the Monte Carlo method optimized for a voxelized geometry. However only photons are modelled and essentially only KERMA is scored. The current code is oriented towards modelling radiation doses from conventional CT scanners. The application supports dose calculations on a) CT series of individiual patients and b) digitial phantoms from Virtual Human Database created by Helmholtz Zentrum research center (http://www.helmholtz-muenchen.de/en/amsd/service/scientific-services/virtual-human-database/). The phantoms are not distributed along with the application due to licensing, but a free license and permission to download the phantoms can be found at Virtual Human Database homepage. 

A small guide on how to use the aplication and description on some implementation details can be found in [this document](readme.pdf)

##Installation
OpenDXMC is currently only available on MS Windows. It is tested on Windows 10 and 7 both 32bits and 64bits. While the core MOnte Carlo dose engine is written in C, all helper functions and GUI is written in Python3. If you have a working Python3 installation you may install OpenDXMC by:

`>pip install opendxmc`

in a windows commmand prompt/console. To run OpenDXMC, start a python interpreter and type `>>>from opendxmc.app import start; start()` or from the Windows console `python -c "from opendxmc.app import start; start()"` 

Alternativly you may download a standalone release from [here.](insert link here) Unzip the folder and run OpenDXMC.exe

##Using the Monte Carlo library to construct your own simulations.
The Monte Carlo dose engine can be used independently of the GUI application to run other types of simulations. The library is written in C, while you can use the C header file and source code in opendxmc/engine/src, this example will focus on creating a simple simulation using the python wrapper. 







