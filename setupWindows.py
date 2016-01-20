
"""
Created on Thu Jan 16 10:41:23 2014

@author: erlean

"""
###############cx_freeze#####################################################
""" Hacked cx_freeze\windist.py to allow no administrator rights install
    In def add_properties(self):
        props = [...
        - ('ALLUSERS', '1')
        + ('ALLUSERS', '2'),
        + ('MSIINSTALLPERUSER','1')
        ]
"""

import sys
from cx_Freeze import setup, Executable
from Cython.Build import cythonize

import numpy as np

app_version = '0.1'

#must edit cx_freeze.hooks line 548 from 
#finder.IncludePackage("scipy.lib")
#to
#finder.IncludePackage("scipy._lib")
#    
#base = None
#if sys.platform == 'win32':
#    base = 'Win32GUI'
#    if len(sys.argv) == 1:
#        sys.argv.append("bdist_msi")
#else:
if len(sys.argv) == 1:
    sys.argv.append("build")

exe = Executable(
    script="testapp.py",
    base='Win32GUI',
    icon='ic.ico'
    )

# http://msdn.microsoft.com/en-us/library/windows/desktop/aa371847(v=vs.85).aspx
shortcut_table = [
    ("DesktopShortcut",        # Shortcut
     "DesktopFolder",          # Directory_
     "OpenDXMC",                # Name
     "TARGETDIR",              # Component_
     "[TARGETDIR]OpenDXMC.exe",  # Target
     None,                     # Arguments
     None,                     # Description
     None,                     # Hotkey
     None,                     # Icon
     None,                     # IconIndex
     None,                     # ShowCmd
     'TARGETDIR'               # WkDir
     )
    ]

# Now create the table dictionary
msi_data = {"Shortcut": shortcut_table}



includefiles = []#('path2python\\Lib\\site-packages\\scipy\\special\\_ufuncs.pyd','_ufuncs.pyd')]

# modules not included automatical by cx_freeze
#includes = ['skimage.draw', 'skimage.draw._draw','skimage._shared.geometry','scipy.sparse','skimage.filter','skimage.feature',
#            'scipy.ndimage','scipy.special', 'scipy.linalg', 'scipy.integrate']#,'scipy.special._ufuncs_cxx', 'scipy.linalg']
#includes = ['scipy', 'scipy.ndimage','scipy.linalg', 'scipy.integrate']
includes = []

excludes = ['curses', 'email', 'ttk', 'PIL', 'matplotlib','tcl', 'tk']
#            'tzdata']

tk_excludes = ["pywin", "pywin.debugger", "pywin.debugger.dbgcon",
               "pywin.dialogs", "pywin.dialogs.list",
               "Tkconstants", "Tkinter", "tcl"]
excludes += tk_excludes

build_exe_options = {'packages': ['scipy'],
                     'excludes': excludes,
                     'includes': includes,
                     'include_files': includefiles}
bdist_msi_options = {'upgrade_code': "{901f8668-1a53-478f-8499-d8735b2eef5b}",
                     'data': msi_data}
setup(
    name="OpenDXMC",
    version=app_version,
    description="Monte Carlo CT tool",
    executables=[exe],
    include_dirs=np.get_include(),
    options={
        "build_exe": build_exe_options,
        "bdist_msi": bdist_msi_options})
