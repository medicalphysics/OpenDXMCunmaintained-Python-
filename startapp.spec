# -*- mode: python -*-

block_cipher = None

extra_data = [('opendxmc/data/materials/*.txt', 'opendxmc/data/materials'),
              ('opendxmc/data/materials/attinuation/*.txt', 'opendxmc/data/materials/attinuation'),
              ('opendxmc/engine/*.dll', 'opendxmc/engine'),
              ('opendxmc/app/icon.png', 'opendxmc/app')]

a = Analysis(['startapp.py'],
             pathex=['C:\\Users\\ander\\Documents\\GitHub\\OpenDXMC'],
             binaries=None,
             datas=extra_data,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='startapp',
          debug=False,
          strip=False,
          upx=True,
          console=False , icon='icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='startapp')
