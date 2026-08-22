# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['formant_editor.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('formant_studio.ico', '.'),
        ('Docs/ipa_symbol_chart.csv', 'Docs'),
        ('phone_class_model.npz', '.'),
    ],
    hiddenimports=['PyQt6.QtMultimedia', 'scipy.signal.windows'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='FormantStudio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['formant_studio.ico'],
)
