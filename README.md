# MRS LightCurve Calibration Pipeline

This repository contains the code used in Michail et al. (2025) to calibrate JWST/MIRI MRS data, producing time-resolved spectral cubes. This code builds upon the method developed by von Fellenberg et al. (2025a) and should be generalized enough to work with any MRS data. A requirements.txt file is provided for quick setup inside a conda environment. 

Running `RunPipeline.py` in its current form with the following directory structure will calibrate any data inside the ``./JWST/jw*`` directory tree:

MRS_LightCurve_Calibration_Pipeline/
|-----> MRS_LC_Calibration.py
|-----> RunPipeline.py
|-----> CombineCalibration.py
|-----> README
|-----> JWST/
        |-----> jw04572003001_02101_00001-seg001_mirifulong/
                |-----> jw04572003001_02101_00001-seg001_mirifulong_rateints.fits
        |-----> jw04572003001_02101_00001-seg001_mirifushort/
                |-----> jw04572003001_02101_00001-seg001_mirifushort_rateints.fits
        |-----> Other directories like this

If this code is used in your work, you are morally obligated to cite the following papers:
1. von Fellenberg et al. (2025a), [DOI:10.3847/2041-8213/ada3d2](https://ui.adsabs.harvard.edu/abs/2025ApJ...979L..20V/abstract)
2. Michail et al. (2025, submitted), DOI/ADS link forthcoming
3. Bushouse et al. (2025): [https://zenodo.org/records/17101851](https://ui.adsabs.harvard.edu/abs/2023zndo...6984365B/abstract)

This work is supported by an NSF Astronomy and Astrophysics Postdoctoral Fellowship grant provided to JMM under award AST-2401752.
