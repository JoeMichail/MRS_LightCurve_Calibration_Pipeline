from MRS_LC_Calibration import MRS_LightCurve_Calibration

# Instantiate light curve processing object
processor = MRS_LightCurve_Calibration("./JWST/jw*")

# Go through each folder starting with above code and calibrate through Level 3
processor.process()
