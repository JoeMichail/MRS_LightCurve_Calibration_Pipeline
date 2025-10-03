from MRS_LC_Calibration import MRS_LightCurve_Calibration

# Instantiate light curve processing object
processor = MRS_LightCurve_Calibration("./jw04572003001*")

# Go through each folder starting with above code and calibrate through Level 3
processor.process()
