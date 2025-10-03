# Pipeline specific imports
import jwst
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline
from jwst.associations import asn_from_list as afl  # Tools for creating association files
from jwst.associations.lib.rules_level2_base import DMSLevel2bBase  # Definition of a Lvl2 association file
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
import crds
from jwst.residual_fringe.utils import fit_residual_fringes_1d as rf1d
import asdf

# Functions needed for commandas
import astropy.io.fits as fits
import astropy.units as u
import glob
import os
import numpy as np
import tqdm

class MRS_LightCurve_Calibration(object):
    """
    All-in-one class to calibrate JWST MIRI/MRS data, specifically tailored to timeseries measurements

    This class is meant to serve as a one-stop-shop to calibrate MRS light curves of general sources starting from either RATEINT (Level 1) or
    CALINT (Level 2) files, with the end of producing time-resolved flux and spectral measurements. You should always check your results! YMMV. 
    Adaptable for future functional additions such as extinction correction.
    
    Level 3 reduction path was developed by Tamojeet Roychowdhury given the current limitations of the Level 3 spectroscopy pipeline. 

    Attributes:
        working_dir             (str) : Working directory
        init_dir                (str) : Initial directory where __init__ was called

        _ch1_spec               (list of str) : List of spectral lines in ch1 to flag
        _ch2_spec               (list of str) : List of spectral lines in ch2 to flag
        _ch3_spec               (list of str) : List of spectral lines in ch3 to flag
        _ch4_spec               (list of str) : List of spectral lines in ch4 to flag
        _ch_dict                (dict) : Dictionary of spectral line lists above

        _spec2dict              (dict) : Dictionary of level 2 spectroscopy pipeline steps
        _spec3dict              (dict) : Dictionary of level 3 spectroscopy pipeline steps

        _jwst_version_          (str) : JWST pipeline version
        _pmap_version_          (str) : JWST pipeline CRDS version

        _build_4d_files         (bool) : Produce position x wavelength x time cubes
        _build_timeseries       (bool) : Product position x time cubes
        _wave_collapse_method   (str)  : Method to get per-channel flux
        _flag_line_chans        (bool) : Flag line channels before measuring continuum flux
        _flag_edge_chans        (bool) : Flag edge channels before measuring continuum flux
        _no_edge_chans          (int)  : Number of edge channels to flag

        _cleanup                (bool) : Delete intermediate files
        _clean_regex_files      (list of str) : Regex statements of files to clean 

        _lv2_asn                (str)  : Path to Level 2 ASN file for calibration
        _defringe               (bool) : Run 1D defringing step on S3D files
        _flag_unfringeable       (bool) : Flag sightlines where 1D defringe did not work?

        _correct_drift          (bool): Correct potential telescope drifts using predetermined values
        _drift_file             (str) : (Relative or absolute) location of drift rate file containing (RA_m, RA_b, Dec_m, Dec_b)

    Methods:
        __init__                :   Initialize object
        _write_log_cfg          :   Write pipeline config file to supress output to file
        _set_spectral_features  :   Add items to or replace line lists
        _run_level2_pipeline    :   Run level 2 pipeline on RATEINT files
        _run_level3_pipeline    :   Run level 3 pipeline on CAL files
        _calint_splitter        :   Split CALINT files into individual CAL files
        _build_4d_int_files     :   Build position x wavelength x time cubes
        _build_timeseries_files :   Build position x time cubes
        process                 :   Process the pipeline autonomously
    """
    def __init__(self, working_dir, spec2dict = None, spec3dict = None, build_4d_files = True, 
                 build_timeseries = True, wave_collapse_method = "mean", flag_line_chans = True,
                 flag_edge_chans = True, no_edge_chans = 10, cleanup = True,  
                 clean_regex_files = ['*crf*', '*x1d*', '*_cal_int*'], lv2_asn = None, 
                 defringe = False, flag_unfringeable = True, correct_drift = False, drift_file = None):
        """ Initialize an instance of the MRS_LightCurve_Calibration class

        Arguments:
            `working_dir`   :   str
                Relative or absolute location of Level 1/2 files
            `spec2dict` :   dict
                Dictionary of steps for Level 2 Spectroscopy pipeline. If none, use default values
            `spec3dict` :   dict
                Dictionary of steps for Level 2 Spectroscopy pipeline. If none, use default values
            `build_4d_files`    :   bool
                Build position x wavelength x time cubes
            `build_timeseries`  :   bool
                Build position x time cubes
            `wave_collapse_method`  :   str
                Method to obtain per-band flux measurement. Only "mean" is currently implemented
            `flag_line_chans`   :   bool
                Flag bright spectral lines before obtaining per-band flux measurement (i.e., continuum only)
            `flag_edge_chans`   :   bool
                Flag certain number of edge channels at beginning and end of spectrum before collapsing over wavelength
            `no_edge_chans` :   int
                Number of edge channels to flag
            `cleanup`   :   bool
                Delete intermediate files?
            `clean_regex_files` :   np.ndarray or list of strings
                rm-like regex file descriptors to remove
            `lv2_asn`   :   str
                Level 2 ASN file to use/edit as base during calibration
            `defringing`    :   bool
                Run 1D defringing step per spaxel in S3D cubes
            `flag_unfringeable` :   bool
                Flag spaxels where 1D defringing didn't work (if defringing == True)
            `correct_drift`     :   bool
                Correct drift in telescope position before building cubes in Level 3
            `drift_file`        :   str
                File name containing pointing/drift rate parameters to correct in Level 3 cubes
        """
        self.working_dir = working_dir
        self.init_dir    = os.getcwd()

        self._ch1_spec = ["5.335-5.345", "5.51-5.515", "6.907-6.915", "6.969-7.000", "7.335-7.358", "7.443-7.468", "7.497-7.512"]
        self._ch2_spec = ["8.018-8.038", "8.965-9.003"]
        self._ch3_spec = ["12.774-12.834", "15.511-15.579"]
        self._ch4_spec = ["18.675-18.747", "22.875-22.965", "25.959-26.013"]
        self._ch_dict = {'1' : self._ch1_spec, '2' : self._ch2_spec, '3' : self._ch3_spec, '4' : self._ch4_spec}

        self._spec2dict = spec2dict
        self._spec3dict = spec3dict

        self._jwst_version_ = jwst.__version__
        self._pmap_version_ = crds.get_context_name('jwst')

        self._build_4d_files = build_4d_files
        self._build_timeseries = build_timeseries
        self._wave_collapse_method = wave_collapse_method
        self._flag_line_chans = flag_line_chans
        self._flag_edge_chans = flag_edge_chans
        self._no_edge_chans   = no_edge_chans

        self._cleanup = cleanup
        self._clean_regex_files = clean_regex_files

        self._lv2_asn = lv2_asn

        self._defringe = defringe
        self._flag_unfringeable = flag_unfringeable

        self._correct_drift = correct_drift
        self._drift_file    = drift_file

        # HARD CODED: TODO MAKE THIS NOT
        self._integ_per_seg = np.array([18, 18, 18, 18, 13])

    def _write_log_cfg(self):
        """Produce a configuration file for the pipeline pass all JWST pipeline outputs to an external folder

        Arguments: none
        """
        with open("./log.control", 'w') as file:
            file.write("[*]\n")
            file.write("level = INFO\n")
            file.write("handler = append:pipeline.logs")
        file.close()

    def _set_spectral_features(self, line_list, chan, mode = "replace"):
        """Append or change the default line list ranges to flag before spectrally averaging

        Arguments:
            `line_list` :   np.ndarray or list
                List of line ranges
            `chan`  :   int
                Channel list to edit
            `mode`  :   string
                Replace or append line_list to internal channel list
        """
        if mode == "replace":
            self._ch_dict[str(chan)] = np.copy(line_list)

        elif mode == "append":
            self._ch_dict[str(chan)].extend(line_list)

    def _run_level2_pipeline(self):
        """ Run the RATEINT file(s) through the Level 2 Spectroscopy pipeline to produce CALINTS

        Arguments: None
        """
        # Get all RATEINT file names (likely only 1)
        ratefiles = [glob.glob(ending) for ending in ['*rateints.fits']][0]

        # If none, this was a waste of time, go back and start from level 3
        if len(ratefiles) == 0:
            print("Critical! No RATEINT files found -- assuming you're starting from Level 2 products instead")
            return 0
        
        # Otherwise tell the user that rateints were found
        else:
            print("Found %g RATEINT file(s)" % len(ratefiles))

        # Default Spec2 parameter steps but can also customize 
        if self._spec2dict == None:
            self._spec2dict = {}

        # Now go through all RATEINTS
        for ratefile in ratefiles:
            # Give good, unique product names
            product_name = ratefile.split("_rateints")[0]

            # If no Level 2 ASN file is given, write a default one 
            if self._lv2_asn == None:
                #Write out file that pipeline needs to run
                asn_file = afl.asn_from_list([ratefile], rule = DMSLevel2bBase, product_name = product_name)
                _, serialized = asn_file.dump()
                
                asn_outname = "./%s_Level2_asn.json" % product_name
                with open(asn_outname, 'w') as outfile:
                   outfile.write(serialized)
            
            else:
                asn_outname = self._lv2_asn

            # And run Spec2 pipeline on each file
            Spec2Pipeline.call(asn_outname, steps = self._spec2dict, save_results=True, output_dir = ".", logcfg = "./log.control")

    def _run_level3_pipeline(self):
        """ Run the splitted CAL files (from CALINT) through the Level 3 Spectroscopy pipeline

        Arguments: None
        """
        # If drift correction is required, open drift file
        # Pipeline design doesn't concanenate all data until 
        # last step, so it doesn't know the "true" integration
        # number (i.e., it's also based on segXXX number)
        if self._correct_drift:
            if self._drift_file is None:
                raise ValueError("You must supply an astrometry correction file to correct drifts!")
            else:
                drift_file = np.loadtxt(self._drift_file, delimiter=',')
                dra  = lambda nfile: drift_file[0] * nfile + drift_file[1]
                ddec = lambda nfile: drift_file[2] * nfile + drift_file[3]
                
        # Get all CAL file names
        calfiles = np.sort([glob.glob(ending) for ending in ['*cal_int*.fits']][0])

        # If none, this was a waste of time, go back and start from level 3
        if len(calfiles) == 0:
            print("Critical! No CAL files found, are you in the wrong directory?")
            return 0
        
        # Otherwise tell the user that rateints were found
        else:
            print("Found %g CAL file(s)" % len(calfiles))

        # Default Spec3 parameter steps but can also customize
        if self._spec3dict == None:
            self._spec3dict = {}

        # If Level 2 ASN is given, use this as a base for Level 3
        # These are variable names we'll need later
        if self._lv2_asn is not None:
            lv3_asn = 'spec3'.join(self._lv2_asn.split("spec2"))
            product_name_original = glob.glob("*rateints.fits")[0].split("_rateints.fits")[0]

        # Now go through all CAL files
        for num, calfile in enumerate(calfiles):
            print("Processing file %g / %g" % (num + 1, len(calfiles)))

            if self._lv2_asn is not None:
                os.system("cp %s %s" % (self._lv2_asn, lv3_asn))
                os.system("sed -i 's/rateints.fits/cal_int%g.fits/g' ./%s" % (num, lv3_asn))
                os.system("sed -i '0,/%s/{s/%s/%s/}' ./%s" % (product_name_original, product_name_original, product_name_original + "_cal_int%g" % num, lv3_asn))

            else:
                # Give good, unique product names
                product_name = calfile.split(".fits")[0]

                # Write out file that pipeline needs to run
                asn_file = afl.asn_from_list([calfile], rule = DMS_Level3_Base, product_name = product_name)
                _, serialized = asn_file.dump()
                
                lv3_asn = "./%s_Level3_asn.json" % product_name
                with open(lv3_asn, 'w') as outfile:
                    outfile.write(serialized)

            # Including drift corrections
            if self._correct_drift:
                # Integration within a single segment file
                intno = int(calfile.split("_int")[1].split(".fits")[0])
                # Segment number within a single observation
                segno = int(calfile.split("-seg")[1].split("_mirifu")[0])
                # Overall integration number within the observation
                integ = self._integ_per_seg[:(segno - 1)].sum() + intno

                # Build pointing correction file
                tree = {
                'units' : str(u.arcsec),
                'filename' : list(np.array([calfile])),
                'raoffset' : list(np.array([dra(intno)])),
                'decoffset': list(np.array([ddec(intno)])),
                }
                pointing_file = asdf.AsdfFile(tree)
                pointing_file.write_to("./DriftCorrection_%g.json" % integ)
                pointing_file.close()

                # Tell pipeline to apply the correction
                self._spec3dict['cube_build'] = {}
                self._spec3dict['cube_build']['offset_file'] = ("./DriftCorrection_%g.json" % integ)

            # And run Spec3 pipeline on each file -- then clear spec3dict back to beginning
            Spec3Pipeline.call(lv3_asn, steps = self._spec3dict, save_results=True, output_dir = ".", logcfg = "./log.control")

    def _calint_splitter(self, calint_files):
        """ Helper function that creates separate CAL files from each integration in a CALINT file to pass through the Level 3 pipeline

        Arguments:
            `calint_files`  :   np.ndarray or list
                List of CALINT files to split into separate CAL files
        """
        # Go through each file
        for filename in calint_files:
            calint_file = fits.open(filename)
            num_ints    = calint_file['SCI'].data.shape[0]

            # Go through each integration in that file (note: not always the same)
            for int_num in range(num_ints):
                output_name = ('cal_int%g' % int_num).join(filename.split("calints"))

                hdus = []

                # Primary HDU has no integration specific info so it's ok to copy directly
                primary_hdu = calint_file[0].copy()
                primary_hdu.header['INTSTART'] = 1 * int_num
                primary_hdu.header['INTEND'] = 1 * int_num
                primary_hdu.header['DATAMODL'] = "IFUImageModel" #Change otherwise the pipeline doesn't pass it through level 3
                hdus.append(primary_hdu)

                # Copy int_num-th data from each HDU and add it to the new file
                for i in range(1, 8):
                    copy_hdu = calint_file[i].copy()

                    # 4th HDU is table
                    if i != 4:
                        copy_hdu.data = 1 * copy_hdu.data[int_num, :, :]
                    hdus.append(copy_hdu)

                # [8] is ASDF HDU (needed?)
                hdus.append(calint_file[8].copy())

                # Write to new file
                hdus = fits.HDUList(hdus)
                hdus.writeto("./" + output_name, overwrite=True)

    def _build_4d_int_files(self, filenames):
        """ Helper function to build position x wavelength x time cubes

        Arguments:
            `filenames` :   np.ndarray or list
                List of S3D files (individual level 3 processed integrations of CALINT)
        """
        # Number of S3D files = number of integrations
        num_ints = len(filenames)
        filename_base = sorted(filenames)[0]
        filename_base = filename_base.split("_int0_")

        # Get integration time table from CALINTs file
        int_times_file = glob.glob(filenames[0].split("_cal_")[0] + "_calints.fits")[0]
        int_timetable = fits.open(int_times_file)['INT_TIMES']

        hdus = []

        # Go through each integration 
        for int_num in range(num_ints):
            s3d_filename = ("_int%g_" % int_num).join(filename_base)
            s3d_file = fits.open(s3d_filename)

            # Build SCI and ERR arrays from first integration
            if int_num == 0:
                # Primary HDU has no integration specific info so it's ok to copy directly
                primary_hdu = s3d_file[0].copy()
                primary_hdu.header['INTSTART'] = 1
                primary_hdu.header['INTEND'] = num_ints
                hdus.append(primary_hdu)
            
                sci_hdu = s3d_file['SCI'].copy()
                err_hdu = s3d_file['ERR'].copy()

                # Add a 4th dimension (time) if this is the first integration
                sci_hdu.data = sci_hdu.data[np.newaxis, :, :, :]
                err_hdu.data = err_hdu.data[np.newaxis, :, :, :]
            else:
                # Otherwise, copy other integrations along the time axis
                sci_hdu.data = np.concatenate((sci_hdu.data, s3d_file['SCI'].data.copy()[np.newaxis, :, :, :]), axis = 0)
                err_hdu.data = np.concatenate((err_hdu.data, s3d_file['ERR'].data.copy()[np.newaxis, :, :, :]), axis = 0)
        
        hdus.extend([sci_hdu, err_hdu, int_timetable])
        hdus = fits.HDUList(hdus)
        
        # Then save
        outname = filenames[0].split("_cal_")[0] + ("_calints_ch%s_4D.fits" % primary_hdu.header['CHANNEL'])
        hdus.writeto("./" + outname, overwrite=True)

    def _build_timeseries_files(self, filenames):
        """ Helper function to build position x time cubes to probe band-averaged flux 

        Arguments:
            `filenames` :   np.ndarray or list
                List of S3D files (individual CALINT integrations passed through level 3 pipeline) to process
        """
        # Number of CAL files = number of integrations
        num_ints = len(filenames)
        filename_base = sorted(filenames)[0]
        filename_base = filename_base.split("_int0_")

        # Get integration time table from CALINT
        int_times_file = glob.glob(filenames[0].split("_cal_")[0] + "_calints.fits")[0]
        int_timetable = fits.open(int_times_file)['INT_TIMES']

        hdus = []

        # Go through each S3D file
        for int_num in range(num_ints):
            s3d_filename = ("_int%g_" % int_num).join(filename_base)
            s3d_file = fits.open(s3d_filename)

            # Primary HDU has no integration specific info so it's ok to copy directly
            if int_num == 0:
                primary_hdu = s3d_file[0].copy()
                primary_hdu.header['INTSTART'] = 1
                primary_hdu.header['INTEND'] = num_ints
                hdus.append(primary_hdu)
            
            # Get copy of SCI and ERR HDUs
            sci_tab = s3d_file['SCI'].copy()
            err_tab = s3d_file['ERR'].copy()

            # Flag lines as specified from internal line database
            if self._flag_line_chans:
                
                # Build wavelength array from header info
                wavelengths = s3d_file['SCI'].header['CRVAL3'] + np.arange(1, s3d_file['SCI'].header['NAXIS3'] + 1) * s3d_file['SCI'].header['CDELT3']

                # Mask wavelength channels within that range
                for wave_range in self._ch_dict[primary_hdu.header['CHANNEL']]:
                    start, end = wave_range.split("-")
                    start, end = float(start), float(end)
                    flag_range = np.greater_equal(wavelengths, start) & np.less_equal(wavelengths, end)

                    print("Flagged %g channels in %s micron range" % (np.sum(flag_range), wave_range))
                    sci_tab.data[flag_range, :, :] = np.nan
                    err_tab.data[flag_range, :, :] = np.nan

            # Flag edge channels in case of roll-off
            if self._flag_edge_chans:
                sci_tab.data[:self._no_edge_chans, :, :] = np.nan
                err_tab.data[:self._no_edge_chans, :, :] = np.nan
                sci_tab.data[-self._no_edge_chans:, :, :] = np.nan
                err_tab.data[-self._no_edge_chans:, :, :] = np.nan

            # Use mean to obtain a single flux measurement and propagate error
            if self._wave_collapse_method == "mean":
                sci_tab.data = np.nanmean(sci_tab.data, axis = 0)
                err_tab.data = np.sqrt(np.nansum(err_tab.data**2.0, axis = 0)) / np.isfinite(err_tab.data).sum(axis = 0)

            # Add your favorite other methods here!!
            else:
                raise NotImplementedError(self._wave_collapse_method + "not yet implemented!")

            # Build 3D cube
            if int_num == 0:
                sci_hdu = sci_tab.copy()
                err_hdu = err_tab.copy()

                # Create new time axis
                sci_hdu.data = sci_hdu.data[np.newaxis, :, :]
                err_hdu.data = err_hdu.data[np.newaxis, :, :]

                #Fix headers on time axis
                sci_hdu.header['CDELT3'] = 1
                sci_hdu.header['CRVAL3'] = 1
                sci_hdu.header['CTYPE3'] = "INTEGRATION"
                sci_hdu.header['CUNIT3'] = ""

            # Otherwise just append along already-create time axis 
            else:
                sci_hdu.data = np.concatenate((sci_hdu.data, sci_tab.data.copy()[np.newaxis, :, :]), axis = 0)
                err_hdu.data = np.concatenate((err_hdu.data, err_tab.data.copy()[np.newaxis, :, :]), axis = 0)

        hdus.extend([sci_hdu, err_hdu, int_timetable])
        hdus = fits.HDUList(hdus)

        # Write the file out
        outname = filenames[0].split("_cal_")[0] + ("_calints_ch%s_3D.fits" % primary_hdu.header['CHANNEL'])
        hdus.writeto("./" + outname, overwrite=True)

    def _s3d_defringer(self, filenames):
        """ Helper function to complete 1D residual fringe correction per spaxel in an S3D file

        Arguments:
            `filenames` :   np.ndarray or list of strings
                List of S3D filenames to defringe
        """
        # Go through each S3D file
        for file in tqdm.tqdm(filenames):

            bad_pix_count = 0

            # We will overwrite the data in the file to make it less work later
            s3d_file = fits.open(file, mode='update')

            # Define the wavelength axis
            wavs = s3d_file['SCI'].header['CRVAL3'] + np.arange(0, s3d_file['SCI'].header['NAXIS3']) * s3d_file['SCI'].header['CDELT3']
            # Hold copy of non-fringed data
            corrected_data = s3d_file['SCI'].data.copy()

            # Now loop through each axis and do correction
            for xpix in range(corrected_data.shape[2]):
                for ypix in range(corrected_data.shape[1]):
                    # All NaN slice (likely toward edges), skip since there's no data
                    if np.all(~np.isfinite(s3d_file['SCI'].data[:, ypix, xpix])):
                        pass
                    else:
                        try:
                            # Only pass finite spectral points to defringer function
                            finite_filter = np.isfinite(s3d_file['SCI'].data[:, ypix, xpix])
                            corrected_data[finite_filter, ypix, xpix] = 1 * rf1d(s3d_file['SCI'].data[finite_filter, ypix, xpix], wavs[finite_filter], int(s3d_file[0].header['CHANNEL']))
                        except ValueError:
                            # If there's an issue, just flag the spaxel (if requested)
                            if self._flag_unfringeable:
                                corrected_data[:, ypix, xpix] = np.nan
                                bad_pix_count += 1
                            else:
                                pass
            print("%g / %g spaxels flagged: %0.1f percent" % (bad_pix_count, corrected_data.shape[2] * corrected_data.shape[1], 
                                                              100 * bad_pix_count / (corrected_data.shape[2] * corrected_data.shape[1])))
            # Replace uncorrected data with corrected one
            s3d_file['SCI'].data = corrected_data.copy()

            # Write corrections and close file
            s3d_file.flush()

    def process(self):
        """Run the JWST pipeline as configured in __init__

        Arguments: None
        """
        # Print out useful information about pipeline and PMAP products for documentation
        print(" ======================== ")
        print("JWST Calibration Pipeline Version = {}".format(self._jwst_version_))
        print("Using CRDS Context = {}".format(self._pmap_version_))
        print(" ======================== ")

        # Go into directory where files are
        print("Moving into working directory: %s" % self.working_dir)
        os.chdir(self.working_dir)
        print(" ======================== ")
        # Write out config file to pipe all JWST log information into a file rather than STDOUT
        self._write_log_cfg()
        
        # Assume we start from Level 2 RATEINT files
        print("Beginning Level 2 Processing...")
        self._run_level2_pipeline()
        print("Level 2 Processing Complete!")
        print(" ======================== ")

        
        # Now we need to split the CALINT files into CAL files to trick the
        # Level 3 pipeline into calibrating them
        print("Splitting CALINTs to separate CAL files")
        self._calint_splitter(glob.glob("./*calints.fits"))
        print("Finished splitting up CALINT files")
        print(" ======================== ")
       
        # Pass separated cal files through Level 3 pipeline
        print("Beginning Level 3 Processing...")
        self._run_level3_pipeline()
        print("Level 3 Processing Complete!")
        print(" ======================== ")
        
        # Run 1D defringing step on data before building cubes (if requested)
        if self._defringe:
            print("Completing 1D defringing on spaxels")
            print("!! Warning !! This may take an excessive amount of time -- go get a coffee?")
            ch1_s3d_ints, ch2_s3d_ints = glob.glob("*cal_int*ch1*s3d.fits"), glob.glob("*cal_int*ch2*s3d.fits")
            ch3_s3d_ints, ch4_s3d_ints = glob.glob("*cal_int*ch3*s3d.fits"), glob.glob("*cal_int*ch4*s3d.fits") 

            print("Found %g channel 1 S3D file(s)" % len(ch1_s3d_ints))
            print("Found %g channel 2 S3D file(s)" % len(ch2_s3d_ints))
            print("Found %g channel 3 S3D file(s)" % len(ch3_s3d_ints))
            print("Found %g channel 4 S3D file(s)" % len(ch4_s3d_ints))
            print(" ======================== ")

            if len(ch1_s3d_ints) != 0:
                self._s3d_defringer(ch1_s3d_ints)
            if len(ch2_s3d_ints) != 0:
                self._s3d_defringer(ch2_s3d_ints)
            if len(ch3_s3d_ints) != 0:
                self._s3d_defringer(ch3_s3d_ints)
            if len(ch4_s3d_ints) != 0:
                self._s3d_defringer(ch4_s3d_ints)

            print("Defringing Complete!")
        print(" ======================== ")

        # Build 4D files (RA x Dec x Wavelength x Time) if requested
        if self._build_4d_files:
            print("Building Position x Spectral x Time Cubes...")
            ch1_s3d_ints, ch2_s3d_ints = glob.glob("*cal_int*ch1*s3d.fits"), glob.glob("*cal_int*ch2*s3d.fits")
            ch3_s3d_ints, ch4_s3d_ints = glob.glob("*cal_int*ch3*s3d.fits"), glob.glob("*cal_int*ch4*s3d.fits") 

            print("Found %g channel 1 S3D file(s)" % len(ch1_s3d_ints))
            print("Found %g channel 2 S3D file(s)" % len(ch2_s3d_ints))
            print("Found %g channel 3 S3D file(s)" % len(ch3_s3d_ints))
            print("Found %g channel 4 S3D file(s)" % len(ch4_s3d_ints))
            print(" ======================== ")

            if len(ch1_s3d_ints) != 0:
                self._build_4d_int_files(ch1_s3d_ints)
            if len(ch2_s3d_ints) != 0:
                self._build_4d_int_files(ch2_s3d_ints)
            if len(ch3_s3d_ints) != 0:
                self._build_4d_int_files(ch3_s3d_ints)
            if len(ch4_s3d_ints) != 0:
                self._build_4d_int_files(ch4_s3d_ints)

            print("4D Cubes Complete!")
        print(" ======================== ")

        # Build 3D files (RA x Dec x Time) if requested
        if self._build_timeseries:
            print("Building Position x Time Cubes...")
            ch1_s3d_ints, ch2_s3d_ints = glob.glob("*cal_int*ch1*s3d.fits"), glob.glob("*cal_int*ch2*s3d.fits")
            ch3_s3d_ints, ch4_s3d_ints = glob.glob("*cal_int*ch3*s3d.fits"), glob.glob("*cal_int*ch4*s3d.fits") 

            print("Found %g channel 1 S3D file(s)" % len(ch1_s3d_ints))
            print("Found %g channel 2 S3D file(s)" % len(ch2_s3d_ints))
            print("Found %g channel 3 S3D file(s)" % len(ch3_s3d_ints))
            print("Found %g channel 4 S3D file(s)" % len(ch4_s3d_ints))
            print(" ======================== ")

            if len(ch1_s3d_ints) != 0:
                self._build_timeseries_files(ch1_s3d_ints)
            if len(ch2_s3d_ints) != 0:
                self._build_timeseries_files(ch2_s3d_ints)
            if len(ch3_s3d_ints) != 0:
                self._build_timeseries_files(ch3_s3d_ints)
            if len(ch4_s3d_ints) != 0:
                self._build_timeseries_files(ch4_s3d_ints)

            print("3D Cubes Complete!")
        print(" ======================== ")

        # Clean the mess of intermediate files to save space
        if self._cleanup:
            print("Cleaning this mess of files...")
            for ext in self._clean_regex_files:
                os.system("rm ./%s" % ext)
            print("Done!")
        print(" ======================== ")
        
        # Go back to original directory once finished
        print("Pipeline complete, happy hunting! Moving back to original directory")
        os.chdir(self.init_dir)
        print("Directory is now: ", os.getcwd())
        print(" ======================== ")
        

def combine_segments(project_id, exec_id, band, combine_4d = True, combine_3d = True, save_dir = "./", search_dir = "./"):
    """ Helper function to combine individual 3D and 4D segments (built in pipeline) into single project cubes

    Basic download from MAST has folders with JWST codes:
        jwPPPPPEEE*
            PPPPP: project_id code below
            EEE  : exec_id code below

    Arguments:
        `project_id`    :   str
            Up to 5 digit project code
        `exec_id`   :   str
            Up to 3 digit execution code (I don't actually know what this is)
        `band`  :   str
            IFU band ("short" or "long")
        `combine_4d`    :   bool
            Combine 4D cubes (position x wavelength x time) into single FITS file?
        `combine_3d`    :   bool  
            Combine 3D cubes (position x time) into single FITS file?
        `save_dir`  :   str
            Path to saving directory
        `search_dir`    :   str
            Path for glob to search for 3D and 4D segment FITS files
    """
    # Pad project and exec IDs to proper(?) length
    project_id    = "0" * (5 - len(project_id)) + project_id
    exec_id    = "0" * (3 - len(exec_id)) + exec_id

    # Build glob folder name to search for data
    precode = "jw" + project_id + exec_id

    # Define channel numbers for each band and append to data path
    chans = {'short' : ['1', '2'], 'long' : ['3', '4']}
    precode += ("*%s*/" % band)

    # Go through each channel in band
    for channel in chans[band]:
        # Then go through each segment in project/exec ID found
        for seg, folder in enumerate(sorted(glob.glob(search_dir + "/" + precode))):
            # Combine data along time axis for 3D
            if combine_3d:
                combined_3d_file = glob.glob("%s/*ch%s*3D*" % (folder, channel))[0]
                ddd_segment_file = fits.open(combined_3d_file)

                if seg == 0:
                    primary_hdu_3d = ddd_segment_file[0].copy()

                    sci_hdu_3d = ddd_segment_file['SCI'].copy()
                    err_hdu_3d = ddd_segment_file['ERR'].copy()
                    tab_hdu_3d = ddd_segment_file['INT_TIMES'].copy()

                else:
                    sci_hdu_3d.data = np.concatenate((sci_hdu_3d.data, ddd_segment_file['SCI'].data), axis = 0)
                    err_hdu_3d.data = np.concatenate((err_hdu_3d.data, ddd_segment_file['ERR'].data), axis = 0)
                    tab_hdu_3d.data = np.append(tab_hdu_3d.data, ddd_segment_file['INT_TIMES'].data)

            # Combine data along time axis for 4D
            if combine_4d:
                combined_4d_file = glob.glob("%s/*ch%s*4D*" % (folder, channel))[0]
                dddd_segment_file = fits.open(combined_4d_file)

                if seg == 0:
                    primary_hdu_4d = dddd_segment_file[0].copy()

                    sci_hdu_4d = dddd_segment_file['SCI'].copy()
                    err_hdu_4d = dddd_segment_file['ERR'].copy()
                    tab_hdu_4d = dddd_segment_file['INT_TIMES'].copy()

                else:
                    sci_hdu_4d.data = np.concatenate((sci_hdu_4d.data, dddd_segment_file['SCI'].data), axis = 0)
                    err_hdu_4d.data = np.concatenate((err_hdu_4d.data, dddd_segment_file['ERR'].data), axis = 0)
                    tab_hdu_4d.data = np.append(tab_hdu_4d.data, dddd_segment_file['INT_TIMES'].data)
        
        # Save 3D and 4D files in save_dir
        if combine_3d:
            outfile_3d = fits.HDUList([primary_hdu_3d, sci_hdu_3d, err_hdu_3d, tab_hdu_3d])
            #                                                             This stripping gets rid of search directory path at beginning of glob filename
            combined_file_name = save_dir + combined_3d_file.split("_")[0][len(search_dir):] + "_AllSeg_" + "mirifu" + band + ("_ch%s_3D.fits" % channel)
            outfile_3d.writeto(combined_file_name)

        if combine_4d:
            outfile_4d = fits.HDUList([primary_hdu_4d, sci_hdu_4d, err_hdu_4d, tab_hdu_4d])
            combined_file_name = save_dir + combined_4d_file.split("_")[0][len(search_dir):] + "_AllSeg_" + "mirifu" + band + ("_ch%s_4D.fits" % channel)
            outfile_4d.writeto(combined_file_name)
