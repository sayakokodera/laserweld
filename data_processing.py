#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Processing of the laser measurements for a SINGLE channel
"""

import numpy as np
import scipy.signal as scsig
from scipy.ndimage import median_filter


def process_laser_data(sig, f_range, dt, w_duration):
    """ 
    Data processing function using the util class defined below.
    Steps:
        (1) Bandpass
        (2) Take envelope
        (3) Smoothing via moving averaging 
        
    Prameters
    ---------
        sig: array(N), vector!
            Signal to process
        f_range: = [f_min, f_max] in [Hz]
            Specifying the upper and lower frequency bound
        dt: float [s]
            Temporal interval 
        w_duration: float [s]
            Window duration required for smoothing
            (= longer the duration, smoother the signal, 
            yet the resolution is reduced as a tradeoff)
    """
    util = ProcessingUtil()
    # (1) BP
    sig_bp = util.apply_bandpass(sig, f_range, dt)
    # (2) Take the envelope
    env = np.abs(scsig.hilbert(sig_bp))
    # (3) Smoothing
    # Convert the window duration into the window length
    l_window = int(w_duration/dt)
    # Smoothing via moving average
    env_smt = util.moving_average(env, l_window)
    return env_smt



class ProcessingUtil():

    # ----------------------------
    # Butterworth LP
    # ----------------------------
    @staticmethod
    def apply_lowpass(sig, cutoff, dt, order=5):
        b, a = scsig.butter(N=order, Wn=cutoff, 
                            fs=1/dt, btype='low')
        # Apply the filter
        sig_lp = scsig.filtfilt(b, a, sig, axis = -1)
        return sig_lp
    
    # ----------------------------
    # Butterworth BP
    # ----------------------------
    @staticmethod
    def apply_bandpass(sig, f_range, dt, order=5):
        # Unpack
        lowcut, highcut = f_range
        # Filter coefficitns
        b, a = scsig.butter(N=order, Wn=[lowcut, highcut], 
                            fs=1/dt, btype='bandpass')
        # Apply the filter
        sig_bp = scsig.filtfilt(b, a, sig, axis = -1)
        return sig_bp

    # ----------------------------
    # Downsampling
    # ----------------------------
    @staticmethod
    def downsample(sig_proc, ds, ret_freps=False):
        # Downsampling in frequency domain
        # FT 
        freps =  np.fft.rfft(sig_proc)
        # Downsample by taking only the first N/ds
        # !!! amplitude needs to be adjusted because we virtually created the replicas for ds times in the
        # same frequency range (due to periodicity in freq. domain)
        # ---> (e.g.) bw=1kHz, ds=4: before={energy=1.0 in bw=1kHz}, after={energy=ds*1.0=4.0 in bw=1kHz}
        end = int((len(freps)-1)/ds)
        freps_new = freps[:end]/np.sqrt(ds)
        # IFT
        sig_new = np.fft.irfft(freps_new)
        if ret_freps == True:
            return sig_new, freps_new
        else:
            return sig_new
        
    # ----------------------------
    # Smoothing via moving window averaging
    # ----------------------------
    @staticmethod
    def moving_average(sig, L):
        """
        Parameters
        ----------
        signal : np.ndarray
            Time series.
        L : int (odd!)
            window length.
        """
        # Convolve with a sliding rectangular window
        sig_conv = np.convolve(sig, np.ones(L), 'same')
        # Averaging array: 
        # Elements of the valid region = 1/L
        arr = np.ones(len(sig)) / L
        # Elements outside the valid region
        for idx in range(len(arr)):
            # Left-hand side
            if idx < int(L/2):
                arr[idx] = 1 / (int(L/2) + 1 + idx)
            # Right-hand side
            elif idx > (len(arr) - int(L/2)) - 1:
                arr[idx] = 1 / (int(L/2)  + len(arr) - idx)

        # Moving average = element wise multiplication with arr
        return sig_conv* arr

    # ----------------------------
    # Outlier detection using MAD
    # ----------------------------
    @staticmethod
    def is_outlier(sig, l_window, rel_threshold):
        # Compute median
        median = median_filter(sig, l_window)
        # Compute MAD
        mad = median_filter(np.abs(sig - median), size = l_window)
        # True-False array: True, if the deviation to the median is larger than the scaled MAD
        tf = np.abs(sig - median) > rel_threshold* mad
        return (tf).astype(int), median, mad
    
    