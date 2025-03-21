"""
Data Processing of the laser measurements for a SINGLE channel
"""

import numpy as np
import scipy.signal as scsig

#=======================================================
# Automatic identification of pulse-like WST components
#=======================================================
class ACFAnalysis():
    def __init__(self):
        pass
    
    def comp_acf(self, _signal, _shift_start, _shift_end):
        # (0) Unit-energy normalization
        signal = _signal / np.linalg.norm(_signal, ord=2)
        # (1) ACF
        acf = np.correlate(signal, signal, mode='full')
        # Keep only the lag >= 0 (Cf: np.correlate doc)
        mid = len(acf) // 2
        acf = acf[mid:]
        # Take only the range of interest
        acf_roi = acf[_shift_start:_shift_end]
        # Normalize s.t. the max == 1
        acf_roi /= np.max(acf_roi)
        return acf_roi
    
    def ls_line_fit(self, x, y):
        # y = ax + b --> y = X \cdot v, with the coefficients v = [a, b]
        # (1) Zero-mean
        y_cent = y - np.mean(y)
        # (2) Array formatting
        X = np.stack((x, np.ones(x.shape[0]))).T
        # (3) Estimate the coefficients v via LS (pseudo-inverse): v_hat = X_pinv \cdot y
        [a_hat, b_hat] = np.dot(np.linalg.pinv(X), y_cent)
        return a_hat, b_hat+np.mean(y)
    
    
    def error_line_fit(self, _signal):
        # line fit
        a, b = self.ls_line_fit(np.arange(len(_signal)), _signal)
        # Model error
        error = (a*np.arange(len(_signal)) + b) - _signal
        return error

    def crest_factor(self, _signal):
        return np.max(np.abs(_signal))/np.linalg.norm(_signal, ord=2)

    def comp_criterion(self, _results, _shift_start, _shift_end):
        """
        Comment: 
            this criterion is selected empirically; I analyzed the ACFs and this particular quantity seems
            to easily threshold 
                * ACFs of the noise components do not have any peaks after time_shift = 0
                    => Their ACFs decay with the time shift -> easily to approximate with a descnding line
                * On the other hand, ACFs of the pulse-like components have multiple peaks even time_shift >> 0
                    => Their ACFs are very different from a line 
            Hence, I decided to used this quantity as the criterion to identify the pulse-like components
        """
        # (1) ACF
        acf = self.comp_acf(_results, _shift_start, _shift_end)
        # (2) Model error: ACF vs line fit
        error = self.error_line_fit(acf)
        # Crierion := inf_error / cf_acf 
        inf_error = np.abs(error).max()
        cf_acf = self.crest_factor(acf)
        return inf_error/cf_acf


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

    
    