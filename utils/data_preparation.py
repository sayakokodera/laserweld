import numpy as np
from kymatio.scattering1d import Scattering1D
from data_handling import LaserDataInteractorESRF
import data_processing as processor
from data_processing import WSTAnalysis901  

class DataPrepper901():
    def __init__(self):
        # Global params: known measurement parameters
        self.fs = 6.25* 10**6 #[Hz]
        self.dt = 1/self.fs
        self.T_pulse = 11.25*10**-3 #[s], interval of the pulse welding (NEW 250318: seems like 11.25 ms!!)

    def load(self, chNo):
        fileNo = 18
        interactor = LaserDataInteractorESRF()
        interactor.path_rel = '/Volumes/Sandisk_SD/Work/IZFP/Laser/03_ESRF/Code/measurements/ESRF_TU_Ilmenau_IZFP/Messdaten_QASS'
        # Ch.1
        interactor.fname = (chNo, fileNo) # ChNo, fileNo
        sig = interactor.load_data(ret_dt=False)
        sig -= sig.mean()
        sig /= np.max(np.abs(sig))
        return sig

    def preprocess(self, sig, fmin = 10*10**3, fmax = 400*10**3, ds = 4):
        """
        Parameters
        ----------
            fmin, fmax: freq. range for band-pass [Hz]
            ds: downsampling factor (int>=1)
        """
        # Register the global param
        self.ds = ds
        # Band-pass -> down sampling
        sig_bp = processor.apply_bandpass(sig, f_range=[fmin, fmax], dt=self.dt)
        sig_bp_ds = processor.downsample(sig_bp, ds, ret_freps=False)
        # Returne the normalized version
        return sig_bp_ds / np.max(np.abs(sig_bp_ds))

    def comp_wst(self, data, J=6, Q=(16,3)):
        """
        Comments on WST's hypreparmeters
        --------------------------------
            T = number of time samples
            2**J = "window length" of the moving window averaging + Downsampling operation 
            Q = nunber of wavelets per octave -> for the 1st and 2nd order scattering (Q1, Q2)
            
        Parameters
        ----------
            data: np.array, signal after preprocessing
            
        """
        T = data.shape[0] # number of time samples
        self.dt_wst = self.dt* self.ds* (2**J) # new time intervals after WST
        # Instantiate
        scattering = Scattering1D(J, T, Q)
        # WST + meta data
        Sx = scattering(data)
        meta = scattering.meta()
        self.order0 = np.where(meta['order'] == 0)
        self.order1 = np.where(meta['order'] == 1)
        self.order2 = np.where(meta['order'] == 2)
        # Return only the 1st and 2nd order WSTs
        return Sx[np.concatenate((self.order1[0], self.order2[0]))]


    def get_pulse_like_components(self, _Sx, thres=1.0):
        # (0) Params
        self.N_pulse = int(self.T_pulse/self.dt_wst)
        shift_start, shift_end = (self.N_pulse, int(8*self.N_pulse))
        L = 8000
        # (1) Compute the criterion values
        analyzer = WSTAnalysis901()
        res_criterion = np.apply_along_axis(
            func1d=analyzer.comp_criterion, 
            axis=1, 
            arr=_Sx[:, :L],
            _shift_start=shift_start,
            _shift_end=shift_end,
        )
        # (2) Threshold: res_criterion >= 1.0 are pulse-like
        return _Sx[(res_criterion>thres), ...]
        



    