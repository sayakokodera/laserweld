# Laser data interactor for the ESRF data

import numpy as np


class LaserDataInteractorESRF():
    
    def __init__(self):
        pass
    
    @property
    def path_rel(self):
        return self._path_rel
    
    @path_rel.setter
    def path_rel(self, path_rel):
        self._path_rel = path_rel
    
    @property
    def fname(self):
        return self._fname
    
    @fname.setter
    def fname(self, info):
        """
        Parameter
        ---------
            info: tuple of size 2, containing the following elemtns:
                chNo: channel number (int)
                qassNo: QASS data ID (int)
        """
        (chNo, qassNo) = info
        self._fname = f'{self.path_rel}/Ch.{chNo}/' + \
                      f'Messungen_TU_Ilmenau_ESRF_Process_{str(qassNo).zfill(3)}_Ch{chNo}_SIG_Raw_compress_1'
        
    
    def load_data(self, ret_dt=True):
        data = np.fromfile(f'{self.fname}.bin', dtype = '<i2').astype(float)
        if ret_dt == False:
            return data
        # Duration & sampling rate
        else:
            T = self.load_duration() #[s]
            dt = T / len(data) # [s]
            return (data, dt)
            
    
    def load_duration(self):
        f = open(f'{self.fname}.txt', 'r')
        content = f.read()
        # Split with the lines
        content_list = content.split("\n")
        # Start time = index 2
        start = float(content_list[2].split(":")[1])
        # End time = index 3
        end = float(content_list[3].split(":")[1])
        # return = duration
        return end - start