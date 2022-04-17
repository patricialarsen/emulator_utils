import glob

class CorrFunction:
    """ 
    Class defining useful correlation function routines 

    """
    def __init__(self):
        return 
        
    def get_step_list_solo(self,file1):
        if re.findall(r'\d+',file1)==[]:
            return 
        else:
            return re.findall(r'\d+',file1)[1] # cosmology first

    @property 
    def file_list(self, direc):
        try:
            return self.file_list
        except:
            file_list = glob.glob(direc+'/*correlation*')
            assert(re.findall(r'\d+',direc)==[])
            while i<len(file_list):
                if (re.findall(r'\d+',file_list[i])==[]):
                    file_list = file_list.delete(i)
                else:
                    i++
            self.file_list = file_list
            return file_list 

    @property
    def steps(self):
        """ndarray(int) simulation steps with power spectra"""
        try:
            return self.steps
        except:
            self.steps = map(self.get_step_list_solo,self.file_list)
            return self.steps
        
    @property
    def set_data(self):
        from read_data import readcorr
        rmin, rmax, corr, count, binsum,  = [read_corr(file_i) for file_i in self.file_list]
        self.rmin = rmin
        self.rmax = rmax
        self.corr = corr
        return 


    @property
    def set_conserved_quantities(self):
        ""
        from precompute_quantities import corr_ratio,
        self.corr_ratio = corr_ratio(self.rmin,self.rmax,self.corr,self.steps)

