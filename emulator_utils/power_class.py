import glob

class PowerSpectrum:
    """ 
    Class defining useful power spectrum routines 

    """
    def __init__(self, direc):
        file_list = glob.glob(direc+'/*.pk.*')
        assert(re.findall(r'\d+',direc)==[])
        while i<len(file_list):
            if (re.findall(r'\d+',file_list[i])==[]):
                file_list = file_list.delete(i)
            else:
                i++
        self.file_list = file_list 


        
    def get_step_list_solo(self,file1)
        if re.findall(r'\d+',file1)==[]:
            return 
        else:
            return re.findall(r'\d+',file1)[-1]

    @property 
    def file_list(self, direc):
        try:
            return self.file_list
        except:
            file_list = glob.glob(direc+'/*.pk.*')
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



