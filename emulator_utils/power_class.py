import glob
import re 

class PowerSpectrum:
    """ 
    Class defining useful power spectrum routines 

    """
    def __init__(self):
        """
        initialize power spectrum class

        """
        return 
        
    def get_step_list_solo(self,file1):
        """
        get step for file

        Parameters
        ----------
        file1: str
           path to file

        Returns
        -------
        step: float 
            simulation step

        """
        if re.findall(r'\d+',file1)==[]:
            return 
        else:
            return re.findall(r'\d+',file1)[-1]

    def set_file_list(self, direc):
        """
        get full file list 

        Parameters
        ----------
        direc: str
            path to all files 

        Returns
        -------
        file_list: array(str)
            list of paths to all files

        """
        file_list = glob.glob(direc+'/*.pk.*')
        assert(re.findall(r'\d+',direc)==[])
        i=0
        while i<len(file_list):
            if (re.findall(r'\d+',file_list[i])[-1]=='000'):
                file_list.pop(i)
            else:
                i+=1
        self.file_list = file_list
        return file_list 

    def set_steps(self):
        """
        Get simulation steps for the power spectrum files 

        Returns
        --------
        steps: ndarray(int)
            simulation steps with power spectra

        """
        try:
            return self.steps
        except:
            self.steps = list(map(self.get_step_list_solo,self.file_list))
            return self.steps
        
    def set_data(self):
        """
        Get data from files 

        """
        from emulator_utils.read_data import readpowerspec
        k=[];pk=[];err=[];npairs=[]
        for file_i in self.file_list:
            k_tmp, pk_tmp, err_tmp, npairs_tmp = readpowerspec(file_i)
            k.append(k_tmp);pk.append(pk_tmp);err.append(err_tmp);npairs.append(npairs_tmp)
        self.k = k 
        self.pk = pk
        self.err = err
        self.npairs = npairs
        return 


    def set_conserved_quantities(self):
        """
        set conserved quantities

        """
        from emulator_utils.precompute_quantities import pk_ratio
        self.pk_ratio = pk_ratio(self.k,self.pk,self.steps)
        return self.pk_ratio

    def extend_k(self, lowk=True, highk=True):
        """
        extend low k using linear theory
        extend high k using pade approximants

        do this before pre-processing

        """
        return

    def pre_process_power(self):
        """
        take the log value and then scale?

        """ 
        return 
