"""
validation.py
=============
Validation metrics

"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


__all__ = ("pk_cross_3d", )

def pk_cross_3d(dim, L, bin_dim, data1, data2):
    """
    Validate power spectrum of 3 dimensional data

    Parameters
    ----------
    dim: int
        number of frequency bins
    L: float
        box size
    bin_dim: float
        a value
    data1: ndarray(float)
        data to validate
    data2: ndarray(float)
        data to compare against 

    Returns
    -------
    k_spec: ndarray(float)
        k values output
    power_spec: ndarray(float)
        output power spectrum

    """

    #dim = 128
    #bins = 30
    freqs = np.fft.fftfreq(dim)
    binning = (np.amax(freqs)-np.amin(freqs))*np.sqrt(2)/bin_dim

    pow_spec = np.zeros(bin_dim)
    cnt_spec = np.zeros(bin_dim)
    k_spec = np.zeros(bin_dim)
    k_matrix = np.zeros((dim,dim,dim))
    index_matrix = np.zeros((dim,dim,dim))



    #### FILE PATH ##################
    #path = filepath

    #### LOAD DATA #################
    #data = np.load(path)

    #img = data[0]
    img = data1
    img2= data2

    ######## FOURIER TRANSFORM OVERDENSITY FIELD ############
    overdensity = img - np.ones(img.shape,dtype=float)*np.average(img)
    #print(overdensity.shape)
    #TRY RFFT2 LATER
    fft = np.fft.fftn(overdensity)
    flat = fft.flatten()

    overdensity2 = img2 - np.ones(img2.shape,dtype=float)*np.average(img2)
    #print(overdensity.shape)
    #TRY RFFT2 LATER
    fft2 = np.fft.fftn(overdensity2)
    flat2 = fft2.flatten()

    print(flat.shape)

    freqs = np.fft.fftfreq(img.shape[0])
    #freqs.shape
    #print(freqs)

    dim = img.shape[0]
    print("image dimension = " + str(dim))


    ##### SETTING BINNING PARAMETERS ##########
    #bin_dim = 30
    binning = (np.amax(freqs)-np.amin(freqs))*np.sqrt(2)/bin_dim
    print("power spectrum binning="+str(binning))


    pow_spec = np.zeros(bin_dim)
    cnt_spec = np.zeros(bin_dim)
    #k_spec = np.zeros(bin_dim)
    k_spec = np.arange(bin_dim)*binning*2*np.pi
    #pow_comp = np.zeros(20)
    #cnt_comp = np.zeros(20)
    #k_comp = np.zeros(20)

    freq_ax=np.arange(bin_dim)*binning*np.pi*2
    #print(freq_ax)


    for i in range(0,dim):
        for j in range(0,dim):
            for k in range(0,dim):

              k_val = np.sqrt((freqs[i]-freqs[0])**2+(freqs[j]-freqs[0])**2+(freqs[k]-freqs[0])**2) #### k_val is the magnitude of |k|
              k_matrix[i][j][k]=k_val

              if int(k_val/binning)> (bin_dim-1) :
                    print("k_value="+ str(distance) + ",error!")

              else:
                    index_matrix[i][j][k]= int(k_val/binning)
                    #print(k_val)
                    #print(int(k_val/binning))
    #print(index_matrix)
    np.max(index_matrix)

    Mult_matrix = np.zeros((int(np.max(index_matrix))+1,dim,dim,dim))
    Fourier_matrix = fft2*np.conj(fft)
    B=np.zeros((dim,dim,dim))
    for i in range(int(np.max(index_matrix))+1):
        A= np.where(index_matrix==i,1,0)
        B+=A
        Mult_matrix[i,:,:,:] = A
        #component = np.sum(np.abs(np.multiply(A,Fourier_matrix)))
        component = np.abs(np.sum(np.multiply(A,Fourier_matrix)))

        pow_spec[i]+=component/(L*L*L*L*np.sum(A))

        cnt_spec[i]+=np.sum(A)
        #k_spec[i]+=i*binning*2*np.pi



    return k_spec,pow_spec



