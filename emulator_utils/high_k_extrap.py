"""
need to import pade libraries, so far this is what we were doing for the extension of pks

"""

import numpy as np
from scipy.interpolate import*

def pade_coeffs (n, l ,m ,xis, rhs): #n: number of points
    
    a_coeffs=[]
    b_coeffs=[]
    
    for i in range(0, n):
        a_coeffs.append([])
        for j in range(0, l+1):
            a_coeffs[i].append(xis[i]**j)
            
    
    for i in range(0,n):
        b_coeffs.append([])        
        
        for j in range(0, m):
            t=-(xis[i]**(j+1))*rhs[i]
            b_coeffs[i].append(t)
        
    a_coeffs=np.array(a_coeffs)
    b_coeffs=np.array(b_coeffs)

    A=np.array(np.concatenate((a_coeffs, b_coeffs), axis=1))
    soln=np.linalg.solve(A, rhs)
    pcs=soln[:l+1]
    qcs=soln[l+1:]
    p_c=np.flipud(pcs)
    q_c= np.concatenate(([1], qcs))
    return p_c, np.flipud(q_c)


def n_point_pade (x, p_c, q_c):
    return (np.poly1d(p_c)(x))/(np.poly1d(q_c)(x))


def extend_pk(k,pk,k_pnts,k_new)
    npts = len(k_pts)
    idx_arr = np.array([np.where(k==ktmp)[0] for ktmp in k_pnts])
    pts = k[idx_arr]
    rhs = pk[idx_arr]
    # double check the 0,2 here - power level of the approximant
    p,q = pade_coeffs(npts,0,2,pts,rhs)
    kmax = np.max(k)
    k_ext = k_new[k_new>kmax]
    pk_ext = n_point_pade(k_ext, p, q)
    pk_int = interp1d(np.log10(k),np.log10(pk))
    pk_new = np.zeros_like(k_new)
    pk_new[k_new<=kmax]=10**pk_int(np.log10(k_new[k_new<=kmax]))
    pk_new[k_new>kmax] = pk_ext
    return pk_new
    

"""
    def get_pk_array_emu(cosmo,file_list,min_k,nk):
    zl_array = []
    pk_array = []
    lk_array = []

    for i in range(len(file_list)):
        # high k (small scale)
        filename = file_list[i]
        k_tmp,pk_tmp = np.loadtxt(filename, dtype="double", usecols=(0,1),unpack=True)
        pts=[k_tmp[480],k_tmp[521], k_tmp[564]]
        rhs=[pk_tmp[480],pk_tmp[521], pk_tmp[564]]
        p,q=pade_coeffs(3,0,2, pts, rhs)
        k_ext=np.linspace(8.5692+0.034, 250, 500)
        pk_ext=n_point_pade(k_ext, p, q)
        k_tmp=np.append(k_tmp, k_ext)
        pk_tmp=np.append(pk_tmp, pk_ext)
        zl_tmp = np.double(file_list[i].split('_')[-5])
        dzl = np.double(file_list[i].split('_')[-4])

        # large scales (small k)
        k_lo = np.logspace(np.log10(min_k),np.log10(np.min(k_tmp)-1.e-5),nk)
        linear = pyccl.linear_matter_power(cosmo,k_lo,1./(1.+zl_tmp))
        kfin = np.concatenate((k_lo,k_tmp))
        pkfin = np.concatenate((linear,pk_tmp))
        zl_array.append(zl_tmp)
        lk_array.append(kfin)
        pk_array.append(pkfin)
    return zl_array,lk_array,pk_array
"""
