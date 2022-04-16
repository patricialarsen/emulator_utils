"""
need to import pade libraries, so far this is what we were doing for the extension of pks

"""

import numpy as np
from scipy.interpolate import*
from numba import jit

@jit(nopython=True)
def numb_extrap1d(x, xs, ys):
    ptr=0
    y=x*0.0
    for i in range(len(x)):
        if x[i]<xs[0]:
            y[i]=ys[0]+(x[i]-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        else:
            while (ptr!=len(xs)) & (x[i]>=xs[ptr]):
                ptr=ptr+1
            if ptr==len(xs):
                y[i]=ys[-1]+(x[i]-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            else:
                y[i]=ys[ptr]+ (x[i]-xs[ptr])*(ys[ptr]-ys[ptr-1])/(xs[ptr]-xs[ptr-1])

    return y



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
    


