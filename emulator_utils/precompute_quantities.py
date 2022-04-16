"""
precompute_quantities.py
========================
Pre-compute conserved quantities

"""

def pk_ratio(k_vals,pk_vals,steps):
    nsteps = len(steps)
    assert([k_vals[i]==k_vals[j] for i,j in range(nsteps)].all())
    base_step = np.max(steps)
    base_idx = np.where(steps==base_step)[0]
    return pk_vals/pk_vals[base_idx]


def corr_ratio(r_min, r_max, corr_vals, steps):
    nsteps = len(steps)
    assert([r_min[i]==r_min[j] for i,j in range(nsteps)].all())
    assert([r_max[i]==r_max[j] for i,j in range(nsteps)].all())
    base_step = np.max(steps)
    base_idx = np.where(steps==base_step)[0]
    return corr_vals/corr_vals[base_idx]


