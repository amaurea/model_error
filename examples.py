import numpy as np, toy, os, sys
from matplotlib import pyplot

def get_outside(map, R):
	pos  = np.array(map.shape)/2
	r2   = np.sum((np.mgrid[:map.shape[0],:map.shape[1]]-pos[:,None,None])**2,0)
	mask = r2>R**2
	return map[mask]

def demean(m): return m - 0.25*(m[0,0]+m[-1,0]+m[0,-1]+m[-1,-1])

def measure_leakage(map, dmap, R=0):
	map, dmap = demean(map), demean(dmap)
	return np.max(np.abs(get_outside(dmap, R)))/np.max(np.abs(map))

def dump_tod(fname, data, map, pmat):
	ndim  = pointing.shape[-1]
	model = pmat.map2tod(map)
	print("Writing %s" % fname)
	np.save(fname, np.concatenate([
		pointing.reshape(-1,ndim), data.reshape(-1,1), model.reshape(-1,1)],-1))

def dump_map(fname, data, map, pmat, pwhite):
	model = pmat.map2tod(map)
	rmap  = toy.solve_white((data-model)**2, pwhite)**0.5
	dmap  = map - toy.solve_white(data, pwhite)
	print("Writing %s" % fname)
	np.save(fname, np.array([map, rmap, dmap]))
	leak_full = measure_leakage(map, dmap)
	leak_far  = measure_leakage(map, dmap, 8)
	print("Leakage %-50s %15.7e %15.7e" % (fname, leak_full, leak_far))
	sys.stdout.flush()

def run_1d(prefix, shape, pointing, data):
	# Solve using standard nearest neighbor and a white noise model
	pmat     = toy.PmatNearest(pointing, shape)
	omap     = toy.solve_white(data, pmat)
	dump_tod(prefix + "_1d_uncorr_tod.npy", data, omap, pmat)

	# Uncorrelated linear interpolation
	pmat_lin = toy.PmatSplinePixell(pointing, shape, order=1)
	omap     = toy.solve_brute(data, pmat_lin)
	dump_tod(prefix + "_1d_uncorr_lin_tod.npy", data, omap, pmat_lin)

	# Uncorrelated cubic spline interpolation
	pmat_cubic = toy.PmatSplinePixell(pointing, shape, order=3)
	omap       = toy.solve_brute(data, pmat_cubic)
	dump_tod(prefix + "_1d_uncorr_cubic_tod.npy", data, omap, pmat_cubic)

	# Solve using correlated noise model instead
	nmat     = toy.NmatOneoverf(data.shape[-1], nsub=nsub)
	omap     = toy.solve_brute(data, pmat, nmat)
	dump_tod(prefix + "_1d_corr_tod.npy", data, omap, pmat)

	# Solve using linear interpolation
	pmat_lin = toy.PmatSplinePixell(pointing, shape, order=1)
	omap     = toy.solve_brute(data, pmat_lin, nmat)
	dump_tod(prefix + "_1d_corr_lin_tod.npy", data, omap, pmat_lin)

	# Solve using cubic spline interpolation
	pmat_cubic = toy.PmatSplinePixell(pointing, shape, order=3)
	omap     = toy.solve_brute(data, pmat_cubic, nmat)
	dump_tod(prefix + "_1d_corr_cubic_tod.npy", data, omap, pmat_cubic)

	# Iterative linear (approximation to proper linear)
	omap     = toy.solve_iterative(data, pmat, pmat_lin, nmat)
	dump_tod(prefix + "_1d_corr_itlin_tod.npy", data, omap, pmat_lin)

	# Iterative cubic (approximation to proper cubic)
	omap     = toy.solve_iterative(data, pmat, pmat_cubic, nmat)
	dump_tod(prefix + "_1d_corr_itcubic_tod.npy", data, omap, pmat_cubic)

	# local white model
	mask     = toy.build_mask_disk(pointing, pos, 5)
	omap     = toy.solve_mask_white(data, pmat, mask, nmat)
	dump_tod(prefix + "_1d_corr_srcwhite_tod.npy", data, omap, pmat)

	# local extra degrees of freedom
	omap     = toy.solve_cg_mask_persamp(data, pmat, mask, nmat)
	dump_tod(prefix + "_1d_corr_srcsamp_tod.npy", data, omap, pmat)

def run_2d(prefix, shape, pointing, data, amp=0, pos=[0,0], sigma=1):
	# Solve using standard nearest neighbor and a white noise model
	pmat     = toy.PmatNearest(pointing, shape)
	omap     = toy.solve_white(data, pmat)
	dump_map(prefix + "_2d_uncorr_map.npy", data, omap, pmat, pmat)

	# Correlated noise model
	nmat     = toy.NmatOneoverf(data.shape[-1], nsub=nsub)
	omap     = toy.solve_cg(data, pmat, nmat)
	model    = pmat.map2tod(omap)
	dump_map(prefix + "_2d_corr_map.npy", data, omap, pmat, pmat)

	# bilinear interpolation
	pmat_lin = toy.PmatSplinePixell(pointing, shape, order=1)
	omap     = toy.solve_cg(data, pmat_lin, nmat)
	dump_map(prefix + "_2d_corr_lin_map.npy", data, omap, pmat_lin, pmat)

	# bicubic interpolation
	pmat_cubic = toy.PmatSplinePixell(pointing, shape, order=3)
	omap     = toy.solve_cg(data, pmat_cubic, nmat)
	dump_map(prefix + "_2d_corr_cubic_map.npy", data, omap, pmat_cubic, pmat)

	# local white model
	mask     = toy.build_mask_disk(pointing, pos, 5)
	omap     = toy.solve_mask_white(data, pmat, mask, nmat)
	dump_map(prefix + "_2d_corr_srcwhite_map.npy", data, omap, pmat, pmat)

	# local extra degrees of freedom
	omap     = toy.solve_cg_mask_persamp(data, pmat, mask, nmat)
	dump_map(prefix + "_2d_corr_srcsamp_map.npy", data, omap, pmat, pmat)

	# source cutting
	omap     = toy.solve_cg_mask(data, pmat, mask, nmat)
	dump_map(prefix + "_2d_corr_srccut_map.npy", data, omap, pmat, pmat)

	# Iterative linear (approximation to proper linear)
	omap     = toy.solve_iterative(data, pmat, pmat_lin, nmat)
	dump_map(prefix + "_2d_corr_itlin_map.npy", data, omap, pmat_lin, pmat)

	# Iterative cubic (approximation to proper cubic)
	omap     = toy.solve_iterative(data, pmat, pmat_cubic, nmat)
	dump_map(prefix + "_2d_corr_itcubic_map.npy", data, omap, pmat_cubic, pmat)

	# source subtraction
	signal_src= toy.build_signal_src(pointing, pos, amp=amp, sigma=sigma)
	omap     = toy.solve_cg(data-signal_src, pmat, nmat)
	omap    += toy.solve_white(signal_src, pmat)
	dump_map(prefix + "_2d_corr_srcsub_map.npy", data, omap, pmat, pmat)

np.random.seed(1)
toy.mpl_setdefault("planck")
toy.mkdir("examples")

# 1d examples. 100 pixels with 11 samples per pixel, simple point source
# in the middle with gaussian profile. Single TOD.
shape    = np.array([100])
pos      = shape/2
nsub     = 11
pointing = toy.build_pointing_1d(shape, nsub=nsub)
data     = toy.build_signal_src(pointing, pos, amp=1, sigma=1.0)
run_1d("examples/src", shape, pointing, data)

# 2d examples. 81x81 pixels. Simple point source in the middle with gaussian profile.
# 2 tods: 1 vertical and 1 horizontal. 3 samples per pixel in each direction.
shape    = np.array([81,81])
pos      = shape/2
nsub     = 3
amp      = 2e4
sigma    = 1.0
pointing = toy.build_pointing_2d(shape, nsub=nsub, ntod=2)
# Consdier src, cmb and noise
signal_src  = toy.build_signal_src(pointing, pos, amp=amp, sigma=sigma)
cmb_highres = toy.build_cmblike_map(shape, nsub=nsub, sigma=sigma)
signal_cmb  = toy.build_signal_from_map(pointing, cmb_highres, nsub=nsub)
nmat        = toy.NmatOneoverf(signal_src.shape[-1], nsub=nsub)
noise       = nmat.sim(*signal_src.shape)

run_2d("examples/src", shape, pointing, signal_src, pos=pos, amp=amp, sigma=sigma)
run_2d("examples/cmb", shape, pointing, signal_cmb, pos=pos)
run_2d("examples/src_cmb", shape, pointing, signal_src+signal_cmb, pos=pos, amp=amp, sigma=sigma)

run_2d("examples/src_noise", shape, pointing, signal_src+noise, pos=pos, amp=amp, sigma=sigma)
run_2d("examples/cmb_noise", shape, pointing, signal_cmb+noise, pos=pos)
run_2d("examples/src_cmb_noise", shape, pointing, signal_src+signal_cmb+noise, pos=pos, amp=amp, sigma=sigma)

# Pointing offset
offsets = np.array([[0,0],[1,1]])[:,None,:]/2**0.5 * 1e-2
signal_src = toy.build_signal_src(pointing+offsets, pos, amp=amp, sigma=sigma)
signal_cmb = toy.build_signal_from_map(pointing+offsets, cmb_highres, nsub=nsub)

run_2d("examples/src_ptoff", shape, pointing, signal_src, pos=pos, amp=amp, sigma=sigma)
run_2d("examples/cmb_ptoff", shape, pointing, signal_cmb, pos=pos)
run_2d("examples/src_cmb_ptoff", shape, pointing, signal_src+signal_cmb, pos=pos, amp=amp, sigma=sigma)

run_2d("examples/src_noise_ptoff", shape, pointing, signal_src+noise, pos=pos, amp=amp, sigma=sigma)
run_2d("examples/cmb_noise_ptoff", shape, pointing, signal_cmb+noise, pos=pos)
run_2d("examples/src_cmb_noise_ptoff", shape, pointing, signal_src+signal_cmb+noise, pos=pos, amp=amp, sigma=sigma)

# Amplitude variability
gains = np.array([0.995,1.005])[:,None]
signal_src = toy.build_signal_src(pointing, pos, amp=amp, sigma=sigma)*gains
signal_cmb = toy.build_signal_from_map(pointing, cmb_highres, nsub=nsub)*gains

run_2d("examples/src_gain", shape, pointing, signal_src, pos=pos, amp=amp, sigma=sigma)
run_2d("examples/cmb_gain", shape, pointing, signal_cmb, pos=pos)
run_2d("examples/src_cmb_gain", shape, pointing, signal_src+signal_cmb, pos=pos, amp=amp, sigma=sigma)

run_2d("examples/src_noise_gain", shape, pointing, signal_src+noise, pos=pos, amp=amp, sigma=sigma)
run_2d("examples/cmb_noise_gain", shape, pointing, signal_cmb+noise, pos=pos)
run_2d("examples/src_cmb_noise_gain", shape, pointing, signal_src+signal_cmb+noise, pos=pos, amp=amp, sigma=sigma)
