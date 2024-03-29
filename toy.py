# Toy mapmaker. T-only, everything is in pixel coordinates, so no
# geometry needed. Maps are Nd and are simple numpy arrays. TODs
# are [ntod,nsamp]. Pointing is [ntod,nsamp,ndim], and is floating point.
# The sample rate is always 1, so the highest frequency will be 0.5.
# We won't care much about efficienty, and will use numpy/scipy functions
# instead of special purpose code, and interfaces will keep as simple
# as possible, even at the cost of some recomputation.
from __future__ import division, print_function
import numpy as np, warnings, os, errno, copy, time
from scipy import ndimage
from scipy.sparse import linalg

# Pointing matrices. All should have this interface

class PmatNearest:
	"""Nearest neighbor pointing matrix"""
	def __init__(self, pointing, shape):
		self.shape  = shape
		ipoint      = np.round(pointing).astype(int)
		self.fpoint = np.ravel_multi_index(np.rollaxis(ipoint,-1), shape, mode="wrap")
	def tod2map(self, tod):
		return np.bincount(self.fpoint.reshape(-1), tod.reshape(-1), minlength=prod(self.shape)).reshape(self.shape)
	def map2tod(self, map):
		return map.reshape(-1)[self.fpoint]

class PmatSpline:
	"""Spline interpolation pointing matrix"""
	def __init__(self, pointing, shape, order=3):
		self.shape  = shape
		self.fpoint = np.rollaxis(pointing,2)
		self.order  = order
		self.ndim   = len(shape)
	def tod2map(self, tod): raise NotImplementedError
	def map2tod(self, map):
		tod = np.zeros(self.fpoint.shape[1:])
		return ndimage.map_coordinates(map, self.fpoint.reshape(self.ndim,-1), order=self.order, mode="wrap").reshape(self.fpoint.shape[1:])

class PmatSplinePixell:
	"""Spline interpolation pointing matrix"""
	def __init__(self, pointing, shape, order=3):
		self.shape  = shape
		self.fpoint = np.ascontiguousarray(np.rollaxis(pointing,2))
		self.order  = order
		self.ndim   = len(shape)
	def tod2map(self, tod):
		from pixell import interpol
		map = np.zeros(self.shape)
		tod = np.ascontiguousarray(tod)
		interpol.map_coordinates(map, self.fpoint.reshape(self.ndim,-1), odata=tod.reshape(-1), order=self.order, border="cyclic", trans=True)
		return map
	def map2tod(self, map):
		from pixell import interpol
		tod = np.zeros(self.fpoint.shape[1:])
		interpol.map_coordinates(map, self.fpoint.reshape(self.ndim,-1), odata=tod, order=self.order, border="cyclic").reshape(self.fpoint.shape[1:])
		return tod

class NmatIdentity:
	def apply(self, tod): return tod.copy()
	def white(self, tod): return tod.copy()
	def sim(self, ntod, nsamp): return np.random.standard_normal([ntod,nsamp])
	def inverse(self): return NmatIdentity()

class NmatOneoverf:
	"""Simple noise model with white noise plus an atmosphere-like 1/f profile"""
	def __init__(self, nsamp, fknee=0.02, alpha=-4, sigma=1, nsub=1):
		self.fknee, self.alpha, self.sigma = fknee, alpha, sigma
		self.freqs   = np.fft.fftfreq(nsamp)*nsub
		# numpy will generate a useless warning here, even though the
		# floating point math all works out correctly
		with nowarn():
			self.profile = self.sigma**-2*(1 + (np.abs(self.freqs)/fknee)**alpha)**-1
		self.profile = np.maximum(self.profile, 1e-10)
	def apply(self, tod, pow=1):
		ft  = np.fft.fft(tod)
		ft *= self.profile**pow
		return np.fft.ifft(ft, n=tod.shape[-1]).real
	def white(self, tod):
		return tod * self.sigma**-2
	def sim(self, ntod, nsamp):
		return self.apply(np.random.standard_normal([ntod,nsamp]), pow=-0.5)
	def inverse(self):
		res = copy.copy(self)
		res.profile = 1/self.profile
		return res

class NmatNotch:
	"""Simple noise model representing white noise, but with downweighting of signal
	in a narrow frequency band, corresponding to ML treatment of a notch filter."""
	def __init__(self, nsamp, fnotch=0.1, width=0.005, nsub=1):
		self.fnotch, self.width = fnotch, width
		self.freqs   = np.fft.fftfreq(nsamp)*nsub
		self.profile = np.abs(self.freqs-fnotch)>width
	def apply(self, tod):
		ft  = np.fft.fft(tod)
		ft *= self.profile
		return np.fft.ifft(ft, n=tod.shape[-1]).real
	def white(self, tod):
		return tod
	def sim(self, ntod, nsamp):
		return np.random.standard_normal([ntod,nsamp])

def build_pointing_1d(shape, nsub=1, nscan=1, ntod=1):
	"""Build a simple 1d pointing [ntod,nsamp,1] where nsamp=npix*nsub*nscan.
	This represents equisampled scanning that wraps around from one
	side to the other nscan times, with nsub samples per pixel each time.
	Ntod specifies how many times this whole pattern should repeat.
	The first argument specifies the shape of the array. It can either
	be (npix,) or, for convenience, npix can be passed directly."""
	# Allow passing npix either directly or as a shape (i.e. a tuple)
	try:              npix = shape[0]
	except TypeError: npix = shape
	nsamp    = npix*nsub*nscan
	pointing = np.linspace(0, npix*nscan, nsamp, endpoint=False)%npix
	pointing = np.repeat(pointing[None,:,None],ntod,1)
	return pointing

def build_pointing_2d(shape, nsub=1, ntod=2, independent_rows=False):
	"""Build a simple 2d pointing [ntod,nsamp,2] representing a
	horizontal and vertical scanning along each row and column
	in the map. Even tods will be vertical and odd tods will be horizontal.
	shape gives the (ny,nx) size of the map."""
	nypix, nxpix = shape
	x = np.arange(nxpix*nsub)/nsub
	y = np.arange(nypix*nsub)/nsub
	#            vertical scanning            horizontal scanning
	patterns = [ np.dstack(np.meshgrid(x,y)), np.dstack(np.meshgrid(y,x))[:,:,::-1] ]
	pointing = []
	for i in range(ntod):
		pointing.append(patterns[i%2].reshape(-1,2))
	pointing = np.array(pointing)
	if independent_rows:
		# This assumes same length of rows and columns
		pointing = np.reshape(pointing, (-1,nypix,2))
	return pointing

def build_signal_src(pointing, pos, sigma=1, amp=1):
	"""Simulate a gaussian point source with standard deviation sigma
	and max-amplitude amp as at the given pixel position pos[ndim]
	as observed by the given pointing [ntod,nsamp,ndim]."""
	r2 = np.sum((pointing-pos)**2,-1)
	return amp*np.exp(-0.5*r2/sigma**2)

def build_cmblike_map(shape, alpha=-1, amp=1.0, sigma=1, nsub=1):
	shape  = np.array(shape)*nsub
	rmap   = np.random.standard_normal(shape)
	ky, kx = [np.fft.fftfreq(n) for n in shape]
	k      = (ky[:,None]**2 + kx[None,:]**2)**0.5
	kmin   = min(k[0,1],k[1,0])
	scale  = np.maximum(k/kmin,1)**alpha
	scale *= amp / np.mean(scale**2)**0.5
	fmap   = np.fft.fft2(rmap)
	fmap  *= scale
	# Apply beam
	fmap  *= np.exp(-0.5*(k*nsub*sigma*2*np.pi)**2)
	omap   = np.fft.ifft2(fmap, shape).real
	return omap

def build_signal_from_map(pointing, map, nsub=1):
	return ndimage.map_coordinates(map, np.rollaxis(pointing*nsub,2).reshape(2,-1), mode="wrap").reshape(pointing.shape[:2])

def build_mask_disk(pointing, pos, rad):
	return np.sum((pointing - pos)**2,-1) < rad**2

def solve_white(data, pmat):
	map = pmat.tod2map(data)
	div = pmat.tod2map(data*0+1)
	map[div>0] /= div[div>0]
	return map

def solve_cg(data, pmat, nmat=NmatIdentity(), callback=None):
	"""Find the map that best reproduces the data by solving
	the equation system map = (P'N"P)"P'N"d using conjugate gradients,
	where ' means transpose and " means inverse, and d,P,N correspond to data,pmat,nmat."""
	rhs = pmat.tod2map(nmat.apply(data))
	def Afun(x):
		return pmat.tod2map(nmat.apply(pmat.map2tod(x.reshape(rhs.shape)))).reshape(-1)
	A = linalg.LinearOperator((rhs.size, rhs.size), matvec=Afun)
	callfun = lambda x: callback(x.reshape(rhs.shape)) if callback else None
	x, info = linalg.cg(A, rhs.reshape(-1), callback=callfun)
	return x.reshape(rhs.shape)

def solve_brute(data, pmat, nmat=NmatIdentity(), callback=None, verbose=False):
	"""Find the map that best reproduces the data by solving
	the equation system map = (P'N"P)"P'N"d brute force, where ' means transpose
	and " means inverse, and d,P,N correspond to data,pmat,nmat."""
	rhs  = pmat.tod2map(nmat.apply(data))
	npix = rhs.size
	def Afun(x):
		return pmat.tod2map(nmat.apply(pmat.map2tod(x.reshape(rhs.shape)))).reshape(-1)
	Amat = build_matrix(Afun, (npix,npix), verbose=verbose)
	omap = np.linalg.solve(Amat, rhs.reshape(-1)).reshape(rhs.shape)
	return omap

def solve_cg_mask_persamp(data, pmat, mask, nmat=NmatIdentity(), callback=None):
	"""Find the map that best reproduces the data by solving
	the equation system map = (P'N"P)"P'N"d using conjugate gradients,
	where ' means transpose and " means inverse, and d,P,N correspond to data,pmat,nmat."""
	# This could be implemented two ways:
	# 1. redundantly solve for both the pixels in the hole and the samples in the hole
	# 2. only solve for the samples in the hole
	# For #1 there the system is degenerate and there are many possible solutions, but
	# all of them should be the same after combining the samples and the map. However,
	# what I see in practice is that #1 has O(1e-6) residual nearby, while #2 has O(1e-8).
	# This is larger than what I would expect from numerical issues.
	Nd        = nmat.apply(data)
	rhs_map   = pmat.tod2map(Nd)
	rhs_samps = Nd[mask]
	npix, nsamp = rhs_map.size, rhs_samps.size
	def zip(map, samps): return np.concatenate([map.reshape(-1),samps])
	def unzip(x): return x[:npix].reshape(pmat.shape), x[npix:]
	def combine(map, samps):
		return map + solve_white(expand_samps(samps, mask), pmat)
	def Afun(x):
		imap, isamps = unzip(x)
		tod          = pmat.map2tod(imap)
		tod[mask]   += isamps
		tod          = nmat.apply(tod)
		omap         = pmat.tod2map(tod)
		osamps       = tod[mask]
		return zip(omap, osamps)
	rhs     = zip(rhs_map, rhs_samps)
	A       = linalg.LinearOperator((rhs.size, rhs.size), matvec=Afun)
	callfun = lambda x: callback(combine(*unzip(x))) if callback else None
	# The source-only case needs lower tolerance here in order to reach the right solution
	# for some reason - even for the white noise case.
	x, info = linalg.cg(A, rhs.reshape(-1), callback=callfun, tol=1e-15, maxiter=200)
	return combine(*unzip(x))

#def solve_cg_mask_persamp2(data, pmat, mask, nmat=NmatIdentity(), callback=None):
#	"""Find the map that best reproduces the data by solving
#	the equation system map = (P'N"P)"P'N"d using conjugate gradients,
#	where ' means transpose and " means inverse, and d,P,N correspond to data,pmat,nmat."""
#	# Unlike persamp, this one solves only for non-redundant degrees of freedom
#	Nd        = nmat.apply(data)
#	rhs_map   = pmat.tod2map(Nd*~mask)
#	rhs_samps = Nd[mask]
#	npix, nsamp = rhs_map.size, rhs_samps.size
#	def zip(map, samps): return np.concatenate([map.reshape(-1),samps])
#	def unzip(x): return x[:npix].reshape(pmat.shape), x[npix:]
#	def combine(map, samps):
#		tod = pmat.map2tod(map)
#		tod[mask] = samps
#		return solve_white(tod, pmat)
#	def Afun(x):
#		imap, isamps = unzip(x)
#		tod          = pmat.map2tod(imap)
#		tod[mask]    = isamps
#		tod          = nmat.apply(tod)
#		omap         = pmat.tod2map(tod*~mask)
#		osamps       = tod[mask]
#		return zip(omap, osamps)
#	rhs     = zip(rhs_map, rhs_samps)
#	A       = linalg.LinearOperator((rhs.size, rhs.size), matvec=Afun)
#	callfun = lambda x: callback(combine(*unzip(x))) if callback else None
#	x, info = linalg.cg(A, rhs.reshape(-1), callback=callfun, tol=1e-15, maxiter=200)
#	return combine(*unzip(x))

def solve_cg_mask(data, pmat, mask, nmat=NmatIdentity(), callback=None):
	"""Find the map that best reproduces the data by solving
	the equation system map = (P'N"P)"P'N"d using conjugate gradients,
	where ' means transpose and " means inverse, and d,P,N correspond to data,pmat,nmat."""
	Nd        = nmat.apply(data)
	rhs_map   = pmat.tod2map(Nd*~mask)
	rhs_samps = Nd[mask]
	npix, nsamp = rhs_map.size, rhs_samps.size
	def zip(map, samps): return np.concatenate([map.reshape(-1),samps])
	def unzip(x): return x[:npix].reshape(pmat.shape), x[npix:]
	def combine(map, samps):
		return map + solve_white(expand_samps(samps, mask), pmat)
	def Afun(x):
		imap, isamps = unzip(x)
		tod          = pmat.map2tod(imap)*~mask
		tod[mask]   += isamps
		tod          = nmat.apply(tod)
		omap         = pmat.tod2map(tod*~mask)
		osamps       = tod[mask]
		return zip(omap, osamps)
	rhs     = zip(rhs_map, rhs_samps)
	A       = linalg.LinearOperator((rhs.size, rhs.size), matvec=Afun)
	callfun = lambda x: callback(combine(*unzip(x))) if callback else None
	x, info = linalg.cg(A, rhs.reshape(-1), callback=callfun)
	map, samps = unzip(x)
	map[map==0] = np.nan
	return map

def solve_mask_white(data, pmat, mask, nmat=NmatIdentity(), callback=None, solver=solve_cg):
	"""Solve for the map in two steps:
	1. Make a gapfilled version of the data and solve that normally
	2. Map the difference between the original and gapfilled tod using a white noise model.
	3. Add them to get the final map."""
	data_filled = fill_mask_constrained(data, mask, nmat)
	map_filled  = solver(data_filled, pmat, nmat, callback=callback)
	map_white   = solve_white(data-data_filled, pmat)
	return map_filled + map_white

def solve_iterative(data, pmat, pmat2, nmat=NmatIdentity(), callback=None, niter=4, solver=solve_cg):
	map = solver(data, pmat, nmat=nmat)
	if callback: callback(map)
	for i in range(1, niter):
		resid = data - pmat2.map2tod(map)
		map  += solver(resid, pmat, nmat=nmat)
		if callback: callback(map)
	return map

def pgls_filter(data, h):
	from scipy.signal import medfilt
	lpass = data*0
	for i in range(data.shape[0]):
		lpass[i] = medfilt(data[i], 2*h+1)
	return data-lpass

def solve_pgls(data, pmat, nmat=NmatIdentity(), h=10, niter=5, verbose=False):
	"""Solve the ML equation using the PGLS method. It is a blind method that's
	a bit similar to white source mapping, at the cost of being slower"""
	map   = solve_cg(data, pmat, nmat)
	for i in range(niter):
		resid = pmat.map2tod(map) - data
		clean = pgls_filter(resid, h)
		artifacts = solve_white(clean, pmat)
		map  -= artifacts
	return map

def solve_xgls(data, pmat, signal, nmat=NmatIdentity(), verbose=False):
	"""Solve the ML equation with an extra noise term given by the sub-pixel
	variance of the signal itself. We assume that we already have an approximation
	for the signal.

	The equation is (P'(N+X)"P)m = P'(N+X)"d, where X is the sub-pixel variance,
	which is diagonal in time domain. pmat is assumed to be nearest neighbor.

	XGLS paper recommends:
	1. (N+X)" = X"(N"+X")"N"
	2. starting point should be binned map
	"""
	from enlib import cg
	# Compute the sub-pixel variance sample matrix X from the signal
	sighit = np.maximum(1,pmat.tod2map(signal*0+1))
	sigmap = pmat.tod2map(signal)/sighit
	sigres = signal - pmat.map2tod(sigmap)
	varmap = pmat.tod2map(sigres**2)/sighit
	X      = pmat.map2tod(varmap)
	iX     = 1/np.maximum(X, np.max(X)*1e-5**2)
	iN     = nmat
	# Define the heavy operator (N+X)"
	class Callback:
		def __init__(self, tag, rhs, dump=None):
			self.i = 0
			self.tag = tag
			self.rhs = rhs
			self.t   = time.time()
			self.dump = dump
		def __call__(self, err, x):
			t = time.time()
			print("%5d %10s %8.3f %15.7e" % (self.i, self.tag, t-self.t, err))
			self.i += 1
			self.t  = t
			if self.dump: self.dump(x, self.i)
	def cgsolve(Afun, rhs, x0=None, Mfun=None, tol=1e-14, maxiter=1000000, miniter=0, callback=None):
		if Mfun is None: Mfun = cg.default_M
		solver = cg.CG(Afun, rhs, x0=x0, M=Mfun)
		for i in range(maxiter):
			solver.step()
			if callback: callback(solver.err, solver.x)
			if solver.err < tol and i > miniter: break
		return solver.x
	def apply_iNiX(x):
		d = x.reshape(data.shape)
		return (iN.apply(d) + iX*d).reshape(-1)
	#op_NX = linalg.LinearOperator((data.size, data.size), matvec=apply_NX)
	#op_iN = linalg.LinearOperator((data.size, data.size), matvec=lambda x: nmat.apply(x.reshape(data.shape)).reshape(-1))
	def apply_iNX(x):
		callback = None #if not verbose else Callback("iNX", x)
		a  = iN.apply(x.reshape(data.shape)).reshape(-1)
		b  = cgsolve(apply_iNiX, a, callback=callback, miniter=100).reshape(data.shape)
		c  = (iX*b).reshape(-1)
		return c

	# Compute the rhs
	rhs = pmat.tod2map(apply_iNX(data.reshape(-1)).reshape(data.shape))
	def apply_A(x):
		return pmat.tod2map(apply_iNX(pmat.map2tod(x.reshape(rhs.shape)).reshape(-1)).reshape(data.shape)).reshape(-1)
	#op_A    = linalg.LinearOperator((rhs.size, rhs.size), matvec=apply_A)
	def dumper(x, i):
		if i % 10 == 0:
			np.save("test_step_%04d.npy" % i, x.reshape(rhs.shape))
	callback= None if not verbose else Callback("A", rhs, dump=dumper)
	x0 = pmat.tod2map(data)/pmat.tod2map(data*0+1)
	x  = cgsolve(apply_A, rhs.reshape(-1), tol=1e-8, callback=callback, miniter=50, maxiter=500)
	#x, info = linalg.cg(op_A, rhs.reshape(-1), callback=callback)
	return x.reshape(rhs.shape)

def filter_subpix(data, pointing, shape):
	"""Replace all samples that consecutively hit the same pixel with their mean"""
	ipoint = np.round(pointing).astype(int)
	fpoint = np.ravel_multi_index(np.rollaxis(ipoint,-1), shape, mode="wrap")
	ntod   = len(data)
	odata  = data.copy()
	for ti in range(ntod):
		tdata, tpoint = data[ti], fpoint[ti]
		# Build a group index based on where the pixel index changes
		edges = tpoint[1:]!=tpoint[:-1]
		index = np.concatenate([[0], np.cumsum(edges)])
		avgs  = np.bincount(index, tdata)/np.bincount(index)
		odata[ti] = avgs[index]
	return odata

def build_matrix(func, shape, verbose=False):
	mat = np.zeros(shape)
	v = np.zeros(shape[1])
	for i in range(shape[1]):
		v[:] = 0; v[i] = 1
		mat[:,i] = func(v)
		if verbose and i % 100 == 0: print("%6d/%d" % (i,shape[1]))
	return mat

def expand_samps(samps, mask):
	tod = np.zeros(mask.shape)
	tod[mask] = samps
	return tod

def fill_mask_constrained(data, mask, nmat, known_tol=100, cg_tol=1e-11):
	"""Given data[ntod,nsamp] and a boolean mask of the same shape
	selecting some of those samples, return a new array where the
	selected samples have been replaced by the ML prediction based
	on the given noise matrix nmax and the unmasked samples.

	This works by solving the equation system (N"+M") odata = M" idata.
	Here N" = nmat is the inverse noise covariance matrix, and M" is the
	masking matrix. Conceptually this is 0 for the samples we wish to
	fill in and infinity for those we know, but we replace the infinity
	with known_tol * white_ivar to speed up convergence."""
	M   = (1-mask)*nmat.white(np.full(data.shape, 1.0))*known_tol
	rhs = data*M
	def Afun(x):
		x   = x.reshape(data.shape)
		Ax  = nmat.apply(x) + M*x
		resid = (data-x)
		return Ax.reshape(-1)
	A = linalg.LinearOperator((rhs.size, rhs.size), matvec=Afun)
	x, info = linalg.cg(A, rhs.reshape(-1), tol=cg_tol)
	odata = x.reshape(rhs.shape)
	# Copy over known data from input, so it doesn't change
	odata[~mask] = data[~mask]
	return odata

# General utilities

def prod(shape):
	res = 1
	for s in shape: res *= s
	return s

def plot_map(map, range=4, sub=True):
	from matplotlib import pyplot
	if sub: map = map - map[0,0]
	pyplot.matshow(map, vmin=-range, vmax=range)
	pyplot.show()

class nowarn:
	"""Use in with block to suppress warnings inside that block."""
	def __enter__(self):
		self.filters = list(warnings.filters)
		warnings.filterwarnings("ignore")
		return self
	def __exit__(self, type, value, traceback):
		warnings.filters = self.filters

def mkdir(path):
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise

# Matplotlib color stuff

colormaps = {
	"planck": [(0,"#0000ffff"),(0.332,"#00d7ffff"),(0.5,"#ffedd9ff"),(0.664,"#ffb400ff"),(0.828,"#ff4b00ff"),(1,"#640000ff")],
}

def mpl_register(names=None):
	import matplotlib.cm, matplotlib.pyplot, matplotlib.colors
	if names is None: names = colormaps.keys()
	if isinstance(names, str): names = [names]
	for name in names:
		cmap = matplotlib.colors.LinearSegmentedColormap.from_list(name, colormaps[name])
		matplotlib.cm.register_cmap(name, cmap)

def mpl_setdefault(name):
	import matplotlib.pyplot
	mpl_register(name)
	matplotlib.pyplot.rcParams['image.cmap'] = name
