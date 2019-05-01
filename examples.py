import numpy as np, toy
from matplotlib import pyplot

np.random.seed(1)
toy.mpl_setdefault("planck")
shape    = np.array([81,81])
#shape    = np.array([9,9])
pos      = (shape-1)/2
nsub     = 3
sigma    = 1.0

pointing = toy.build_pointing_2d(shape, nsub=nsub, ntod=2)#, independent_rows=True)
#data     = toy.build_signal_src(pointing, pos, amp=np.array([10000,9900])[:,None], sigma=sigma)
#data     = toy.build_signal_src(pointing, np.array([pos,pos+[0.1,0.1]])[:,None,:], amp=10000, sigma=sigma)
data     = toy.build_signal_src(pointing, pos, amp=10000, sigma=sigma)
cmb_high = toy.build_cmblike_map(shape, nsub=nsub, sigma=sigma)
data    += toy.build_signal_from_map(pointing, cmb_high, nsub=nsub)

mask     = toy.build_mask_disk(pointing, pos, 5)

# Subpixel filtering does not work. It removes subpixel aliasing along
# the scanning direction, but not in the perpendicular direction.
#data     = toy.filter_subpix(data, pointing, shape)

nmat       = toy.NmatOneoverf(data.shape[-1], nsub=nsub)
#nmat       = toy.NmatNotch(data.shape[-1], nsub=nsub)
pmat       = toy.PmatNearest(pointing, shape)
pmat_spline= toy.PmatSplinePixell(pointing, shape, order=1)

#data       = toy.fill_mask_constrained(data, mask, nmat)

omap_white = toy.solve_cg(data, pmat)
#omap_oof   = toy.solve_brute(data, pmat, nmat)
#omap_oof   = toy.solve_cg(data, pmat, nmat)
#omap_oof    = toy.solve_iterative(data, pmat, pmat_spline, nmat, callback=toy.plot_map)

#omap_oof   = toy.solve_cg(data, pmat_spline, nmat)
#omap_oof   = toy.solve_cg_mask_persamp(data, pmat, mask, nmat)
omap_oof   = toy.solve_mask_white(data, pmat, mask, nmat)
#omap_oof  -= omap_oof[0,0]

toy.plot_map(omap_oof,   range=4)
#toy.plot_map(omap_white, range=4)

model_white = pmat.map2tod(omap_white)
model_oof   = pmat.map2tod(omap_oof)
#model_oof   = pmat_spline.map2tod(omap_oof)

for i in range(2):
	np.savetxt("tod%d.txt" % i, np.concatenate([
		pointing[i], data[i,:,None], model_white[i,:,None], model_oof[i,:,None]],-1), fmt="%15.7e")

#pyplot.matshow(omap_white, vmin=-4, vmax=4)
#pyplot.matshow(omap_oof,   vmin=-4, vmax=4)
#pyplot.show()



