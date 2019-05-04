import numpy as np, argparse
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
args = parser.parse_args()

for ifile in args.ifiles:
	ofile = ifile[:-4] + ".txt"
	print(ofile)
	data  = np.load(ifile)
	np.savetxt(ofile, data, fmt="%15.7e")
