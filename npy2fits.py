import numpy as np, argparse
from pixell import enmap
parser = argparse.ArgumentParser()
parser.add_argument("ifiles", nargs="+")
args = parser.parse_args()

for ifile in args.ifiles:
	ofile = ifile[:-4] + ".fits"
	print(ofile)
	data  = np.load(ifile)
	enmap.write_map(ofile, enmap.enmap(data))
