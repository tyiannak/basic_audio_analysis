import numpy as np
from numpy import genfromtxt
import glob
import os
import random

# Download dataset from here https://zenodo.org/record/1286570#.X0WGsi17FGo
# then run this script to generate 0.5 second segments and csv files of respective
# f0 and f0_std values (for each segment)
# Then you can use this to train regression models

file_list = [f for f in glob.glob("ChoralSingingDataset/*.wav")]

for iF, f in enumerate(file_list):
	f0_file = f.replace('.wav', '.f0')
	d = genfromtxt(f0_file, delimiter=' ')
	time = d[:, 0]
	f0 = d[:, 1]

	seg_window = 0.5

	duration = time[-1]

	d = 0
	while d + seg_window < duration:
		start_index = np.argmin(np.abs(d - time))
		end_index =  np.argmin(np.abs((d + seg_window) - time))
		cur_f0s = f0[start_index: end_index]
		if np.count_nonzero(cur_f0s < 1) < len(cur_f0s) / 10:
			# uncomment this for skipping 1 in 20 files 
#			if random.randint(1, 20)==7:
				# keep the average (actually median to avoid noise) f0 
				# and the std of all f0 values in the 500 msec segment 
				cur_f0 = np.median(cur_f0s)
				cur_f0_std = np.std(cur_f0s)
				new_file = f'{f}_segments_{cur_f0}.wav'
				command = f"ffmpeg -i {f} -ar 8000 -ac 1 -ss {d} -to {d + seg_window} {new_file} -y -loglevel panic"
				with open("f0.csv", "a") as f1:
					f1.write(f'{os.path.basename(new_file)}, {cur_f0}\n')
				with open("f0_std.csv", "a") as f2:
					f2.write(f'{os.path.basename(new_file)}, {cur_f0_std}\n')
				os.system(command)
		d += seg_window