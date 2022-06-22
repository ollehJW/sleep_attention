# %%
from glob import glob
import os
import argparse
import numpy as np
import shutil
import ntpath
from scipy import io
import tqdm
from scipy import signal

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, 
                        default="./new_data", 
                        help="File path to the mat files.")
    parser.add_argument("--output_dir", type=str,
                        default="numpy_data",
                        help="Directory where to save numpy files outputs.")
    parser.add_argument("--sampling_rate", type=int,
                        default=100,
                        help="The sampling rate of EEG signal.")
    parser.add_argument("--epoch_sec_size", type=int,
                        default=30,
                        help="Epoch second size of EEG signal.")
    args = parser.parse_args()

    # Output dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    # Signal length size
    signal_length = 100 * args.epoch_sec_size

    # Read mat files
    mat_fnames = glob(os.path.join(args.data_dir, "*.mat"))
    mat_fnames.sort()
    psg_fnames = np.asarray(mat_fnames)

    for i in tqdm.tqdm(range(len(mat_fnames))):
        filename = ntpath.basename(psg_fnames[i]).replace(".mat", ".npz")

        data = io.loadmat(psg_fnames[i])

        x = [data['eeg_data'][i:i + signal_length] for i in range(0, len(data['eeg_data']), signal_length)]
        x = np.stack(x)
        x=x.astype(np.float32)
        y = data['label'][:, 0] - 1

        save_dict = {
            "x": x, 
            "y": y, 
            "fs": args.sampling_rate,
        }
        np.savez(os.path.join(args.output_dir, filename), **save_dict)


if __name__ == "__main__":
    main()


