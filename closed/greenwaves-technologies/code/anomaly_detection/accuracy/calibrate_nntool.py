import os
from nntool.api import NNGraph
from glob import glob
from tqdm import tqdm
import numpy as np
import argparse
import pickle

class MyDataLoader():
    def __init__(self, image_files, max_idx=None):      
        self._file_list = image_files
        self._idx = 0
        self._max_idx = max_idx if max_idx is not None else len(image_files) - 1
        self._idy = 0

    def __len__(self):
        return (self._max_idx + 1)

    def __iter__(self):
        self._idx = 0
        return self

    def __next__(self):
        if self._idx > self._max_idx:
            raise StopIteration()
        filename = self._file_list[self._idx]

        # Here we read the image and make it a numpy array
        in_data = np.fromfile(filename, dtype=np.float32)

        self._idx += 1
        return in_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_model', type=str, default='../model/ad01_fp32.tflite', 
                        help='Model to use for KWS.')
    parser.add_argument('--calibration_samples', type=str, default="../datasets/calibration_samples/",
                        help='Path to calibration samples for NNTool quantization')
    parser.add_argument('--calibration_file', type=str, default="nntool_calibration.pickle",
                        help='Path to calibration samples for NNTool quantization')
    parser.add_argument('--n_samples', type=int, default=1000,
                        help='Max number of samples from calibration dataset')
    parser.add_argument('--load_tflite_quant', action='store_true',
                        help='Load tflite quantization')
    args, unparsed = parser.parse_known_args()

    # Open the model in NNTool
    G = NNGraph.load_graph(args.tflite_model, load_quantization=args.load_tflite_quant)
    G.adjust_order()
    G.fusions('scaled_match_group')
    data_loader = MyDataLoader(glob(args.calibration_samples + "/*"), max_idx=args.n_samples)
    statistics = G.collect_statistics(data_loader)

    with open(os.path.splitext(args.calibration_file)[0] + ".txt", "w") as f:
        f.write(f"{'key':50}\t{'min':>8}\t{'max':>8}\t{'mean':>8}\t{'std':>8}\t\n")
        for k, v in statistics.items():
            tmp = v['range_out'][0]
            key = k if not isinstance(k, tuple) else k[-1]
            out_str = f"{key:50}\t{float(tmp['min']):8.2f}\t{float(tmp['max']):8.2f}\t{float(tmp['mean']):8.2f}\t{float(tmp['std']):8.2f}\t\n"
            f.write(out_str)
    with open(args.calibration_file, 'wb') as f:
        pickle.dump(statistics, f)

if __name__ == '__main__':
    main()
