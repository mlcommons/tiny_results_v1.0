import os
from nntool.api import NNGraph
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='BUILD_MODEL_NE16_QUANT/ad01.json',
                        help='Model ready for inference --> NNTool state.')
    parser.add_argument('--input_path', type=str, default='../dataset/perf_samples',
                        help='Path to input file')
    args, unparsed = parser.parse_known_args()

    # Open the model in NNTool
    if os.path.splitext(args.model_path)[-1] == ".json":
        G = NNGraph.load_graph(args.model_path)
        print(G.show())
        print(G.qshow())
    else:
        raise ValueError("For this test provide the final json model from NNTool")

    transpose_to_chw = G[0].out_dims[0].order == ["c", "h", "w"]
    if transpose_to_chw:
        print("Transposing to CHW")

    win_len = int(int(2560) / 4) #row[3]
    stride = int(int(512) / 4) #row[4]
    file_name = args.input_path
    if os.path.isfile(file_name):
        print(f"Testing image: {file_name}")

        # Load input and convert to float (Test samples are generated in uint8)
        dat = np.fromfile(file_name, dtype=np.float32)
        sz, = dat.shape
        num_wind = int((sz - win_len + stride) / stride)
        win_data = np.zeros( (num_wind, win_len), dtype=np.float32)
        output_data = np.empty_like(win_data)

        # fill win data
        #for i in range(win_data.shape[0]):
        for i in range(0, 1):
            win_data[i:i+1, :] = dat[i*stride:i*stride+win_len]
            output = G.execute([win_data[i:i+1, :]], dequantize=True)
            #for i, out in enumerate(output):
            #    print(f"{G[i].name}:\n\t{np.sum(out[0].astype(np.int8))}")
            output_data[i:i+1, :] = output[-1][0].flatten()

        # Code written in the submission
        diffsum = 0
        for w, o in zip(win_data[0], output_data[0]):
            diff = w - o
            diffsum += diff*diff
        diffsum = diffsum / len(win_data[0])
        print(f"[{diffsum:.3f}]")

    else:
        ValueError(f"Could not open inut_path: {file_name}")

if __name__ == '__main__':
    main()
