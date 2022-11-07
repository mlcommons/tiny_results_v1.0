import os
from nntool.api import NNGraph
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='BUILD_MODEL_NE16_QUANT/kws.json',
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

    file_name = args.input_path
    if os.path.isfile(file_name):

        # Load input and convert to float (Test samples are generated in float range already)
        dat = np.fromfile(file_name, dtype=np.uint8).astype(np.float32).reshape([32, 32, 3])

        if transpose_to_chw:
            dat = dat.transpose(2, 0, 1)

        output = G.execute([dat], dequantize=True)[-1][0].flatten()

        str_out = "Predicted output:\n["
        for out in output:
            str_out += f"{out:.3f},"
        str_out = str_out[:-1] + "]\n"
        print(str_out)

    else:
        ValueError(f"Could not open inut_path: {file_name}")

if __name__ == '__main__':
    main()
