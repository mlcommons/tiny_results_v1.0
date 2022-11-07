import os
from PIL import Image
from nntool.api import NNGraph
from nntool.graph.types import ConstantInputNode
from glob import glob
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import csv
import argparse
from eval_functions_eembc import calculate_accuracy
import pickle
from calibrate_nntool import MyDataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='../model/pretrainedResnet_quant.tflite', 
                        help='Model to use for KWS.')
    parser.add_argument('--dataset_path', type=str, default='../dataset/perf_samples',
                        help='Path to VWW dataset')
    parser.add_argument('--calibration_samples', type=str, default="../datasets/calibration_samples",
                        help='Path to calibration samples for NNTool quantization')
    parser.add_argument('--calibration_file', type=str, default="nntool_calibration.pickle",
                        help='Path to calibration samples for NNTool quantization')
    parser.add_argument('--ann_file', type=str, default='../datasets/y_labels.csv', 
                        help='Path to the VWW annotation file')
    parser.add_argument('--num_classes', type=int, default=10, 
                        help='Number of classes!')
    parser.add_argument('--run_float', action='store_true',
                        help='Run execution in NNTool in float')
    parser.add_argument('--nntool_quant', type=str, default="ne16_quant", choices=["ne16", "ne16_quant", "int8", "int8_quant", "fp16", "bfp16"],
                        help='Type of quantization options for nntool [ne16, ne16_quant, int8, int8_quant, fp16, bfp16]')
    parser.add_argument('--weight_bits', type=int, default=8,
                        help='Number of weight bits to quantize when ne16 enabled')
    parser.add_argument('--log_file', type=str, default=None,
                        help='write accuracy log to file')
    parser.add_argument('--clip_type', type=str, default="none",
                        help='clip type option for quantization')
    parser.add_argument('--clip_type_weights', type=str, default="none",
                        help='clip type option for quantization')
    args, unparsed = parser.parse_known_args()


    if args.dataset_path == '':
        print('Error! Missing path to COCO')
        exit()

    # Configurable Parameters
    ANNOTATION_FILE_PATH = args.ann_file
    BASEPATH = args.dataset_path

    # Open the model in NNTool
    if os.path.splitext(args.model_path)[-1] == ".json":
        G = NNGraph.load_graph(args.model_path)
        print(G.show())
        print(G.qshow())
    else:
        G = NNGraph.load_graph(args.model_path, load_quantization="quant" in args.nntool_quant)
        G.adjust_order()
        G.fusions('scaled_match_group')
        print(G.show())

        if not args.run_float:
            statistics = None
            if not "quant" in args.nntool_quant:
                if not args.calibration_file:
                    data_loader = MyDataLoader(glob(args.calibration_samples))
                    statistics = G.collect_statistics(data_loader)
                else:
                    filehandler = open(args.calibration_file, 'rb')
                    statistics = pickle.load(filehandler)

            node_options = {
                node.name: {'clip_type': args.clip_type_weights}
                for node in G.nodes(node_classes=(ConstantInputNode))
            }

            G.quantize(
                statistics=statistics,
                schemes=['float'] if args.nntool_quant in ["bfp16", "fp16"] else ["scaled"],
                graph_options={
                    'scheme' : 'FLOAT' if args.nntool_quant in ["bfp16", "fp16"] else None,
                    'float_type':'bfloat16',
                    'use_ne16': args.nntool_quant.startswith("ne16"),
                    'weight_bits': args.weight_bits if args.nntool_quant.startswith("ne16") else 8,
                    'hwc': True,
                    'clip_type': args.clip_type
                },
                node_options=node_options
                    
            )
            print(G.qshow())

    transpose_to_chw = G[0].out_dims[0].order == ["c", "h", "w"]
    if transpose_to_chw:
        print("Transposing to CHW")

    NUM_CLASSES = args.num_classes
    pred = [0 for x in range(NUM_CLASSES)]
    classes = [0 for x in range(NUM_CLASSES)]
    accuracy_classes = [0 for x in range(NUM_CLASSES)]
    tot_samples = 0

    list_pred = []
    list_target = []

    n_files = 0
    for row in open(ANNOTATION_FILE_PATH):
        n_files += 1

    with open(ANNOTATION_FILE_PATH) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for index, row in tqdm(enumerate(csv_reader), total=n_files):
            entry = row[0]
            target = int(row[2])
            file_name = os.path.join(BASEPATH, entry)
            if os.path.isfile(file_name):

                # Load input and convert to float (Test samples are generated in float range already)
                dat = np.fromfile(os.path.join(BASEPATH, entry), dtype=np.uint8).astype(np.float32).reshape([32, 32, 3])
                if transpose_to_chw:
                    dat = dat.transpose(2, 0, 1)

                output = G.execute([dat], dequantize=not args.run_float)[-1][0].flatten()

                list_pred.append(output.tolist())
                list_target.append(target)

                predicted_class = np.argmax(output)

                classes[target] += 1
                tot_samples +=1
                if predicted_class >= NUM_CLASSES:
                    print('Not recognized')
                elif target == predicted_class:
                    pred[target] += 1 

                accuracy_classes = [pred[i]/classes[i] if classes[i]>0 else 0 for i in range(NUM_CLASSES)]

    list_pred = np.asarray(list_pred)
    list_target = np.asarray(list_target)
    acc = calculate_accuracy(list_pred, list_target)
    print("EEMBC accuracy: ", acc)

    str_out = ''
    # Print final results
    sum_pred = 0
    for i in range(NUM_CLASSES):
        sum_pred += pred[i]
        str_out += f'Class {i:3}: {100*pred[i]/classes[i]:.2f}% ({pred[i]}/{classes[i]})\n'
    str_out += f"The average accuracy is: {100*sum_pred/tot_samples:.2f}% on {tot_samples} samples\n"

    print(str_out)
    if args.log_file:
        with open(args.log_file, "w") as f:
            f.write(G.show())
            f.write("\n\n")
            if not args.run_float:
                f.write(G.qshow())
                f.write("\n\n")
            f.write(str_out)

if __name__ == '__main__':
    main()
