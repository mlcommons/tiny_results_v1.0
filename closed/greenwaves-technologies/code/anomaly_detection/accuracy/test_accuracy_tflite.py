import os
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
import numpy as np
import csv
import argparse
from eval_functions_eembc import calculate_ae_pr_accuracy, calculate_ae_accuracy, calculate_ae_auc
#from sklearn import metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_model', type=str, default='../model/ad01_int8.tflite', 
                        help='Model to use for IC.')
    parser.add_argument('--dataset_path', type=str, default='../datasets/perf_samples/',
                        help='Path to dataset')
    parser.add_argument('--ann_file', type=str, default='../datasets/y_labels.csv', 
                        help='Path to the annotation file')
    parser.add_argument('--num_classes', type=int, default=2, 
                        help='Number of classes!')
    parser.add_argument('--log_file', type=str, default=None,
                        help='write accuracy log to file')
    args, unparsed = parser.parse_known_args()

    if args.dataset_path == '':
        print('Error! Missing path to COCO')
        exit()

    # Configurable Parameters
    ANNOTATION_FILE_PATH = args.ann_file
    BASEPATH = args.dataset_path
    TFLITE_MODEL_PATH = args.tflite_model

    # Configura the TF Interpreter
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()

    # Get input and output tensors of TFLite
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print('Input details: ', input_details)
    print('Output details: ', output_details)
    scale, zero_point = input_details[0]['quantization']
    scale_out, zero_point_out = output_details[0]['quantization']
    input_shape = input_details[0]['shape']
    print('The input parameters: ', scale, zero_point, input_shape)

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
            win_len = int (int(row[3]) / 4)
            stride = int(int(row[4]) / 4)
            file_name = os.path.join(BASEPATH, entry)
            if os.path.isfile(file_name):

                # Load input and convert to float (Test samples are generated in uint8)
                dat = np.fromfile(os.path.join(BASEPATH, entry), dtype=np.float32)
                sz, = dat.shape
                num_wind = int((sz - win_len + stride) / stride)
                win_data = np.zeros( (num_wind, win_len), dtype=np.float32)
                output_data = np.empty_like(win_data)

                # fill win data
                for i in range(win_data.shape[0]):
                    win_data[i:i+1, :] = dat[i*stride:i*stride+win_len]

                # quantization
                if scale:
                    data_q = (win_data / scale) + zero_point 
                    data_q = np.round(data_q).astype(np.int8)
                else:
                    data_q = win_data

                # run inference
                for i in range(win_data.shape[0]):
                    interpreter.set_tensor(input_details[0]['index'], data_q[i:i+1, :] )
                    interpreter.invoke()
                    output_data[i:i+1, :] = interpreter.get_tensor(output_details[0]['index'])

                # de quantize output
                if scale_out:
                    output_data = scale_out * (output_data.astype(np.float32) - zero_point_out)

                errors = np.mean(np.square(win_data - output_data), axis=1)
                errors =  np.mean(errors)
                list_pred.append(errors.tolist())
                list_target.append(target)

    #auc = metrics.roc_auc_score(list_target, list_pred)

    list_pred = np.asarray(list_pred)
    list_target = np.asarray(list_target)
    acc = calculate_ae_accuracy(list_pred, list_target)
    auc = calculate_ae_auc(list_pred, list_target, 'test')
    str_out = f"ACC: {acc:.2f}\nAUC: {auc:.2f}\n"

    print(str_out)
    if args.log_file:
        with open(args.log_file, "w") as f:
            f.write(str_out) 

if __name__ == '__main__':
    main()
