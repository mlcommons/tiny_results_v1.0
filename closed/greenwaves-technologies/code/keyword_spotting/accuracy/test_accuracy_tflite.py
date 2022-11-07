import os
from PIL import Image
import tensorflow as tf
import numpy as np
import csv
from tqdm import tqdm
import argparse
from eval_functions_eembc import calculate_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite_model', type=str, default='../model/kws_ref_model.tflite', 
                        help='Model to use for KWS.')
    parser.add_argument('--dataset_path', type=str, default='../datasets/perf_samples/',
                        help='Path to dataset')
    parser.add_argument('--ann_file', type=str, default='../datasets/y_labels.csv', 
                        help='Path to the annotation file')
    parser.add_argument('--num_classes', type=int, default=12, 
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
    input_shape = input_details[0]['shape']
    [nb,h_tflite,w_tflite,c_tflite] = input_shape
    print('The input parameters: ', scale, zero_point, nb, h_tflite, w_tflite, c_tflite)

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

                # Load input and convert to float (test samples are provided quantized as in the platform)
                dat = (np.fromfile(os.path.join(BASEPATH, entry), dtype=np.int8).reshape(input_details[0]['shape']).astype(np.float32) - 83) * 0.5847029089927673

                if input_details[0]['dtype'] == np.float32:
                    interpreter.set_tensor(input_details[0]['index'], dat)
                elif input_details[0]['dtype'] == np.int8:
                    dat_q = np.clip((np.round(dat / scale) + zero_point), -128, 127).astype(np.int8) # should match input type in quantize.py
                    interpreter.set_tensor(input_details[0]['index'], dat_q)
                else:
                    raise ValueError("TFLite file has input dtype {:}.  Only np.int8 and np.float32 are supported".format(input_details[0]['dtype']))

                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                list_pred.append(output[0].tolist())
                list_target.append(target)

                predicted_class = np.argmax(output)

                classes[target] += 1
                tot_samples +=1
                if predicted_class >= NUM_CLASSES:
                    print('Not recognized')
                elif target == predicted_class:
                    pred[target] += 1 

                tflite_pred = predicted_class
                accuracy_classes = [pred[i]/classes[i] if classes[i]>0 else 0 for i in range(NUM_CLASSES)]
            else:
                print(f"problem with {file_name}")

    list_pred = np.asarray(list_pred)
    list_target = np.asarray(list_target)
    acc = calculate_accuracy(list_pred,list_target)
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
            f.write(str_out) 

if __name__ == '__main__':
    main()
