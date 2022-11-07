python train.py --model_init_path=trained_models/kws_ref_model
echo "Done converting standard keras model to syntiant float"
python package_tinyml_perf.py
echo "Done creating synpkg from syntiant float model"
xxd -i tinyMLPerf.synpkg > dnn.c
echo "Done converting synpkg to c file"
