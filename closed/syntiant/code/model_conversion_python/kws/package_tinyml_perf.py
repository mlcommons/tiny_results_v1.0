import json
from syntiant_networks.architectures.ndp12x.ndp12X import NDP12X
from syntiant_packager.packages import SyntiantPackageFactory
from tinyml_model import TinyMlPerf


model = TinyMlPerf()
model.load_weights('syntiant_float.h5')
for layer in model.layers:
    if layer.is_quantizeable():
        layer.set_layer_quantization_args(q_scheme='uniform',
                                    q_bits = 8,
                                    q_range = 'no_clip')
model.summary()

bit_matched_model = model.get_bitmatch_architecture('NDP120_B0')
flat_configs = bit_matched_model.get_flattened_serialization()
model_labels = ["down", "go", "left", "off", "no", "on", "right", "stop", "up", "yes", "silence", "unknown"]

posterior_json_path =  'posterior_mlperf.json'
with open(posterior_json_path, "r") as f:
    posterior_handler_output_json = json.load(f)
package_filename = 'tinyMLPerf.synpkg'
ph_params_filename = "ph_params.json"

package = SyntiantPackageFactory.build_package(chip='NDP120_B0',
                                    version_string="mlperf tiny april 2021",
                                    config_data=[flat_configs],
                                    class_labels=[model_labels])

with open(ph_params_filename, "w") as ph_params_file:
    ph_params_file.write(json.dumps(posterior_handler_output_json))
package.copy_from_package(file_name=ph_params_filename)

package.to_file(filename=package_filename)
