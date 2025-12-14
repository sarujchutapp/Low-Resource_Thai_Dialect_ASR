from espnet2.bin.asr_inference import Speech2Text

config_path = "/mnt/c/users/Saruj/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/espnet_model_zoo/models--SLSCU--thai-dialect_thai-central_model/snapshots/ecc43657e3720390b366672f3bbe67a2504d040b/thai-central/config.yaml"
model_path = "/mnt/c/users/Saruj/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/espnet_model_zoo/models--SLSCU--thai-dialect_thai-central_model/snapshots/ecc43657e3720390b366672f3bbe67a2504d040b/thai-central/valid.acc.ave_10best.pth"

# Load model
s2t = Speech2Text(
    asr_train_config=config_path, 
    asr_model_file=model_path, 
    minlenratio=0.0, maxlenratio=0.0
)
model = s2t.asr_model

# Drill down into the FIRST layer of the encoder
encoder_layer_0 = model.encoder.encoders.encoder.layers[0]

print("--- INSIDE ENCODER LAYER 0 ---")
for name, module in encoder_layer_0.named_modules():
    # Only print it if it is a Linear layer (the ones we want to LoRA)
    if "Linear" in str(type(module)):
        print(f"Name: {name}  |  Type: {type(module).__name__}")