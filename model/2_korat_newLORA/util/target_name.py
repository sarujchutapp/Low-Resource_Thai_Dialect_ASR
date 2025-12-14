from espnet2.bin.asr_inference import Speech2Text

config_path = "/mnt/c/users/Saruj/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/espnet_model_zoo/models--SLSCU--thai-dialect_thai-central_model/snapshots/ecc43657e3720390b366672f3bbe67a2504d040b/thai-central/config.yaml"
model_path = "/mnt/c/users/Saruj/espnet/tools/miniconda/envs/espnet/lib/python3.11/site-packages/espnet_model_zoo/models--SLSCU--thai-dialect_thai-central_model/snapshots/ecc43657e3720390b366672f3bbe67a2504d040b/thai-central/valid.acc.ave_10best.pth"

# Load your model (using the same path logic you used in train.py)
s2t = Speech2Text(
    asr_train_config=config_path, 
    asr_model_file=model_path, 
    minlenratio=0.0, maxlenratio=0.0
)
model = s2t.asr_model

# PRINT ALL MODULE NAMES
print("--- List of potential LoRA Targets ---")
for name, module in model.named_modules():
    # We only print "top level" layers to keep the list readable
    if len(name.split(".")) < 6: 
        print(name)