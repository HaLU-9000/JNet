import os
import argparse
import json

import utils

# get args
parser = argparse.ArgumentParser(description='')
parser.add_argument('image_name')
parser.add_argument('model_name')
parser.add_argument('-dna')
parser.add_argument('-t', '--train_mode')
args = parser.parse_args()
configs = open(os.path.join("experiments/configs",f"{args.model_name}.json"))
configs = json.load(configs)
params  = configs["params"]
shape = [80, 112, 112]
image_basename = utils.get_basename(args.image_name)

# image load
image_org  = utils.load_anything(args.image_name)
image      = utils.ImageProcessing(image_org)

# model load
model = utils.init_model(params, is_finetuning = True)
utils.load_model_weight(model, model_name = args.model_name)
utils.mount_model_to_device(model, configs = configs)
model.eval()
image.deconv_model = model

# align model load (if neccesary) and apply
if args.dna is not None:
    align_params  = configs["align_params"]
    de_noise_aligner = utils.init_dna(align_params)
    utils.load_model_weight(de_noise_aligner, model_name = args.dna)
    utils.mount_model_to_device(de_noise_aligner, configs=configs)
    de_noise_aligner.eval()
    image.align_model = de_noise_aligner
    
    image.apply_both(
        align_model  = de_noise_aligner                                  ,
        deconv_model = model                                             ,
        align_params = align_params                                      ,
        params       = params                                            ,
        chunk_shape  = shape                                             ,
        overlap      = [0,0,0]                                           ,
        file_align   = f"_apply_test/{image_basename}_{args.dna}"        ,
        file_deconv  = f"_apply_test/{image_basename}_{args.model_name}" ,
        format       = "tif"                                             ,
        bit          =  12
    )

else:
    image.process_image(model, params, shape, "enhanced_image",
                        overlap=[0, 0, 0], apply_hill=True)
    image.save_processed_image(
        file   = f"_apply_test/{image_basename}_{args.model_name}",
        format = "tif",
        bit    = 12)

# example usage:
# python3 apply.py  /home/haruhiko/Downloads/Set_03/MDA15_20230915.nd2 JNet_510
# python3 apply.py  _wakelabdata_processed/1_Spine_structure_AD_175-11w-D3-xyz6-020C2-T1.tif JNet_510