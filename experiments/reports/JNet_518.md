



# JNet_518 Report
  
training with danet 517  
pretrained model : JNet_487_pretrain
## Model Parameters
  

|Parameter|Value|Comment|
| :--- | :--- | :--- |
|hidden_channels_list|[16, 32, 64, 128, 256]||
|attn_list|[False, False, False, False, False]||
|nblocks|2||
|activation|nn.ReLU(inplace=True)||
|dropout|0.5||
|superres|True||
|partial|None||
|reconstruct|False||
|apply_vq|False||
|use_x_quantized|False||
|threshold|0.5||
|use_fftconv|True||
|mu_z|1.2||
|sig_z|0.3||
|blur_mode|gibsonlanni|`gaussian` or `gibsonlanni`|
|size_x|51||
|size_y|51||
|size_z|201||
|NA|0.3||
|wavelength|0.5|microns|
|M|25|magnification|
|ns|1.4|specimen refractive index (RI)|
|ng0|1.5|coverslip RI design value|
|ng|1.5|coverslip RI experimental value|
|ni0|1.33|immersion medium RI design value|
|ni|1.33|immersion medium RI experimental value|
|ti0|150|microns, working distance (immersion medium thickness) design value|
|tg0|170|microns, coverslip thickness design value|
|tg|170|microns, coverslip thickness experimental value|
|res_lateral|0.31|microns|
|res_axial|1.0|microns|
|pZ|0|microns, particle distance from coverslip|
|bet_z|30.0||
|bet_xy|3.0||
|poisson_weight|0.1||
|sig_eps|0.01||
|background|0.01||
|scale|3||
|mid|20|num of NeurIPSF middle channel|
|loss_fn|nn.MSELoss()|loss func for NeurIPSF|
|lr|0.01|lr for pre-training NeurIPSF|
|num_iter_psf_pretrain|1000|epoch for pre-training of NeurIPSF|
|device|cuda:6||

## Datasets and other training details

### simulation_data_generation

|Parameter|Value|
| :--- | :--- |
|dataset_name|_var_num_realisticdata3|
|train_num|16|
|valid_num|4|
|image_size|[1200, 500, 500]|
|train_object_num_min|1500|
|train_object_num_max|2500|
|valid_object_num_min|1500|
|valid_object_num_max|2500|

### pretrain_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_var_num_realisticdata3|
|labelname|_label|
|size|[1200, 500, 500]|
|cropsize|[240, 112, 112]|
|I|200|
|low|0|
|high|16|
|scale|3|
|mask|True|
|mask_size|[1, 10, 10]|
|mask_num|30|
|surround|False|
|surround_size|[32, 4, 4]|

### pretrain_val_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_var_num_realisticdata3|
|labelname|_label|
|size|[1200, 500, 500]|
|cropsize|[240, 112, 112]|
|I|20|
|low|16|
|high|20|
|scale|3|
|mask|False|
|mask_size|[1, 10, 10]|
|mask_num|False|
|surround|False|
|surround_size|[32, 4, 4]|
|seed|907|

### train_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|/home/morita/home/Downloads/Set_03|
|size|[333, 1024, 1024]|
|cropsize|[240, 112, 112]|
|I|200|
|scale|3|
|train|True|
|mask|True|
|mask_size|[1, 10, 10]|
|mask_num|10|
|surround|False|
|surround_size|[32, 4, 4]|

### val_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|/home/morita/home/Downloads/Set_03|
|size|[333, 1024, 1024]|
|cropsize|[240, 112, 112]|
|I|20|
|scale|3|
|train|False|
|mask|False|
|mask_size|[1, 10, 10]|
|mask_num|10|
|surround|False|
|surround_size|[32, 4, 4]|
|seed|1204|

### pretrain_loop

|Parameter|Value|
| :--- | :--- |
|batch_size|1|
|n_epochs|50|
|lr|0.001|
|loss_fnx|nn.BCELoss()|
|loss_fnz|nn.BCELoss()|
|path|model|
|savefig_path|train|
|partial|params['partial']|
|ewc|None|
|es_patience|10|
|is_vibrate|False|
|weight_x|1|
|weight_z|1|

### train_loop

|Parameter|Value|
| :--- | :--- |
|batch_size|1|
|n_epochs|50|
|lr|0.001|
|loss_fn|nn.MSELoss()|
|path|model|
|savefig_path|train|
|partial|None|
|ewc|True|
|params|params|
|es_patience|10|
|reconstruct|True|
|is_instantblur|False|
|is_vibrate|False|
|adjust_luminance|False|
|zloss_weight|1|
|ewc_weight|0.1|
|qloss_weight|1|
|ploss_weight|1.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results

### Pretraining
  
Segmentation: mean MSE: 0.01835298165678978, mean BCE: 0.09800174087285995  
Luminance Estimation: mean MSE: 0.9610462188720703, mean BCE: inf
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_0_original_plane]|![JNet_487_pretrain_0_novibrate_plane]|![JNet_487_pretrain_0_aligned_plane]|![JNet_487_pretrain_0_outputx_plane]|![JNet_487_pretrain_0_labelx_plane]|![JNet_487_pretrain_0_outputz_plane]|![JNet_487_pretrain_0_labelz_plane]|
  
MSEx: 0.015531417913734913, BCEx: 0.09179729968309402  
MSEz: 0.9700160622596741, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_0_original_depth]|![JNet_487_pretrain_0_novibrate_depth]|![JNet_487_pretrain_0_aligned_depth]|![JNet_487_pretrain_0_outputx_depth]|![JNet_487_pretrain_0_labelx_depth]|![JNet_487_pretrain_0_outputz_depth]|![JNet_487_pretrain_0_labelz_depth]|
  
MSEx: 0.015531417913734913, BCEx: 0.09179729968309402  
MSEz: 0.9700160622596741, BCEz: inf  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_1_original_plane]|![JNet_487_pretrain_1_novibrate_plane]|![JNet_487_pretrain_1_aligned_plane]|![JNet_487_pretrain_1_outputx_plane]|![JNet_487_pretrain_1_labelx_plane]|![JNet_487_pretrain_1_outputz_plane]|![JNet_487_pretrain_1_labelz_plane]|
  
MSEx: 0.016710296273231506, BCEx: 0.0911002829670906  
MSEz: 0.9638848900794983, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_1_original_depth]|![JNet_487_pretrain_1_novibrate_depth]|![JNet_487_pretrain_1_aligned_depth]|![JNet_487_pretrain_1_outputx_depth]|![JNet_487_pretrain_1_labelx_depth]|![JNet_487_pretrain_1_outputz_depth]|![JNet_487_pretrain_1_labelz_depth]|
  
MSEx: 0.016710296273231506, BCEx: 0.0911002829670906  
MSEz: 0.9638848900794983, BCEz: inf  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_2_original_plane]|![JNet_487_pretrain_2_novibrate_plane]|![JNet_487_pretrain_2_aligned_plane]|![JNet_487_pretrain_2_outputx_plane]|![JNet_487_pretrain_2_labelx_plane]|![JNet_487_pretrain_2_outputz_plane]|![JNet_487_pretrain_2_labelz_plane]|
  
MSEx: 0.015841756016016006, BCEx: 0.06817679852247238  
MSEz: 0.9544236660003662, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_2_original_depth]|![JNet_487_pretrain_2_novibrate_depth]|![JNet_487_pretrain_2_aligned_depth]|![JNet_487_pretrain_2_outputx_depth]|![JNet_487_pretrain_2_labelx_depth]|![JNet_487_pretrain_2_outputz_depth]|![JNet_487_pretrain_2_labelz_depth]|
  
MSEx: 0.015841756016016006, BCEx: 0.06817679852247238  
MSEz: 0.9544236660003662, BCEz: inf  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_3_original_plane]|![JNet_487_pretrain_3_novibrate_plane]|![JNet_487_pretrain_3_aligned_plane]|![JNet_487_pretrain_3_outputx_plane]|![JNet_487_pretrain_3_labelx_plane]|![JNet_487_pretrain_3_outputz_plane]|![JNet_487_pretrain_3_labelz_plane]|
  
MSEx: 0.016223685815930367, BCEx: 0.11346706748008728  
MSEz: 0.9622484445571899, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_3_original_depth]|![JNet_487_pretrain_3_novibrate_depth]|![JNet_487_pretrain_3_aligned_depth]|![JNet_487_pretrain_3_outputx_depth]|![JNet_487_pretrain_3_labelx_depth]|![JNet_487_pretrain_3_outputz_depth]|![JNet_487_pretrain_3_labelz_depth]|
  
MSEx: 0.016223685815930367, BCEx: 0.11346706748008728  
MSEz: 0.9622484445571899, BCEz: inf  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_4_original_plane]|![JNet_487_pretrain_4_novibrate_plane]|![JNet_487_pretrain_4_aligned_plane]|![JNet_487_pretrain_4_outputx_plane]|![JNet_487_pretrain_4_labelx_plane]|![JNet_487_pretrain_4_outputz_plane]|![JNet_487_pretrain_4_labelz_plane]|
  
MSEx: 0.027457749471068382, BCEx: 0.12546728551387787  
MSEz: 0.9546579122543335, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_4_original_depth]|![JNet_487_pretrain_4_novibrate_depth]|![JNet_487_pretrain_4_aligned_depth]|![JNet_487_pretrain_4_outputx_depth]|![JNet_487_pretrain_4_labelx_depth]|![JNet_487_pretrain_4_outputz_depth]|![JNet_487_pretrain_4_labelz_depth]|
  
MSEx: 0.027457749471068382, BCEx: 0.12546728551387787  
MSEz: 0.9546579122543335, BCEz: inf  

### Finetuning Results with Simulation

### image 0

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_0_original_plane]|![JNet_518_0_novibrate_plane]|![JNet_518_0_aligned_plane]|![JNet_518_0_reconst_plane]|![JNet_518_0_heatmap_plane]|![JNet_518_0_outputx_plane]|![JNet_518_0_labelx_plane]|![JNet_518_0_outputz_plane]|![JNet_518_0_labelz_plane]|
  
MSEz: 0.917123556137085, quantized loss: 0.0007686115568503737  

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_0_original_depth]|![JNet_518_0_novibrate_depth]|![JNet_518_0_aligned_depth]|![JNet_518_0_reconst_depth]|![JNet_518_0_heatmap_depth]|![JNet_518_0_outputx_depth]|![JNet_518_0_labelx_depth]|![JNet_518_0_outputz_depth]|![JNet_518_0_labelz_depth]|
  
MSEz: 0.917123556137085, quantized loss: 0.0007686115568503737  

### image 1

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_1_original_plane]|![JNet_518_1_novibrate_plane]|![JNet_518_1_aligned_plane]|![JNet_518_1_reconst_plane]|![JNet_518_1_heatmap_plane]|![JNet_518_1_outputx_plane]|![JNet_518_1_labelx_plane]|![JNet_518_1_outputz_plane]|![JNet_518_1_labelz_plane]|
  
MSEz: 0.9877804517745972, quantized loss: 6.122546619735658e-05  

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_1_original_depth]|![JNet_518_1_novibrate_depth]|![JNet_518_1_aligned_depth]|![JNet_518_1_reconst_depth]|![JNet_518_1_heatmap_depth]|![JNet_518_1_outputx_depth]|![JNet_518_1_labelx_depth]|![JNet_518_1_outputz_depth]|![JNet_518_1_labelz_depth]|
  
MSEz: 0.9877804517745972, quantized loss: 6.122546619735658e-05  

### image 2

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_2_original_plane]|![JNet_518_2_novibrate_plane]|![JNet_518_2_aligned_plane]|![JNet_518_2_reconst_plane]|![JNet_518_2_heatmap_plane]|![JNet_518_2_outputx_plane]|![JNet_518_2_labelx_plane]|![JNet_518_2_outputz_plane]|![JNet_518_2_labelz_plane]|
  
MSEz: 0.9767466187477112, quantized loss: 0.00044308698852546513  

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_2_original_depth]|![JNet_518_2_novibrate_depth]|![JNet_518_2_aligned_depth]|![JNet_518_2_reconst_depth]|![JNet_518_2_heatmap_depth]|![JNet_518_2_outputx_depth]|![JNet_518_2_labelx_depth]|![JNet_518_2_outputz_depth]|![JNet_518_2_labelz_depth]|
  
MSEz: 0.9767466187477112, quantized loss: 0.00044308698852546513  

### image 3

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_3_original_plane]|![JNet_518_3_novibrate_plane]|![JNet_518_3_aligned_plane]|![JNet_518_3_reconst_plane]|![JNet_518_3_heatmap_plane]|![JNet_518_3_outputx_plane]|![JNet_518_3_labelx_plane]|![JNet_518_3_outputz_plane]|![JNet_518_3_labelz_plane]|
  
MSEz: 0.984675943851471, quantized loss: 0.0004151031607761979  

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_3_original_depth]|![JNet_518_3_novibrate_depth]|![JNet_518_3_aligned_depth]|![JNet_518_3_reconst_depth]|![JNet_518_3_heatmap_depth]|![JNet_518_3_outputx_depth]|![JNet_518_3_labelx_depth]|![JNet_518_3_outputz_depth]|![JNet_518_3_labelz_depth]|
  
MSEz: 0.984675943851471, quantized loss: 0.0004151031607761979  

### image 4

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_4_original_plane]|![JNet_518_4_novibrate_plane]|![JNet_518_4_aligned_plane]|![JNet_518_4_reconst_plane]|![JNet_518_4_heatmap_plane]|![JNet_518_4_outputx_plane]|![JNet_518_4_labelx_plane]|![JNet_518_4_outputz_plane]|![JNet_518_4_labelz_plane]|
  
MSEz: 0.9912257790565491, quantized loss: 5.41476983926259e-05  

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_4_original_depth]|![JNet_518_4_novibrate_depth]|![JNet_518_4_aligned_depth]|![JNet_518_4_reconst_depth]|![JNet_518_4_heatmap_depth]|![JNet_518_4_outputx_depth]|![JNet_518_4_labelx_depth]|![JNet_518_4_outputz_depth]|![JNet_518_4_labelz_depth]|
  
MSEz: 0.9912257790565491, quantized loss: 5.41476983926259e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
### Finetuning Results with Microglia

#### finetuning == False

### image 0

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_microglia_0_original_plane]|![JNet_487_pretrain_microglia_0_aligned_plane]|![JNet_487_pretrain_microglia_0_outputx_plane]|![JNet_487_pretrain_microglia_0_outputz_plane]|![JNet_487_pretrain_microglia_0_reconst_plane]|![JNet_487_pretrain_microglia_0_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_microglia_0_original_depth]|![JNet_487_pretrain_microglia_0_aligned_depth]|![JNet_487_pretrain_microglia_0_outputx_depth]|![JNet_487_pretrain_microglia_0_outputz_depth]|![JNet_487_pretrain_microglia_0_reconst_depth]|![JNet_487_pretrain_microglia_0_heatmap_depth]|
  

### image 1

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_microglia_1_original_plane]|![JNet_487_pretrain_microglia_1_aligned_plane]|![JNet_487_pretrain_microglia_1_outputx_plane]|![JNet_487_pretrain_microglia_1_outputz_plane]|![JNet_487_pretrain_microglia_1_reconst_plane]|![JNet_487_pretrain_microglia_1_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_microglia_1_original_depth]|![JNet_487_pretrain_microglia_1_aligned_depth]|![JNet_487_pretrain_microglia_1_outputx_depth]|![JNet_487_pretrain_microglia_1_outputz_depth]|![JNet_487_pretrain_microglia_1_reconst_depth]|![JNet_487_pretrain_microglia_1_heatmap_depth]|
  

### image 2

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_microglia_2_original_plane]|![JNet_487_pretrain_microglia_2_aligned_plane]|![JNet_487_pretrain_microglia_2_outputx_plane]|![JNet_487_pretrain_microglia_2_outputz_plane]|![JNet_487_pretrain_microglia_2_reconst_plane]|![JNet_487_pretrain_microglia_2_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_microglia_2_original_depth]|![JNet_487_pretrain_microglia_2_aligned_depth]|![JNet_487_pretrain_microglia_2_outputx_depth]|![JNet_487_pretrain_microglia_2_outputz_depth]|![JNet_487_pretrain_microglia_2_reconst_depth]|![JNet_487_pretrain_microglia_2_heatmap_depth]|
  

### image 3

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_microglia_3_original_plane]|![JNet_487_pretrain_microglia_3_aligned_plane]|![JNet_487_pretrain_microglia_3_outputx_plane]|![JNet_487_pretrain_microglia_3_outputz_plane]|![JNet_487_pretrain_microglia_3_reconst_plane]|![JNet_487_pretrain_microglia_3_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_microglia_3_original_depth]|![JNet_487_pretrain_microglia_3_aligned_depth]|![JNet_487_pretrain_microglia_3_outputx_depth]|![JNet_487_pretrain_microglia_3_outputz_depth]|![JNet_487_pretrain_microglia_3_reconst_depth]|![JNet_487_pretrain_microglia_3_heatmap_depth]|
  

### image 4

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_microglia_4_original_plane]|![JNet_487_pretrain_microglia_4_aligned_plane]|![JNet_487_pretrain_microglia_4_outputx_plane]|![JNet_487_pretrain_microglia_4_outputz_plane]|![JNet_487_pretrain_microglia_4_reconst_plane]|![JNet_487_pretrain_microglia_4_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_487_pretrain_microglia_4_original_depth]|![JNet_487_pretrain_microglia_4_aligned_depth]|![JNet_487_pretrain_microglia_4_outputx_depth]|![JNet_487_pretrain_microglia_4_outputz_depth]|![JNet_487_pretrain_microglia_4_reconst_depth]|![JNet_487_pretrain_microglia_4_heatmap_depth]|
  

#### finetuning == True

### image 0

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_microglia_0_original_plane]|![JNet_518_microglia_0_aligned_plane]|![JNet_518_microglia_0_outputx_plane]|![JNet_518_microglia_0_outputz_plane]|![JNet_518_microglia_0_reconst_plane]|![JNet_518_microglia_0_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_microglia_0_original_depth]|![JNet_518_microglia_0_aligned_depth]|![JNet_518_microglia_0_outputx_depth]|![JNet_518_microglia_0_outputz_depth]|![JNet_518_microglia_0_reconst_depth]|![JNet_518_microglia_0_heatmap_depth]|
  

### image 1

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_microglia_1_original_plane]|![JNet_518_microglia_1_aligned_plane]|![JNet_518_microglia_1_outputx_plane]|![JNet_518_microglia_1_outputz_plane]|![JNet_518_microglia_1_reconst_plane]|![JNet_518_microglia_1_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_microglia_1_original_depth]|![JNet_518_microglia_1_aligned_depth]|![JNet_518_microglia_1_outputx_depth]|![JNet_518_microglia_1_outputz_depth]|![JNet_518_microglia_1_reconst_depth]|![JNet_518_microglia_1_heatmap_depth]|
  

### image 2

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_microglia_2_original_plane]|![JNet_518_microglia_2_aligned_plane]|![JNet_518_microglia_2_outputx_plane]|![JNet_518_microglia_2_outputz_plane]|![JNet_518_microglia_2_reconst_plane]|![JNet_518_microglia_2_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_microglia_2_original_depth]|![JNet_518_microglia_2_aligned_depth]|![JNet_518_microglia_2_outputx_depth]|![JNet_518_microglia_2_outputz_depth]|![JNet_518_microglia_2_reconst_depth]|![JNet_518_microglia_2_heatmap_depth]|
  

### image 3

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_microglia_3_original_plane]|![JNet_518_microglia_3_aligned_plane]|![JNet_518_microglia_3_outputx_plane]|![JNet_518_microglia_3_outputz_plane]|![JNet_518_microglia_3_reconst_plane]|![JNet_518_microglia_3_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_microglia_3_original_depth]|![JNet_518_microglia_3_aligned_depth]|![JNet_518_microglia_3_outputx_depth]|![JNet_518_microglia_3_outputz_depth]|![JNet_518_microglia_3_reconst_depth]|![JNet_518_microglia_3_heatmap_depth]|
  

### image 4

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_microglia_4_original_plane]|![JNet_518_microglia_4_aligned_plane]|![JNet_518_microglia_4_outputx_plane]|![JNet_518_microglia_4_outputz_plane]|![JNet_518_microglia_4_reconst_plane]|![JNet_518_microglia_4_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_518_microglia_4_original_depth]|![JNet_518_microglia_4_aligned_depth]|![JNet_518_microglia_4_outputx_depth]|![JNet_518_microglia_4_outputz_depth]|![JNet_518_microglia_4_reconst_depth]|![JNet_518_microglia_4_heatmap_depth]|
  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_518_psf_pre]|![JNet_518_psf_post]|

## Architecture
  
  
```  
JNet(  
  (prev0): JNetBlock0(  
    (conv): Conv3d(1, 16, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
  )  
  (prev): ModuleList(  
    (0-1): 2 x JNetBlock(  
      (bn1): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu1): ReLU(inplace=True)  
      (conv1): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
      (bn2): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu2): ReLU(inplace=True)  
      (dropout1): Dropout(p=0.5, inplace=False)  
      (conv2): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
    )  
  )  
  (mid): JNetLayer(  
    (pool): JNetPooling(  
      (maxpool): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
      (conv): Conv3d(16, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
      (relu): ReLU(inplace=True)  
    )  
    (conv): Conv3d(32, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
    (prev): ModuleList(  
      (0-1): 2 x JNetBlock(  
        (bn1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
        (relu1): ReLU(inplace=True)  
        (conv1): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
        (bn2): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
        (relu2): ReLU(inplace=True)  
        (dropout1): Dropout(p=0.5, inplace=False)  
        (conv2): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
      )  
    )  
    (mid): JNetLayer(  
      (pool): JNetPooling(  
        (maxpool): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
        (conv): Conv3d(32, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
        (relu): ReLU(inplace=True)  
      )  
      (conv): Conv3d(64, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
      (prev): ModuleList(  
        (0-1): 2 x JNetBlock(  
          (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
          (relu1): ReLU(inplace=True)  
          (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
          (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
          (relu2): ReLU(inplace=True)  
          (dropout1): Dropout(p=0.5, inplace=False)  
          (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
        )  
      )  
      (mid): JNetLayer(  
        (pool): JNetPooling(  
          (maxpool): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
          (conv): Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
          (relu): ReLU(inplace=True)  
        )  
        (conv): Conv3d(128, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
        (prev): ModuleList(  
          (0-1): 2 x JNetBlock(  
            (bn1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            (relu1): ReLU(inplace=True)  
            (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
            (bn2): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            (relu2): ReLU(inplace=True)  
            (dropout1): Dropout(p=0.5, inplace=False)  
            (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
          )  
        )  
        (mid): JNetLayer(  
          (pool): JNetPooling(  
            (maxpool): MaxPool3d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)  
            (conv): Conv3d(128, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
            (relu): ReLU(inplace=True)  
          )  
          (conv): Conv3d(256, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
          (prev): ModuleList(  
            (0-1): 2 x JNetBlock(  
              (bn1): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
              (relu1): ReLU(inplace=True)  
              (conv1): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
              (bn2): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
              (relu2): ReLU(inplace=True)  
              (dropout1): Dropout(p=0.5, inplace=False)  
              (conv2): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
            )  
          )  
          (mid): Identity()  
          (attn): Identity()  
          (post): ModuleList(  
            (0-1): 2 x JNetBlock(  
              (bn1): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
              (relu1): ReLU(inplace=True)  
              (conv1): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
              (bn2): BatchNorm3d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
              (relu2): ReLU(inplace=True)  
              (dropout1): Dropout(p=0.5, inplace=False)  
              (conv2): Conv3d(256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
            )  
          )  
          (unpool): JNetUnpooling(  
            (upsample): Upsample(scale_factor=2.0, mode='trilinear')  
            (conv): Conv3d(256, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
            (relu): ReLU(inplace=True)  
          )  
        )  
        (attn): Identity()  
        (post): ModuleList(  
          (0-1): 2 x JNetBlock(  
            (bn1): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            (relu1): ReLU(inplace=True)  
            (conv1): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
            (bn2): BatchNorm3d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
            (relu2): ReLU(inplace=True)  
            (dropout1): Dropout(p=0.5, inplace=False)  
            (conv2): Conv3d(128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
          )  
        )  
        (unpool): JNetUnpooling(  
          (upsample): Upsample(scale_factor=2.0, mode='trilinear')  
          (conv): Conv3d(128, 64, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
          (relu): ReLU(inplace=True)  
        )  
      )  
      (attn): Identity()  
      (post): ModuleList(  
        (0-1): 2 x JNetBlock(  
          (bn1): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
          (relu1): ReLU(inplace=True)  
          (conv1): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
          (bn2): BatchNorm3d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
          (relu2): ReLU(inplace=True)  
          (dropout1): Dropout(p=0.5, inplace=False)  
          (conv2): Conv3d(64, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
        )  
      )  
      (unpool): JNetUnpooling(  
        (upsample): Upsample(scale_factor=2.0, mode='trilinear')  
        (conv): Conv3d(64, 32, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
        (relu): ReLU(inplace=True)  
      )  
    )  
    (attn): Identity()  
    (post): ModuleList(  
      (0-1): 2 x JNetBlock(  
        (bn1): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
        (relu1): ReLU(inplace=True)  
        (conv1): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
        (bn2): BatchNorm3d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
        (relu2): ReLU(inplace=True)  
        (dropout1): Dropout(p=0.5, inplace=False)  
        (conv2): Conv3d(32, 32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
      )  
    )  
    (unpool): JNetUnpooling(  
      (upsample): Upsample(scale_factor=2.0, mode='trilinear')  
      (conv): Conv3d(32, 16, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
      (relu): ReLU(inplace=True)  
    )  
  )  
  (postx): ModuleList(  
    (0-1): 2 x JNetBlock(  
      (bn1): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu1): ReLU(inplace=True)  
      (conv1): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
      (bn2): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu2): ReLU(inplace=True)  
      (dropout1): Dropout(p=0.5, inplace=False)  
      (conv2): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
    )  
    (2): JNetBlockN(  
      (conv): Conv3d(16, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
      (sigm): Sigmoid()  
    )  
  )  
  (postz): ModuleList(  
    (0-1): 2 x JNetBlock(  
      (bn1): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu1): ReLU(inplace=True)  
      (conv1): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
      (bn2): BatchNorm3d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
      (relu2): ReLU(inplace=True)  
      (dropout1): Dropout(p=0.5, inplace=False)  
      (conv2): Conv3d(16, 16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
    )  
    (2): JNetBlockN(  
      (conv): Conv3d(16, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
      (sigm): Sigmoid()  
    )  
  )  
  (image): ImagingProcess(  
    (emission): Emission()  
    (blur): Blur(  
      (neuripsf): NeuralImplicitPSF(  
        (layers): Sequential(  
          (0): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
          (1): Linear(in_features=2, out_features=20, bias=True)  
          (2): Sigmoid()  
          (3): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
          (4): Linear(in_features=20, out_features=1, bias=True)  
          (5): Sigmoid()  
        )  
      )  
    )  
    (noise): Noise()  
    (preprocess): PreProcess()  
    (hill): Hill()  
  )  
  (upsample): JNetUpsample(  
    (upsample): Upsample(scale_factor=(3.0, 1.0, 1.0), mode='trilinear')  
  )  
  (vq): VectorQuantizer()  
)  
```  
  



[JNet_487_pretrain_0_aligned_depth]: /experiments/images/JNet_487_pretrain_0_aligned_depth.png
[JNet_487_pretrain_0_aligned_plane]: /experiments/images/JNet_487_pretrain_0_aligned_plane.png
[JNet_487_pretrain_0_labelx_depth]: /experiments/images/JNet_487_pretrain_0_labelx_depth.png
[JNet_487_pretrain_0_labelx_plane]: /experiments/images/JNet_487_pretrain_0_labelx_plane.png
[JNet_487_pretrain_0_labelz_depth]: /experiments/images/JNet_487_pretrain_0_labelz_depth.png
[JNet_487_pretrain_0_labelz_plane]: /experiments/images/JNet_487_pretrain_0_labelz_plane.png
[JNet_487_pretrain_0_novibrate_depth]: /experiments/images/JNet_487_pretrain_0_novibrate_depth.png
[JNet_487_pretrain_0_novibrate_plane]: /experiments/images/JNet_487_pretrain_0_novibrate_plane.png
[JNet_487_pretrain_0_original_depth]: /experiments/images/JNet_487_pretrain_0_original_depth.png
[JNet_487_pretrain_0_original_plane]: /experiments/images/JNet_487_pretrain_0_original_plane.png
[JNet_487_pretrain_0_outputx_depth]: /experiments/images/JNet_487_pretrain_0_outputx_depth.png
[JNet_487_pretrain_0_outputx_plane]: /experiments/images/JNet_487_pretrain_0_outputx_plane.png
[JNet_487_pretrain_0_outputz_depth]: /experiments/images/JNet_487_pretrain_0_outputz_depth.png
[JNet_487_pretrain_0_outputz_plane]: /experiments/images/JNet_487_pretrain_0_outputz_plane.png
[JNet_487_pretrain_1_aligned_depth]: /experiments/images/JNet_487_pretrain_1_aligned_depth.png
[JNet_487_pretrain_1_aligned_plane]: /experiments/images/JNet_487_pretrain_1_aligned_plane.png
[JNet_487_pretrain_1_labelx_depth]: /experiments/images/JNet_487_pretrain_1_labelx_depth.png
[JNet_487_pretrain_1_labelx_plane]: /experiments/images/JNet_487_pretrain_1_labelx_plane.png
[JNet_487_pretrain_1_labelz_depth]: /experiments/images/JNet_487_pretrain_1_labelz_depth.png
[JNet_487_pretrain_1_labelz_plane]: /experiments/images/JNet_487_pretrain_1_labelz_plane.png
[JNet_487_pretrain_1_novibrate_depth]: /experiments/images/JNet_487_pretrain_1_novibrate_depth.png
[JNet_487_pretrain_1_novibrate_plane]: /experiments/images/JNet_487_pretrain_1_novibrate_plane.png
[JNet_487_pretrain_1_original_depth]: /experiments/images/JNet_487_pretrain_1_original_depth.png
[JNet_487_pretrain_1_original_plane]: /experiments/images/JNet_487_pretrain_1_original_plane.png
[JNet_487_pretrain_1_outputx_depth]: /experiments/images/JNet_487_pretrain_1_outputx_depth.png
[JNet_487_pretrain_1_outputx_plane]: /experiments/images/JNet_487_pretrain_1_outputx_plane.png
[JNet_487_pretrain_1_outputz_depth]: /experiments/images/JNet_487_pretrain_1_outputz_depth.png
[JNet_487_pretrain_1_outputz_plane]: /experiments/images/JNet_487_pretrain_1_outputz_plane.png
[JNet_487_pretrain_2_aligned_depth]: /experiments/images/JNet_487_pretrain_2_aligned_depth.png
[JNet_487_pretrain_2_aligned_plane]: /experiments/images/JNet_487_pretrain_2_aligned_plane.png
[JNet_487_pretrain_2_labelx_depth]: /experiments/images/JNet_487_pretrain_2_labelx_depth.png
[JNet_487_pretrain_2_labelx_plane]: /experiments/images/JNet_487_pretrain_2_labelx_plane.png
[JNet_487_pretrain_2_labelz_depth]: /experiments/images/JNet_487_pretrain_2_labelz_depth.png
[JNet_487_pretrain_2_labelz_plane]: /experiments/images/JNet_487_pretrain_2_labelz_plane.png
[JNet_487_pretrain_2_novibrate_depth]: /experiments/images/JNet_487_pretrain_2_novibrate_depth.png
[JNet_487_pretrain_2_novibrate_plane]: /experiments/images/JNet_487_pretrain_2_novibrate_plane.png
[JNet_487_pretrain_2_original_depth]: /experiments/images/JNet_487_pretrain_2_original_depth.png
[JNet_487_pretrain_2_original_plane]: /experiments/images/JNet_487_pretrain_2_original_plane.png
[JNet_487_pretrain_2_outputx_depth]: /experiments/images/JNet_487_pretrain_2_outputx_depth.png
[JNet_487_pretrain_2_outputx_plane]: /experiments/images/JNet_487_pretrain_2_outputx_plane.png
[JNet_487_pretrain_2_outputz_depth]: /experiments/images/JNet_487_pretrain_2_outputz_depth.png
[JNet_487_pretrain_2_outputz_plane]: /experiments/images/JNet_487_pretrain_2_outputz_plane.png
[JNet_487_pretrain_3_aligned_depth]: /experiments/images/JNet_487_pretrain_3_aligned_depth.png
[JNet_487_pretrain_3_aligned_plane]: /experiments/images/JNet_487_pretrain_3_aligned_plane.png
[JNet_487_pretrain_3_labelx_depth]: /experiments/images/JNet_487_pretrain_3_labelx_depth.png
[JNet_487_pretrain_3_labelx_plane]: /experiments/images/JNet_487_pretrain_3_labelx_plane.png
[JNet_487_pretrain_3_labelz_depth]: /experiments/images/JNet_487_pretrain_3_labelz_depth.png
[JNet_487_pretrain_3_labelz_plane]: /experiments/images/JNet_487_pretrain_3_labelz_plane.png
[JNet_487_pretrain_3_novibrate_depth]: /experiments/images/JNet_487_pretrain_3_novibrate_depth.png
[JNet_487_pretrain_3_novibrate_plane]: /experiments/images/JNet_487_pretrain_3_novibrate_plane.png
[JNet_487_pretrain_3_original_depth]: /experiments/images/JNet_487_pretrain_3_original_depth.png
[JNet_487_pretrain_3_original_plane]: /experiments/images/JNet_487_pretrain_3_original_plane.png
[JNet_487_pretrain_3_outputx_depth]: /experiments/images/JNet_487_pretrain_3_outputx_depth.png
[JNet_487_pretrain_3_outputx_plane]: /experiments/images/JNet_487_pretrain_3_outputx_plane.png
[JNet_487_pretrain_3_outputz_depth]: /experiments/images/JNet_487_pretrain_3_outputz_depth.png
[JNet_487_pretrain_3_outputz_plane]: /experiments/images/JNet_487_pretrain_3_outputz_plane.png
[JNet_487_pretrain_4_aligned_depth]: /experiments/images/JNet_487_pretrain_4_aligned_depth.png
[JNet_487_pretrain_4_aligned_plane]: /experiments/images/JNet_487_pretrain_4_aligned_plane.png
[JNet_487_pretrain_4_labelx_depth]: /experiments/images/JNet_487_pretrain_4_labelx_depth.png
[JNet_487_pretrain_4_labelx_plane]: /experiments/images/JNet_487_pretrain_4_labelx_plane.png
[JNet_487_pretrain_4_labelz_depth]: /experiments/images/JNet_487_pretrain_4_labelz_depth.png
[JNet_487_pretrain_4_labelz_plane]: /experiments/images/JNet_487_pretrain_4_labelz_plane.png
[JNet_487_pretrain_4_novibrate_depth]: /experiments/images/JNet_487_pretrain_4_novibrate_depth.png
[JNet_487_pretrain_4_novibrate_plane]: /experiments/images/JNet_487_pretrain_4_novibrate_plane.png
[JNet_487_pretrain_4_original_depth]: /experiments/images/JNet_487_pretrain_4_original_depth.png
[JNet_487_pretrain_4_original_plane]: /experiments/images/JNet_487_pretrain_4_original_plane.png
[JNet_487_pretrain_4_outputx_depth]: /experiments/images/JNet_487_pretrain_4_outputx_depth.png
[JNet_487_pretrain_4_outputx_plane]: /experiments/images/JNet_487_pretrain_4_outputx_plane.png
[JNet_487_pretrain_4_outputz_depth]: /experiments/images/JNet_487_pretrain_4_outputz_depth.png
[JNet_487_pretrain_4_outputz_plane]: /experiments/images/JNet_487_pretrain_4_outputz_plane.png
[JNet_487_pretrain_microglia_0_aligned_depth]: /experiments/images/JNet_487_pretrain_microglia_0_aligned_depth.png
[JNet_487_pretrain_microglia_0_aligned_plane]: /experiments/images/JNet_487_pretrain_microglia_0_aligned_plane.png
[JNet_487_pretrain_microglia_0_heatmap_depth]: /experiments/images/JNet_487_pretrain_microglia_0_heatmap_depth.png
[JNet_487_pretrain_microglia_0_heatmap_plane]: /experiments/images/JNet_487_pretrain_microglia_0_heatmap_plane.png
[JNet_487_pretrain_microglia_0_original_depth]: /experiments/images/JNet_487_pretrain_microglia_0_original_depth.png
[JNet_487_pretrain_microglia_0_original_plane]: /experiments/images/JNet_487_pretrain_microglia_0_original_plane.png
[JNet_487_pretrain_microglia_0_outputx_depth]: /experiments/images/JNet_487_pretrain_microglia_0_outputx_depth.png
[JNet_487_pretrain_microglia_0_outputx_plane]: /experiments/images/JNet_487_pretrain_microglia_0_outputx_plane.png
[JNet_487_pretrain_microglia_0_outputz_depth]: /experiments/images/JNet_487_pretrain_microglia_0_outputz_depth.png
[JNet_487_pretrain_microglia_0_outputz_plane]: /experiments/images/JNet_487_pretrain_microglia_0_outputz_plane.png
[JNet_487_pretrain_microglia_0_reconst_depth]: /experiments/images/JNet_487_pretrain_microglia_0_reconst_depth.png
[JNet_487_pretrain_microglia_0_reconst_plane]: /experiments/images/JNet_487_pretrain_microglia_0_reconst_plane.png
[JNet_487_pretrain_microglia_1_aligned_depth]: /experiments/images/JNet_487_pretrain_microglia_1_aligned_depth.png
[JNet_487_pretrain_microglia_1_aligned_plane]: /experiments/images/JNet_487_pretrain_microglia_1_aligned_plane.png
[JNet_487_pretrain_microglia_1_heatmap_depth]: /experiments/images/JNet_487_pretrain_microglia_1_heatmap_depth.png
[JNet_487_pretrain_microglia_1_heatmap_plane]: /experiments/images/JNet_487_pretrain_microglia_1_heatmap_plane.png
[JNet_487_pretrain_microglia_1_original_depth]: /experiments/images/JNet_487_pretrain_microglia_1_original_depth.png
[JNet_487_pretrain_microglia_1_original_plane]: /experiments/images/JNet_487_pretrain_microglia_1_original_plane.png
[JNet_487_pretrain_microglia_1_outputx_depth]: /experiments/images/JNet_487_pretrain_microglia_1_outputx_depth.png
[JNet_487_pretrain_microglia_1_outputx_plane]: /experiments/images/JNet_487_pretrain_microglia_1_outputx_plane.png
[JNet_487_pretrain_microglia_1_outputz_depth]: /experiments/images/JNet_487_pretrain_microglia_1_outputz_depth.png
[JNet_487_pretrain_microglia_1_outputz_plane]: /experiments/images/JNet_487_pretrain_microglia_1_outputz_plane.png
[JNet_487_pretrain_microglia_1_reconst_depth]: /experiments/images/JNet_487_pretrain_microglia_1_reconst_depth.png
[JNet_487_pretrain_microglia_1_reconst_plane]: /experiments/images/JNet_487_pretrain_microglia_1_reconst_plane.png
[JNet_487_pretrain_microglia_2_aligned_depth]: /experiments/images/JNet_487_pretrain_microglia_2_aligned_depth.png
[JNet_487_pretrain_microglia_2_aligned_plane]: /experiments/images/JNet_487_pretrain_microglia_2_aligned_plane.png
[JNet_487_pretrain_microglia_2_heatmap_depth]: /experiments/images/JNet_487_pretrain_microglia_2_heatmap_depth.png
[JNet_487_pretrain_microglia_2_heatmap_plane]: /experiments/images/JNet_487_pretrain_microglia_2_heatmap_plane.png
[JNet_487_pretrain_microglia_2_original_depth]: /experiments/images/JNet_487_pretrain_microglia_2_original_depth.png
[JNet_487_pretrain_microglia_2_original_plane]: /experiments/images/JNet_487_pretrain_microglia_2_original_plane.png
[JNet_487_pretrain_microglia_2_outputx_depth]: /experiments/images/JNet_487_pretrain_microglia_2_outputx_depth.png
[JNet_487_pretrain_microglia_2_outputx_plane]: /experiments/images/JNet_487_pretrain_microglia_2_outputx_plane.png
[JNet_487_pretrain_microglia_2_outputz_depth]: /experiments/images/JNet_487_pretrain_microglia_2_outputz_depth.png
[JNet_487_pretrain_microglia_2_outputz_plane]: /experiments/images/JNet_487_pretrain_microglia_2_outputz_plane.png
[JNet_487_pretrain_microglia_2_reconst_depth]: /experiments/images/JNet_487_pretrain_microglia_2_reconst_depth.png
[JNet_487_pretrain_microglia_2_reconst_plane]: /experiments/images/JNet_487_pretrain_microglia_2_reconst_plane.png
[JNet_487_pretrain_microglia_3_aligned_depth]: /experiments/images/JNet_487_pretrain_microglia_3_aligned_depth.png
[JNet_487_pretrain_microglia_3_aligned_plane]: /experiments/images/JNet_487_pretrain_microglia_3_aligned_plane.png
[JNet_487_pretrain_microglia_3_heatmap_depth]: /experiments/images/JNet_487_pretrain_microglia_3_heatmap_depth.png
[JNet_487_pretrain_microglia_3_heatmap_plane]: /experiments/images/JNet_487_pretrain_microglia_3_heatmap_plane.png
[JNet_487_pretrain_microglia_3_original_depth]: /experiments/images/JNet_487_pretrain_microglia_3_original_depth.png
[JNet_487_pretrain_microglia_3_original_plane]: /experiments/images/JNet_487_pretrain_microglia_3_original_plane.png
[JNet_487_pretrain_microglia_3_outputx_depth]: /experiments/images/JNet_487_pretrain_microglia_3_outputx_depth.png
[JNet_487_pretrain_microglia_3_outputx_plane]: /experiments/images/JNet_487_pretrain_microglia_3_outputx_plane.png
[JNet_487_pretrain_microglia_3_outputz_depth]: /experiments/images/JNet_487_pretrain_microglia_3_outputz_depth.png
[JNet_487_pretrain_microglia_3_outputz_plane]: /experiments/images/JNet_487_pretrain_microglia_3_outputz_plane.png
[JNet_487_pretrain_microglia_3_reconst_depth]: /experiments/images/JNet_487_pretrain_microglia_3_reconst_depth.png
[JNet_487_pretrain_microglia_3_reconst_plane]: /experiments/images/JNet_487_pretrain_microglia_3_reconst_plane.png
[JNet_487_pretrain_microglia_4_aligned_depth]: /experiments/images/JNet_487_pretrain_microglia_4_aligned_depth.png
[JNet_487_pretrain_microglia_4_aligned_plane]: /experiments/images/JNet_487_pretrain_microglia_4_aligned_plane.png
[JNet_487_pretrain_microglia_4_heatmap_depth]: /experiments/images/JNet_487_pretrain_microglia_4_heatmap_depth.png
[JNet_487_pretrain_microglia_4_heatmap_plane]: /experiments/images/JNet_487_pretrain_microglia_4_heatmap_plane.png
[JNet_487_pretrain_microglia_4_original_depth]: /experiments/images/JNet_487_pretrain_microglia_4_original_depth.png
[JNet_487_pretrain_microglia_4_original_plane]: /experiments/images/JNet_487_pretrain_microglia_4_original_plane.png
[JNet_487_pretrain_microglia_4_outputx_depth]: /experiments/images/JNet_487_pretrain_microglia_4_outputx_depth.png
[JNet_487_pretrain_microglia_4_outputx_plane]: /experiments/images/JNet_487_pretrain_microglia_4_outputx_plane.png
[JNet_487_pretrain_microglia_4_outputz_depth]: /experiments/images/JNet_487_pretrain_microglia_4_outputz_depth.png
[JNet_487_pretrain_microglia_4_outputz_plane]: /experiments/images/JNet_487_pretrain_microglia_4_outputz_plane.png
[JNet_487_pretrain_microglia_4_reconst_depth]: /experiments/images/JNet_487_pretrain_microglia_4_reconst_depth.png
[JNet_487_pretrain_microglia_4_reconst_plane]: /experiments/images/JNet_487_pretrain_microglia_4_reconst_plane.png
[JNet_518_0_aligned_depth]: /experiments/images/JNet_518_0_aligned_depth.png
[JNet_518_0_aligned_plane]: /experiments/images/JNet_518_0_aligned_plane.png
[JNet_518_0_heatmap_depth]: /experiments/images/JNet_518_0_heatmap_depth.png
[JNet_518_0_heatmap_plane]: /experiments/images/JNet_518_0_heatmap_plane.png
[JNet_518_0_labelx_depth]: /experiments/images/JNet_518_0_labelx_depth.png
[JNet_518_0_labelx_plane]: /experiments/images/JNet_518_0_labelx_plane.png
[JNet_518_0_labelz_depth]: /experiments/images/JNet_518_0_labelz_depth.png
[JNet_518_0_labelz_plane]: /experiments/images/JNet_518_0_labelz_plane.png
[JNet_518_0_novibrate_depth]: /experiments/images/JNet_518_0_novibrate_depth.png
[JNet_518_0_novibrate_plane]: /experiments/images/JNet_518_0_novibrate_plane.png
[JNet_518_0_original_depth]: /experiments/images/JNet_518_0_original_depth.png
[JNet_518_0_original_plane]: /experiments/images/JNet_518_0_original_plane.png
[JNet_518_0_outputx_depth]: /experiments/images/JNet_518_0_outputx_depth.png
[JNet_518_0_outputx_plane]: /experiments/images/JNet_518_0_outputx_plane.png
[JNet_518_0_outputz_depth]: /experiments/images/JNet_518_0_outputz_depth.png
[JNet_518_0_outputz_plane]: /experiments/images/JNet_518_0_outputz_plane.png
[JNet_518_0_reconst_depth]: /experiments/images/JNet_518_0_reconst_depth.png
[JNet_518_0_reconst_plane]: /experiments/images/JNet_518_0_reconst_plane.png
[JNet_518_1_aligned_depth]: /experiments/images/JNet_518_1_aligned_depth.png
[JNet_518_1_aligned_plane]: /experiments/images/JNet_518_1_aligned_plane.png
[JNet_518_1_heatmap_depth]: /experiments/images/JNet_518_1_heatmap_depth.png
[JNet_518_1_heatmap_plane]: /experiments/images/JNet_518_1_heatmap_plane.png
[JNet_518_1_labelx_depth]: /experiments/images/JNet_518_1_labelx_depth.png
[JNet_518_1_labelx_plane]: /experiments/images/JNet_518_1_labelx_plane.png
[JNet_518_1_labelz_depth]: /experiments/images/JNet_518_1_labelz_depth.png
[JNet_518_1_labelz_plane]: /experiments/images/JNet_518_1_labelz_plane.png
[JNet_518_1_novibrate_depth]: /experiments/images/JNet_518_1_novibrate_depth.png
[JNet_518_1_novibrate_plane]: /experiments/images/JNet_518_1_novibrate_plane.png
[JNet_518_1_original_depth]: /experiments/images/JNet_518_1_original_depth.png
[JNet_518_1_original_plane]: /experiments/images/JNet_518_1_original_plane.png
[JNet_518_1_outputx_depth]: /experiments/images/JNet_518_1_outputx_depth.png
[JNet_518_1_outputx_plane]: /experiments/images/JNet_518_1_outputx_plane.png
[JNet_518_1_outputz_depth]: /experiments/images/JNet_518_1_outputz_depth.png
[JNet_518_1_outputz_plane]: /experiments/images/JNet_518_1_outputz_plane.png
[JNet_518_1_reconst_depth]: /experiments/images/JNet_518_1_reconst_depth.png
[JNet_518_1_reconst_plane]: /experiments/images/JNet_518_1_reconst_plane.png
[JNet_518_2_aligned_depth]: /experiments/images/JNet_518_2_aligned_depth.png
[JNet_518_2_aligned_plane]: /experiments/images/JNet_518_2_aligned_plane.png
[JNet_518_2_heatmap_depth]: /experiments/images/JNet_518_2_heatmap_depth.png
[JNet_518_2_heatmap_plane]: /experiments/images/JNet_518_2_heatmap_plane.png
[JNet_518_2_labelx_depth]: /experiments/images/JNet_518_2_labelx_depth.png
[JNet_518_2_labelx_plane]: /experiments/images/JNet_518_2_labelx_plane.png
[JNet_518_2_labelz_depth]: /experiments/images/JNet_518_2_labelz_depth.png
[JNet_518_2_labelz_plane]: /experiments/images/JNet_518_2_labelz_plane.png
[JNet_518_2_novibrate_depth]: /experiments/images/JNet_518_2_novibrate_depth.png
[JNet_518_2_novibrate_plane]: /experiments/images/JNet_518_2_novibrate_plane.png
[JNet_518_2_original_depth]: /experiments/images/JNet_518_2_original_depth.png
[JNet_518_2_original_plane]: /experiments/images/JNet_518_2_original_plane.png
[JNet_518_2_outputx_depth]: /experiments/images/JNet_518_2_outputx_depth.png
[JNet_518_2_outputx_plane]: /experiments/images/JNet_518_2_outputx_plane.png
[JNet_518_2_outputz_depth]: /experiments/images/JNet_518_2_outputz_depth.png
[JNet_518_2_outputz_plane]: /experiments/images/JNet_518_2_outputz_plane.png
[JNet_518_2_reconst_depth]: /experiments/images/JNet_518_2_reconst_depth.png
[JNet_518_2_reconst_plane]: /experiments/images/JNet_518_2_reconst_plane.png
[JNet_518_3_aligned_depth]: /experiments/images/JNet_518_3_aligned_depth.png
[JNet_518_3_aligned_plane]: /experiments/images/JNet_518_3_aligned_plane.png
[JNet_518_3_heatmap_depth]: /experiments/images/JNet_518_3_heatmap_depth.png
[JNet_518_3_heatmap_plane]: /experiments/images/JNet_518_3_heatmap_plane.png
[JNet_518_3_labelx_depth]: /experiments/images/JNet_518_3_labelx_depth.png
[JNet_518_3_labelx_plane]: /experiments/images/JNet_518_3_labelx_plane.png
[JNet_518_3_labelz_depth]: /experiments/images/JNet_518_3_labelz_depth.png
[JNet_518_3_labelz_plane]: /experiments/images/JNet_518_3_labelz_plane.png
[JNet_518_3_novibrate_depth]: /experiments/images/JNet_518_3_novibrate_depth.png
[JNet_518_3_novibrate_plane]: /experiments/images/JNet_518_3_novibrate_plane.png
[JNet_518_3_original_depth]: /experiments/images/JNet_518_3_original_depth.png
[JNet_518_3_original_plane]: /experiments/images/JNet_518_3_original_plane.png
[JNet_518_3_outputx_depth]: /experiments/images/JNet_518_3_outputx_depth.png
[JNet_518_3_outputx_plane]: /experiments/images/JNet_518_3_outputx_plane.png
[JNet_518_3_outputz_depth]: /experiments/images/JNet_518_3_outputz_depth.png
[JNet_518_3_outputz_plane]: /experiments/images/JNet_518_3_outputz_plane.png
[JNet_518_3_reconst_depth]: /experiments/images/JNet_518_3_reconst_depth.png
[JNet_518_3_reconst_plane]: /experiments/images/JNet_518_3_reconst_plane.png
[JNet_518_4_aligned_depth]: /experiments/images/JNet_518_4_aligned_depth.png
[JNet_518_4_aligned_plane]: /experiments/images/JNet_518_4_aligned_plane.png
[JNet_518_4_heatmap_depth]: /experiments/images/JNet_518_4_heatmap_depth.png
[JNet_518_4_heatmap_plane]: /experiments/images/JNet_518_4_heatmap_plane.png
[JNet_518_4_labelx_depth]: /experiments/images/JNet_518_4_labelx_depth.png
[JNet_518_4_labelx_plane]: /experiments/images/JNet_518_4_labelx_plane.png
[JNet_518_4_labelz_depth]: /experiments/images/JNet_518_4_labelz_depth.png
[JNet_518_4_labelz_plane]: /experiments/images/JNet_518_4_labelz_plane.png
[JNet_518_4_novibrate_depth]: /experiments/images/JNet_518_4_novibrate_depth.png
[JNet_518_4_novibrate_plane]: /experiments/images/JNet_518_4_novibrate_plane.png
[JNet_518_4_original_depth]: /experiments/images/JNet_518_4_original_depth.png
[JNet_518_4_original_plane]: /experiments/images/JNet_518_4_original_plane.png
[JNet_518_4_outputx_depth]: /experiments/images/JNet_518_4_outputx_depth.png
[JNet_518_4_outputx_plane]: /experiments/images/JNet_518_4_outputx_plane.png
[JNet_518_4_outputz_depth]: /experiments/images/JNet_518_4_outputz_depth.png
[JNet_518_4_outputz_plane]: /experiments/images/JNet_518_4_outputz_plane.png
[JNet_518_4_reconst_depth]: /experiments/images/JNet_518_4_reconst_depth.png
[JNet_518_4_reconst_plane]: /experiments/images/JNet_518_4_reconst_plane.png
[JNet_518_microglia_0_aligned_depth]: /experiments/images/JNet_518_microglia_0_aligned_depth.png
[JNet_518_microglia_0_aligned_plane]: /experiments/images/JNet_518_microglia_0_aligned_plane.png
[JNet_518_microglia_0_heatmap_depth]: /experiments/images/JNet_518_microglia_0_heatmap_depth.png
[JNet_518_microglia_0_heatmap_plane]: /experiments/images/JNet_518_microglia_0_heatmap_plane.png
[JNet_518_microglia_0_original_depth]: /experiments/images/JNet_518_microglia_0_original_depth.png
[JNet_518_microglia_0_original_plane]: /experiments/images/JNet_518_microglia_0_original_plane.png
[JNet_518_microglia_0_outputx_depth]: /experiments/images/JNet_518_microglia_0_outputx_depth.png
[JNet_518_microglia_0_outputx_plane]: /experiments/images/JNet_518_microglia_0_outputx_plane.png
[JNet_518_microglia_0_outputz_depth]: /experiments/images/JNet_518_microglia_0_outputz_depth.png
[JNet_518_microglia_0_outputz_plane]: /experiments/images/JNet_518_microglia_0_outputz_plane.png
[JNet_518_microglia_0_reconst_depth]: /experiments/images/JNet_518_microglia_0_reconst_depth.png
[JNet_518_microglia_0_reconst_plane]: /experiments/images/JNet_518_microglia_0_reconst_plane.png
[JNet_518_microglia_1_aligned_depth]: /experiments/images/JNet_518_microglia_1_aligned_depth.png
[JNet_518_microglia_1_aligned_plane]: /experiments/images/JNet_518_microglia_1_aligned_plane.png
[JNet_518_microglia_1_heatmap_depth]: /experiments/images/JNet_518_microglia_1_heatmap_depth.png
[JNet_518_microglia_1_heatmap_plane]: /experiments/images/JNet_518_microglia_1_heatmap_plane.png
[JNet_518_microglia_1_original_depth]: /experiments/images/JNet_518_microglia_1_original_depth.png
[JNet_518_microglia_1_original_plane]: /experiments/images/JNet_518_microglia_1_original_plane.png
[JNet_518_microglia_1_outputx_depth]: /experiments/images/JNet_518_microglia_1_outputx_depth.png
[JNet_518_microglia_1_outputx_plane]: /experiments/images/JNet_518_microglia_1_outputx_plane.png
[JNet_518_microglia_1_outputz_depth]: /experiments/images/JNet_518_microglia_1_outputz_depth.png
[JNet_518_microglia_1_outputz_plane]: /experiments/images/JNet_518_microglia_1_outputz_plane.png
[JNet_518_microglia_1_reconst_depth]: /experiments/images/JNet_518_microglia_1_reconst_depth.png
[JNet_518_microglia_1_reconst_plane]: /experiments/images/JNet_518_microglia_1_reconst_plane.png
[JNet_518_microglia_2_aligned_depth]: /experiments/images/JNet_518_microglia_2_aligned_depth.png
[JNet_518_microglia_2_aligned_plane]: /experiments/images/JNet_518_microglia_2_aligned_plane.png
[JNet_518_microglia_2_heatmap_depth]: /experiments/images/JNet_518_microglia_2_heatmap_depth.png
[JNet_518_microglia_2_heatmap_plane]: /experiments/images/JNet_518_microglia_2_heatmap_plane.png
[JNet_518_microglia_2_original_depth]: /experiments/images/JNet_518_microglia_2_original_depth.png
[JNet_518_microglia_2_original_plane]: /experiments/images/JNet_518_microglia_2_original_plane.png
[JNet_518_microglia_2_outputx_depth]: /experiments/images/JNet_518_microglia_2_outputx_depth.png
[JNet_518_microglia_2_outputx_plane]: /experiments/images/JNet_518_microglia_2_outputx_plane.png
[JNet_518_microglia_2_outputz_depth]: /experiments/images/JNet_518_microglia_2_outputz_depth.png
[JNet_518_microglia_2_outputz_plane]: /experiments/images/JNet_518_microglia_2_outputz_plane.png
[JNet_518_microglia_2_reconst_depth]: /experiments/images/JNet_518_microglia_2_reconst_depth.png
[JNet_518_microglia_2_reconst_plane]: /experiments/images/JNet_518_microglia_2_reconst_plane.png
[JNet_518_microglia_3_aligned_depth]: /experiments/images/JNet_518_microglia_3_aligned_depth.png
[JNet_518_microglia_3_aligned_plane]: /experiments/images/JNet_518_microglia_3_aligned_plane.png
[JNet_518_microglia_3_heatmap_depth]: /experiments/images/JNet_518_microglia_3_heatmap_depth.png
[JNet_518_microglia_3_heatmap_plane]: /experiments/images/JNet_518_microglia_3_heatmap_plane.png
[JNet_518_microglia_3_original_depth]: /experiments/images/JNet_518_microglia_3_original_depth.png
[JNet_518_microglia_3_original_plane]: /experiments/images/JNet_518_microglia_3_original_plane.png
[JNet_518_microglia_3_outputx_depth]: /experiments/images/JNet_518_microglia_3_outputx_depth.png
[JNet_518_microglia_3_outputx_plane]: /experiments/images/JNet_518_microglia_3_outputx_plane.png
[JNet_518_microglia_3_outputz_depth]: /experiments/images/JNet_518_microglia_3_outputz_depth.png
[JNet_518_microglia_3_outputz_plane]: /experiments/images/JNet_518_microglia_3_outputz_plane.png
[JNet_518_microglia_3_reconst_depth]: /experiments/images/JNet_518_microglia_3_reconst_depth.png
[JNet_518_microglia_3_reconst_plane]: /experiments/images/JNet_518_microglia_3_reconst_plane.png
[JNet_518_microglia_4_aligned_depth]: /experiments/images/JNet_518_microglia_4_aligned_depth.png
[JNet_518_microglia_4_aligned_plane]: /experiments/images/JNet_518_microglia_4_aligned_plane.png
[JNet_518_microglia_4_heatmap_depth]: /experiments/images/JNet_518_microglia_4_heatmap_depth.png
[JNet_518_microglia_4_heatmap_plane]: /experiments/images/JNet_518_microglia_4_heatmap_plane.png
[JNet_518_microglia_4_original_depth]: /experiments/images/JNet_518_microglia_4_original_depth.png
[JNet_518_microglia_4_original_plane]: /experiments/images/JNet_518_microglia_4_original_plane.png
[JNet_518_microglia_4_outputx_depth]: /experiments/images/JNet_518_microglia_4_outputx_depth.png
[JNet_518_microglia_4_outputx_plane]: /experiments/images/JNet_518_microglia_4_outputx_plane.png
[JNet_518_microglia_4_outputz_depth]: /experiments/images/JNet_518_microglia_4_outputz_depth.png
[JNet_518_microglia_4_outputz_plane]: /experiments/images/JNet_518_microglia_4_outputz_plane.png
[JNet_518_microglia_4_reconst_depth]: /experiments/images/JNet_518_microglia_4_reconst_depth.png
[JNet_518_microglia_4_reconst_plane]: /experiments/images/JNet_518_microglia_4_reconst_plane.png
[JNet_518_psf_post]: /experiments/images/JNet_518_psf_post.png
[JNet_518_psf_pre]: /experiments/images/JNet_518_psf_pre.png
[finetuned]: /experiments/tmp/JNet_518_train.png
[pretrained_model]: /experiments/tmp/JNet_487_pretrain_train.png
