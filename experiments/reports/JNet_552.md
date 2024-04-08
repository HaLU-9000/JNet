



# JNet_552 Report
  
pretrain with vibration, with gradient clipping, finetuning with mrf loss x20 as 523, PSF loss x10, finetuning is also with gradient clipping, with separated traindataset on finetuning, after stackreg  
pretrained model : JNet_542_pretrain
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
|mid|40|num of NeurIPSF middle channel|
|loss_fn|nn.MSELoss()|loss func for NeurIPSF|
|lr|0.01|lr for pre-training NeurIPSF|
|num_iter_psf_pretrain|20000|epoch for pre-training of NeurIPSF|
|nipsf_loss_target|1e-05|epoch for pre-training of NeurIPSF|
|device|cuda||

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
|folderpath|/home/haruhiko/Downloads/Set_03_stackreg_z|
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
|folderpath|/home/haruhiko/Downloads/Set_03_stackreg_z|
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
|n_epochs|100|
|lr|0.001|
|loss_fnx|nn.BCELoss()|
|loss_fnz|nn.BCELoss()|
|path|model|
|savefig_path|train|
|partial|params['partial']|
|ewc|None|
|es_patience|10|
|is_vibrate|True|
|weight_x|1|
|weight_z|1|
|without_noise|False|

### train_loop

|Parameter|Value|
| :--- | :--- |
|batch_size|1|
|n_epochs|100|
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
|ploss_weight|10.0|
|mrfloss_order|1|
|mrfloss_dilation|1|
|mrfloss_weights|{'l_00': -1.0, 'l_01': 1.0, 'l_10': 2.0, 'l_11': -2.0}|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results

### Pretraining
  
Segmentation: mean MSE: 0.01515874732285738, mean BCE: 0.05933680012822151  
Luminance Estimation: mean MSE: 0.9641395807266235, mean BCE: nan
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_0_original_plane]|![JNet_542_pretrain_0_novibrate_plane]|![JNet_542_pretrain_0_aligned_plane]|![JNet_542_pretrain_0_outputx_plane]|![JNet_542_pretrain_0_labelx_plane]|![JNet_542_pretrain_0_outputz_plane]|![JNet_542_pretrain_0_labelz_plane]|
  
MSEx: 0.012865607626736164, BCEx: 0.0457005649805069  
MSEz: 0.9589962363243103, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_0_original_depth]|![JNet_542_pretrain_0_novibrate_depth]|![JNet_542_pretrain_0_aligned_depth]|![JNet_542_pretrain_0_outputx_depth]|![JNet_542_pretrain_0_labelx_depth]|![JNet_542_pretrain_0_outputz_depth]|![JNet_542_pretrain_0_labelz_depth]|
  
MSEx: 0.012865607626736164, BCEx: 0.0457005649805069  
MSEz: 0.9589962363243103, BCEz: nan  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_1_original_plane]|![JNet_542_pretrain_1_novibrate_plane]|![JNet_542_pretrain_1_aligned_plane]|![JNet_542_pretrain_1_outputx_plane]|![JNet_542_pretrain_1_labelx_plane]|![JNet_542_pretrain_1_outputz_plane]|![JNet_542_pretrain_1_labelz_plane]|
  
MSEx: 0.014446188695728779, BCEx: 0.06490708142518997  
MSEz: 0.9803880453109741, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_1_original_depth]|![JNet_542_pretrain_1_novibrate_depth]|![JNet_542_pretrain_1_aligned_depth]|![JNet_542_pretrain_1_outputx_depth]|![JNet_542_pretrain_1_labelx_depth]|![JNet_542_pretrain_1_outputz_depth]|![JNet_542_pretrain_1_labelz_depth]|
  
MSEx: 0.014446188695728779, BCEx: 0.06490708142518997  
MSEz: 0.9803880453109741, BCEz: inf  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_2_original_plane]|![JNet_542_pretrain_2_novibrate_plane]|![JNet_542_pretrain_2_aligned_plane]|![JNet_542_pretrain_2_outputx_plane]|![JNet_542_pretrain_2_labelx_plane]|![JNet_542_pretrain_2_outputz_plane]|![JNet_542_pretrain_2_labelz_plane]|
  
MSEx: 0.012331780977547169, BCEx: 0.048847492784261703  
MSEz: 0.9779883027076721, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_2_original_depth]|![JNet_542_pretrain_2_novibrate_depth]|![JNet_542_pretrain_2_aligned_depth]|![JNet_542_pretrain_2_outputx_depth]|![JNet_542_pretrain_2_labelx_depth]|![JNet_542_pretrain_2_outputz_depth]|![JNet_542_pretrain_2_labelz_depth]|
  
MSEx: 0.012331780977547169, BCEx: 0.048847492784261703  
MSEz: 0.9779883027076721, BCEz: inf  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_3_original_plane]|![JNet_542_pretrain_3_novibrate_plane]|![JNet_542_pretrain_3_aligned_plane]|![JNet_542_pretrain_3_outputx_plane]|![JNet_542_pretrain_3_labelx_plane]|![JNet_542_pretrain_3_outputz_plane]|![JNet_542_pretrain_3_labelz_plane]|
  
MSEx: 0.028073493391275406, BCEx: 0.10974408686161041  
MSEz: 0.9308543801307678, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_3_original_depth]|![JNet_542_pretrain_3_novibrate_depth]|![JNet_542_pretrain_3_aligned_depth]|![JNet_542_pretrain_3_outputx_depth]|![JNet_542_pretrain_3_labelx_depth]|![JNet_542_pretrain_3_outputz_depth]|![JNet_542_pretrain_3_labelz_depth]|
  
MSEx: 0.028073493391275406, BCEx: 0.10974408686161041  
MSEz: 0.9308543801307678, BCEz: inf  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_4_original_plane]|![JNet_542_pretrain_4_novibrate_plane]|![JNet_542_pretrain_4_aligned_plane]|![JNet_542_pretrain_4_outputx_plane]|![JNet_542_pretrain_4_labelx_plane]|![JNet_542_pretrain_4_outputz_plane]|![JNet_542_pretrain_4_labelz_plane]|
  
MSEx: 0.008076662197709084, BCEx: 0.02748476155102253  
MSEz: 0.9724709987640381, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_4_original_depth]|![JNet_542_pretrain_4_novibrate_depth]|![JNet_542_pretrain_4_aligned_depth]|![JNet_542_pretrain_4_outputx_depth]|![JNet_542_pretrain_4_labelx_depth]|![JNet_542_pretrain_4_outputz_depth]|![JNet_542_pretrain_4_labelz_depth]|
  
MSEx: 0.008076662197709084, BCEx: 0.02748476155102253  
MSEz: 0.9724709987640381, BCEz: inf  

### Finetuning Results with Simulation

### image 0

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_0_original_plane]|![JNet_552_0_novibrate_plane]|![JNet_552_0_aligned_plane]|![JNet_552_0_reconst_plane]|![JNet_552_0_heatmap_plane]|![JNet_552_0_outputx_plane]|![JNet_552_0_labelx_plane]|![JNet_552_0_outputz_plane]|![JNet_552_0_labelz_plane]|
  
MSEz: 0.9956100583076477, quantized loss: 9.379621523919468e-09  

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_0_original_depth]|![JNet_552_0_novibrate_depth]|![JNet_552_0_aligned_depth]|![JNet_552_0_reconst_depth]|![JNet_552_0_heatmap_depth]|![JNet_552_0_outputx_depth]|![JNet_552_0_labelx_depth]|![JNet_552_0_outputz_depth]|![JNet_552_0_labelz_depth]|
  
MSEz: 0.9956100583076477, quantized loss: 9.379621523919468e-09  

### image 1

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_1_original_plane]|![JNet_552_1_novibrate_plane]|![JNet_552_1_aligned_plane]|![JNet_552_1_reconst_plane]|![JNet_552_1_heatmap_plane]|![JNet_552_1_outputx_plane]|![JNet_552_1_labelx_plane]|![JNet_552_1_outputz_plane]|![JNet_552_1_labelz_plane]|
  
MSEz: 0.9853872656822205, quantized loss: 0.0002531253849156201  

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_1_original_depth]|![JNet_552_1_novibrate_depth]|![JNet_552_1_aligned_depth]|![JNet_552_1_reconst_depth]|![JNet_552_1_heatmap_depth]|![JNet_552_1_outputx_depth]|![JNet_552_1_labelx_depth]|![JNet_552_1_outputz_depth]|![JNet_552_1_labelz_depth]|
  
MSEz: 0.9853872656822205, quantized loss: 0.0002531253849156201  

### image 2

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_2_original_plane]|![JNet_552_2_novibrate_plane]|![JNet_552_2_aligned_plane]|![JNet_552_2_reconst_plane]|![JNet_552_2_heatmap_plane]|![JNet_552_2_outputx_plane]|![JNet_552_2_labelx_plane]|![JNet_552_2_outputz_plane]|![JNet_552_2_labelz_plane]|
  
MSEz: 0.9922517538070679, quantized loss: 8.620716585028276e-09  

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_2_original_depth]|![JNet_552_2_novibrate_depth]|![JNet_552_2_aligned_depth]|![JNet_552_2_reconst_depth]|![JNet_552_2_heatmap_depth]|![JNet_552_2_outputx_depth]|![JNet_552_2_labelx_depth]|![JNet_552_2_outputz_depth]|![JNet_552_2_labelz_depth]|
  
MSEz: 0.9922517538070679, quantized loss: 8.620716585028276e-09  

### image 3

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_3_original_plane]|![JNet_552_3_novibrate_plane]|![JNet_552_3_aligned_plane]|![JNet_552_3_reconst_plane]|![JNet_552_3_heatmap_plane]|![JNet_552_3_outputx_plane]|![JNet_552_3_labelx_plane]|![JNet_552_3_outputz_plane]|![JNet_552_3_labelz_plane]|
  
MSEz: 0.9965091347694397, quantized loss: 8.209155133442891e-09  

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_3_original_depth]|![JNet_552_3_novibrate_depth]|![JNet_552_3_aligned_depth]|![JNet_552_3_reconst_depth]|![JNet_552_3_heatmap_depth]|![JNet_552_3_outputx_depth]|![JNet_552_3_labelx_depth]|![JNet_552_3_outputz_depth]|![JNet_552_3_labelz_depth]|
  
MSEz: 0.9965091347694397, quantized loss: 8.209155133442891e-09  

### image 4

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_4_original_plane]|![JNet_552_4_novibrate_plane]|![JNet_552_4_aligned_plane]|![JNet_552_4_reconst_plane]|![JNet_552_4_heatmap_plane]|![JNet_552_4_outputx_plane]|![JNet_552_4_labelx_plane]|![JNet_552_4_outputz_plane]|![JNet_552_4_labelz_plane]|
  
MSEz: 0.992272675037384, quantized loss: 5.191365115564395e-09  

|original|novibrate|aligned|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_4_original_depth]|![JNet_552_4_novibrate_depth]|![JNet_552_4_aligned_depth]|![JNet_552_4_reconst_depth]|![JNet_552_4_heatmap_depth]|![JNet_552_4_outputx_depth]|![JNet_552_4_labelx_depth]|![JNet_552_4_outputz_depth]|![JNet_552_4_labelz_depth]|
  
MSEz: 0.992272675037384, quantized loss: 5.191365115564395e-09  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
### Finetuning Results with Microglia

#### finetuning == False

### image 0

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_microglia_0_original_plane]|![JNet_542_pretrain_microglia_0_aligned_plane]|![JNet_542_pretrain_microglia_0_outputx_plane]|![JNet_542_pretrain_microglia_0_outputz_plane]|![JNet_542_pretrain_microglia_0_reconst_plane]|![JNet_542_pretrain_microglia_0_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_microglia_0_original_depth]|![JNet_542_pretrain_microglia_0_aligned_depth]|![JNet_542_pretrain_microglia_0_outputx_depth]|![JNet_542_pretrain_microglia_0_outputz_depth]|![JNet_542_pretrain_microglia_0_reconst_depth]|![JNet_542_pretrain_microglia_0_heatmap_depth]|
  

### image 1

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_microglia_1_original_plane]|![JNet_542_pretrain_microglia_1_aligned_plane]|![JNet_542_pretrain_microglia_1_outputx_plane]|![JNet_542_pretrain_microglia_1_outputz_plane]|![JNet_542_pretrain_microglia_1_reconst_plane]|![JNet_542_pretrain_microglia_1_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_microglia_1_original_depth]|![JNet_542_pretrain_microglia_1_aligned_depth]|![JNet_542_pretrain_microglia_1_outputx_depth]|![JNet_542_pretrain_microglia_1_outputz_depth]|![JNet_542_pretrain_microglia_1_reconst_depth]|![JNet_542_pretrain_microglia_1_heatmap_depth]|
  

### image 2

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_microglia_2_original_plane]|![JNet_542_pretrain_microglia_2_aligned_plane]|![JNet_542_pretrain_microglia_2_outputx_plane]|![JNet_542_pretrain_microglia_2_outputz_plane]|![JNet_542_pretrain_microglia_2_reconst_plane]|![JNet_542_pretrain_microglia_2_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_microglia_2_original_depth]|![JNet_542_pretrain_microglia_2_aligned_depth]|![JNet_542_pretrain_microglia_2_outputx_depth]|![JNet_542_pretrain_microglia_2_outputz_depth]|![JNet_542_pretrain_microglia_2_reconst_depth]|![JNet_542_pretrain_microglia_2_heatmap_depth]|
  

### image 3

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_microglia_3_original_plane]|![JNet_542_pretrain_microglia_3_aligned_plane]|![JNet_542_pretrain_microglia_3_outputx_plane]|![JNet_542_pretrain_microglia_3_outputz_plane]|![JNet_542_pretrain_microglia_3_reconst_plane]|![JNet_542_pretrain_microglia_3_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_microglia_3_original_depth]|![JNet_542_pretrain_microglia_3_aligned_depth]|![JNet_542_pretrain_microglia_3_outputx_depth]|![JNet_542_pretrain_microglia_3_outputz_depth]|![JNet_542_pretrain_microglia_3_reconst_depth]|![JNet_542_pretrain_microglia_3_heatmap_depth]|
  

### image 4

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_microglia_4_original_plane]|![JNet_542_pretrain_microglia_4_aligned_plane]|![JNet_542_pretrain_microglia_4_outputx_plane]|![JNet_542_pretrain_microglia_4_outputz_plane]|![JNet_542_pretrain_microglia_4_reconst_plane]|![JNet_542_pretrain_microglia_4_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_542_pretrain_microglia_4_original_depth]|![JNet_542_pretrain_microglia_4_aligned_depth]|![JNet_542_pretrain_microglia_4_outputx_depth]|![JNet_542_pretrain_microglia_4_outputz_depth]|![JNet_542_pretrain_microglia_4_reconst_depth]|![JNet_542_pretrain_microglia_4_heatmap_depth]|
  

#### finetuning == True

### image 0

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_microglia_0_original_plane]|![JNet_552_microglia_0_aligned_plane]|![JNet_552_microglia_0_outputx_plane]|![JNet_552_microglia_0_outputz_plane]|![JNet_552_microglia_0_reconst_plane]|![JNet_552_microglia_0_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_microglia_0_original_depth]|![JNet_552_microglia_0_aligned_depth]|![JNet_552_microglia_0_outputx_depth]|![JNet_552_microglia_0_outputz_depth]|![JNet_552_microglia_0_reconst_depth]|![JNet_552_microglia_0_heatmap_depth]|
  

### image 1

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_microglia_1_original_plane]|![JNet_552_microglia_1_aligned_plane]|![JNet_552_microglia_1_outputx_plane]|![JNet_552_microglia_1_outputz_plane]|![JNet_552_microglia_1_reconst_plane]|![JNet_552_microglia_1_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_microglia_1_original_depth]|![JNet_552_microglia_1_aligned_depth]|![JNet_552_microglia_1_outputx_depth]|![JNet_552_microglia_1_outputz_depth]|![JNet_552_microglia_1_reconst_depth]|![JNet_552_microglia_1_heatmap_depth]|
  

### image 2

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_microglia_2_original_plane]|![JNet_552_microglia_2_aligned_plane]|![JNet_552_microglia_2_outputx_plane]|![JNet_552_microglia_2_outputz_plane]|![JNet_552_microglia_2_reconst_plane]|![JNet_552_microglia_2_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_microglia_2_original_depth]|![JNet_552_microglia_2_aligned_depth]|![JNet_552_microglia_2_outputx_depth]|![JNet_552_microglia_2_outputz_depth]|![JNet_552_microglia_2_reconst_depth]|![JNet_552_microglia_2_heatmap_depth]|
  

### image 3

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_microglia_3_original_plane]|![JNet_552_microglia_3_aligned_plane]|![JNet_552_microglia_3_outputx_plane]|![JNet_552_microglia_3_outputz_plane]|![JNet_552_microglia_3_reconst_plane]|![JNet_552_microglia_3_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_microglia_3_original_depth]|![JNet_552_microglia_3_aligned_depth]|![JNet_552_microglia_3_outputx_depth]|![JNet_552_microglia_3_outputz_depth]|![JNet_552_microglia_3_reconst_depth]|![JNet_552_microglia_3_heatmap_depth]|
  

### image 4

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_microglia_4_original_plane]|![JNet_552_microglia_4_aligned_plane]|![JNet_552_microglia_4_outputx_plane]|![JNet_552_microglia_4_outputz_plane]|![JNet_552_microglia_4_reconst_plane]|![JNet_552_microglia_4_heatmap_plane]|
  

|original|aligned|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_552_microglia_4_original_depth]|![JNet_552_microglia_4_aligned_depth]|![JNet_552_microglia_4_outputx_depth]|![JNet_552_microglia_4_outputz_depth]|![JNet_552_microglia_4_reconst_depth]|![JNet_552_microglia_4_heatmap_depth]|
  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_552_psf_pre]|![JNet_552_psf_post]|

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
          (1): Linear(in_features=2, out_features=40, bias=True)  
          (2): Sigmoid()  
          (3): BatchNorm1d(40, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
          (4): Linear(in_features=40, out_features=1, bias=True)  
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
  



[JNet_542_pretrain_0_aligned_depth]: /experiments/images/JNet_542_pretrain_0_aligned_depth.png
[JNet_542_pretrain_0_aligned_plane]: /experiments/images/JNet_542_pretrain_0_aligned_plane.png
[JNet_542_pretrain_0_labelx_depth]: /experiments/images/JNet_542_pretrain_0_labelx_depth.png
[JNet_542_pretrain_0_labelx_plane]: /experiments/images/JNet_542_pretrain_0_labelx_plane.png
[JNet_542_pretrain_0_labelz_depth]: /experiments/images/JNet_542_pretrain_0_labelz_depth.png
[JNet_542_pretrain_0_labelz_plane]: /experiments/images/JNet_542_pretrain_0_labelz_plane.png
[JNet_542_pretrain_0_novibrate_depth]: /experiments/images/JNet_542_pretrain_0_novibrate_depth.png
[JNet_542_pretrain_0_novibrate_plane]: /experiments/images/JNet_542_pretrain_0_novibrate_plane.png
[JNet_542_pretrain_0_original_depth]: /experiments/images/JNet_542_pretrain_0_original_depth.png
[JNet_542_pretrain_0_original_plane]: /experiments/images/JNet_542_pretrain_0_original_plane.png
[JNet_542_pretrain_0_outputx_depth]: /experiments/images/JNet_542_pretrain_0_outputx_depth.png
[JNet_542_pretrain_0_outputx_plane]: /experiments/images/JNet_542_pretrain_0_outputx_plane.png
[JNet_542_pretrain_0_outputz_depth]: /experiments/images/JNet_542_pretrain_0_outputz_depth.png
[JNet_542_pretrain_0_outputz_plane]: /experiments/images/JNet_542_pretrain_0_outputz_plane.png
[JNet_542_pretrain_1_aligned_depth]: /experiments/images/JNet_542_pretrain_1_aligned_depth.png
[JNet_542_pretrain_1_aligned_plane]: /experiments/images/JNet_542_pretrain_1_aligned_plane.png
[JNet_542_pretrain_1_labelx_depth]: /experiments/images/JNet_542_pretrain_1_labelx_depth.png
[JNet_542_pretrain_1_labelx_plane]: /experiments/images/JNet_542_pretrain_1_labelx_plane.png
[JNet_542_pretrain_1_labelz_depth]: /experiments/images/JNet_542_pretrain_1_labelz_depth.png
[JNet_542_pretrain_1_labelz_plane]: /experiments/images/JNet_542_pretrain_1_labelz_plane.png
[JNet_542_pretrain_1_novibrate_depth]: /experiments/images/JNet_542_pretrain_1_novibrate_depth.png
[JNet_542_pretrain_1_novibrate_plane]: /experiments/images/JNet_542_pretrain_1_novibrate_plane.png
[JNet_542_pretrain_1_original_depth]: /experiments/images/JNet_542_pretrain_1_original_depth.png
[JNet_542_pretrain_1_original_plane]: /experiments/images/JNet_542_pretrain_1_original_plane.png
[JNet_542_pretrain_1_outputx_depth]: /experiments/images/JNet_542_pretrain_1_outputx_depth.png
[JNet_542_pretrain_1_outputx_plane]: /experiments/images/JNet_542_pretrain_1_outputx_plane.png
[JNet_542_pretrain_1_outputz_depth]: /experiments/images/JNet_542_pretrain_1_outputz_depth.png
[JNet_542_pretrain_1_outputz_plane]: /experiments/images/JNet_542_pretrain_1_outputz_plane.png
[JNet_542_pretrain_2_aligned_depth]: /experiments/images/JNet_542_pretrain_2_aligned_depth.png
[JNet_542_pretrain_2_aligned_plane]: /experiments/images/JNet_542_pretrain_2_aligned_plane.png
[JNet_542_pretrain_2_labelx_depth]: /experiments/images/JNet_542_pretrain_2_labelx_depth.png
[JNet_542_pretrain_2_labelx_plane]: /experiments/images/JNet_542_pretrain_2_labelx_plane.png
[JNet_542_pretrain_2_labelz_depth]: /experiments/images/JNet_542_pretrain_2_labelz_depth.png
[JNet_542_pretrain_2_labelz_plane]: /experiments/images/JNet_542_pretrain_2_labelz_plane.png
[JNet_542_pretrain_2_novibrate_depth]: /experiments/images/JNet_542_pretrain_2_novibrate_depth.png
[JNet_542_pretrain_2_novibrate_plane]: /experiments/images/JNet_542_pretrain_2_novibrate_plane.png
[JNet_542_pretrain_2_original_depth]: /experiments/images/JNet_542_pretrain_2_original_depth.png
[JNet_542_pretrain_2_original_plane]: /experiments/images/JNet_542_pretrain_2_original_plane.png
[JNet_542_pretrain_2_outputx_depth]: /experiments/images/JNet_542_pretrain_2_outputx_depth.png
[JNet_542_pretrain_2_outputx_plane]: /experiments/images/JNet_542_pretrain_2_outputx_plane.png
[JNet_542_pretrain_2_outputz_depth]: /experiments/images/JNet_542_pretrain_2_outputz_depth.png
[JNet_542_pretrain_2_outputz_plane]: /experiments/images/JNet_542_pretrain_2_outputz_plane.png
[JNet_542_pretrain_3_aligned_depth]: /experiments/images/JNet_542_pretrain_3_aligned_depth.png
[JNet_542_pretrain_3_aligned_plane]: /experiments/images/JNet_542_pretrain_3_aligned_plane.png
[JNet_542_pretrain_3_labelx_depth]: /experiments/images/JNet_542_pretrain_3_labelx_depth.png
[JNet_542_pretrain_3_labelx_plane]: /experiments/images/JNet_542_pretrain_3_labelx_plane.png
[JNet_542_pretrain_3_labelz_depth]: /experiments/images/JNet_542_pretrain_3_labelz_depth.png
[JNet_542_pretrain_3_labelz_plane]: /experiments/images/JNet_542_pretrain_3_labelz_plane.png
[JNet_542_pretrain_3_novibrate_depth]: /experiments/images/JNet_542_pretrain_3_novibrate_depth.png
[JNet_542_pretrain_3_novibrate_plane]: /experiments/images/JNet_542_pretrain_3_novibrate_plane.png
[JNet_542_pretrain_3_original_depth]: /experiments/images/JNet_542_pretrain_3_original_depth.png
[JNet_542_pretrain_3_original_plane]: /experiments/images/JNet_542_pretrain_3_original_plane.png
[JNet_542_pretrain_3_outputx_depth]: /experiments/images/JNet_542_pretrain_3_outputx_depth.png
[JNet_542_pretrain_3_outputx_plane]: /experiments/images/JNet_542_pretrain_3_outputx_plane.png
[JNet_542_pretrain_3_outputz_depth]: /experiments/images/JNet_542_pretrain_3_outputz_depth.png
[JNet_542_pretrain_3_outputz_plane]: /experiments/images/JNet_542_pretrain_3_outputz_plane.png
[JNet_542_pretrain_4_aligned_depth]: /experiments/images/JNet_542_pretrain_4_aligned_depth.png
[JNet_542_pretrain_4_aligned_plane]: /experiments/images/JNet_542_pretrain_4_aligned_plane.png
[JNet_542_pretrain_4_labelx_depth]: /experiments/images/JNet_542_pretrain_4_labelx_depth.png
[JNet_542_pretrain_4_labelx_plane]: /experiments/images/JNet_542_pretrain_4_labelx_plane.png
[JNet_542_pretrain_4_labelz_depth]: /experiments/images/JNet_542_pretrain_4_labelz_depth.png
[JNet_542_pretrain_4_labelz_plane]: /experiments/images/JNet_542_pretrain_4_labelz_plane.png
[JNet_542_pretrain_4_novibrate_depth]: /experiments/images/JNet_542_pretrain_4_novibrate_depth.png
[JNet_542_pretrain_4_novibrate_plane]: /experiments/images/JNet_542_pretrain_4_novibrate_plane.png
[JNet_542_pretrain_4_original_depth]: /experiments/images/JNet_542_pretrain_4_original_depth.png
[JNet_542_pretrain_4_original_plane]: /experiments/images/JNet_542_pretrain_4_original_plane.png
[JNet_542_pretrain_4_outputx_depth]: /experiments/images/JNet_542_pretrain_4_outputx_depth.png
[JNet_542_pretrain_4_outputx_plane]: /experiments/images/JNet_542_pretrain_4_outputx_plane.png
[JNet_542_pretrain_4_outputz_depth]: /experiments/images/JNet_542_pretrain_4_outputz_depth.png
[JNet_542_pretrain_4_outputz_plane]: /experiments/images/JNet_542_pretrain_4_outputz_plane.png
[JNet_542_pretrain_microglia_0_aligned_depth]: /experiments/images/JNet_542_pretrain_microglia_0_aligned_depth.png
[JNet_542_pretrain_microglia_0_aligned_plane]: /experiments/images/JNet_542_pretrain_microglia_0_aligned_plane.png
[JNet_542_pretrain_microglia_0_heatmap_depth]: /experiments/images/JNet_542_pretrain_microglia_0_heatmap_depth.png
[JNet_542_pretrain_microglia_0_heatmap_plane]: /experiments/images/JNet_542_pretrain_microglia_0_heatmap_plane.png
[JNet_542_pretrain_microglia_0_original_depth]: /experiments/images/JNet_542_pretrain_microglia_0_original_depth.png
[JNet_542_pretrain_microglia_0_original_plane]: /experiments/images/JNet_542_pretrain_microglia_0_original_plane.png
[JNet_542_pretrain_microglia_0_outputx_depth]: /experiments/images/JNet_542_pretrain_microglia_0_outputx_depth.png
[JNet_542_pretrain_microglia_0_outputx_plane]: /experiments/images/JNet_542_pretrain_microglia_0_outputx_plane.png
[JNet_542_pretrain_microglia_0_outputz_depth]: /experiments/images/JNet_542_pretrain_microglia_0_outputz_depth.png
[JNet_542_pretrain_microglia_0_outputz_plane]: /experiments/images/JNet_542_pretrain_microglia_0_outputz_plane.png
[JNet_542_pretrain_microglia_0_reconst_depth]: /experiments/images/JNet_542_pretrain_microglia_0_reconst_depth.png
[JNet_542_pretrain_microglia_0_reconst_plane]: /experiments/images/JNet_542_pretrain_microglia_0_reconst_plane.png
[JNet_542_pretrain_microglia_1_aligned_depth]: /experiments/images/JNet_542_pretrain_microglia_1_aligned_depth.png
[JNet_542_pretrain_microglia_1_aligned_plane]: /experiments/images/JNet_542_pretrain_microglia_1_aligned_plane.png
[JNet_542_pretrain_microglia_1_heatmap_depth]: /experiments/images/JNet_542_pretrain_microglia_1_heatmap_depth.png
[JNet_542_pretrain_microglia_1_heatmap_plane]: /experiments/images/JNet_542_pretrain_microglia_1_heatmap_plane.png
[JNet_542_pretrain_microglia_1_original_depth]: /experiments/images/JNet_542_pretrain_microglia_1_original_depth.png
[JNet_542_pretrain_microglia_1_original_plane]: /experiments/images/JNet_542_pretrain_microglia_1_original_plane.png
[JNet_542_pretrain_microglia_1_outputx_depth]: /experiments/images/JNet_542_pretrain_microglia_1_outputx_depth.png
[JNet_542_pretrain_microglia_1_outputx_plane]: /experiments/images/JNet_542_pretrain_microglia_1_outputx_plane.png
[JNet_542_pretrain_microglia_1_outputz_depth]: /experiments/images/JNet_542_pretrain_microglia_1_outputz_depth.png
[JNet_542_pretrain_microglia_1_outputz_plane]: /experiments/images/JNet_542_pretrain_microglia_1_outputz_plane.png
[JNet_542_pretrain_microglia_1_reconst_depth]: /experiments/images/JNet_542_pretrain_microglia_1_reconst_depth.png
[JNet_542_pretrain_microglia_1_reconst_plane]: /experiments/images/JNet_542_pretrain_microglia_1_reconst_plane.png
[JNet_542_pretrain_microglia_2_aligned_depth]: /experiments/images/JNet_542_pretrain_microglia_2_aligned_depth.png
[JNet_542_pretrain_microglia_2_aligned_plane]: /experiments/images/JNet_542_pretrain_microglia_2_aligned_plane.png
[JNet_542_pretrain_microglia_2_heatmap_depth]: /experiments/images/JNet_542_pretrain_microglia_2_heatmap_depth.png
[JNet_542_pretrain_microglia_2_heatmap_plane]: /experiments/images/JNet_542_pretrain_microglia_2_heatmap_plane.png
[JNet_542_pretrain_microglia_2_original_depth]: /experiments/images/JNet_542_pretrain_microglia_2_original_depth.png
[JNet_542_pretrain_microglia_2_original_plane]: /experiments/images/JNet_542_pretrain_microglia_2_original_plane.png
[JNet_542_pretrain_microglia_2_outputx_depth]: /experiments/images/JNet_542_pretrain_microglia_2_outputx_depth.png
[JNet_542_pretrain_microglia_2_outputx_plane]: /experiments/images/JNet_542_pretrain_microglia_2_outputx_plane.png
[JNet_542_pretrain_microglia_2_outputz_depth]: /experiments/images/JNet_542_pretrain_microglia_2_outputz_depth.png
[JNet_542_pretrain_microglia_2_outputz_plane]: /experiments/images/JNet_542_pretrain_microglia_2_outputz_plane.png
[JNet_542_pretrain_microglia_2_reconst_depth]: /experiments/images/JNet_542_pretrain_microglia_2_reconst_depth.png
[JNet_542_pretrain_microglia_2_reconst_plane]: /experiments/images/JNet_542_pretrain_microglia_2_reconst_plane.png
[JNet_542_pretrain_microglia_3_aligned_depth]: /experiments/images/JNet_542_pretrain_microglia_3_aligned_depth.png
[JNet_542_pretrain_microglia_3_aligned_plane]: /experiments/images/JNet_542_pretrain_microglia_3_aligned_plane.png
[JNet_542_pretrain_microglia_3_heatmap_depth]: /experiments/images/JNet_542_pretrain_microglia_3_heatmap_depth.png
[JNet_542_pretrain_microglia_3_heatmap_plane]: /experiments/images/JNet_542_pretrain_microglia_3_heatmap_plane.png
[JNet_542_pretrain_microglia_3_original_depth]: /experiments/images/JNet_542_pretrain_microglia_3_original_depth.png
[JNet_542_pretrain_microglia_3_original_plane]: /experiments/images/JNet_542_pretrain_microglia_3_original_plane.png
[JNet_542_pretrain_microglia_3_outputx_depth]: /experiments/images/JNet_542_pretrain_microglia_3_outputx_depth.png
[JNet_542_pretrain_microglia_3_outputx_plane]: /experiments/images/JNet_542_pretrain_microglia_3_outputx_plane.png
[JNet_542_pretrain_microglia_3_outputz_depth]: /experiments/images/JNet_542_pretrain_microglia_3_outputz_depth.png
[JNet_542_pretrain_microglia_3_outputz_plane]: /experiments/images/JNet_542_pretrain_microglia_3_outputz_plane.png
[JNet_542_pretrain_microglia_3_reconst_depth]: /experiments/images/JNet_542_pretrain_microglia_3_reconst_depth.png
[JNet_542_pretrain_microglia_3_reconst_plane]: /experiments/images/JNet_542_pretrain_microglia_3_reconst_plane.png
[JNet_542_pretrain_microglia_4_aligned_depth]: /experiments/images/JNet_542_pretrain_microglia_4_aligned_depth.png
[JNet_542_pretrain_microglia_4_aligned_plane]: /experiments/images/JNet_542_pretrain_microglia_4_aligned_plane.png
[JNet_542_pretrain_microglia_4_heatmap_depth]: /experiments/images/JNet_542_pretrain_microglia_4_heatmap_depth.png
[JNet_542_pretrain_microglia_4_heatmap_plane]: /experiments/images/JNet_542_pretrain_microglia_4_heatmap_plane.png
[JNet_542_pretrain_microglia_4_original_depth]: /experiments/images/JNet_542_pretrain_microglia_4_original_depth.png
[JNet_542_pretrain_microglia_4_original_plane]: /experiments/images/JNet_542_pretrain_microglia_4_original_plane.png
[JNet_542_pretrain_microglia_4_outputx_depth]: /experiments/images/JNet_542_pretrain_microglia_4_outputx_depth.png
[JNet_542_pretrain_microglia_4_outputx_plane]: /experiments/images/JNet_542_pretrain_microglia_4_outputx_plane.png
[JNet_542_pretrain_microglia_4_outputz_depth]: /experiments/images/JNet_542_pretrain_microglia_4_outputz_depth.png
[JNet_542_pretrain_microglia_4_outputz_plane]: /experiments/images/JNet_542_pretrain_microglia_4_outputz_plane.png
[JNet_542_pretrain_microglia_4_reconst_depth]: /experiments/images/JNet_542_pretrain_microglia_4_reconst_depth.png
[JNet_542_pretrain_microglia_4_reconst_plane]: /experiments/images/JNet_542_pretrain_microglia_4_reconst_plane.png
[JNet_552_0_aligned_depth]: /experiments/images/JNet_552_0_aligned_depth.png
[JNet_552_0_aligned_plane]: /experiments/images/JNet_552_0_aligned_plane.png
[JNet_552_0_heatmap_depth]: /experiments/images/JNet_552_0_heatmap_depth.png
[JNet_552_0_heatmap_plane]: /experiments/images/JNet_552_0_heatmap_plane.png
[JNet_552_0_labelx_depth]: /experiments/images/JNet_552_0_labelx_depth.png
[JNet_552_0_labelx_plane]: /experiments/images/JNet_552_0_labelx_plane.png
[JNet_552_0_labelz_depth]: /experiments/images/JNet_552_0_labelz_depth.png
[JNet_552_0_labelz_plane]: /experiments/images/JNet_552_0_labelz_plane.png
[JNet_552_0_novibrate_depth]: /experiments/images/JNet_552_0_novibrate_depth.png
[JNet_552_0_novibrate_plane]: /experiments/images/JNet_552_0_novibrate_plane.png
[JNet_552_0_original_depth]: /experiments/images/JNet_552_0_original_depth.png
[JNet_552_0_original_plane]: /experiments/images/JNet_552_0_original_plane.png
[JNet_552_0_outputx_depth]: /experiments/images/JNet_552_0_outputx_depth.png
[JNet_552_0_outputx_plane]: /experiments/images/JNet_552_0_outputx_plane.png
[JNet_552_0_outputz_depth]: /experiments/images/JNet_552_0_outputz_depth.png
[JNet_552_0_outputz_plane]: /experiments/images/JNet_552_0_outputz_plane.png
[JNet_552_0_reconst_depth]: /experiments/images/JNet_552_0_reconst_depth.png
[JNet_552_0_reconst_plane]: /experiments/images/JNet_552_0_reconst_plane.png
[JNet_552_1_aligned_depth]: /experiments/images/JNet_552_1_aligned_depth.png
[JNet_552_1_aligned_plane]: /experiments/images/JNet_552_1_aligned_plane.png
[JNet_552_1_heatmap_depth]: /experiments/images/JNet_552_1_heatmap_depth.png
[JNet_552_1_heatmap_plane]: /experiments/images/JNet_552_1_heatmap_plane.png
[JNet_552_1_labelx_depth]: /experiments/images/JNet_552_1_labelx_depth.png
[JNet_552_1_labelx_plane]: /experiments/images/JNet_552_1_labelx_plane.png
[JNet_552_1_labelz_depth]: /experiments/images/JNet_552_1_labelz_depth.png
[JNet_552_1_labelz_plane]: /experiments/images/JNet_552_1_labelz_plane.png
[JNet_552_1_novibrate_depth]: /experiments/images/JNet_552_1_novibrate_depth.png
[JNet_552_1_novibrate_plane]: /experiments/images/JNet_552_1_novibrate_plane.png
[JNet_552_1_original_depth]: /experiments/images/JNet_552_1_original_depth.png
[JNet_552_1_original_plane]: /experiments/images/JNet_552_1_original_plane.png
[JNet_552_1_outputx_depth]: /experiments/images/JNet_552_1_outputx_depth.png
[JNet_552_1_outputx_plane]: /experiments/images/JNet_552_1_outputx_plane.png
[JNet_552_1_outputz_depth]: /experiments/images/JNet_552_1_outputz_depth.png
[JNet_552_1_outputz_plane]: /experiments/images/JNet_552_1_outputz_plane.png
[JNet_552_1_reconst_depth]: /experiments/images/JNet_552_1_reconst_depth.png
[JNet_552_1_reconst_plane]: /experiments/images/JNet_552_1_reconst_plane.png
[JNet_552_2_aligned_depth]: /experiments/images/JNet_552_2_aligned_depth.png
[JNet_552_2_aligned_plane]: /experiments/images/JNet_552_2_aligned_plane.png
[JNet_552_2_heatmap_depth]: /experiments/images/JNet_552_2_heatmap_depth.png
[JNet_552_2_heatmap_plane]: /experiments/images/JNet_552_2_heatmap_plane.png
[JNet_552_2_labelx_depth]: /experiments/images/JNet_552_2_labelx_depth.png
[JNet_552_2_labelx_plane]: /experiments/images/JNet_552_2_labelx_plane.png
[JNet_552_2_labelz_depth]: /experiments/images/JNet_552_2_labelz_depth.png
[JNet_552_2_labelz_plane]: /experiments/images/JNet_552_2_labelz_plane.png
[JNet_552_2_novibrate_depth]: /experiments/images/JNet_552_2_novibrate_depth.png
[JNet_552_2_novibrate_plane]: /experiments/images/JNet_552_2_novibrate_plane.png
[JNet_552_2_original_depth]: /experiments/images/JNet_552_2_original_depth.png
[JNet_552_2_original_plane]: /experiments/images/JNet_552_2_original_plane.png
[JNet_552_2_outputx_depth]: /experiments/images/JNet_552_2_outputx_depth.png
[JNet_552_2_outputx_plane]: /experiments/images/JNet_552_2_outputx_plane.png
[JNet_552_2_outputz_depth]: /experiments/images/JNet_552_2_outputz_depth.png
[JNet_552_2_outputz_plane]: /experiments/images/JNet_552_2_outputz_plane.png
[JNet_552_2_reconst_depth]: /experiments/images/JNet_552_2_reconst_depth.png
[JNet_552_2_reconst_plane]: /experiments/images/JNet_552_2_reconst_plane.png
[JNet_552_3_aligned_depth]: /experiments/images/JNet_552_3_aligned_depth.png
[JNet_552_3_aligned_plane]: /experiments/images/JNet_552_3_aligned_plane.png
[JNet_552_3_heatmap_depth]: /experiments/images/JNet_552_3_heatmap_depth.png
[JNet_552_3_heatmap_plane]: /experiments/images/JNet_552_3_heatmap_plane.png
[JNet_552_3_labelx_depth]: /experiments/images/JNet_552_3_labelx_depth.png
[JNet_552_3_labelx_plane]: /experiments/images/JNet_552_3_labelx_plane.png
[JNet_552_3_labelz_depth]: /experiments/images/JNet_552_3_labelz_depth.png
[JNet_552_3_labelz_plane]: /experiments/images/JNet_552_3_labelz_plane.png
[JNet_552_3_novibrate_depth]: /experiments/images/JNet_552_3_novibrate_depth.png
[JNet_552_3_novibrate_plane]: /experiments/images/JNet_552_3_novibrate_plane.png
[JNet_552_3_original_depth]: /experiments/images/JNet_552_3_original_depth.png
[JNet_552_3_original_plane]: /experiments/images/JNet_552_3_original_plane.png
[JNet_552_3_outputx_depth]: /experiments/images/JNet_552_3_outputx_depth.png
[JNet_552_3_outputx_plane]: /experiments/images/JNet_552_3_outputx_plane.png
[JNet_552_3_outputz_depth]: /experiments/images/JNet_552_3_outputz_depth.png
[JNet_552_3_outputz_plane]: /experiments/images/JNet_552_3_outputz_plane.png
[JNet_552_3_reconst_depth]: /experiments/images/JNet_552_3_reconst_depth.png
[JNet_552_3_reconst_plane]: /experiments/images/JNet_552_3_reconst_plane.png
[JNet_552_4_aligned_depth]: /experiments/images/JNet_552_4_aligned_depth.png
[JNet_552_4_aligned_plane]: /experiments/images/JNet_552_4_aligned_plane.png
[JNet_552_4_heatmap_depth]: /experiments/images/JNet_552_4_heatmap_depth.png
[JNet_552_4_heatmap_plane]: /experiments/images/JNet_552_4_heatmap_plane.png
[JNet_552_4_labelx_depth]: /experiments/images/JNet_552_4_labelx_depth.png
[JNet_552_4_labelx_plane]: /experiments/images/JNet_552_4_labelx_plane.png
[JNet_552_4_labelz_depth]: /experiments/images/JNet_552_4_labelz_depth.png
[JNet_552_4_labelz_plane]: /experiments/images/JNet_552_4_labelz_plane.png
[JNet_552_4_novibrate_depth]: /experiments/images/JNet_552_4_novibrate_depth.png
[JNet_552_4_novibrate_plane]: /experiments/images/JNet_552_4_novibrate_plane.png
[JNet_552_4_original_depth]: /experiments/images/JNet_552_4_original_depth.png
[JNet_552_4_original_plane]: /experiments/images/JNet_552_4_original_plane.png
[JNet_552_4_outputx_depth]: /experiments/images/JNet_552_4_outputx_depth.png
[JNet_552_4_outputx_plane]: /experiments/images/JNet_552_4_outputx_plane.png
[JNet_552_4_outputz_depth]: /experiments/images/JNet_552_4_outputz_depth.png
[JNet_552_4_outputz_plane]: /experiments/images/JNet_552_4_outputz_plane.png
[JNet_552_4_reconst_depth]: /experiments/images/JNet_552_4_reconst_depth.png
[JNet_552_4_reconst_plane]: /experiments/images/JNet_552_4_reconst_plane.png
[JNet_552_microglia_0_aligned_depth]: /experiments/images/JNet_552_microglia_0_aligned_depth.png
[JNet_552_microglia_0_aligned_plane]: /experiments/images/JNet_552_microglia_0_aligned_plane.png
[JNet_552_microglia_0_heatmap_depth]: /experiments/images/JNet_552_microglia_0_heatmap_depth.png
[JNet_552_microglia_0_heatmap_plane]: /experiments/images/JNet_552_microglia_0_heatmap_plane.png
[JNet_552_microglia_0_original_depth]: /experiments/images/JNet_552_microglia_0_original_depth.png
[JNet_552_microglia_0_original_plane]: /experiments/images/JNet_552_microglia_0_original_plane.png
[JNet_552_microglia_0_outputx_depth]: /experiments/images/JNet_552_microglia_0_outputx_depth.png
[JNet_552_microglia_0_outputx_plane]: /experiments/images/JNet_552_microglia_0_outputx_plane.png
[JNet_552_microglia_0_outputz_depth]: /experiments/images/JNet_552_microglia_0_outputz_depth.png
[JNet_552_microglia_0_outputz_plane]: /experiments/images/JNet_552_microglia_0_outputz_plane.png
[JNet_552_microglia_0_reconst_depth]: /experiments/images/JNet_552_microglia_0_reconst_depth.png
[JNet_552_microglia_0_reconst_plane]: /experiments/images/JNet_552_microglia_0_reconst_plane.png
[JNet_552_microglia_1_aligned_depth]: /experiments/images/JNet_552_microglia_1_aligned_depth.png
[JNet_552_microglia_1_aligned_plane]: /experiments/images/JNet_552_microglia_1_aligned_plane.png
[JNet_552_microglia_1_heatmap_depth]: /experiments/images/JNet_552_microglia_1_heatmap_depth.png
[JNet_552_microglia_1_heatmap_plane]: /experiments/images/JNet_552_microglia_1_heatmap_plane.png
[JNet_552_microglia_1_original_depth]: /experiments/images/JNet_552_microglia_1_original_depth.png
[JNet_552_microglia_1_original_plane]: /experiments/images/JNet_552_microglia_1_original_plane.png
[JNet_552_microglia_1_outputx_depth]: /experiments/images/JNet_552_microglia_1_outputx_depth.png
[JNet_552_microglia_1_outputx_plane]: /experiments/images/JNet_552_microglia_1_outputx_plane.png
[JNet_552_microglia_1_outputz_depth]: /experiments/images/JNet_552_microglia_1_outputz_depth.png
[JNet_552_microglia_1_outputz_plane]: /experiments/images/JNet_552_microglia_1_outputz_plane.png
[JNet_552_microglia_1_reconst_depth]: /experiments/images/JNet_552_microglia_1_reconst_depth.png
[JNet_552_microglia_1_reconst_plane]: /experiments/images/JNet_552_microglia_1_reconst_plane.png
[JNet_552_microglia_2_aligned_depth]: /experiments/images/JNet_552_microglia_2_aligned_depth.png
[JNet_552_microglia_2_aligned_plane]: /experiments/images/JNet_552_microglia_2_aligned_plane.png
[JNet_552_microglia_2_heatmap_depth]: /experiments/images/JNet_552_microglia_2_heatmap_depth.png
[JNet_552_microglia_2_heatmap_plane]: /experiments/images/JNet_552_microglia_2_heatmap_plane.png
[JNet_552_microglia_2_original_depth]: /experiments/images/JNet_552_microglia_2_original_depth.png
[JNet_552_microglia_2_original_plane]: /experiments/images/JNet_552_microglia_2_original_plane.png
[JNet_552_microglia_2_outputx_depth]: /experiments/images/JNet_552_microglia_2_outputx_depth.png
[JNet_552_microglia_2_outputx_plane]: /experiments/images/JNet_552_microglia_2_outputx_plane.png
[JNet_552_microglia_2_outputz_depth]: /experiments/images/JNet_552_microglia_2_outputz_depth.png
[JNet_552_microglia_2_outputz_plane]: /experiments/images/JNet_552_microglia_2_outputz_plane.png
[JNet_552_microglia_2_reconst_depth]: /experiments/images/JNet_552_microglia_2_reconst_depth.png
[JNet_552_microglia_2_reconst_plane]: /experiments/images/JNet_552_microglia_2_reconst_plane.png
[JNet_552_microglia_3_aligned_depth]: /experiments/images/JNet_552_microglia_3_aligned_depth.png
[JNet_552_microglia_3_aligned_plane]: /experiments/images/JNet_552_microglia_3_aligned_plane.png
[JNet_552_microglia_3_heatmap_depth]: /experiments/images/JNet_552_microglia_3_heatmap_depth.png
[JNet_552_microglia_3_heatmap_plane]: /experiments/images/JNet_552_microglia_3_heatmap_plane.png
[JNet_552_microglia_3_original_depth]: /experiments/images/JNet_552_microglia_3_original_depth.png
[JNet_552_microglia_3_original_plane]: /experiments/images/JNet_552_microglia_3_original_plane.png
[JNet_552_microglia_3_outputx_depth]: /experiments/images/JNet_552_microglia_3_outputx_depth.png
[JNet_552_microglia_3_outputx_plane]: /experiments/images/JNet_552_microglia_3_outputx_plane.png
[JNet_552_microglia_3_outputz_depth]: /experiments/images/JNet_552_microglia_3_outputz_depth.png
[JNet_552_microglia_3_outputz_plane]: /experiments/images/JNet_552_microglia_3_outputz_plane.png
[JNet_552_microglia_3_reconst_depth]: /experiments/images/JNet_552_microglia_3_reconst_depth.png
[JNet_552_microglia_3_reconst_plane]: /experiments/images/JNet_552_microglia_3_reconst_plane.png
[JNet_552_microglia_4_aligned_depth]: /experiments/images/JNet_552_microglia_4_aligned_depth.png
[JNet_552_microglia_4_aligned_plane]: /experiments/images/JNet_552_microglia_4_aligned_plane.png
[JNet_552_microglia_4_heatmap_depth]: /experiments/images/JNet_552_microglia_4_heatmap_depth.png
[JNet_552_microglia_4_heatmap_plane]: /experiments/images/JNet_552_microglia_4_heatmap_plane.png
[JNet_552_microglia_4_original_depth]: /experiments/images/JNet_552_microglia_4_original_depth.png
[JNet_552_microglia_4_original_plane]: /experiments/images/JNet_552_microglia_4_original_plane.png
[JNet_552_microglia_4_outputx_depth]: /experiments/images/JNet_552_microglia_4_outputx_depth.png
[JNet_552_microglia_4_outputx_plane]: /experiments/images/JNet_552_microglia_4_outputx_plane.png
[JNet_552_microglia_4_outputz_depth]: /experiments/images/JNet_552_microglia_4_outputz_depth.png
[JNet_552_microglia_4_outputz_plane]: /experiments/images/JNet_552_microglia_4_outputz_plane.png
[JNet_552_microglia_4_reconst_depth]: /experiments/images/JNet_552_microglia_4_reconst_depth.png
[JNet_552_microglia_4_reconst_plane]: /experiments/images/JNet_552_microglia_4_reconst_plane.png
[JNet_552_psf_post]: /experiments/images/JNet_552_psf_post.png
[JNet_552_psf_pre]: /experiments/images/JNet_552_psf_pre.png
[finetuned]: /experiments/tmp/JNet_552_train.png
[pretrained_model]: /experiments/tmp/JNet_542_pretrain_train.png
