



# JNet_513 Report
  
vibration without warming up slowly pretraining and without vib in finetuning  
pretrained model : JNet_512_pretrain
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
|folderpath|/home/haruhiko/Downloads/Set_03|
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
|folderpath|/home/haruhiko/Downloads/Set_03|
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
|n_epochs|200|
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

### train_loop

|Parameter|Value|
| :--- | :--- |
|batch_size|1|
|n_epochs|200|
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
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results

### Pretraining
  
Segmentation: mean MSE: 0.012104468420147896, mean BCE: 0.044863197952508926  
Luminance Estimation: mean MSE: 0.9682755470275879, mean BCE: 6.29713249206543
### 0

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_0_original_plane]|![JNet_512_pretrain_0_outputx_plane]|![JNet_512_pretrain_0_labelx_plane]|![JNet_512_pretrain_0_outputz_plane]|![JNet_512_pretrain_0_labelz_plane]|
  
MSEx: 0.015696678310632706, BCEx: 0.055252037942409515  
MSEz: 0.9544225335121155, BCEz: 6.186501502990723  

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_0_original_depth]|![JNet_512_pretrain_0_outputx_depth]|![JNet_512_pretrain_0_labelx_depth]|![JNet_512_pretrain_0_outputz_depth]|![JNet_512_pretrain_0_labelz_depth]|
  
MSEx: 0.015696678310632706, BCEx: 0.055252037942409515  
MSEz: 0.9544225335121155, BCEz: 6.186501502990723  

### 1

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_1_original_plane]|![JNet_512_pretrain_1_outputx_plane]|![JNet_512_pretrain_1_labelx_plane]|![JNet_512_pretrain_1_outputz_plane]|![JNet_512_pretrain_1_labelz_plane]|
  
MSEx: 0.01261675264686346, BCEx: 0.047051891684532166  
MSEz: 0.9712703824043274, BCEz: 6.503024101257324  

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_1_original_depth]|![JNet_512_pretrain_1_outputx_depth]|![JNet_512_pretrain_1_labelx_depth]|![JNet_512_pretrain_1_outputz_depth]|![JNet_512_pretrain_1_labelz_depth]|
  
MSEx: 0.01261675264686346, BCEx: 0.047051891684532166  
MSEz: 0.9712703824043274, BCEz: 6.503024101257324  

### 2

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_2_original_plane]|![JNet_512_pretrain_2_outputx_plane]|![JNet_512_pretrain_2_labelx_plane]|![JNet_512_pretrain_2_outputz_plane]|![JNet_512_pretrain_2_labelz_plane]|
  
MSEx: 0.010328221134841442, BCEx: 0.04015090689063072  
MSEz: 0.9680277705192566, BCEz: 6.222410678863525  

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_2_original_depth]|![JNet_512_pretrain_2_outputx_depth]|![JNet_512_pretrain_2_labelx_depth]|![JNet_512_pretrain_2_outputz_depth]|![JNet_512_pretrain_2_labelz_depth]|
  
MSEx: 0.010328221134841442, BCEx: 0.04015090689063072  
MSEz: 0.9680277705192566, BCEz: 6.222410678863525  

### 3

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_3_original_plane]|![JNet_512_pretrain_3_outputx_plane]|![JNet_512_pretrain_3_labelx_plane]|![JNet_512_pretrain_3_outputz_plane]|![JNet_512_pretrain_3_labelz_plane]|
  
MSEx: 0.012524819932878017, BCEx: 0.04687507450580597  
MSEz: 0.9738572835922241, BCEz: 6.316730499267578  

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_3_original_depth]|![JNet_512_pretrain_3_outputx_depth]|![JNet_512_pretrain_3_labelx_depth]|![JNet_512_pretrain_3_outputz_depth]|![JNet_512_pretrain_3_labelz_depth]|
  
MSEx: 0.012524819932878017, BCEx: 0.04687507450580597  
MSEz: 0.9738572835922241, BCEz: 6.316730499267578  

### 4

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_4_original_plane]|![JNet_512_pretrain_4_outputx_plane]|![JNet_512_pretrain_4_labelx_plane]|![JNet_512_pretrain_4_outputz_plane]|![JNet_512_pretrain_4_labelz_plane]|
  
MSEx: 0.00935586728155613, BCEx: 0.03498607873916626  
MSEz: 0.9737995862960815, BCEz: 6.256996154785156  

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_4_original_depth]|![JNet_512_pretrain_4_outputx_depth]|![JNet_512_pretrain_4_labelx_depth]|![JNet_512_pretrain_4_outputz_depth]|![JNet_512_pretrain_4_labelz_depth]|
  
MSEx: 0.00935586728155613, BCEx: 0.03498607873916626  
MSEz: 0.9737995862960815, BCEz: 6.256996154785156  

### Finetuning Results with Simulation

### image 0

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_513_0_original_plane]|![JNet_513_0_reconst_plane]|![JNet_513_0_heatmap_plane]|![JNet_513_0_outputx_plane]|![JNet_513_0_labelx_plane]|![JNet_513_0_outputz_plane]|![JNet_513_0_labelz_plane]|
  
MSEz: 0.9541367888450623, quantized loss: 0.010010411031544209  

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_513_0_original_depth]|![JNet_513_0_reconst_depth]|![JNet_513_0_heatmap_depth]|![JNet_513_0_outputx_depth]|![JNet_513_0_labelx_depth]|![JNet_513_0_outputz_depth]|![JNet_513_0_labelz_depth]|
  
MSEz: 0.9541367888450623, quantized loss: 0.010010411031544209  

### image 1

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_513_1_original_plane]|![JNet_513_1_reconst_plane]|![JNet_513_1_heatmap_plane]|![JNet_513_1_outputx_plane]|![JNet_513_1_labelx_plane]|![JNet_513_1_outputz_plane]|![JNet_513_1_labelz_plane]|
  
MSEz: 0.9524046182632446, quantized loss: 0.009426719509065151  

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_513_1_original_depth]|![JNet_513_1_reconst_depth]|![JNet_513_1_heatmap_depth]|![JNet_513_1_outputx_depth]|![JNet_513_1_labelx_depth]|![JNet_513_1_outputz_depth]|![JNet_513_1_labelz_depth]|
  
MSEz: 0.9524046182632446, quantized loss: 0.009426719509065151  

### image 2

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_513_2_original_plane]|![JNet_513_2_reconst_plane]|![JNet_513_2_heatmap_plane]|![JNet_513_2_outputx_plane]|![JNet_513_2_labelx_plane]|![JNet_513_2_outputz_plane]|![JNet_513_2_labelz_plane]|
  
MSEz: 0.8992493748664856, quantized loss: 0.019186407327651978  

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_513_2_original_depth]|![JNet_513_2_reconst_depth]|![JNet_513_2_heatmap_depth]|![JNet_513_2_outputx_depth]|![JNet_513_2_labelx_depth]|![JNet_513_2_outputz_depth]|![JNet_513_2_labelz_depth]|
  
MSEz: 0.8992493748664856, quantized loss: 0.019186407327651978  

### image 3

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_513_3_original_plane]|![JNet_513_3_reconst_plane]|![JNet_513_3_heatmap_plane]|![JNet_513_3_outputx_plane]|![JNet_513_3_labelx_plane]|![JNet_513_3_outputz_plane]|![JNet_513_3_labelz_plane]|
  
MSEz: 0.93842613697052, quantized loss: 0.0108004966750741  

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_513_3_original_depth]|![JNet_513_3_reconst_depth]|![JNet_513_3_heatmap_depth]|![JNet_513_3_outputx_depth]|![JNet_513_3_labelx_depth]|![JNet_513_3_outputz_depth]|![JNet_513_3_labelz_depth]|
  
MSEz: 0.93842613697052, quantized loss: 0.0108004966750741  

### image 4

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_513_4_original_plane]|![JNet_513_4_reconst_plane]|![JNet_513_4_heatmap_plane]|![JNet_513_4_outputx_plane]|![JNet_513_4_labelx_plane]|![JNet_513_4_outputz_plane]|![JNet_513_4_labelz_plane]|
  
MSEz: 0.9614249467849731, quantized loss: 0.008609408512711525  

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_513_4_original_depth]|![JNet_513_4_reconst_depth]|![JNet_513_4_heatmap_depth]|![JNet_513_4_outputx_depth]|![JNet_513_4_labelx_depth]|![JNet_513_4_outputz_depth]|![JNet_513_4_labelz_depth]|
  
MSEz: 0.9614249467849731, quantized loss: 0.008609408512711525  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
### Finetuning Results with Microglia

#### finetuning == False

### image 0

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_microglia_0_original_plane]|![JNet_512_pretrain_microglia_0_outputx_plane]|![JNet_512_pretrain_microglia_0_outputz_plane]|![JNet_512_pretrain_microglia_0_reconst_plane]|![JNet_512_pretrain_microglia_0_heatmap_plane]|
  

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_microglia_0_original_depth]|![JNet_512_pretrain_microglia_0_outputx_depth]|![JNet_512_pretrain_microglia_0_outputz_depth]|![JNet_512_pretrain_microglia_0_reconst_depth]|![JNet_512_pretrain_microglia_0_heatmap_depth]|
  

### image 1

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_microglia_1_original_plane]|![JNet_512_pretrain_microglia_1_outputx_plane]|![JNet_512_pretrain_microglia_1_outputz_plane]|![JNet_512_pretrain_microglia_1_reconst_plane]|![JNet_512_pretrain_microglia_1_heatmap_plane]|
  

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_microglia_1_original_depth]|![JNet_512_pretrain_microglia_1_outputx_depth]|![JNet_512_pretrain_microglia_1_outputz_depth]|![JNet_512_pretrain_microglia_1_reconst_depth]|![JNet_512_pretrain_microglia_1_heatmap_depth]|
  

### image 2

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_microglia_2_original_plane]|![JNet_512_pretrain_microglia_2_outputx_plane]|![JNet_512_pretrain_microglia_2_outputz_plane]|![JNet_512_pretrain_microglia_2_reconst_plane]|![JNet_512_pretrain_microglia_2_heatmap_plane]|
  

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_microglia_2_original_depth]|![JNet_512_pretrain_microglia_2_outputx_depth]|![JNet_512_pretrain_microglia_2_outputz_depth]|![JNet_512_pretrain_microglia_2_reconst_depth]|![JNet_512_pretrain_microglia_2_heatmap_depth]|
  

### image 3

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_microglia_3_original_plane]|![JNet_512_pretrain_microglia_3_outputx_plane]|![JNet_512_pretrain_microglia_3_outputz_plane]|![JNet_512_pretrain_microglia_3_reconst_plane]|![JNet_512_pretrain_microglia_3_heatmap_plane]|
  

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_microglia_3_original_depth]|![JNet_512_pretrain_microglia_3_outputx_depth]|![JNet_512_pretrain_microglia_3_outputz_depth]|![JNet_512_pretrain_microglia_3_reconst_depth]|![JNet_512_pretrain_microglia_3_heatmap_depth]|
  

### image 4

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_microglia_4_original_plane]|![JNet_512_pretrain_microglia_4_outputx_plane]|![JNet_512_pretrain_microglia_4_outputz_plane]|![JNet_512_pretrain_microglia_4_reconst_plane]|![JNet_512_pretrain_microglia_4_heatmap_plane]|
  

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_512_pretrain_microglia_4_original_depth]|![JNet_512_pretrain_microglia_4_outputx_depth]|![JNet_512_pretrain_microglia_4_outputz_depth]|![JNet_512_pretrain_microglia_4_reconst_depth]|![JNet_512_pretrain_microglia_4_heatmap_depth]|
  

#### finetuning == True

### image 0

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_513_microglia_0_original_plane]|![JNet_513_microglia_0_outputx_plane]|![JNet_513_microglia_0_outputz_plane]|![JNet_513_microglia_0_reconst_plane]|![JNet_513_microglia_0_heatmap_plane]|
  

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_513_microglia_0_original_depth]|![JNet_513_microglia_0_outputx_depth]|![JNet_513_microglia_0_outputz_depth]|![JNet_513_microglia_0_reconst_depth]|![JNet_513_microglia_0_heatmap_depth]|
  

### image 1

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_513_microglia_1_original_plane]|![JNet_513_microglia_1_outputx_plane]|![JNet_513_microglia_1_outputz_plane]|![JNet_513_microglia_1_reconst_plane]|![JNet_513_microglia_1_heatmap_plane]|
  

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_513_microglia_1_original_depth]|![JNet_513_microglia_1_outputx_depth]|![JNet_513_microglia_1_outputz_depth]|![JNet_513_microglia_1_reconst_depth]|![JNet_513_microglia_1_heatmap_depth]|
  

### image 2

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_513_microglia_2_original_plane]|![JNet_513_microglia_2_outputx_plane]|![JNet_513_microglia_2_outputz_plane]|![JNet_513_microglia_2_reconst_plane]|![JNet_513_microglia_2_heatmap_plane]|
  

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_513_microglia_2_original_depth]|![JNet_513_microglia_2_outputx_depth]|![JNet_513_microglia_2_outputz_depth]|![JNet_513_microglia_2_reconst_depth]|![JNet_513_microglia_2_heatmap_depth]|
  

### image 3

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_513_microglia_3_original_plane]|![JNet_513_microglia_3_outputx_plane]|![JNet_513_microglia_3_outputz_plane]|![JNet_513_microglia_3_reconst_plane]|![JNet_513_microglia_3_heatmap_plane]|
  

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_513_microglia_3_original_depth]|![JNet_513_microglia_3_outputx_depth]|![JNet_513_microglia_3_outputz_depth]|![JNet_513_microglia_3_reconst_depth]|![JNet_513_microglia_3_heatmap_depth]|
  

### image 4

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_513_microglia_4_original_plane]|![JNet_513_microglia_4_outputx_plane]|![JNet_513_microglia_4_outputz_plane]|![JNet_513_microglia_4_reconst_plane]|![JNet_513_microglia_4_heatmap_plane]|
  

|original|outputx|outputz|reconst|heatmap|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_513_microglia_4_original_depth]|![JNet_513_microglia_4_outputx_depth]|![JNet_513_microglia_4_outputz_depth]|![JNet_513_microglia_4_reconst_depth]|![JNet_513_microglia_4_heatmap_depth]|
  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_513_psf_pre]|![JNet_513_psf_post]|

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
  



[JNet_512_pretrain_0_labelx_depth]: /experiments/images/JNet_512_pretrain_0_labelx_depth.png
[JNet_512_pretrain_0_labelx_plane]: /experiments/images/JNet_512_pretrain_0_labelx_plane.png
[JNet_512_pretrain_0_labelz_depth]: /experiments/images/JNet_512_pretrain_0_labelz_depth.png
[JNet_512_pretrain_0_labelz_plane]: /experiments/images/JNet_512_pretrain_0_labelz_plane.png
[JNet_512_pretrain_0_original_depth]: /experiments/images/JNet_512_pretrain_0_original_depth.png
[JNet_512_pretrain_0_original_plane]: /experiments/images/JNet_512_pretrain_0_original_plane.png
[JNet_512_pretrain_0_outputx_depth]: /experiments/images/JNet_512_pretrain_0_outputx_depth.png
[JNet_512_pretrain_0_outputx_plane]: /experiments/images/JNet_512_pretrain_0_outputx_plane.png
[JNet_512_pretrain_0_outputz_depth]: /experiments/images/JNet_512_pretrain_0_outputz_depth.png
[JNet_512_pretrain_0_outputz_plane]: /experiments/images/JNet_512_pretrain_0_outputz_plane.png
[JNet_512_pretrain_1_labelx_depth]: /experiments/images/JNet_512_pretrain_1_labelx_depth.png
[JNet_512_pretrain_1_labelx_plane]: /experiments/images/JNet_512_pretrain_1_labelx_plane.png
[JNet_512_pretrain_1_labelz_depth]: /experiments/images/JNet_512_pretrain_1_labelz_depth.png
[JNet_512_pretrain_1_labelz_plane]: /experiments/images/JNet_512_pretrain_1_labelz_plane.png
[JNet_512_pretrain_1_original_depth]: /experiments/images/JNet_512_pretrain_1_original_depth.png
[JNet_512_pretrain_1_original_plane]: /experiments/images/JNet_512_pretrain_1_original_plane.png
[JNet_512_pretrain_1_outputx_depth]: /experiments/images/JNet_512_pretrain_1_outputx_depth.png
[JNet_512_pretrain_1_outputx_plane]: /experiments/images/JNet_512_pretrain_1_outputx_plane.png
[JNet_512_pretrain_1_outputz_depth]: /experiments/images/JNet_512_pretrain_1_outputz_depth.png
[JNet_512_pretrain_1_outputz_plane]: /experiments/images/JNet_512_pretrain_1_outputz_plane.png
[JNet_512_pretrain_2_labelx_depth]: /experiments/images/JNet_512_pretrain_2_labelx_depth.png
[JNet_512_pretrain_2_labelx_plane]: /experiments/images/JNet_512_pretrain_2_labelx_plane.png
[JNet_512_pretrain_2_labelz_depth]: /experiments/images/JNet_512_pretrain_2_labelz_depth.png
[JNet_512_pretrain_2_labelz_plane]: /experiments/images/JNet_512_pretrain_2_labelz_plane.png
[JNet_512_pretrain_2_original_depth]: /experiments/images/JNet_512_pretrain_2_original_depth.png
[JNet_512_pretrain_2_original_plane]: /experiments/images/JNet_512_pretrain_2_original_plane.png
[JNet_512_pretrain_2_outputx_depth]: /experiments/images/JNet_512_pretrain_2_outputx_depth.png
[JNet_512_pretrain_2_outputx_plane]: /experiments/images/JNet_512_pretrain_2_outputx_plane.png
[JNet_512_pretrain_2_outputz_depth]: /experiments/images/JNet_512_pretrain_2_outputz_depth.png
[JNet_512_pretrain_2_outputz_plane]: /experiments/images/JNet_512_pretrain_2_outputz_plane.png
[JNet_512_pretrain_3_labelx_depth]: /experiments/images/JNet_512_pretrain_3_labelx_depth.png
[JNet_512_pretrain_3_labelx_plane]: /experiments/images/JNet_512_pretrain_3_labelx_plane.png
[JNet_512_pretrain_3_labelz_depth]: /experiments/images/JNet_512_pretrain_3_labelz_depth.png
[JNet_512_pretrain_3_labelz_plane]: /experiments/images/JNet_512_pretrain_3_labelz_plane.png
[JNet_512_pretrain_3_original_depth]: /experiments/images/JNet_512_pretrain_3_original_depth.png
[JNet_512_pretrain_3_original_plane]: /experiments/images/JNet_512_pretrain_3_original_plane.png
[JNet_512_pretrain_3_outputx_depth]: /experiments/images/JNet_512_pretrain_3_outputx_depth.png
[JNet_512_pretrain_3_outputx_plane]: /experiments/images/JNet_512_pretrain_3_outputx_plane.png
[JNet_512_pretrain_3_outputz_depth]: /experiments/images/JNet_512_pretrain_3_outputz_depth.png
[JNet_512_pretrain_3_outputz_plane]: /experiments/images/JNet_512_pretrain_3_outputz_plane.png
[JNet_512_pretrain_4_labelx_depth]: /experiments/images/JNet_512_pretrain_4_labelx_depth.png
[JNet_512_pretrain_4_labelx_plane]: /experiments/images/JNet_512_pretrain_4_labelx_plane.png
[JNet_512_pretrain_4_labelz_depth]: /experiments/images/JNet_512_pretrain_4_labelz_depth.png
[JNet_512_pretrain_4_labelz_plane]: /experiments/images/JNet_512_pretrain_4_labelz_plane.png
[JNet_512_pretrain_4_original_depth]: /experiments/images/JNet_512_pretrain_4_original_depth.png
[JNet_512_pretrain_4_original_plane]: /experiments/images/JNet_512_pretrain_4_original_plane.png
[JNet_512_pretrain_4_outputx_depth]: /experiments/images/JNet_512_pretrain_4_outputx_depth.png
[JNet_512_pretrain_4_outputx_plane]: /experiments/images/JNet_512_pretrain_4_outputx_plane.png
[JNet_512_pretrain_4_outputz_depth]: /experiments/images/JNet_512_pretrain_4_outputz_depth.png
[JNet_512_pretrain_4_outputz_plane]: /experiments/images/JNet_512_pretrain_4_outputz_plane.png
[JNet_512_pretrain_microglia_0_heatmap_depth]: /experiments/images/JNet_512_pretrain_microglia_0_heatmap_depth.png
[JNet_512_pretrain_microglia_0_heatmap_plane]: /experiments/images/JNet_512_pretrain_microglia_0_heatmap_plane.png
[JNet_512_pretrain_microglia_0_original_depth]: /experiments/images/JNet_512_pretrain_microglia_0_original_depth.png
[JNet_512_pretrain_microglia_0_original_plane]: /experiments/images/JNet_512_pretrain_microglia_0_original_plane.png
[JNet_512_pretrain_microglia_0_outputx_depth]: /experiments/images/JNet_512_pretrain_microglia_0_outputx_depth.png
[JNet_512_pretrain_microglia_0_outputx_plane]: /experiments/images/JNet_512_pretrain_microglia_0_outputx_plane.png
[JNet_512_pretrain_microglia_0_outputz_depth]: /experiments/images/JNet_512_pretrain_microglia_0_outputz_depth.png
[JNet_512_pretrain_microglia_0_outputz_plane]: /experiments/images/JNet_512_pretrain_microglia_0_outputz_plane.png
[JNet_512_pretrain_microglia_0_reconst_depth]: /experiments/images/JNet_512_pretrain_microglia_0_reconst_depth.png
[JNet_512_pretrain_microglia_0_reconst_plane]: /experiments/images/JNet_512_pretrain_microglia_0_reconst_plane.png
[JNet_512_pretrain_microglia_1_heatmap_depth]: /experiments/images/JNet_512_pretrain_microglia_1_heatmap_depth.png
[JNet_512_pretrain_microglia_1_heatmap_plane]: /experiments/images/JNet_512_pretrain_microglia_1_heatmap_plane.png
[JNet_512_pretrain_microglia_1_original_depth]: /experiments/images/JNet_512_pretrain_microglia_1_original_depth.png
[JNet_512_pretrain_microglia_1_original_plane]: /experiments/images/JNet_512_pretrain_microglia_1_original_plane.png
[JNet_512_pretrain_microglia_1_outputx_depth]: /experiments/images/JNet_512_pretrain_microglia_1_outputx_depth.png
[JNet_512_pretrain_microglia_1_outputx_plane]: /experiments/images/JNet_512_pretrain_microglia_1_outputx_plane.png
[JNet_512_pretrain_microglia_1_outputz_depth]: /experiments/images/JNet_512_pretrain_microglia_1_outputz_depth.png
[JNet_512_pretrain_microglia_1_outputz_plane]: /experiments/images/JNet_512_pretrain_microglia_1_outputz_plane.png
[JNet_512_pretrain_microglia_1_reconst_depth]: /experiments/images/JNet_512_pretrain_microglia_1_reconst_depth.png
[JNet_512_pretrain_microglia_1_reconst_plane]: /experiments/images/JNet_512_pretrain_microglia_1_reconst_plane.png
[JNet_512_pretrain_microglia_2_heatmap_depth]: /experiments/images/JNet_512_pretrain_microglia_2_heatmap_depth.png
[JNet_512_pretrain_microglia_2_heatmap_plane]: /experiments/images/JNet_512_pretrain_microglia_2_heatmap_plane.png
[JNet_512_pretrain_microglia_2_original_depth]: /experiments/images/JNet_512_pretrain_microglia_2_original_depth.png
[JNet_512_pretrain_microglia_2_original_plane]: /experiments/images/JNet_512_pretrain_microglia_2_original_plane.png
[JNet_512_pretrain_microglia_2_outputx_depth]: /experiments/images/JNet_512_pretrain_microglia_2_outputx_depth.png
[JNet_512_pretrain_microglia_2_outputx_plane]: /experiments/images/JNet_512_pretrain_microglia_2_outputx_plane.png
[JNet_512_pretrain_microglia_2_outputz_depth]: /experiments/images/JNet_512_pretrain_microglia_2_outputz_depth.png
[JNet_512_pretrain_microglia_2_outputz_plane]: /experiments/images/JNet_512_pretrain_microglia_2_outputz_plane.png
[JNet_512_pretrain_microglia_2_reconst_depth]: /experiments/images/JNet_512_pretrain_microglia_2_reconst_depth.png
[JNet_512_pretrain_microglia_2_reconst_plane]: /experiments/images/JNet_512_pretrain_microglia_2_reconst_plane.png
[JNet_512_pretrain_microglia_3_heatmap_depth]: /experiments/images/JNet_512_pretrain_microglia_3_heatmap_depth.png
[JNet_512_pretrain_microglia_3_heatmap_plane]: /experiments/images/JNet_512_pretrain_microglia_3_heatmap_plane.png
[JNet_512_pretrain_microglia_3_original_depth]: /experiments/images/JNet_512_pretrain_microglia_3_original_depth.png
[JNet_512_pretrain_microglia_3_original_plane]: /experiments/images/JNet_512_pretrain_microglia_3_original_plane.png
[JNet_512_pretrain_microglia_3_outputx_depth]: /experiments/images/JNet_512_pretrain_microglia_3_outputx_depth.png
[JNet_512_pretrain_microglia_3_outputx_plane]: /experiments/images/JNet_512_pretrain_microglia_3_outputx_plane.png
[JNet_512_pretrain_microglia_3_outputz_depth]: /experiments/images/JNet_512_pretrain_microglia_3_outputz_depth.png
[JNet_512_pretrain_microglia_3_outputz_plane]: /experiments/images/JNet_512_pretrain_microglia_3_outputz_plane.png
[JNet_512_pretrain_microglia_3_reconst_depth]: /experiments/images/JNet_512_pretrain_microglia_3_reconst_depth.png
[JNet_512_pretrain_microglia_3_reconst_plane]: /experiments/images/JNet_512_pretrain_microglia_3_reconst_plane.png
[JNet_512_pretrain_microglia_4_heatmap_depth]: /experiments/images/JNet_512_pretrain_microglia_4_heatmap_depth.png
[JNet_512_pretrain_microglia_4_heatmap_plane]: /experiments/images/JNet_512_pretrain_microglia_4_heatmap_plane.png
[JNet_512_pretrain_microglia_4_original_depth]: /experiments/images/JNet_512_pretrain_microglia_4_original_depth.png
[JNet_512_pretrain_microglia_4_original_plane]: /experiments/images/JNet_512_pretrain_microglia_4_original_plane.png
[JNet_512_pretrain_microglia_4_outputx_depth]: /experiments/images/JNet_512_pretrain_microglia_4_outputx_depth.png
[JNet_512_pretrain_microglia_4_outputx_plane]: /experiments/images/JNet_512_pretrain_microglia_4_outputx_plane.png
[JNet_512_pretrain_microglia_4_outputz_depth]: /experiments/images/JNet_512_pretrain_microglia_4_outputz_depth.png
[JNet_512_pretrain_microglia_4_outputz_plane]: /experiments/images/JNet_512_pretrain_microglia_4_outputz_plane.png
[JNet_512_pretrain_microglia_4_reconst_depth]: /experiments/images/JNet_512_pretrain_microglia_4_reconst_depth.png
[JNet_512_pretrain_microglia_4_reconst_plane]: /experiments/images/JNet_512_pretrain_microglia_4_reconst_plane.png
[JNet_513_0_heatmap_depth]: /experiments/images/JNet_513_0_heatmap_depth.png
[JNet_513_0_heatmap_plane]: /experiments/images/JNet_513_0_heatmap_plane.png
[JNet_513_0_labelx_depth]: /experiments/images/JNet_513_0_labelx_depth.png
[JNet_513_0_labelx_plane]: /experiments/images/JNet_513_0_labelx_plane.png
[JNet_513_0_labelz_depth]: /experiments/images/JNet_513_0_labelz_depth.png
[JNet_513_0_labelz_plane]: /experiments/images/JNet_513_0_labelz_plane.png
[JNet_513_0_original_depth]: /experiments/images/JNet_513_0_original_depth.png
[JNet_513_0_original_plane]: /experiments/images/JNet_513_0_original_plane.png
[JNet_513_0_outputx_depth]: /experiments/images/JNet_513_0_outputx_depth.png
[JNet_513_0_outputx_plane]: /experiments/images/JNet_513_0_outputx_plane.png
[JNet_513_0_outputz_depth]: /experiments/images/JNet_513_0_outputz_depth.png
[JNet_513_0_outputz_plane]: /experiments/images/JNet_513_0_outputz_plane.png
[JNet_513_0_reconst_depth]: /experiments/images/JNet_513_0_reconst_depth.png
[JNet_513_0_reconst_plane]: /experiments/images/JNet_513_0_reconst_plane.png
[JNet_513_1_heatmap_depth]: /experiments/images/JNet_513_1_heatmap_depth.png
[JNet_513_1_heatmap_plane]: /experiments/images/JNet_513_1_heatmap_plane.png
[JNet_513_1_labelx_depth]: /experiments/images/JNet_513_1_labelx_depth.png
[JNet_513_1_labelx_plane]: /experiments/images/JNet_513_1_labelx_plane.png
[JNet_513_1_labelz_depth]: /experiments/images/JNet_513_1_labelz_depth.png
[JNet_513_1_labelz_plane]: /experiments/images/JNet_513_1_labelz_plane.png
[JNet_513_1_original_depth]: /experiments/images/JNet_513_1_original_depth.png
[JNet_513_1_original_plane]: /experiments/images/JNet_513_1_original_plane.png
[JNet_513_1_outputx_depth]: /experiments/images/JNet_513_1_outputx_depth.png
[JNet_513_1_outputx_plane]: /experiments/images/JNet_513_1_outputx_plane.png
[JNet_513_1_outputz_depth]: /experiments/images/JNet_513_1_outputz_depth.png
[JNet_513_1_outputz_plane]: /experiments/images/JNet_513_1_outputz_plane.png
[JNet_513_1_reconst_depth]: /experiments/images/JNet_513_1_reconst_depth.png
[JNet_513_1_reconst_plane]: /experiments/images/JNet_513_1_reconst_plane.png
[JNet_513_2_heatmap_depth]: /experiments/images/JNet_513_2_heatmap_depth.png
[JNet_513_2_heatmap_plane]: /experiments/images/JNet_513_2_heatmap_plane.png
[JNet_513_2_labelx_depth]: /experiments/images/JNet_513_2_labelx_depth.png
[JNet_513_2_labelx_plane]: /experiments/images/JNet_513_2_labelx_plane.png
[JNet_513_2_labelz_depth]: /experiments/images/JNet_513_2_labelz_depth.png
[JNet_513_2_labelz_plane]: /experiments/images/JNet_513_2_labelz_plane.png
[JNet_513_2_original_depth]: /experiments/images/JNet_513_2_original_depth.png
[JNet_513_2_original_plane]: /experiments/images/JNet_513_2_original_plane.png
[JNet_513_2_outputx_depth]: /experiments/images/JNet_513_2_outputx_depth.png
[JNet_513_2_outputx_plane]: /experiments/images/JNet_513_2_outputx_plane.png
[JNet_513_2_outputz_depth]: /experiments/images/JNet_513_2_outputz_depth.png
[JNet_513_2_outputz_plane]: /experiments/images/JNet_513_2_outputz_plane.png
[JNet_513_2_reconst_depth]: /experiments/images/JNet_513_2_reconst_depth.png
[JNet_513_2_reconst_plane]: /experiments/images/JNet_513_2_reconst_plane.png
[JNet_513_3_heatmap_depth]: /experiments/images/JNet_513_3_heatmap_depth.png
[JNet_513_3_heatmap_plane]: /experiments/images/JNet_513_3_heatmap_plane.png
[JNet_513_3_labelx_depth]: /experiments/images/JNet_513_3_labelx_depth.png
[JNet_513_3_labelx_plane]: /experiments/images/JNet_513_3_labelx_plane.png
[JNet_513_3_labelz_depth]: /experiments/images/JNet_513_3_labelz_depth.png
[JNet_513_3_labelz_plane]: /experiments/images/JNet_513_3_labelz_plane.png
[JNet_513_3_original_depth]: /experiments/images/JNet_513_3_original_depth.png
[JNet_513_3_original_plane]: /experiments/images/JNet_513_3_original_plane.png
[JNet_513_3_outputx_depth]: /experiments/images/JNet_513_3_outputx_depth.png
[JNet_513_3_outputx_plane]: /experiments/images/JNet_513_3_outputx_plane.png
[JNet_513_3_outputz_depth]: /experiments/images/JNet_513_3_outputz_depth.png
[JNet_513_3_outputz_plane]: /experiments/images/JNet_513_3_outputz_plane.png
[JNet_513_3_reconst_depth]: /experiments/images/JNet_513_3_reconst_depth.png
[JNet_513_3_reconst_plane]: /experiments/images/JNet_513_3_reconst_plane.png
[JNet_513_4_heatmap_depth]: /experiments/images/JNet_513_4_heatmap_depth.png
[JNet_513_4_heatmap_plane]: /experiments/images/JNet_513_4_heatmap_plane.png
[JNet_513_4_labelx_depth]: /experiments/images/JNet_513_4_labelx_depth.png
[JNet_513_4_labelx_plane]: /experiments/images/JNet_513_4_labelx_plane.png
[JNet_513_4_labelz_depth]: /experiments/images/JNet_513_4_labelz_depth.png
[JNet_513_4_labelz_plane]: /experiments/images/JNet_513_4_labelz_plane.png
[JNet_513_4_original_depth]: /experiments/images/JNet_513_4_original_depth.png
[JNet_513_4_original_plane]: /experiments/images/JNet_513_4_original_plane.png
[JNet_513_4_outputx_depth]: /experiments/images/JNet_513_4_outputx_depth.png
[JNet_513_4_outputx_plane]: /experiments/images/JNet_513_4_outputx_plane.png
[JNet_513_4_outputz_depth]: /experiments/images/JNet_513_4_outputz_depth.png
[JNet_513_4_outputz_plane]: /experiments/images/JNet_513_4_outputz_plane.png
[JNet_513_4_reconst_depth]: /experiments/images/JNet_513_4_reconst_depth.png
[JNet_513_4_reconst_plane]: /experiments/images/JNet_513_4_reconst_plane.png
[JNet_513_microglia_0_heatmap_depth]: /experiments/images/JNet_513_microglia_0_heatmap_depth.png
[JNet_513_microglia_0_heatmap_plane]: /experiments/images/JNet_513_microglia_0_heatmap_plane.png
[JNet_513_microglia_0_original_depth]: /experiments/images/JNet_513_microglia_0_original_depth.png
[JNet_513_microglia_0_original_plane]: /experiments/images/JNet_513_microglia_0_original_plane.png
[JNet_513_microglia_0_outputx_depth]: /experiments/images/JNet_513_microglia_0_outputx_depth.png
[JNet_513_microglia_0_outputx_plane]: /experiments/images/JNet_513_microglia_0_outputx_plane.png
[JNet_513_microglia_0_outputz_depth]: /experiments/images/JNet_513_microglia_0_outputz_depth.png
[JNet_513_microglia_0_outputz_plane]: /experiments/images/JNet_513_microglia_0_outputz_plane.png
[JNet_513_microglia_0_reconst_depth]: /experiments/images/JNet_513_microglia_0_reconst_depth.png
[JNet_513_microglia_0_reconst_plane]: /experiments/images/JNet_513_microglia_0_reconst_plane.png
[JNet_513_microglia_1_heatmap_depth]: /experiments/images/JNet_513_microglia_1_heatmap_depth.png
[JNet_513_microglia_1_heatmap_plane]: /experiments/images/JNet_513_microglia_1_heatmap_plane.png
[JNet_513_microglia_1_original_depth]: /experiments/images/JNet_513_microglia_1_original_depth.png
[JNet_513_microglia_1_original_plane]: /experiments/images/JNet_513_microglia_1_original_plane.png
[JNet_513_microglia_1_outputx_depth]: /experiments/images/JNet_513_microglia_1_outputx_depth.png
[JNet_513_microglia_1_outputx_plane]: /experiments/images/JNet_513_microglia_1_outputx_plane.png
[JNet_513_microglia_1_outputz_depth]: /experiments/images/JNet_513_microglia_1_outputz_depth.png
[JNet_513_microglia_1_outputz_plane]: /experiments/images/JNet_513_microglia_1_outputz_plane.png
[JNet_513_microglia_1_reconst_depth]: /experiments/images/JNet_513_microglia_1_reconst_depth.png
[JNet_513_microglia_1_reconst_plane]: /experiments/images/JNet_513_microglia_1_reconst_plane.png
[JNet_513_microglia_2_heatmap_depth]: /experiments/images/JNet_513_microglia_2_heatmap_depth.png
[JNet_513_microglia_2_heatmap_plane]: /experiments/images/JNet_513_microglia_2_heatmap_plane.png
[JNet_513_microglia_2_original_depth]: /experiments/images/JNet_513_microglia_2_original_depth.png
[JNet_513_microglia_2_original_plane]: /experiments/images/JNet_513_microglia_2_original_plane.png
[JNet_513_microglia_2_outputx_depth]: /experiments/images/JNet_513_microglia_2_outputx_depth.png
[JNet_513_microglia_2_outputx_plane]: /experiments/images/JNet_513_microglia_2_outputx_plane.png
[JNet_513_microglia_2_outputz_depth]: /experiments/images/JNet_513_microglia_2_outputz_depth.png
[JNet_513_microglia_2_outputz_plane]: /experiments/images/JNet_513_microglia_2_outputz_plane.png
[JNet_513_microglia_2_reconst_depth]: /experiments/images/JNet_513_microglia_2_reconst_depth.png
[JNet_513_microglia_2_reconst_plane]: /experiments/images/JNet_513_microglia_2_reconst_plane.png
[JNet_513_microglia_3_heatmap_depth]: /experiments/images/JNet_513_microglia_3_heatmap_depth.png
[JNet_513_microglia_3_heatmap_plane]: /experiments/images/JNet_513_microglia_3_heatmap_plane.png
[JNet_513_microglia_3_original_depth]: /experiments/images/JNet_513_microglia_3_original_depth.png
[JNet_513_microglia_3_original_plane]: /experiments/images/JNet_513_microglia_3_original_plane.png
[JNet_513_microglia_3_outputx_depth]: /experiments/images/JNet_513_microglia_3_outputx_depth.png
[JNet_513_microglia_3_outputx_plane]: /experiments/images/JNet_513_microglia_3_outputx_plane.png
[JNet_513_microglia_3_outputz_depth]: /experiments/images/JNet_513_microglia_3_outputz_depth.png
[JNet_513_microglia_3_outputz_plane]: /experiments/images/JNet_513_microglia_3_outputz_plane.png
[JNet_513_microglia_3_reconst_depth]: /experiments/images/JNet_513_microglia_3_reconst_depth.png
[JNet_513_microglia_3_reconst_plane]: /experiments/images/JNet_513_microglia_3_reconst_plane.png
[JNet_513_microglia_4_heatmap_depth]: /experiments/images/JNet_513_microglia_4_heatmap_depth.png
[JNet_513_microglia_4_heatmap_plane]: /experiments/images/JNet_513_microglia_4_heatmap_plane.png
[JNet_513_microglia_4_original_depth]: /experiments/images/JNet_513_microglia_4_original_depth.png
[JNet_513_microglia_4_original_plane]: /experiments/images/JNet_513_microglia_4_original_plane.png
[JNet_513_microglia_4_outputx_depth]: /experiments/images/JNet_513_microglia_4_outputx_depth.png
[JNet_513_microglia_4_outputx_plane]: /experiments/images/JNet_513_microglia_4_outputx_plane.png
[JNet_513_microglia_4_outputz_depth]: /experiments/images/JNet_513_microglia_4_outputz_depth.png
[JNet_513_microglia_4_outputz_plane]: /experiments/images/JNet_513_microglia_4_outputz_plane.png
[JNet_513_microglia_4_reconst_depth]: /experiments/images/JNet_513_microglia_4_reconst_depth.png
[JNet_513_microglia_4_reconst_plane]: /experiments/images/JNet_513_microglia_4_reconst_plane.png
[JNet_513_psf_post]: /experiments/images/JNet_513_psf_post.png
[JNet_513_psf_pre]: /experiments/images/JNet_513_psf_pre.png
[finetuned]: /experiments/tmp/JNet_513_train.png
[pretrained_model]: /experiments/tmp/JNet_512_pretrain_train.png
