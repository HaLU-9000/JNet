



# JNet_472 Report
  
new data generation with more objects. axon deconv  
pretrained model : JNet_471_pretrain
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
|NA|1.0||
|wavelength|2.0|microns|
|M|25|magnification|
|ns|1.4|specimen refractive index (RI)|
|ng0|1.5|coverslip RI design value|
|ng|1.5|coverslip RI experimental value|
|ni0|1.33|immersion medium RI design value|
|ni|1.33|immersion medium RI experimental value|
|ti0|150|microns, working distance (immersion medium thickness) design value|
|tg0|170|microns, coverslip thickness design value|
|tg|170|microns, coverslip thickness experimental value|
|res_lateral|0.05|microns|
|res_axial|0.5|microns|
|pZ|0|microns, particle distance from coverslip|
|bet_z|30.0||
|bet_xy|3.0||
|sig_eps|0.01||
|background|0.01||
|scale|10||
|mid|20|num of NeurIPSF middle channel|
|loss_fn|nn.MSELoss()|loss func for NeurIPSF|
|lr|0.01|lr for pre-training NeurIPSF|
|num_iter_psf_pretrain|1000|epoch for pre-training of NeurIPSF|
|device|cuda||

## Datasets and other training details

### simulation_data_generation

|Parameter|Value|
| :--- | :--- |
|dataset_name|_var_num_realisticdataset|
|train_num|16|
|valid_num|4|
|image_size|[1200, 500, 500]|
|train_object_num_min|2000|
|train_object_num_max|18000|
|valid_object_num_min|6000|
|valid_object_num_max|10000|

### pretrain_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_var_num_realisticdata|
|labelname|_label|
|size|[1200, 500, 500]|
|cropsize|[240, 112, 112]|
|I|200|
|low|0|
|high|16|
|scale|10|
|mask|True|
|mask_size|[1, 10, 10]|
|mask_num|30|
|surround|False|
|surround_size|[32, 4, 4]|

### pretrain_val_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_var_num_realisticdata|
|labelname|_label|
|size|[1200, 500, 500]|
|cropsize|[240, 112, 112]|
|I|20|
|low|16|
|high|20|
|scale|10|
|mask|False|
|mask_size|[1, 10, 10]|
|mask_num|False|
|surround|False|
|surround_size|[32, 4, 4]|
|seed|907|

### train_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_20231208_tsuji_beads_stackreged|
|size|[310, 512, 512]|
|cropsize|[240, 112, 112]|
|I|200|
|scale|10|
|train|True|
|mask|True|
|mask_size|[1, 10, 10]|
|mask_num|10|
|surround|False|
|surround_size|[32, 4, 4]|

### val_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_20231208_tsuji_beads_stackreged|
|size|[310, 512, 512]|
|cropsize|[240, 112, 112]|
|I|20|
|scale|10|
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
|loss_fn|nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=params['device']))|
|path|model|
|savefig_path|train|
|partial|params['partial']|
|ewc|None|
|params|params|
|es_patience|10|
|reconstruct|False|
|is_instantblur|True|
|is_vibrate|True|
|loss_weight|1|
|qloss_weight|0|
|ploss_weight|0|

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
|ewc|ewc|
|params|params|
|es_patience|10|
|reconstruct|True|
|is_instantblur|False|
|is_vibrate|True|
|loss_weight|1|
|ewc_weight|1000000|
|qloss_weight|1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.018400708213448524, mean BCE: 0.0756513699889183
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_471_pretrain_0_original_plane]|![JNet_471_pretrain_0_output_plane]|![JNet_471_pretrain_0_label_plane]|
  
MSE: 0.021290073171257973, BCE: 0.08111866563558578  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_471_pretrain_0_original_depth]|![JNet_471_pretrain_0_output_depth]|![JNet_471_pretrain_0_label_depth]|
  
MSE: 0.021290073171257973, BCE: 0.08111866563558578  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_471_pretrain_1_original_plane]|![JNet_471_pretrain_1_output_plane]|![JNet_471_pretrain_1_label_plane]|
  
MSE: 0.01206216961145401, BCE: 0.056011002510786057  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_471_pretrain_1_original_depth]|![JNet_471_pretrain_1_output_depth]|![JNet_471_pretrain_1_label_depth]|
  
MSE: 0.01206216961145401, BCE: 0.056011002510786057  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_471_pretrain_2_original_plane]|![JNet_471_pretrain_2_output_plane]|![JNet_471_pretrain_2_label_plane]|
  
MSE: 0.015060506761074066, BCE: 0.06332483142614365  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_471_pretrain_2_original_depth]|![JNet_471_pretrain_2_output_depth]|![JNet_471_pretrain_2_label_depth]|
  
MSE: 0.015060506761074066, BCE: 0.06332483142614365  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_471_pretrain_3_original_plane]|![JNet_471_pretrain_3_output_plane]|![JNet_471_pretrain_3_label_plane]|
  
MSE: 0.023585274815559387, BCE: 0.09399653971195221  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_471_pretrain_3_original_depth]|![JNet_471_pretrain_3_output_depth]|![JNet_471_pretrain_3_label_depth]|
  
MSE: 0.023585274815559387, BCE: 0.09399653971195221  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_471_pretrain_4_original_plane]|![JNet_471_pretrain_4_output_plane]|![JNet_471_pretrain_4_label_plane]|
  
MSE: 0.020005518570542336, BCE: 0.08380581438541412  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_471_pretrain_4_original_depth]|![JNet_471_pretrain_4_output_depth]|![JNet_471_pretrain_4_label_depth]|
  
MSE: 0.020005518570542336, BCE: 0.08380581438541412  
  
mean MSE: 0.02020266465842724, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_472_0_original_plane]|![JNet_472_0_output_plane]|![JNet_472_0_label_plane]|
  
MSE: 0.01674821972846985, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_472_0_original_depth]|![JNet_472_0_output_depth]|![JNet_472_0_label_depth]|
  
MSE: 0.01674821972846985, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_472_1_original_plane]|![JNet_472_1_output_plane]|![JNet_472_1_label_plane]|
  
MSE: 0.016418805345892906, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_472_1_original_depth]|![JNet_472_1_output_depth]|![JNet_472_1_label_depth]|
  
MSE: 0.016418805345892906, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_472_2_original_plane]|![JNet_472_2_output_plane]|![JNet_472_2_label_plane]|
  
MSE: 0.025712212547659874, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_472_2_original_depth]|![JNet_472_2_output_depth]|![JNet_472_2_label_depth]|
  
MSE: 0.025712212547659874, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_472_3_original_plane]|![JNet_472_3_output_plane]|![JNet_472_3_label_plane]|
  
MSE: 0.025997135788202286, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_472_3_original_depth]|![JNet_472_3_output_depth]|![JNet_472_3_label_depth]|
  
MSE: 0.025997135788202286, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_472_4_original_plane]|![JNet_472_4_output_plane]|![JNet_472_4_label_plane]|
  
MSE: 0.016136951744556427, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_472_4_original_depth]|![JNet_472_4_output_depth]|![JNet_472_4_label_depth]|
  
MSE: 0.016136951744556427, BCE: nan  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi000_im000._original_depth]|![JNet_471_pretrain_beads_roi000_im000._output_depth]|![JNet_471_pretrain_beads_roi000_im000._reconst_depth]|![JNet_471_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 43.62290625000001, MSE: 0.003323987824842334, quantized loss: 0.00849800556898117  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi001_im004._original_depth]|![JNet_471_pretrain_beads_roi001_im004._output_depth]|![JNet_471_pretrain_beads_roi001_im004._reconst_depth]|![JNet_471_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 51.73881250000001, MSE: 0.003982119727879763, quantized loss: 0.011453572660684586  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi002_im005._original_depth]|![JNet_471_pretrain_beads_roi002_im005._output_depth]|![JNet_471_pretrain_beads_roi002_im005._reconst_depth]|![JNet_471_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 48.37288281250001, MSE: 0.0036510711070150137, quantized loss: 0.010663356631994247  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi003_im006._original_depth]|![JNet_471_pretrain_beads_roi003_im006._output_depth]|![JNet_471_pretrain_beads_roi003_im006._reconst_depth]|![JNet_471_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 49.27924218750001, MSE: 0.003619058756157756, quantized loss: 0.010769067332148552  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi004_im006._original_depth]|![JNet_471_pretrain_beads_roi004_im006._output_depth]|![JNet_471_pretrain_beads_roi004_im006._reconst_depth]|![JNet_471_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 49.83918750000001, MSE: 0.003652476705610752, quantized loss: 0.010737334378063679  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi005_im007._original_depth]|![JNet_471_pretrain_beads_roi005_im007._output_depth]|![JNet_471_pretrain_beads_roi005_im007._reconst_depth]|![JNet_471_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 49.37117187500001, MSE: 0.0035935889463871717, quantized loss: 0.010752133093774319  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi006_im008._original_depth]|![JNet_471_pretrain_beads_roi006_im008._output_depth]|![JNet_471_pretrain_beads_roi006_im008._reconst_depth]|![JNet_471_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 51.17719921875001, MSE: 0.003480897517874837, quantized loss: 0.010684103704988956  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi007_im009._original_depth]|![JNet_471_pretrain_beads_roi007_im009._output_depth]|![JNet_471_pretrain_beads_roi007_im009._reconst_depth]|![JNet_471_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 52.02167578125001, MSE: 0.0036067524924874306, quantized loss: 0.010869386605918407  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi008_im010._original_depth]|![JNet_471_pretrain_beads_roi008_im010._output_depth]|![JNet_471_pretrain_beads_roi008_im010._reconst_depth]|![JNet_471_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 50.797542968750015, MSE: 0.0037385544274002314, quantized loss: 0.01172823365777731  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi009_im011._original_depth]|![JNet_471_pretrain_beads_roi009_im011._output_depth]|![JNet_471_pretrain_beads_roi009_im011._reconst_depth]|![JNet_471_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 45.72723828125001, MSE: 0.003348768688738346, quantized loss: 0.010088255628943443  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi010_im012._original_depth]|![JNet_471_pretrain_beads_roi010_im012._output_depth]|![JNet_471_pretrain_beads_roi010_im012._reconst_depth]|![JNet_471_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 51.529785156250014, MSE: 0.003989122342318296, quantized loss: 0.011246761307120323  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi011_im013._original_depth]|![JNet_471_pretrain_beads_roi011_im013._output_depth]|![JNet_471_pretrain_beads_roi011_im013._reconst_depth]|![JNet_471_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 50.01821875000001, MSE: 0.0038788907695561647, quantized loss: 0.010911759920418262  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi012_im014._original_depth]|![JNet_471_pretrain_beads_roi012_im014._output_depth]|![JNet_471_pretrain_beads_roi012_im014._reconst_depth]|![JNet_471_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 42.68424609375001, MSE: 0.0035391219425946474, quantized loss: 0.007729124743491411  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi013_im015._original_depth]|![JNet_471_pretrain_beads_roi013_im015._output_depth]|![JNet_471_pretrain_beads_roi013_im015._reconst_depth]|![JNet_471_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 44.79173046875001, MSE: 0.0035557884257286787, quantized loss: 0.010228700935840607  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi014_im016._original_depth]|![JNet_471_pretrain_beads_roi014_im016._output_depth]|![JNet_471_pretrain_beads_roi014_im016._reconst_depth]|![JNet_471_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 48.92713671875001, MSE: 0.0035087503492832184, quantized loss: 0.01084829680621624  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi015_im017._original_depth]|![JNet_471_pretrain_beads_roi015_im017._output_depth]|![JNet_471_pretrain_beads_roi015_im017._reconst_depth]|![JNet_471_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 47.48164453125001, MSE: 0.003405594499781728, quantized loss: 0.010838128626346588  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi016_im018._original_depth]|![JNet_471_pretrain_beads_roi016_im018._output_depth]|![JNet_471_pretrain_beads_roi016_im018._reconst_depth]|![JNet_471_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 50.976148437500015, MSE: 0.00391437578946352, quantized loss: 0.011096409521996975  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi017_im018._original_depth]|![JNet_471_pretrain_beads_roi017_im018._output_depth]|![JNet_471_pretrain_beads_roi017_im018._reconst_depth]|![JNet_471_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 50.06598437500001, MSE: 0.003981958609074354, quantized loss: 0.011074190028011799  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi018_im022._original_depth]|![JNet_471_pretrain_beads_roi018_im022._output_depth]|![JNet_471_pretrain_beads_roi018_im022._reconst_depth]|![JNet_471_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 39.46751171875001, MSE: 0.003236649325117469, quantized loss: 0.006861796136945486  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi019_im023._original_depth]|![JNet_471_pretrain_beads_roi019_im023._output_depth]|![JNet_471_pretrain_beads_roi019_im023._reconst_depth]|![JNet_471_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 38.54611328125001, MSE: 0.003154727863147855, quantized loss: 0.0067982119508087635  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi020_im024._original_depth]|![JNet_471_pretrain_beads_roi020_im024._output_depth]|![JNet_471_pretrain_beads_roi020_im024._reconst_depth]|![JNet_471_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 44.76976562500001, MSE: 0.003679443383589387, quantized loss: 0.008153492584824562  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi021_im026._original_depth]|![JNet_471_pretrain_beads_roi021_im026._output_depth]|![JNet_471_pretrain_beads_roi021_im026._reconst_depth]|![JNet_471_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 44.87849218750001, MSE: 0.0034971332643181086, quantized loss: 0.008276582695543766  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi022_im027._original_depth]|![JNet_471_pretrain_beads_roi022_im027._output_depth]|![JNet_471_pretrain_beads_roi022_im027._reconst_depth]|![JNet_471_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 43.98499218750001, MSE: 0.0034000228624790907, quantized loss: 0.008239873684942722  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi023_im028._original_depth]|![JNet_471_pretrain_beads_roi023_im028._output_depth]|![JNet_471_pretrain_beads_roi023_im028._reconst_depth]|![JNet_471_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 47.91041015625001, MSE: 0.003412682330235839, quantized loss: 0.009296752512454987  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi024_im028._original_depth]|![JNet_471_pretrain_beads_roi024_im028._output_depth]|![JNet_471_pretrain_beads_roi024_im028._reconst_depth]|![JNet_471_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 46.39181640625001, MSE: 0.003609931096434593, quantized loss: 0.008721886202692986  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi025_im028._original_depth]|![JNet_471_pretrain_beads_roi025_im028._output_depth]|![JNet_471_pretrain_beads_roi025_im028._reconst_depth]|![JNet_471_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 46.39181640625001, MSE: 0.003609931096434593, quantized loss: 0.008721886202692986  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi026_im029._original_depth]|![JNet_471_pretrain_beads_roi026_im029._output_depth]|![JNet_471_pretrain_beads_roi026_im029._reconst_depth]|![JNet_471_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 45.41073828125001, MSE: 0.0036585575435310602, quantized loss: 0.008522904478013515  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi027_im029._original_depth]|![JNet_471_pretrain_beads_roi027_im029._output_depth]|![JNet_471_pretrain_beads_roi027_im029._reconst_depth]|![JNet_471_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 42.42775781250001, MSE: 0.0035712458193302155, quantized loss: 0.00793928001075983  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi028_im030._original_depth]|![JNet_471_pretrain_beads_roi028_im030._output_depth]|![JNet_471_pretrain_beads_roi028_im030._reconst_depth]|![JNet_471_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 41.80003125000001, MSE: 0.003383476985618472, quantized loss: 0.0075195650570094585  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_471_pretrain_beads_roi029_im030._original_depth]|![JNet_471_pretrain_beads_roi029_im030._output_depth]|![JNet_471_pretrain_beads_roi029_im030._reconst_depth]|![JNet_471_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 42.68830468750001, MSE: 0.0035130914766341448, quantized loss: 0.007897594012320042  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi000_im000._original_depth]|![JNet_472_beads_roi000_im000._output_depth]|![JNet_472_beads_roi000_im000._reconst_depth]|![JNet_472_beads_roi000_im000._heatmap_depth]|
  
volume: 12.961138671875004, MSE: 0.00011328465188853443, quantized loss: 1.607336162123829e-05  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi001_im004._original_depth]|![JNet_472_beads_roi001_im004._output_depth]|![JNet_472_beads_roi001_im004._reconst_depth]|![JNet_472_beads_roi001_im004._heatmap_depth]|
  
volume: 14.715768554687504, MSE: 0.0002393380127614364, quantized loss: 1.934640749823302e-05  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi002_im005._original_depth]|![JNet_472_beads_roi002_im005._output_depth]|![JNet_472_beads_roi002_im005._reconst_depth]|![JNet_472_beads_roi002_im005._heatmap_depth]|
  
volume: 13.458758789062504, MSE: 0.00018842467397917062, quantized loss: 1.8484415704733692e-05  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi003_im006._original_depth]|![JNet_472_beads_roi003_im006._output_depth]|![JNet_472_beads_roi003_im006._reconst_depth]|![JNet_472_beads_roi003_im006._heatmap_depth]|
  
volume: 13.344242187500003, MSE: 0.0002362066152272746, quantized loss: 1.7194037354784086e-05  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi004_im006._original_depth]|![JNet_472_beads_roi004_im006._output_depth]|![JNet_472_beads_roi004_im006._reconst_depth]|![JNet_472_beads_roi004_im006._heatmap_depth]|
  
volume: 13.721754882812503, MSE: 0.00025112737785093486, quantized loss: 1.689452983555384e-05  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi005_im007._original_depth]|![JNet_472_beads_roi005_im007._output_depth]|![JNet_472_beads_roi005_im007._reconst_depth]|![JNet_472_beads_roi005_im007._heatmap_depth]|
  
volume: 13.497715820312504, MSE: 0.00023512255575042218, quantized loss: 1.774000520526897e-05  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi006_im008._original_depth]|![JNet_472_beads_roi006_im008._output_depth]|![JNet_472_beads_roi006_im008._reconst_depth]|![JNet_472_beads_roi006_im008._heatmap_depth]|
  
volume: 13.718954101562503, MSE: 0.00027010939083993435, quantized loss: 1.8372174963587895e-05  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi007_im009._original_depth]|![JNet_472_beads_roi007_im009._output_depth]|![JNet_472_beads_roi007_im009._reconst_depth]|![JNet_472_beads_roi007_im009._heatmap_depth]|
  
volume: 13.749738281250004, MSE: 0.0003096640866715461, quantized loss: 1.8817105228663422e-05  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi008_im010._original_depth]|![JNet_472_beads_roi008_im010._output_depth]|![JNet_472_beads_roi008_im010._reconst_depth]|![JNet_472_beads_roi008_im010._heatmap_depth]|
  
volume: 14.152033203125004, MSE: 0.0002184684999519959, quantized loss: 1.722071465337649e-05  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi009_im011._original_depth]|![JNet_472_beads_roi009_im011._output_depth]|![JNet_472_beads_roi009_im011._reconst_depth]|![JNet_472_beads_roi009_im011._heatmap_depth]|
  
volume: 13.260802734375003, MSE: 0.00013606341963168234, quantized loss: 1.5684028767282143e-05  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi010_im012._original_depth]|![JNet_472_beads_roi010_im012._output_depth]|![JNet_472_beads_roi010_im012._reconst_depth]|![JNet_472_beads_roi010_im012._heatmap_depth]|
  
volume: 14.986811523437504, MSE: 0.00018290405569132417, quantized loss: 1.9320426872582175e-05  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi011_im013._original_depth]|![JNet_472_beads_roi011_im013._output_depth]|![JNet_472_beads_roi011_im013._reconst_depth]|![JNet_472_beads_roi011_im013._heatmap_depth]|
  
volume: 14.835778320312503, MSE: 0.00016101768414955586, quantized loss: 1.8012213331530802e-05  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi012_im014._original_depth]|![JNet_472_beads_roi012_im014._output_depth]|![JNet_472_beads_roi012_im014._reconst_depth]|![JNet_472_beads_roi012_im014._heatmap_depth]|
  
volume: 13.645989257812504, MSE: 0.00017281337932217866, quantized loss: 1.7123695215559565e-05  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi013_im015._original_depth]|![JNet_472_beads_roi013_im015._output_depth]|![JNet_472_beads_roi013_im015._reconst_depth]|![JNet_472_beads_roi013_im015._heatmap_depth]|
  
volume: 12.883236328125003, MSE: 0.0001808284578146413, quantized loss: 1.6228439562837593e-05  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi014_im016._original_depth]|![JNet_472_beads_roi014_im016._output_depth]|![JNet_472_beads_roi014_im016._reconst_depth]|![JNet_472_beads_roi014_im016._heatmap_depth]|
  
volume: 13.016940429687503, MSE: 0.00023447549028787762, quantized loss: 1.719435749691911e-05  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi015_im017._original_depth]|![JNet_472_beads_roi015_im017._output_depth]|![JNet_472_beads_roi015_im017._reconst_depth]|![JNet_472_beads_roi015_im017._heatmap_depth]|
  
volume: 13.002566406250002, MSE: 0.0002123791491612792, quantized loss: 1.671352220000699e-05  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi016_im018._original_depth]|![JNet_472_beads_roi016_im018._output_depth]|![JNet_472_beads_roi016_im018._reconst_depth]|![JNet_472_beads_roi016_im018._heatmap_depth]|
  
volume: 14.144511718750003, MSE: 0.0002829194418154657, quantized loss: 1.8598946553538553e-05  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi017_im018._original_depth]|![JNet_472_beads_roi017_im018._output_depth]|![JNet_472_beads_roi017_im018._reconst_depth]|![JNet_472_beads_roi017_im018._heatmap_depth]|
  
volume: 14.087072265625004, MSE: 0.0002719367330428213, quantized loss: 1.7060770915122703e-05  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi018_im022._original_depth]|![JNet_472_beads_roi018_im022._output_depth]|![JNet_472_beads_roi018_im022._reconst_depth]|![JNet_472_beads_roi018_im022._heatmap_depth]|
  
volume: 12.115543945312503, MSE: 0.00010607910371618345, quantized loss: 1.5515277482336387e-05  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi019_im023._original_depth]|![JNet_472_beads_roi019_im023._output_depth]|![JNet_472_beads_roi019_im023._reconst_depth]|![JNet_472_beads_roi019_im023._heatmap_depth]|
  
volume: 11.713786132812503, MSE: 0.00010636208025971428, quantized loss: 1.4778883269173093e-05  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi020_im024._original_depth]|![JNet_472_beads_roi020_im024._output_depth]|![JNet_472_beads_roi020_im024._reconst_depth]|![JNet_472_beads_roi020_im024._heatmap_depth]|
  
volume: 14.291639648437503, MSE: 0.00011123930016765371, quantized loss: 1.6849753592396155e-05  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi021_im026._original_depth]|![JNet_472_beads_roi021_im026._output_depth]|![JNet_472_beads_roi021_im026._reconst_depth]|![JNet_472_beads_roi021_im026._heatmap_depth]|
  
volume: 13.959142578125004, MSE: 0.00011240391177125275, quantized loss: 1.5602787243551575e-05  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi022_im027._original_depth]|![JNet_472_beads_roi022_im027._output_depth]|![JNet_472_beads_roi022_im027._reconst_depth]|![JNet_472_beads_roi022_im027._heatmap_depth]|
  
volume: 13.698161132812503, MSE: 0.00012684495595749468, quantized loss: 1.675706334935967e-05  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi023_im028._original_depth]|![JNet_472_beads_roi023_im028._output_depth]|![JNet_472_beads_roi023_im028._reconst_depth]|![JNet_472_beads_roi023_im028._heatmap_depth]|
  
volume: 14.709957031250003, MSE: 0.00010889858822338283, quantized loss: 1.8468528651283123e-05  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi024_im028._original_depth]|![JNet_472_beads_roi024_im028._output_depth]|![JNet_472_beads_roi024_im028._reconst_depth]|![JNet_472_beads_roi024_im028._heatmap_depth]|
  
volume: 14.278257812500003, MSE: 0.00011048767919419333, quantized loss: 1.9206456272513606e-05  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi025_im028._original_depth]|![JNet_472_beads_roi025_im028._output_depth]|![JNet_472_beads_roi025_im028._reconst_depth]|![JNet_472_beads_roi025_im028._heatmap_depth]|
  
volume: 14.278257812500003, MSE: 0.00011048767919419333, quantized loss: 1.9206456272513606e-05  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi026_im029._original_depth]|![JNet_472_beads_roi026_im029._output_depth]|![JNet_472_beads_roi026_im029._reconst_depth]|![JNet_472_beads_roi026_im029._heatmap_depth]|
  
volume: 14.425152343750003, MSE: 0.0001354128326056525, quantized loss: 1.5974648704286665e-05  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi027_im029._original_depth]|![JNet_472_beads_roi027_im029._output_depth]|![JNet_472_beads_roi027_im029._reconst_depth]|![JNet_472_beads_roi027_im029._heatmap_depth]|
  
volume: 13.203375000000003, MSE: 0.0001325482880929485, quantized loss: 1.581287506269291e-05  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi028_im030._original_depth]|![JNet_472_beads_roi028_im030._output_depth]|![JNet_472_beads_roi028_im030._reconst_depth]|![JNet_472_beads_roi028_im030._heatmap_depth]|
  
volume: 12.923627929687504, MSE: 0.00011060295946663246, quantized loss: 1.638098547118716e-05  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_472_beads_roi029_im030._original_depth]|![JNet_472_beads_roi029_im030._output_depth]|![JNet_472_beads_roi029_im030._reconst_depth]|![JNet_472_beads_roi029_im030._heatmap_depth]|
  
volume: 13.444309570312504, MSE: 0.00011906217696378008, quantized loss: 1.8082031601807103e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_472_psf_pre]|![JNet_472_psf_post]|

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
  (post): ModuleList(  
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
  (post0): JNetBlockN(  
    (conv): Conv3d(16, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
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
  )  
  (upsample): JNetUpsample(  
    (upsample): Upsample(scale_factor=(10.0, 1.0, 1.0), mode='trilinear')  
  )  
  (vq): VectorQuantizer()  
)  
```  
  



[JNet_471_pretrain_0_label_depth]: /experiments/images/JNet_471_pretrain_0_label_depth.png
[JNet_471_pretrain_0_label_plane]: /experiments/images/JNet_471_pretrain_0_label_plane.png
[JNet_471_pretrain_0_original_depth]: /experiments/images/JNet_471_pretrain_0_original_depth.png
[JNet_471_pretrain_0_original_plane]: /experiments/images/JNet_471_pretrain_0_original_plane.png
[JNet_471_pretrain_0_output_depth]: /experiments/images/JNet_471_pretrain_0_output_depth.png
[JNet_471_pretrain_0_output_plane]: /experiments/images/JNet_471_pretrain_0_output_plane.png
[JNet_471_pretrain_1_label_depth]: /experiments/images/JNet_471_pretrain_1_label_depth.png
[JNet_471_pretrain_1_label_plane]: /experiments/images/JNet_471_pretrain_1_label_plane.png
[JNet_471_pretrain_1_original_depth]: /experiments/images/JNet_471_pretrain_1_original_depth.png
[JNet_471_pretrain_1_original_plane]: /experiments/images/JNet_471_pretrain_1_original_plane.png
[JNet_471_pretrain_1_output_depth]: /experiments/images/JNet_471_pretrain_1_output_depth.png
[JNet_471_pretrain_1_output_plane]: /experiments/images/JNet_471_pretrain_1_output_plane.png
[JNet_471_pretrain_2_label_depth]: /experiments/images/JNet_471_pretrain_2_label_depth.png
[JNet_471_pretrain_2_label_plane]: /experiments/images/JNet_471_pretrain_2_label_plane.png
[JNet_471_pretrain_2_original_depth]: /experiments/images/JNet_471_pretrain_2_original_depth.png
[JNet_471_pretrain_2_original_plane]: /experiments/images/JNet_471_pretrain_2_original_plane.png
[JNet_471_pretrain_2_output_depth]: /experiments/images/JNet_471_pretrain_2_output_depth.png
[JNet_471_pretrain_2_output_plane]: /experiments/images/JNet_471_pretrain_2_output_plane.png
[JNet_471_pretrain_3_label_depth]: /experiments/images/JNet_471_pretrain_3_label_depth.png
[JNet_471_pretrain_3_label_plane]: /experiments/images/JNet_471_pretrain_3_label_plane.png
[JNet_471_pretrain_3_original_depth]: /experiments/images/JNet_471_pretrain_3_original_depth.png
[JNet_471_pretrain_3_original_plane]: /experiments/images/JNet_471_pretrain_3_original_plane.png
[JNet_471_pretrain_3_output_depth]: /experiments/images/JNet_471_pretrain_3_output_depth.png
[JNet_471_pretrain_3_output_plane]: /experiments/images/JNet_471_pretrain_3_output_plane.png
[JNet_471_pretrain_4_label_depth]: /experiments/images/JNet_471_pretrain_4_label_depth.png
[JNet_471_pretrain_4_label_plane]: /experiments/images/JNet_471_pretrain_4_label_plane.png
[JNet_471_pretrain_4_original_depth]: /experiments/images/JNet_471_pretrain_4_original_depth.png
[JNet_471_pretrain_4_original_plane]: /experiments/images/JNet_471_pretrain_4_original_plane.png
[JNet_471_pretrain_4_output_depth]: /experiments/images/JNet_471_pretrain_4_output_depth.png
[JNet_471_pretrain_4_output_plane]: /experiments/images/JNet_471_pretrain_4_output_plane.png
[JNet_471_pretrain_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi000_im000._heatmap_depth.png
[JNet_471_pretrain_beads_roi000_im000._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi000_im000._original_depth.png
[JNet_471_pretrain_beads_roi000_im000._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi000_im000._output_depth.png
[JNet_471_pretrain_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi000_im000._reconst_depth.png
[JNet_471_pretrain_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi001_im004._heatmap_depth.png
[JNet_471_pretrain_beads_roi001_im004._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi001_im004._original_depth.png
[JNet_471_pretrain_beads_roi001_im004._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi001_im004._output_depth.png
[JNet_471_pretrain_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi001_im004._reconst_depth.png
[JNet_471_pretrain_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi002_im005._heatmap_depth.png
[JNet_471_pretrain_beads_roi002_im005._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi002_im005._original_depth.png
[JNet_471_pretrain_beads_roi002_im005._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi002_im005._output_depth.png
[JNet_471_pretrain_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi002_im005._reconst_depth.png
[JNet_471_pretrain_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi003_im006._heatmap_depth.png
[JNet_471_pretrain_beads_roi003_im006._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi003_im006._original_depth.png
[JNet_471_pretrain_beads_roi003_im006._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi003_im006._output_depth.png
[JNet_471_pretrain_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi003_im006._reconst_depth.png
[JNet_471_pretrain_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi004_im006._heatmap_depth.png
[JNet_471_pretrain_beads_roi004_im006._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi004_im006._original_depth.png
[JNet_471_pretrain_beads_roi004_im006._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi004_im006._output_depth.png
[JNet_471_pretrain_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi004_im006._reconst_depth.png
[JNet_471_pretrain_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi005_im007._heatmap_depth.png
[JNet_471_pretrain_beads_roi005_im007._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi005_im007._original_depth.png
[JNet_471_pretrain_beads_roi005_im007._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi005_im007._output_depth.png
[JNet_471_pretrain_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi005_im007._reconst_depth.png
[JNet_471_pretrain_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi006_im008._heatmap_depth.png
[JNet_471_pretrain_beads_roi006_im008._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi006_im008._original_depth.png
[JNet_471_pretrain_beads_roi006_im008._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi006_im008._output_depth.png
[JNet_471_pretrain_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi006_im008._reconst_depth.png
[JNet_471_pretrain_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi007_im009._heatmap_depth.png
[JNet_471_pretrain_beads_roi007_im009._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi007_im009._original_depth.png
[JNet_471_pretrain_beads_roi007_im009._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi007_im009._output_depth.png
[JNet_471_pretrain_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi007_im009._reconst_depth.png
[JNet_471_pretrain_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi008_im010._heatmap_depth.png
[JNet_471_pretrain_beads_roi008_im010._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi008_im010._original_depth.png
[JNet_471_pretrain_beads_roi008_im010._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi008_im010._output_depth.png
[JNet_471_pretrain_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi008_im010._reconst_depth.png
[JNet_471_pretrain_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi009_im011._heatmap_depth.png
[JNet_471_pretrain_beads_roi009_im011._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi009_im011._original_depth.png
[JNet_471_pretrain_beads_roi009_im011._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi009_im011._output_depth.png
[JNet_471_pretrain_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi009_im011._reconst_depth.png
[JNet_471_pretrain_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi010_im012._heatmap_depth.png
[JNet_471_pretrain_beads_roi010_im012._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi010_im012._original_depth.png
[JNet_471_pretrain_beads_roi010_im012._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi010_im012._output_depth.png
[JNet_471_pretrain_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi010_im012._reconst_depth.png
[JNet_471_pretrain_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi011_im013._heatmap_depth.png
[JNet_471_pretrain_beads_roi011_im013._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi011_im013._original_depth.png
[JNet_471_pretrain_beads_roi011_im013._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi011_im013._output_depth.png
[JNet_471_pretrain_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi011_im013._reconst_depth.png
[JNet_471_pretrain_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi012_im014._heatmap_depth.png
[JNet_471_pretrain_beads_roi012_im014._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi012_im014._original_depth.png
[JNet_471_pretrain_beads_roi012_im014._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi012_im014._output_depth.png
[JNet_471_pretrain_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi012_im014._reconst_depth.png
[JNet_471_pretrain_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi013_im015._heatmap_depth.png
[JNet_471_pretrain_beads_roi013_im015._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi013_im015._original_depth.png
[JNet_471_pretrain_beads_roi013_im015._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi013_im015._output_depth.png
[JNet_471_pretrain_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi013_im015._reconst_depth.png
[JNet_471_pretrain_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi014_im016._heatmap_depth.png
[JNet_471_pretrain_beads_roi014_im016._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi014_im016._original_depth.png
[JNet_471_pretrain_beads_roi014_im016._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi014_im016._output_depth.png
[JNet_471_pretrain_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi014_im016._reconst_depth.png
[JNet_471_pretrain_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi015_im017._heatmap_depth.png
[JNet_471_pretrain_beads_roi015_im017._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi015_im017._original_depth.png
[JNet_471_pretrain_beads_roi015_im017._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi015_im017._output_depth.png
[JNet_471_pretrain_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi015_im017._reconst_depth.png
[JNet_471_pretrain_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi016_im018._heatmap_depth.png
[JNet_471_pretrain_beads_roi016_im018._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi016_im018._original_depth.png
[JNet_471_pretrain_beads_roi016_im018._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi016_im018._output_depth.png
[JNet_471_pretrain_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi016_im018._reconst_depth.png
[JNet_471_pretrain_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi017_im018._heatmap_depth.png
[JNet_471_pretrain_beads_roi017_im018._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi017_im018._original_depth.png
[JNet_471_pretrain_beads_roi017_im018._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi017_im018._output_depth.png
[JNet_471_pretrain_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi017_im018._reconst_depth.png
[JNet_471_pretrain_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi018_im022._heatmap_depth.png
[JNet_471_pretrain_beads_roi018_im022._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi018_im022._original_depth.png
[JNet_471_pretrain_beads_roi018_im022._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi018_im022._output_depth.png
[JNet_471_pretrain_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi018_im022._reconst_depth.png
[JNet_471_pretrain_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi019_im023._heatmap_depth.png
[JNet_471_pretrain_beads_roi019_im023._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi019_im023._original_depth.png
[JNet_471_pretrain_beads_roi019_im023._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi019_im023._output_depth.png
[JNet_471_pretrain_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi019_im023._reconst_depth.png
[JNet_471_pretrain_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi020_im024._heatmap_depth.png
[JNet_471_pretrain_beads_roi020_im024._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi020_im024._original_depth.png
[JNet_471_pretrain_beads_roi020_im024._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi020_im024._output_depth.png
[JNet_471_pretrain_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi020_im024._reconst_depth.png
[JNet_471_pretrain_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi021_im026._heatmap_depth.png
[JNet_471_pretrain_beads_roi021_im026._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi021_im026._original_depth.png
[JNet_471_pretrain_beads_roi021_im026._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi021_im026._output_depth.png
[JNet_471_pretrain_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi021_im026._reconst_depth.png
[JNet_471_pretrain_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi022_im027._heatmap_depth.png
[JNet_471_pretrain_beads_roi022_im027._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi022_im027._original_depth.png
[JNet_471_pretrain_beads_roi022_im027._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi022_im027._output_depth.png
[JNet_471_pretrain_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi022_im027._reconst_depth.png
[JNet_471_pretrain_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi023_im028._heatmap_depth.png
[JNet_471_pretrain_beads_roi023_im028._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi023_im028._original_depth.png
[JNet_471_pretrain_beads_roi023_im028._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi023_im028._output_depth.png
[JNet_471_pretrain_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi023_im028._reconst_depth.png
[JNet_471_pretrain_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi024_im028._heatmap_depth.png
[JNet_471_pretrain_beads_roi024_im028._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi024_im028._original_depth.png
[JNet_471_pretrain_beads_roi024_im028._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi024_im028._output_depth.png
[JNet_471_pretrain_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi024_im028._reconst_depth.png
[JNet_471_pretrain_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi025_im028._heatmap_depth.png
[JNet_471_pretrain_beads_roi025_im028._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi025_im028._original_depth.png
[JNet_471_pretrain_beads_roi025_im028._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi025_im028._output_depth.png
[JNet_471_pretrain_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi025_im028._reconst_depth.png
[JNet_471_pretrain_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi026_im029._heatmap_depth.png
[JNet_471_pretrain_beads_roi026_im029._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi026_im029._original_depth.png
[JNet_471_pretrain_beads_roi026_im029._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi026_im029._output_depth.png
[JNet_471_pretrain_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi026_im029._reconst_depth.png
[JNet_471_pretrain_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi027_im029._heatmap_depth.png
[JNet_471_pretrain_beads_roi027_im029._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi027_im029._original_depth.png
[JNet_471_pretrain_beads_roi027_im029._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi027_im029._output_depth.png
[JNet_471_pretrain_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi027_im029._reconst_depth.png
[JNet_471_pretrain_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi028_im030._heatmap_depth.png
[JNet_471_pretrain_beads_roi028_im030._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi028_im030._original_depth.png
[JNet_471_pretrain_beads_roi028_im030._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi028_im030._output_depth.png
[JNet_471_pretrain_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi028_im030._reconst_depth.png
[JNet_471_pretrain_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_471_pretrain_beads_roi029_im030._heatmap_depth.png
[JNet_471_pretrain_beads_roi029_im030._original_depth]: /experiments/images/JNet_471_pretrain_beads_roi029_im030._original_depth.png
[JNet_471_pretrain_beads_roi029_im030._output_depth]: /experiments/images/JNet_471_pretrain_beads_roi029_im030._output_depth.png
[JNet_471_pretrain_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_471_pretrain_beads_roi029_im030._reconst_depth.png
[JNet_472_0_label_depth]: /experiments/images/JNet_472_0_label_depth.png
[JNet_472_0_label_plane]: /experiments/images/JNet_472_0_label_plane.png
[JNet_472_0_original_depth]: /experiments/images/JNet_472_0_original_depth.png
[JNet_472_0_original_plane]: /experiments/images/JNet_472_0_original_plane.png
[JNet_472_0_output_depth]: /experiments/images/JNet_472_0_output_depth.png
[JNet_472_0_output_plane]: /experiments/images/JNet_472_0_output_plane.png
[JNet_472_1_label_depth]: /experiments/images/JNet_472_1_label_depth.png
[JNet_472_1_label_plane]: /experiments/images/JNet_472_1_label_plane.png
[JNet_472_1_original_depth]: /experiments/images/JNet_472_1_original_depth.png
[JNet_472_1_original_plane]: /experiments/images/JNet_472_1_original_plane.png
[JNet_472_1_output_depth]: /experiments/images/JNet_472_1_output_depth.png
[JNet_472_1_output_plane]: /experiments/images/JNet_472_1_output_plane.png
[JNet_472_2_label_depth]: /experiments/images/JNet_472_2_label_depth.png
[JNet_472_2_label_plane]: /experiments/images/JNet_472_2_label_plane.png
[JNet_472_2_original_depth]: /experiments/images/JNet_472_2_original_depth.png
[JNet_472_2_original_plane]: /experiments/images/JNet_472_2_original_plane.png
[JNet_472_2_output_depth]: /experiments/images/JNet_472_2_output_depth.png
[JNet_472_2_output_plane]: /experiments/images/JNet_472_2_output_plane.png
[JNet_472_3_label_depth]: /experiments/images/JNet_472_3_label_depth.png
[JNet_472_3_label_plane]: /experiments/images/JNet_472_3_label_plane.png
[JNet_472_3_original_depth]: /experiments/images/JNet_472_3_original_depth.png
[JNet_472_3_original_plane]: /experiments/images/JNet_472_3_original_plane.png
[JNet_472_3_output_depth]: /experiments/images/JNet_472_3_output_depth.png
[JNet_472_3_output_plane]: /experiments/images/JNet_472_3_output_plane.png
[JNet_472_4_label_depth]: /experiments/images/JNet_472_4_label_depth.png
[JNet_472_4_label_plane]: /experiments/images/JNet_472_4_label_plane.png
[JNet_472_4_original_depth]: /experiments/images/JNet_472_4_original_depth.png
[JNet_472_4_original_plane]: /experiments/images/JNet_472_4_original_plane.png
[JNet_472_4_output_depth]: /experiments/images/JNet_472_4_output_depth.png
[JNet_472_4_output_plane]: /experiments/images/JNet_472_4_output_plane.png
[JNet_472_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_472_beads_roi000_im000._heatmap_depth.png
[JNet_472_beads_roi000_im000._original_depth]: /experiments/images/JNet_472_beads_roi000_im000._original_depth.png
[JNet_472_beads_roi000_im000._output_depth]: /experiments/images/JNet_472_beads_roi000_im000._output_depth.png
[JNet_472_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_472_beads_roi000_im000._reconst_depth.png
[JNet_472_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_472_beads_roi001_im004._heatmap_depth.png
[JNet_472_beads_roi001_im004._original_depth]: /experiments/images/JNet_472_beads_roi001_im004._original_depth.png
[JNet_472_beads_roi001_im004._output_depth]: /experiments/images/JNet_472_beads_roi001_im004._output_depth.png
[JNet_472_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_472_beads_roi001_im004._reconst_depth.png
[JNet_472_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_472_beads_roi002_im005._heatmap_depth.png
[JNet_472_beads_roi002_im005._original_depth]: /experiments/images/JNet_472_beads_roi002_im005._original_depth.png
[JNet_472_beads_roi002_im005._output_depth]: /experiments/images/JNet_472_beads_roi002_im005._output_depth.png
[JNet_472_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_472_beads_roi002_im005._reconst_depth.png
[JNet_472_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_472_beads_roi003_im006._heatmap_depth.png
[JNet_472_beads_roi003_im006._original_depth]: /experiments/images/JNet_472_beads_roi003_im006._original_depth.png
[JNet_472_beads_roi003_im006._output_depth]: /experiments/images/JNet_472_beads_roi003_im006._output_depth.png
[JNet_472_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_472_beads_roi003_im006._reconst_depth.png
[JNet_472_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_472_beads_roi004_im006._heatmap_depth.png
[JNet_472_beads_roi004_im006._original_depth]: /experiments/images/JNet_472_beads_roi004_im006._original_depth.png
[JNet_472_beads_roi004_im006._output_depth]: /experiments/images/JNet_472_beads_roi004_im006._output_depth.png
[JNet_472_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_472_beads_roi004_im006._reconst_depth.png
[JNet_472_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_472_beads_roi005_im007._heatmap_depth.png
[JNet_472_beads_roi005_im007._original_depth]: /experiments/images/JNet_472_beads_roi005_im007._original_depth.png
[JNet_472_beads_roi005_im007._output_depth]: /experiments/images/JNet_472_beads_roi005_im007._output_depth.png
[JNet_472_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_472_beads_roi005_im007._reconst_depth.png
[JNet_472_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_472_beads_roi006_im008._heatmap_depth.png
[JNet_472_beads_roi006_im008._original_depth]: /experiments/images/JNet_472_beads_roi006_im008._original_depth.png
[JNet_472_beads_roi006_im008._output_depth]: /experiments/images/JNet_472_beads_roi006_im008._output_depth.png
[JNet_472_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_472_beads_roi006_im008._reconst_depth.png
[JNet_472_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_472_beads_roi007_im009._heatmap_depth.png
[JNet_472_beads_roi007_im009._original_depth]: /experiments/images/JNet_472_beads_roi007_im009._original_depth.png
[JNet_472_beads_roi007_im009._output_depth]: /experiments/images/JNet_472_beads_roi007_im009._output_depth.png
[JNet_472_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_472_beads_roi007_im009._reconst_depth.png
[JNet_472_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_472_beads_roi008_im010._heatmap_depth.png
[JNet_472_beads_roi008_im010._original_depth]: /experiments/images/JNet_472_beads_roi008_im010._original_depth.png
[JNet_472_beads_roi008_im010._output_depth]: /experiments/images/JNet_472_beads_roi008_im010._output_depth.png
[JNet_472_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_472_beads_roi008_im010._reconst_depth.png
[JNet_472_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_472_beads_roi009_im011._heatmap_depth.png
[JNet_472_beads_roi009_im011._original_depth]: /experiments/images/JNet_472_beads_roi009_im011._original_depth.png
[JNet_472_beads_roi009_im011._output_depth]: /experiments/images/JNet_472_beads_roi009_im011._output_depth.png
[JNet_472_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_472_beads_roi009_im011._reconst_depth.png
[JNet_472_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_472_beads_roi010_im012._heatmap_depth.png
[JNet_472_beads_roi010_im012._original_depth]: /experiments/images/JNet_472_beads_roi010_im012._original_depth.png
[JNet_472_beads_roi010_im012._output_depth]: /experiments/images/JNet_472_beads_roi010_im012._output_depth.png
[JNet_472_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_472_beads_roi010_im012._reconst_depth.png
[JNet_472_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_472_beads_roi011_im013._heatmap_depth.png
[JNet_472_beads_roi011_im013._original_depth]: /experiments/images/JNet_472_beads_roi011_im013._original_depth.png
[JNet_472_beads_roi011_im013._output_depth]: /experiments/images/JNet_472_beads_roi011_im013._output_depth.png
[JNet_472_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_472_beads_roi011_im013._reconst_depth.png
[JNet_472_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_472_beads_roi012_im014._heatmap_depth.png
[JNet_472_beads_roi012_im014._original_depth]: /experiments/images/JNet_472_beads_roi012_im014._original_depth.png
[JNet_472_beads_roi012_im014._output_depth]: /experiments/images/JNet_472_beads_roi012_im014._output_depth.png
[JNet_472_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_472_beads_roi012_im014._reconst_depth.png
[JNet_472_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_472_beads_roi013_im015._heatmap_depth.png
[JNet_472_beads_roi013_im015._original_depth]: /experiments/images/JNet_472_beads_roi013_im015._original_depth.png
[JNet_472_beads_roi013_im015._output_depth]: /experiments/images/JNet_472_beads_roi013_im015._output_depth.png
[JNet_472_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_472_beads_roi013_im015._reconst_depth.png
[JNet_472_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_472_beads_roi014_im016._heatmap_depth.png
[JNet_472_beads_roi014_im016._original_depth]: /experiments/images/JNet_472_beads_roi014_im016._original_depth.png
[JNet_472_beads_roi014_im016._output_depth]: /experiments/images/JNet_472_beads_roi014_im016._output_depth.png
[JNet_472_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_472_beads_roi014_im016._reconst_depth.png
[JNet_472_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_472_beads_roi015_im017._heatmap_depth.png
[JNet_472_beads_roi015_im017._original_depth]: /experiments/images/JNet_472_beads_roi015_im017._original_depth.png
[JNet_472_beads_roi015_im017._output_depth]: /experiments/images/JNet_472_beads_roi015_im017._output_depth.png
[JNet_472_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_472_beads_roi015_im017._reconst_depth.png
[JNet_472_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_472_beads_roi016_im018._heatmap_depth.png
[JNet_472_beads_roi016_im018._original_depth]: /experiments/images/JNet_472_beads_roi016_im018._original_depth.png
[JNet_472_beads_roi016_im018._output_depth]: /experiments/images/JNet_472_beads_roi016_im018._output_depth.png
[JNet_472_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_472_beads_roi016_im018._reconst_depth.png
[JNet_472_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_472_beads_roi017_im018._heatmap_depth.png
[JNet_472_beads_roi017_im018._original_depth]: /experiments/images/JNet_472_beads_roi017_im018._original_depth.png
[JNet_472_beads_roi017_im018._output_depth]: /experiments/images/JNet_472_beads_roi017_im018._output_depth.png
[JNet_472_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_472_beads_roi017_im018._reconst_depth.png
[JNet_472_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_472_beads_roi018_im022._heatmap_depth.png
[JNet_472_beads_roi018_im022._original_depth]: /experiments/images/JNet_472_beads_roi018_im022._original_depth.png
[JNet_472_beads_roi018_im022._output_depth]: /experiments/images/JNet_472_beads_roi018_im022._output_depth.png
[JNet_472_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_472_beads_roi018_im022._reconst_depth.png
[JNet_472_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_472_beads_roi019_im023._heatmap_depth.png
[JNet_472_beads_roi019_im023._original_depth]: /experiments/images/JNet_472_beads_roi019_im023._original_depth.png
[JNet_472_beads_roi019_im023._output_depth]: /experiments/images/JNet_472_beads_roi019_im023._output_depth.png
[JNet_472_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_472_beads_roi019_im023._reconst_depth.png
[JNet_472_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_472_beads_roi020_im024._heatmap_depth.png
[JNet_472_beads_roi020_im024._original_depth]: /experiments/images/JNet_472_beads_roi020_im024._original_depth.png
[JNet_472_beads_roi020_im024._output_depth]: /experiments/images/JNet_472_beads_roi020_im024._output_depth.png
[JNet_472_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_472_beads_roi020_im024._reconst_depth.png
[JNet_472_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_472_beads_roi021_im026._heatmap_depth.png
[JNet_472_beads_roi021_im026._original_depth]: /experiments/images/JNet_472_beads_roi021_im026._original_depth.png
[JNet_472_beads_roi021_im026._output_depth]: /experiments/images/JNet_472_beads_roi021_im026._output_depth.png
[JNet_472_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_472_beads_roi021_im026._reconst_depth.png
[JNet_472_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_472_beads_roi022_im027._heatmap_depth.png
[JNet_472_beads_roi022_im027._original_depth]: /experiments/images/JNet_472_beads_roi022_im027._original_depth.png
[JNet_472_beads_roi022_im027._output_depth]: /experiments/images/JNet_472_beads_roi022_im027._output_depth.png
[JNet_472_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_472_beads_roi022_im027._reconst_depth.png
[JNet_472_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_472_beads_roi023_im028._heatmap_depth.png
[JNet_472_beads_roi023_im028._original_depth]: /experiments/images/JNet_472_beads_roi023_im028._original_depth.png
[JNet_472_beads_roi023_im028._output_depth]: /experiments/images/JNet_472_beads_roi023_im028._output_depth.png
[JNet_472_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_472_beads_roi023_im028._reconst_depth.png
[JNet_472_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_472_beads_roi024_im028._heatmap_depth.png
[JNet_472_beads_roi024_im028._original_depth]: /experiments/images/JNet_472_beads_roi024_im028._original_depth.png
[JNet_472_beads_roi024_im028._output_depth]: /experiments/images/JNet_472_beads_roi024_im028._output_depth.png
[JNet_472_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_472_beads_roi024_im028._reconst_depth.png
[JNet_472_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_472_beads_roi025_im028._heatmap_depth.png
[JNet_472_beads_roi025_im028._original_depth]: /experiments/images/JNet_472_beads_roi025_im028._original_depth.png
[JNet_472_beads_roi025_im028._output_depth]: /experiments/images/JNet_472_beads_roi025_im028._output_depth.png
[JNet_472_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_472_beads_roi025_im028._reconst_depth.png
[JNet_472_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_472_beads_roi026_im029._heatmap_depth.png
[JNet_472_beads_roi026_im029._original_depth]: /experiments/images/JNet_472_beads_roi026_im029._original_depth.png
[JNet_472_beads_roi026_im029._output_depth]: /experiments/images/JNet_472_beads_roi026_im029._output_depth.png
[JNet_472_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_472_beads_roi026_im029._reconst_depth.png
[JNet_472_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_472_beads_roi027_im029._heatmap_depth.png
[JNet_472_beads_roi027_im029._original_depth]: /experiments/images/JNet_472_beads_roi027_im029._original_depth.png
[JNet_472_beads_roi027_im029._output_depth]: /experiments/images/JNet_472_beads_roi027_im029._output_depth.png
[JNet_472_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_472_beads_roi027_im029._reconst_depth.png
[JNet_472_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_472_beads_roi028_im030._heatmap_depth.png
[JNet_472_beads_roi028_im030._original_depth]: /experiments/images/JNet_472_beads_roi028_im030._original_depth.png
[JNet_472_beads_roi028_im030._output_depth]: /experiments/images/JNet_472_beads_roi028_im030._output_depth.png
[JNet_472_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_472_beads_roi028_im030._reconst_depth.png
[JNet_472_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_472_beads_roi029_im030._heatmap_depth.png
[JNet_472_beads_roi029_im030._original_depth]: /experiments/images/JNet_472_beads_roi029_im030._original_depth.png
[JNet_472_beads_roi029_im030._output_depth]: /experiments/images/JNet_472_beads_roi029_im030._output_depth.png
[JNet_472_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_472_beads_roi029_im030._reconst_depth.png
[JNet_472_psf_post]: /experiments/images/JNet_472_psf_post.png
[JNet_472_psf_pre]: /experiments/images/JNet_472_psf_pre.png
[finetuned]: /experiments/tmp/JNet_472_train.png
[pretrained_model]: /experiments/tmp/JNet_471_pretrain_train.png
