



# JNet_586 Report
  
psf loss 0.1  
pretrained model : JNet_584_pretrain
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
|NA|0.7||
|wavelength|0.9|microns|
|M|20|magnification|
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
|bet_z|5.0||
|bet_xy|10.0||
|poisson_weight|0.001||
|sig_eps|0.001||
|background|0.01||
|scale|10||
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
|scale|10|
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
|seed|723|

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
|is_vibrate|False|
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
|adjust_luminance|True|
|zloss_weight|1|
|ewc_weight|1|
|qloss_weight|1.0|
|ploss_weight|0.1|
|mrfloss_order|1|
|mrfloss_dilation|1|
|mrfloss_weights|{'l_00': 0, 'l_01': 0, 'l_10': 0, 'l_11': 0}|

## Results

### Pretraining
  
Segmentation: mean MSE: 0.00872753094881773, mean BCE: 0.03432992845773697  
Luminance Estimation: mean MSE: 0.9808942079544067, mean BCE: inf
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_0_original_plane]|![JNet_584_pretrain_0_novibrate_plane]|![JNet_584_pretrain_0_aligned_plane]|![JNet_584_pretrain_0_outputx_plane]|![JNet_584_pretrain_0_labelx_plane]|![JNet_584_pretrain_0_outputz_plane]|![JNet_584_pretrain_0_labelz_plane]|
  
MSEx: 0.008677849546074867, BCEx: 0.03387918695807457  
MSEz: 0.9850048422813416, BCEz: 8.0938720703125  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_0_original_depth]|![JNet_584_pretrain_0_novibrate_depth]|![JNet_584_pretrain_0_aligned_depth]|![JNet_584_pretrain_0_outputx_depth]|![JNet_584_pretrain_0_labelx_depth]|![JNet_584_pretrain_0_outputz_depth]|![JNet_584_pretrain_0_labelz_depth]|
  
MSEx: 0.008677849546074867, BCEx: 0.03387918695807457  
MSEz: 0.9850048422813416, BCEz: 8.0938720703125  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_1_original_plane]|![JNet_584_pretrain_1_novibrate_plane]|![JNet_584_pretrain_1_aligned_plane]|![JNet_584_pretrain_1_outputx_plane]|![JNet_584_pretrain_1_labelx_plane]|![JNet_584_pretrain_1_outputz_plane]|![JNet_584_pretrain_1_labelz_plane]|
  
MSEx: 0.00811035931110382, BCEx: 0.03154226765036583  
MSEz: 0.979112446308136, BCEz: 8.337677001953125  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_1_original_depth]|![JNet_584_pretrain_1_novibrate_depth]|![JNet_584_pretrain_1_aligned_depth]|![JNet_584_pretrain_1_outputx_depth]|![JNet_584_pretrain_1_labelx_depth]|![JNet_584_pretrain_1_outputz_depth]|![JNet_584_pretrain_1_labelz_depth]|
  
MSEx: 0.00811035931110382, BCEx: 0.03154226765036583  
MSEz: 0.979112446308136, BCEz: 8.337677001953125  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_2_original_plane]|![JNet_584_pretrain_2_novibrate_plane]|![JNet_584_pretrain_2_aligned_plane]|![JNet_584_pretrain_2_outputx_plane]|![JNet_584_pretrain_2_labelx_plane]|![JNet_584_pretrain_2_outputz_plane]|![JNet_584_pretrain_2_labelz_plane]|
  
MSEx: 0.012402854859828949, BCEx: 0.05046815052628517  
MSEz: 0.9733222126960754, BCEz: 7.586366653442383  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_2_original_depth]|![JNet_584_pretrain_2_novibrate_depth]|![JNet_584_pretrain_2_aligned_depth]|![JNet_584_pretrain_2_outputx_depth]|![JNet_584_pretrain_2_labelx_depth]|![JNet_584_pretrain_2_outputz_depth]|![JNet_584_pretrain_2_labelz_depth]|
  
MSEx: 0.012402854859828949, BCEx: 0.05046815052628517  
MSEz: 0.9733222126960754, BCEz: 7.586366653442383  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_3_original_plane]|![JNet_584_pretrain_3_novibrate_plane]|![JNet_584_pretrain_3_aligned_plane]|![JNet_584_pretrain_3_outputx_plane]|![JNet_584_pretrain_3_labelx_plane]|![JNet_584_pretrain_3_outputz_plane]|![JNet_584_pretrain_3_labelz_plane]|
  
MSEx: 0.004114519339054823, BCEx: 0.015583401545882225  
MSEz: 0.9937385320663452, BCEz: 8.958773612976074  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_3_original_depth]|![JNet_584_pretrain_3_novibrate_depth]|![JNet_584_pretrain_3_aligned_depth]|![JNet_584_pretrain_3_outputx_depth]|![JNet_584_pretrain_3_labelx_depth]|![JNet_584_pretrain_3_outputz_depth]|![JNet_584_pretrain_3_labelz_depth]|
  
MSEx: 0.004114519339054823, BCEx: 0.015583401545882225  
MSEz: 0.9937385320663452, BCEz: 8.958773612976074  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_4_original_plane]|![JNet_584_pretrain_4_novibrate_plane]|![JNet_584_pretrain_4_aligned_plane]|![JNet_584_pretrain_4_outputx_plane]|![JNet_584_pretrain_4_labelx_plane]|![JNet_584_pretrain_4_outputz_plane]|![JNet_584_pretrain_4_labelz_plane]|
  
MSEx: 0.010332074947655201, BCEx: 0.040176648646593094  
MSEz: 0.9732929468154907, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_4_original_depth]|![JNet_584_pretrain_4_novibrate_depth]|![JNet_584_pretrain_4_aligned_depth]|![JNet_584_pretrain_4_outputx_depth]|![JNet_584_pretrain_4_labelx_depth]|![JNet_584_pretrain_4_outputz_depth]|![JNet_584_pretrain_4_labelz_depth]|
  
MSEx: 0.010332074947655201, BCEx: 0.040176648646593094  
MSEz: 0.9732929468154907, BCEz: inf  

### pretrain
  
volume mean: 3.4894237711588554, volume sd: 0.23449575599800496
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi000_im000._original_depth]|![JNet_584_pretrain_beads_roi000_im000._output_depth]|![JNet_584_pretrain_beads_roi000_im000._reconst_depth]|![JNet_584_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 3.2842309570312507, MSE: 0.001101232715882361, quantized loss: 0.00028521419153548777  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi001_im004._original_depth]|![JNet_584_pretrain_beads_roi001_im004._output_depth]|![JNet_584_pretrain_beads_roi001_im004._reconst_depth]|![JNet_584_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 3.8444382324218758, MSE: 0.0011423139367252588, quantized loss: 0.00033242456265725195  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi002_im005._original_depth]|![JNet_584_pretrain_beads_roi002_im005._output_depth]|![JNet_584_pretrain_beads_roi002_im005._reconst_depth]|![JNet_584_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 3.419813232421876, MSE: 0.0010817451402544975, quantized loss: 0.00030168122611939907  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi003_im006._original_depth]|![JNet_584_pretrain_beads_roi003_im006._output_depth]|![JNet_584_pretrain_beads_roi003_im006._reconst_depth]|![JNet_584_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 3.503425048828126, MSE: 0.001097343978472054, quantized loss: 0.00030660154880024493  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi004_im006._original_depth]|![JNet_584_pretrain_beads_roi004_im006._output_depth]|![JNet_584_pretrain_beads_roi004_im006._reconst_depth]|![JNet_584_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 3.5564414062500007, MSE: 0.0011274017160758376, quantized loss: 0.00031118563492782414  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi005_im007._original_depth]|![JNet_584_pretrain_beads_roi005_im007._output_depth]|![JNet_584_pretrain_beads_roi005_im007._reconst_depth]|![JNet_584_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 3.322938720703126, MSE: 0.0010796904098242521, quantized loss: 0.00029913525213487446  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi006_im008._original_depth]|![JNet_584_pretrain_beads_roi006_im008._output_depth]|![JNet_584_pretrain_beads_roi006_im008._reconst_depth]|![JNet_584_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 3.613320556640626, MSE: 0.001059770118445158, quantized loss: 0.0003323771816212684  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi007_im009._original_depth]|![JNet_584_pretrain_beads_roi007_im009._output_depth]|![JNet_584_pretrain_beads_roi007_im009._reconst_depth]|![JNet_584_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 3.435328613281251, MSE: 0.00111062778159976, quantized loss: 0.0003017105918843299  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi008_im010._original_depth]|![JNet_584_pretrain_beads_roi008_im010._output_depth]|![JNet_584_pretrain_beads_roi008_im010._reconst_depth]|![JNet_584_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 3.5335095214843757, MSE: 0.0010923290392383933, quantized loss: 0.00030853398493491113  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi009_im011._original_depth]|![JNet_584_pretrain_beads_roi009_im011._output_depth]|![JNet_584_pretrain_beads_roi009_im011._reconst_depth]|![JNet_584_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 3.334980224609376, MSE: 0.0010555305052548647, quantized loss: 0.0002989350468851626  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi010_im012._original_depth]|![JNet_584_pretrain_beads_roi010_im012._output_depth]|![JNet_584_pretrain_beads_roi010_im012._reconst_depth]|![JNet_584_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 3.898695312500001, MSE: 0.001117922249250114, quantized loss: 0.00033721773070283234  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi011_im013._original_depth]|![JNet_584_pretrain_beads_roi011_im013._output_depth]|![JNet_584_pretrain_beads_roi011_im013._reconst_depth]|![JNet_584_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 3.940753173828126, MSE: 0.0011050108587369323, quantized loss: 0.00034512538695707917  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi012_im014._original_depth]|![JNet_584_pretrain_beads_roi012_im014._output_depth]|![JNet_584_pretrain_beads_roi012_im014._reconst_depth]|![JNet_584_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 3.4553366699218757, MSE: 0.001213587005622685, quantized loss: 0.0002906744775827974  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi013_im015._original_depth]|![JNet_584_pretrain_beads_roi013_im015._output_depth]|![JNet_584_pretrain_beads_roi013_im015._reconst_depth]|![JNet_584_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 3.208452392578126, MSE: 0.001137966406531632, quantized loss: 0.00028277223464101553  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi014_im016._original_depth]|![JNet_584_pretrain_beads_roi014_im016._output_depth]|![JNet_584_pretrain_beads_roi014_im016._reconst_depth]|![JNet_584_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 3.3563659667968757, MSE: 0.0010714237578213215, quantized loss: 0.00033797870855778456  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi015_im017._original_depth]|![JNet_584_pretrain_beads_roi015_im017._output_depth]|![JNet_584_pretrain_beads_roi015_im017._reconst_depth]|![JNet_584_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 3.246431640625001, MSE: 0.0010734890820458531, quantized loss: 0.0002901337284129113  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi016_im018._original_depth]|![JNet_584_pretrain_beads_roi016_im018._output_depth]|![JNet_584_pretrain_beads_roi016_im018._reconst_depth]|![JNet_584_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 3.5162871093750008, MSE: 0.0011937689268961549, quantized loss: 0.00029882637318223715  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi017_im018._original_depth]|![JNet_584_pretrain_beads_roi017_im018._output_depth]|![JNet_584_pretrain_beads_roi017_im018._reconst_depth]|![JNet_584_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 3.405478515625001, MSE: 0.0012289440492168069, quantized loss: 0.00028909786487929523  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi018_im022._original_depth]|![JNet_584_pretrain_beads_roi018_im022._output_depth]|![JNet_584_pretrain_beads_roi018_im022._reconst_depth]|![JNet_584_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 3.0705837402343756, MSE: 0.001083890674635768, quantized loss: 0.00027944252360612154  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi019_im023._original_depth]|![JNet_584_pretrain_beads_roi019_im023._output_depth]|![JNet_584_pretrain_beads_roi019_im023._reconst_depth]|![JNet_584_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 3.0369553222656256, MSE: 0.0011049945605918765, quantized loss: 0.0002759440103545785  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi020_im024._original_depth]|![JNet_584_pretrain_beads_roi020_im024._output_depth]|![JNet_584_pretrain_beads_roi020_im024._reconst_depth]|![JNet_584_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 3.6872309570312507, MSE: 0.0011134262895211577, quantized loss: 0.00030431768391281366  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi021_im026._original_depth]|![JNet_584_pretrain_beads_roi021_im026._output_depth]|![JNet_584_pretrain_beads_roi021_im026._reconst_depth]|![JNet_584_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 3.5494353027343757, MSE: 0.0010463210055604577, quantized loss: 0.00030415935907512903  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi022_im027._original_depth]|![JNet_584_pretrain_beads_roi022_im027._output_depth]|![JNet_584_pretrain_beads_roi022_im027._reconst_depth]|![JNet_584_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 3.379167724609376, MSE: 0.001115325023420155, quantized loss: 0.0002856942592188716  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi023_im028._original_depth]|![JNet_584_pretrain_beads_roi023_im028._output_depth]|![JNet_584_pretrain_beads_roi023_im028._reconst_depth]|![JNet_584_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 3.887989746093751, MSE: 0.000940138241276145, quantized loss: 0.00036663454375229776  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi024_im028._original_depth]|![JNet_584_pretrain_beads_roi024_im028._output_depth]|![JNet_584_pretrain_beads_roi024_im028._reconst_depth]|![JNet_584_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 3.743247314453126, MSE: 0.0010195090435445309, quantized loss: 0.0003285563725512475  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi025_im028._original_depth]|![JNet_584_pretrain_beads_roi025_im028._output_depth]|![JNet_584_pretrain_beads_roi025_im028._reconst_depth]|![JNet_584_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 3.743247314453126, MSE: 0.0010195090435445309, quantized loss: 0.0003285563725512475  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi026_im029._original_depth]|![JNet_584_pretrain_beads_roi026_im029._output_depth]|![JNet_584_pretrain_beads_roi026_im029._reconst_depth]|![JNet_584_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 3.7226926269531257, MSE: 0.0011379948118701577, quantized loss: 0.00031799066346138716  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi027_im029._original_depth]|![JNet_584_pretrain_beads_roi027_im029._output_depth]|![JNet_584_pretrain_beads_roi027_im029._reconst_depth]|![JNet_584_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 3.2995615234375006, MSE: 0.0011108177714049816, quantized loss: 0.0002911230840254575  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi028_im030._original_depth]|![JNet_584_pretrain_beads_roi028_im030._output_depth]|![JNet_584_pretrain_beads_roi028_im030._reconst_depth]|![JNet_584_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 3.278461914062501, MSE: 0.001091885264031589, quantized loss: 0.00028876453870907426  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_beads_roi029_im030._original_depth]|![JNet_584_pretrain_beads_roi029_im030._output_depth]|![JNet_584_pretrain_beads_roi029_im030._reconst_depth]|![JNet_584_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 3.4079123535156257, MSE: 0.0011299886973574758, quantized loss: 0.0002938350953627378  

### finetuning
  
volume mean: 6.363018994140629, volume sd: 0.3384017568472703
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi000_im000._original_depth]|![JNet_586_beads_roi000_im000._output_depth]|![JNet_586_beads_roi000_im000._reconst_depth]|![JNet_586_beads_roi000_im000._heatmap_depth]|
  
volume: 6.274334960937502, MSE: 0.0016693230718374252, quantized loss: 0.0004901357460767031  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi001_im004._original_depth]|![JNet_586_beads_roi001_im004._output_depth]|![JNet_586_beads_roi001_im004._reconst_depth]|![JNet_586_beads_roi001_im004._heatmap_depth]|
  
volume: 6.826369140625002, MSE: 0.001884306431747973, quantized loss: 0.0006750233587808907  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi002_im005._original_depth]|![JNet_586_beads_roi002_im005._output_depth]|![JNet_586_beads_roi002_im005._reconst_depth]|![JNet_586_beads_roi002_im005._heatmap_depth]|
  
volume: 6.489547363281251, MSE: 0.0017813798040151596, quantized loss: 0.0006498605362139642  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi003_im006._original_depth]|![JNet_586_beads_roi003_im006._output_depth]|![JNet_586_beads_roi003_im006._reconst_depth]|![JNet_586_beads_roi003_im006._heatmap_depth]|
  
volume: 6.537721191406251, MSE: 0.0017256428254768252, quantized loss: 0.0006799050024710596  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi004_im006._original_depth]|![JNet_586_beads_roi004_im006._output_depth]|![JNet_586_beads_roi004_im006._reconst_depth]|![JNet_586_beads_roi004_im006._heatmap_depth]|
  
volume: 6.608992675781251, MSE: 0.0017040908569470048, quantized loss: 0.0006248544086702168  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi005_im007._original_depth]|![JNet_586_beads_roi005_im007._output_depth]|![JNet_586_beads_roi005_im007._reconst_depth]|![JNet_586_beads_roi005_im007._heatmap_depth]|
  
volume: 6.588432617187501, MSE: 0.001712351106107235, quantized loss: 0.0008149397326633334  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi006_im008._original_depth]|![JNet_586_beads_roi006_im008._output_depth]|![JNet_586_beads_roi006_im008._reconst_depth]|![JNet_586_beads_roi006_im008._heatmap_depth]|
  
volume: 6.898621582031252, MSE: 0.0017474283231422305, quantized loss: 0.0007877410971559584  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi007_im009._original_depth]|![JNet_586_beads_roi007_im009._output_depth]|![JNet_586_beads_roi007_im009._reconst_depth]|![JNet_586_beads_roi007_im009._heatmap_depth]|
  
volume: 6.650625000000002, MSE: 0.0017584054730832577, quantized loss: 0.0007202582783065736  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi008_im010._original_depth]|![JNet_586_beads_roi008_im010._output_depth]|![JNet_586_beads_roi008_im010._reconst_depth]|![JNet_586_beads_roi008_im010._heatmap_depth]|
  
volume: 6.420439941406252, MSE: 0.0017772032879292965, quantized loss: 0.0005141851142980158  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi009_im011._original_depth]|![JNet_586_beads_roi009_im011._output_depth]|![JNet_586_beads_roi009_im011._reconst_depth]|![JNet_586_beads_roi009_im011._heatmap_depth]|
  
volume: 6.2866503906250015, MSE: 0.0016898317262530327, quantized loss: 0.0005198849830776453  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi010_im012._original_depth]|![JNet_586_beads_roi010_im012._output_depth]|![JNet_586_beads_roi010_im012._reconst_depth]|![JNet_586_beads_roi010_im012._heatmap_depth]|
  
volume: 6.884286621093752, MSE: 0.0018591267289593816, quantized loss: 0.0006522487383335829  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi011_im013._original_depth]|![JNet_586_beads_roi011_im013._output_depth]|![JNet_586_beads_roi011_im013._reconst_depth]|![JNet_586_beads_roi011_im013._heatmap_depth]|
  
volume: 6.881837890625001, MSE: 0.0018665677634999156, quantized loss: 0.0006414451636373997  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi012_im014._original_depth]|![JNet_586_beads_roi012_im014._output_depth]|![JNet_586_beads_roi012_im014._reconst_depth]|![JNet_586_beads_roi012_im014._heatmap_depth]|
  
volume: 6.096307128906251, MSE: 0.001777026685886085, quantized loss: 0.00040619977517053485  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi013_im015._original_depth]|![JNet_586_beads_roi013_im015._output_depth]|![JNet_586_beads_roi013_im015._reconst_depth]|![JNet_586_beads_roi013_im015._heatmap_depth]|
  
volume: 6.002417480468751, MSE: 0.001654316671192646, quantized loss: 0.0004381223989184946  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi014_im016._original_depth]|![JNet_586_beads_roi014_im016._output_depth]|![JNet_586_beads_roi014_im016._reconst_depth]|![JNet_586_beads_roi014_im016._heatmap_depth]|
  
volume: 6.5726567382812515, MSE: 0.0017505479045212269, quantized loss: 0.0007875228184275329  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi015_im017._original_depth]|![JNet_586_beads_roi015_im017._output_depth]|![JNet_586_beads_roi015_im017._reconst_depth]|![JNet_586_beads_roi015_im017._heatmap_depth]|
  
volume: 6.199398925781251, MSE: 0.001623971271328628, quantized loss: 0.0005461698747240007  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi016_im018._original_depth]|![JNet_586_beads_roi016_im018._output_depth]|![JNet_586_beads_roi016_im018._reconst_depth]|![JNet_586_beads_roi016_im018._heatmap_depth]|
  
volume: 6.727682128906252, MSE: 0.0018425866728648543, quantized loss: 0.000757790170609951  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi017_im018._original_depth]|![JNet_586_beads_roi017_im018._output_depth]|![JNet_586_beads_roi017_im018._reconst_depth]|![JNet_586_beads_roi017_im018._heatmap_depth]|
  
volume: 6.504857421875002, MSE: 0.001877413596957922, quantized loss: 0.0007332188542932272  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi018_im022._original_depth]|![JNet_586_beads_roi018_im022._output_depth]|![JNet_586_beads_roi018_im022._reconst_depth]|![JNet_586_beads_roi018_im022._heatmap_depth]|
  
volume: 5.657861328125001, MSE: 0.001605263096280396, quantized loss: 0.00035076329368166625  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi019_im023._original_depth]|![JNet_586_beads_roi019_im023._output_depth]|![JNet_586_beads_roi019_im023._reconst_depth]|![JNet_586_beads_roi019_im023._heatmap_depth]|
  
volume: 5.729145996093751, MSE: 0.0015353825874626637, quantized loss: 0.00036779558286070824  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi020_im024._original_depth]|![JNet_586_beads_roi020_im024._output_depth]|![JNet_586_beads_roi020_im024._reconst_depth]|![JNet_586_beads_roi020_im024._heatmap_depth]|
  
volume: 6.247126953125002, MSE: 0.0017058715457096696, quantized loss: 0.00041510656592436135  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi021_im026._original_depth]|![JNet_586_beads_roi021_im026._output_depth]|![JNet_586_beads_roi021_im026._reconst_depth]|![JNet_586_beads_roi021_im026._heatmap_depth]|
  
volume: 6.203817871093752, MSE: 0.001668391632847488, quantized loss: 0.00043023063335567713  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi022_im027._original_depth]|![JNet_586_beads_roi022_im027._output_depth]|![JNet_586_beads_roi022_im027._reconst_depth]|![JNet_586_beads_roi022_im027._heatmap_depth]|
  
volume: 6.245432617187501, MSE: 0.0017473372863605618, quantized loss: 0.0004032654978800565  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi023_im028._original_depth]|![JNet_586_beads_roi023_im028._output_depth]|![JNet_586_beads_roi023_im028._reconst_depth]|![JNet_586_beads_roi023_im028._heatmap_depth]|
  
volume: 6.581028320312502, MSE: 0.0017899718368425965, quantized loss: 0.0005341269425116479  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi024_im028._original_depth]|![JNet_586_beads_roi024_im028._output_depth]|![JNet_586_beads_roi024_im028._reconst_depth]|![JNet_586_beads_roi024_im028._heatmap_depth]|
  
volume: 6.376665039062502, MSE: 0.0018024616874754429, quantized loss: 0.0004777366411872208  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi025_im028._original_depth]|![JNet_586_beads_roi025_im028._output_depth]|![JNet_586_beads_roi025_im028._reconst_depth]|![JNet_586_beads_roi025_im028._heatmap_depth]|
  
volume: 6.376665039062502, MSE: 0.0018024616874754429, quantized loss: 0.0004777366411872208  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi026_im029._original_depth]|![JNet_586_beads_roi026_im029._output_depth]|![JNet_586_beads_roi026_im029._reconst_depth]|![JNet_586_beads_roi026_im029._heatmap_depth]|
  
volume: 6.3580395507812515, MSE: 0.00177520711440593, quantized loss: 0.0004137182841077447  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi027_im029._original_depth]|![JNet_586_beads_roi027_im029._output_depth]|![JNet_586_beads_roi027_im029._reconst_depth]|![JNet_586_beads_roi027_im029._heatmap_depth]|
  
volume: 5.738210937500002, MSE: 0.0017041964456439018, quantized loss: 0.000348036817740649  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi028_im030._original_depth]|![JNet_586_beads_roi028_im030._output_depth]|![JNet_586_beads_roi028_im030._reconst_depth]|![JNet_586_beads_roi028_im030._heatmap_depth]|
  
volume: 5.960939941406251, MSE: 0.0016682165442034602, quantized loss: 0.0004234920197632164  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_586_beads_roi029_im030._original_depth]|![JNet_586_beads_roi029_im030._output_depth]|![JNet_586_beads_roi029_im030._reconst_depth]|![JNet_586_beads_roi029_im030._heatmap_depth]|
  
volume: 5.964457031250001, MSE: 0.0017027814174070954, quantized loss: 0.0004163554694969207  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_586_psf_pre]|![JNet_586_psf_post]|

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
    (upsample): Upsample(scale_factor=(10.0, 1.0, 1.0), mode='trilinear')  
  )  
  (vq): VectorQuantizer()  
)  
```  
  



[JNet_584_pretrain_0_aligned_depth]: /experiments/images/JNet_584_pretrain_0_aligned_depth.png
[JNet_584_pretrain_0_aligned_plane]: /experiments/images/JNet_584_pretrain_0_aligned_plane.png
[JNet_584_pretrain_0_labelx_depth]: /experiments/images/JNet_584_pretrain_0_labelx_depth.png
[JNet_584_pretrain_0_labelx_plane]: /experiments/images/JNet_584_pretrain_0_labelx_plane.png
[JNet_584_pretrain_0_labelz_depth]: /experiments/images/JNet_584_pretrain_0_labelz_depth.png
[JNet_584_pretrain_0_labelz_plane]: /experiments/images/JNet_584_pretrain_0_labelz_plane.png
[JNet_584_pretrain_0_novibrate_depth]: /experiments/images/JNet_584_pretrain_0_novibrate_depth.png
[JNet_584_pretrain_0_novibrate_plane]: /experiments/images/JNet_584_pretrain_0_novibrate_plane.png
[JNet_584_pretrain_0_original_depth]: /experiments/images/JNet_584_pretrain_0_original_depth.png
[JNet_584_pretrain_0_original_plane]: /experiments/images/JNet_584_pretrain_0_original_plane.png
[JNet_584_pretrain_0_outputx_depth]: /experiments/images/JNet_584_pretrain_0_outputx_depth.png
[JNet_584_pretrain_0_outputx_plane]: /experiments/images/JNet_584_pretrain_0_outputx_plane.png
[JNet_584_pretrain_0_outputz_depth]: /experiments/images/JNet_584_pretrain_0_outputz_depth.png
[JNet_584_pretrain_0_outputz_plane]: /experiments/images/JNet_584_pretrain_0_outputz_plane.png
[JNet_584_pretrain_1_aligned_depth]: /experiments/images/JNet_584_pretrain_1_aligned_depth.png
[JNet_584_pretrain_1_aligned_plane]: /experiments/images/JNet_584_pretrain_1_aligned_plane.png
[JNet_584_pretrain_1_labelx_depth]: /experiments/images/JNet_584_pretrain_1_labelx_depth.png
[JNet_584_pretrain_1_labelx_plane]: /experiments/images/JNet_584_pretrain_1_labelx_plane.png
[JNet_584_pretrain_1_labelz_depth]: /experiments/images/JNet_584_pretrain_1_labelz_depth.png
[JNet_584_pretrain_1_labelz_plane]: /experiments/images/JNet_584_pretrain_1_labelz_plane.png
[JNet_584_pretrain_1_novibrate_depth]: /experiments/images/JNet_584_pretrain_1_novibrate_depth.png
[JNet_584_pretrain_1_novibrate_plane]: /experiments/images/JNet_584_pretrain_1_novibrate_plane.png
[JNet_584_pretrain_1_original_depth]: /experiments/images/JNet_584_pretrain_1_original_depth.png
[JNet_584_pretrain_1_original_plane]: /experiments/images/JNet_584_pretrain_1_original_plane.png
[JNet_584_pretrain_1_outputx_depth]: /experiments/images/JNet_584_pretrain_1_outputx_depth.png
[JNet_584_pretrain_1_outputx_plane]: /experiments/images/JNet_584_pretrain_1_outputx_plane.png
[JNet_584_pretrain_1_outputz_depth]: /experiments/images/JNet_584_pretrain_1_outputz_depth.png
[JNet_584_pretrain_1_outputz_plane]: /experiments/images/JNet_584_pretrain_1_outputz_plane.png
[JNet_584_pretrain_2_aligned_depth]: /experiments/images/JNet_584_pretrain_2_aligned_depth.png
[JNet_584_pretrain_2_aligned_plane]: /experiments/images/JNet_584_pretrain_2_aligned_plane.png
[JNet_584_pretrain_2_labelx_depth]: /experiments/images/JNet_584_pretrain_2_labelx_depth.png
[JNet_584_pretrain_2_labelx_plane]: /experiments/images/JNet_584_pretrain_2_labelx_plane.png
[JNet_584_pretrain_2_labelz_depth]: /experiments/images/JNet_584_pretrain_2_labelz_depth.png
[JNet_584_pretrain_2_labelz_plane]: /experiments/images/JNet_584_pretrain_2_labelz_plane.png
[JNet_584_pretrain_2_novibrate_depth]: /experiments/images/JNet_584_pretrain_2_novibrate_depth.png
[JNet_584_pretrain_2_novibrate_plane]: /experiments/images/JNet_584_pretrain_2_novibrate_plane.png
[JNet_584_pretrain_2_original_depth]: /experiments/images/JNet_584_pretrain_2_original_depth.png
[JNet_584_pretrain_2_original_plane]: /experiments/images/JNet_584_pretrain_2_original_plane.png
[JNet_584_pretrain_2_outputx_depth]: /experiments/images/JNet_584_pretrain_2_outputx_depth.png
[JNet_584_pretrain_2_outputx_plane]: /experiments/images/JNet_584_pretrain_2_outputx_plane.png
[JNet_584_pretrain_2_outputz_depth]: /experiments/images/JNet_584_pretrain_2_outputz_depth.png
[JNet_584_pretrain_2_outputz_plane]: /experiments/images/JNet_584_pretrain_2_outputz_plane.png
[JNet_584_pretrain_3_aligned_depth]: /experiments/images/JNet_584_pretrain_3_aligned_depth.png
[JNet_584_pretrain_3_aligned_plane]: /experiments/images/JNet_584_pretrain_3_aligned_plane.png
[JNet_584_pretrain_3_labelx_depth]: /experiments/images/JNet_584_pretrain_3_labelx_depth.png
[JNet_584_pretrain_3_labelx_plane]: /experiments/images/JNet_584_pretrain_3_labelx_plane.png
[JNet_584_pretrain_3_labelz_depth]: /experiments/images/JNet_584_pretrain_3_labelz_depth.png
[JNet_584_pretrain_3_labelz_plane]: /experiments/images/JNet_584_pretrain_3_labelz_plane.png
[JNet_584_pretrain_3_novibrate_depth]: /experiments/images/JNet_584_pretrain_3_novibrate_depth.png
[JNet_584_pretrain_3_novibrate_plane]: /experiments/images/JNet_584_pretrain_3_novibrate_plane.png
[JNet_584_pretrain_3_original_depth]: /experiments/images/JNet_584_pretrain_3_original_depth.png
[JNet_584_pretrain_3_original_plane]: /experiments/images/JNet_584_pretrain_3_original_plane.png
[JNet_584_pretrain_3_outputx_depth]: /experiments/images/JNet_584_pretrain_3_outputx_depth.png
[JNet_584_pretrain_3_outputx_plane]: /experiments/images/JNet_584_pretrain_3_outputx_plane.png
[JNet_584_pretrain_3_outputz_depth]: /experiments/images/JNet_584_pretrain_3_outputz_depth.png
[JNet_584_pretrain_3_outputz_plane]: /experiments/images/JNet_584_pretrain_3_outputz_plane.png
[JNet_584_pretrain_4_aligned_depth]: /experiments/images/JNet_584_pretrain_4_aligned_depth.png
[JNet_584_pretrain_4_aligned_plane]: /experiments/images/JNet_584_pretrain_4_aligned_plane.png
[JNet_584_pretrain_4_labelx_depth]: /experiments/images/JNet_584_pretrain_4_labelx_depth.png
[JNet_584_pretrain_4_labelx_plane]: /experiments/images/JNet_584_pretrain_4_labelx_plane.png
[JNet_584_pretrain_4_labelz_depth]: /experiments/images/JNet_584_pretrain_4_labelz_depth.png
[JNet_584_pretrain_4_labelz_plane]: /experiments/images/JNet_584_pretrain_4_labelz_plane.png
[JNet_584_pretrain_4_novibrate_depth]: /experiments/images/JNet_584_pretrain_4_novibrate_depth.png
[JNet_584_pretrain_4_novibrate_plane]: /experiments/images/JNet_584_pretrain_4_novibrate_plane.png
[JNet_584_pretrain_4_original_depth]: /experiments/images/JNet_584_pretrain_4_original_depth.png
[JNet_584_pretrain_4_original_plane]: /experiments/images/JNet_584_pretrain_4_original_plane.png
[JNet_584_pretrain_4_outputx_depth]: /experiments/images/JNet_584_pretrain_4_outputx_depth.png
[JNet_584_pretrain_4_outputx_plane]: /experiments/images/JNet_584_pretrain_4_outputx_plane.png
[JNet_584_pretrain_4_outputz_depth]: /experiments/images/JNet_584_pretrain_4_outputz_depth.png
[JNet_584_pretrain_4_outputz_plane]: /experiments/images/JNet_584_pretrain_4_outputz_plane.png
[JNet_584_pretrain_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi000_im000._heatmap_depth.png
[JNet_584_pretrain_beads_roi000_im000._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi000_im000._original_depth.png
[JNet_584_pretrain_beads_roi000_im000._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi000_im000._output_depth.png
[JNet_584_pretrain_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi000_im000._reconst_depth.png
[JNet_584_pretrain_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi001_im004._heatmap_depth.png
[JNet_584_pretrain_beads_roi001_im004._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi001_im004._original_depth.png
[JNet_584_pretrain_beads_roi001_im004._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi001_im004._output_depth.png
[JNet_584_pretrain_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi001_im004._reconst_depth.png
[JNet_584_pretrain_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi002_im005._heatmap_depth.png
[JNet_584_pretrain_beads_roi002_im005._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi002_im005._original_depth.png
[JNet_584_pretrain_beads_roi002_im005._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi002_im005._output_depth.png
[JNet_584_pretrain_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi002_im005._reconst_depth.png
[JNet_584_pretrain_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi003_im006._heatmap_depth.png
[JNet_584_pretrain_beads_roi003_im006._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi003_im006._original_depth.png
[JNet_584_pretrain_beads_roi003_im006._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi003_im006._output_depth.png
[JNet_584_pretrain_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi003_im006._reconst_depth.png
[JNet_584_pretrain_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi004_im006._heatmap_depth.png
[JNet_584_pretrain_beads_roi004_im006._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi004_im006._original_depth.png
[JNet_584_pretrain_beads_roi004_im006._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi004_im006._output_depth.png
[JNet_584_pretrain_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi004_im006._reconst_depth.png
[JNet_584_pretrain_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi005_im007._heatmap_depth.png
[JNet_584_pretrain_beads_roi005_im007._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi005_im007._original_depth.png
[JNet_584_pretrain_beads_roi005_im007._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi005_im007._output_depth.png
[JNet_584_pretrain_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi005_im007._reconst_depth.png
[JNet_584_pretrain_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi006_im008._heatmap_depth.png
[JNet_584_pretrain_beads_roi006_im008._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi006_im008._original_depth.png
[JNet_584_pretrain_beads_roi006_im008._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi006_im008._output_depth.png
[JNet_584_pretrain_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi006_im008._reconst_depth.png
[JNet_584_pretrain_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi007_im009._heatmap_depth.png
[JNet_584_pretrain_beads_roi007_im009._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi007_im009._original_depth.png
[JNet_584_pretrain_beads_roi007_im009._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi007_im009._output_depth.png
[JNet_584_pretrain_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi007_im009._reconst_depth.png
[JNet_584_pretrain_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi008_im010._heatmap_depth.png
[JNet_584_pretrain_beads_roi008_im010._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi008_im010._original_depth.png
[JNet_584_pretrain_beads_roi008_im010._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi008_im010._output_depth.png
[JNet_584_pretrain_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi008_im010._reconst_depth.png
[JNet_584_pretrain_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi009_im011._heatmap_depth.png
[JNet_584_pretrain_beads_roi009_im011._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi009_im011._original_depth.png
[JNet_584_pretrain_beads_roi009_im011._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi009_im011._output_depth.png
[JNet_584_pretrain_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi009_im011._reconst_depth.png
[JNet_584_pretrain_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi010_im012._heatmap_depth.png
[JNet_584_pretrain_beads_roi010_im012._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi010_im012._original_depth.png
[JNet_584_pretrain_beads_roi010_im012._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi010_im012._output_depth.png
[JNet_584_pretrain_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi010_im012._reconst_depth.png
[JNet_584_pretrain_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi011_im013._heatmap_depth.png
[JNet_584_pretrain_beads_roi011_im013._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi011_im013._original_depth.png
[JNet_584_pretrain_beads_roi011_im013._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi011_im013._output_depth.png
[JNet_584_pretrain_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi011_im013._reconst_depth.png
[JNet_584_pretrain_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi012_im014._heatmap_depth.png
[JNet_584_pretrain_beads_roi012_im014._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi012_im014._original_depth.png
[JNet_584_pretrain_beads_roi012_im014._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi012_im014._output_depth.png
[JNet_584_pretrain_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi012_im014._reconst_depth.png
[JNet_584_pretrain_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi013_im015._heatmap_depth.png
[JNet_584_pretrain_beads_roi013_im015._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi013_im015._original_depth.png
[JNet_584_pretrain_beads_roi013_im015._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi013_im015._output_depth.png
[JNet_584_pretrain_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi013_im015._reconst_depth.png
[JNet_584_pretrain_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi014_im016._heatmap_depth.png
[JNet_584_pretrain_beads_roi014_im016._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi014_im016._original_depth.png
[JNet_584_pretrain_beads_roi014_im016._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi014_im016._output_depth.png
[JNet_584_pretrain_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi014_im016._reconst_depth.png
[JNet_584_pretrain_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi015_im017._heatmap_depth.png
[JNet_584_pretrain_beads_roi015_im017._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi015_im017._original_depth.png
[JNet_584_pretrain_beads_roi015_im017._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi015_im017._output_depth.png
[JNet_584_pretrain_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi015_im017._reconst_depth.png
[JNet_584_pretrain_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi016_im018._heatmap_depth.png
[JNet_584_pretrain_beads_roi016_im018._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi016_im018._original_depth.png
[JNet_584_pretrain_beads_roi016_im018._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi016_im018._output_depth.png
[JNet_584_pretrain_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi016_im018._reconst_depth.png
[JNet_584_pretrain_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi017_im018._heatmap_depth.png
[JNet_584_pretrain_beads_roi017_im018._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi017_im018._original_depth.png
[JNet_584_pretrain_beads_roi017_im018._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi017_im018._output_depth.png
[JNet_584_pretrain_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi017_im018._reconst_depth.png
[JNet_584_pretrain_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi018_im022._heatmap_depth.png
[JNet_584_pretrain_beads_roi018_im022._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi018_im022._original_depth.png
[JNet_584_pretrain_beads_roi018_im022._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi018_im022._output_depth.png
[JNet_584_pretrain_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi018_im022._reconst_depth.png
[JNet_584_pretrain_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi019_im023._heatmap_depth.png
[JNet_584_pretrain_beads_roi019_im023._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi019_im023._original_depth.png
[JNet_584_pretrain_beads_roi019_im023._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi019_im023._output_depth.png
[JNet_584_pretrain_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi019_im023._reconst_depth.png
[JNet_584_pretrain_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi020_im024._heatmap_depth.png
[JNet_584_pretrain_beads_roi020_im024._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi020_im024._original_depth.png
[JNet_584_pretrain_beads_roi020_im024._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi020_im024._output_depth.png
[JNet_584_pretrain_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi020_im024._reconst_depth.png
[JNet_584_pretrain_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi021_im026._heatmap_depth.png
[JNet_584_pretrain_beads_roi021_im026._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi021_im026._original_depth.png
[JNet_584_pretrain_beads_roi021_im026._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi021_im026._output_depth.png
[JNet_584_pretrain_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi021_im026._reconst_depth.png
[JNet_584_pretrain_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi022_im027._heatmap_depth.png
[JNet_584_pretrain_beads_roi022_im027._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi022_im027._original_depth.png
[JNet_584_pretrain_beads_roi022_im027._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi022_im027._output_depth.png
[JNet_584_pretrain_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi022_im027._reconst_depth.png
[JNet_584_pretrain_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi023_im028._heatmap_depth.png
[JNet_584_pretrain_beads_roi023_im028._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi023_im028._original_depth.png
[JNet_584_pretrain_beads_roi023_im028._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi023_im028._output_depth.png
[JNet_584_pretrain_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi023_im028._reconst_depth.png
[JNet_584_pretrain_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi024_im028._heatmap_depth.png
[JNet_584_pretrain_beads_roi024_im028._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi024_im028._original_depth.png
[JNet_584_pretrain_beads_roi024_im028._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi024_im028._output_depth.png
[JNet_584_pretrain_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi024_im028._reconst_depth.png
[JNet_584_pretrain_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi025_im028._heatmap_depth.png
[JNet_584_pretrain_beads_roi025_im028._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi025_im028._original_depth.png
[JNet_584_pretrain_beads_roi025_im028._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi025_im028._output_depth.png
[JNet_584_pretrain_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi025_im028._reconst_depth.png
[JNet_584_pretrain_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi026_im029._heatmap_depth.png
[JNet_584_pretrain_beads_roi026_im029._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi026_im029._original_depth.png
[JNet_584_pretrain_beads_roi026_im029._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi026_im029._output_depth.png
[JNet_584_pretrain_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi026_im029._reconst_depth.png
[JNet_584_pretrain_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi027_im029._heatmap_depth.png
[JNet_584_pretrain_beads_roi027_im029._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi027_im029._original_depth.png
[JNet_584_pretrain_beads_roi027_im029._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi027_im029._output_depth.png
[JNet_584_pretrain_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi027_im029._reconst_depth.png
[JNet_584_pretrain_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi028_im030._heatmap_depth.png
[JNet_584_pretrain_beads_roi028_im030._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi028_im030._original_depth.png
[JNet_584_pretrain_beads_roi028_im030._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi028_im030._output_depth.png
[JNet_584_pretrain_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi028_im030._reconst_depth.png
[JNet_584_pretrain_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_584_pretrain_beads_roi029_im030._heatmap_depth.png
[JNet_584_pretrain_beads_roi029_im030._original_depth]: /experiments/images/JNet_584_pretrain_beads_roi029_im030._original_depth.png
[JNet_584_pretrain_beads_roi029_im030._output_depth]: /experiments/images/JNet_584_pretrain_beads_roi029_im030._output_depth.png
[JNet_584_pretrain_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_584_pretrain_beads_roi029_im030._reconst_depth.png
[JNet_586_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_586_beads_roi000_im000._heatmap_depth.png
[JNet_586_beads_roi000_im000._original_depth]: /experiments/images/JNet_586_beads_roi000_im000._original_depth.png
[JNet_586_beads_roi000_im000._output_depth]: /experiments/images/JNet_586_beads_roi000_im000._output_depth.png
[JNet_586_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_586_beads_roi000_im000._reconst_depth.png
[JNet_586_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_586_beads_roi001_im004._heatmap_depth.png
[JNet_586_beads_roi001_im004._original_depth]: /experiments/images/JNet_586_beads_roi001_im004._original_depth.png
[JNet_586_beads_roi001_im004._output_depth]: /experiments/images/JNet_586_beads_roi001_im004._output_depth.png
[JNet_586_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_586_beads_roi001_im004._reconst_depth.png
[JNet_586_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_586_beads_roi002_im005._heatmap_depth.png
[JNet_586_beads_roi002_im005._original_depth]: /experiments/images/JNet_586_beads_roi002_im005._original_depth.png
[JNet_586_beads_roi002_im005._output_depth]: /experiments/images/JNet_586_beads_roi002_im005._output_depth.png
[JNet_586_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_586_beads_roi002_im005._reconst_depth.png
[JNet_586_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_586_beads_roi003_im006._heatmap_depth.png
[JNet_586_beads_roi003_im006._original_depth]: /experiments/images/JNet_586_beads_roi003_im006._original_depth.png
[JNet_586_beads_roi003_im006._output_depth]: /experiments/images/JNet_586_beads_roi003_im006._output_depth.png
[JNet_586_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_586_beads_roi003_im006._reconst_depth.png
[JNet_586_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_586_beads_roi004_im006._heatmap_depth.png
[JNet_586_beads_roi004_im006._original_depth]: /experiments/images/JNet_586_beads_roi004_im006._original_depth.png
[JNet_586_beads_roi004_im006._output_depth]: /experiments/images/JNet_586_beads_roi004_im006._output_depth.png
[JNet_586_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_586_beads_roi004_im006._reconst_depth.png
[JNet_586_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_586_beads_roi005_im007._heatmap_depth.png
[JNet_586_beads_roi005_im007._original_depth]: /experiments/images/JNet_586_beads_roi005_im007._original_depth.png
[JNet_586_beads_roi005_im007._output_depth]: /experiments/images/JNet_586_beads_roi005_im007._output_depth.png
[JNet_586_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_586_beads_roi005_im007._reconst_depth.png
[JNet_586_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_586_beads_roi006_im008._heatmap_depth.png
[JNet_586_beads_roi006_im008._original_depth]: /experiments/images/JNet_586_beads_roi006_im008._original_depth.png
[JNet_586_beads_roi006_im008._output_depth]: /experiments/images/JNet_586_beads_roi006_im008._output_depth.png
[JNet_586_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_586_beads_roi006_im008._reconst_depth.png
[JNet_586_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_586_beads_roi007_im009._heatmap_depth.png
[JNet_586_beads_roi007_im009._original_depth]: /experiments/images/JNet_586_beads_roi007_im009._original_depth.png
[JNet_586_beads_roi007_im009._output_depth]: /experiments/images/JNet_586_beads_roi007_im009._output_depth.png
[JNet_586_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_586_beads_roi007_im009._reconst_depth.png
[JNet_586_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_586_beads_roi008_im010._heatmap_depth.png
[JNet_586_beads_roi008_im010._original_depth]: /experiments/images/JNet_586_beads_roi008_im010._original_depth.png
[JNet_586_beads_roi008_im010._output_depth]: /experiments/images/JNet_586_beads_roi008_im010._output_depth.png
[JNet_586_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_586_beads_roi008_im010._reconst_depth.png
[JNet_586_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_586_beads_roi009_im011._heatmap_depth.png
[JNet_586_beads_roi009_im011._original_depth]: /experiments/images/JNet_586_beads_roi009_im011._original_depth.png
[JNet_586_beads_roi009_im011._output_depth]: /experiments/images/JNet_586_beads_roi009_im011._output_depth.png
[JNet_586_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_586_beads_roi009_im011._reconst_depth.png
[JNet_586_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_586_beads_roi010_im012._heatmap_depth.png
[JNet_586_beads_roi010_im012._original_depth]: /experiments/images/JNet_586_beads_roi010_im012._original_depth.png
[JNet_586_beads_roi010_im012._output_depth]: /experiments/images/JNet_586_beads_roi010_im012._output_depth.png
[JNet_586_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_586_beads_roi010_im012._reconst_depth.png
[JNet_586_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_586_beads_roi011_im013._heatmap_depth.png
[JNet_586_beads_roi011_im013._original_depth]: /experiments/images/JNet_586_beads_roi011_im013._original_depth.png
[JNet_586_beads_roi011_im013._output_depth]: /experiments/images/JNet_586_beads_roi011_im013._output_depth.png
[JNet_586_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_586_beads_roi011_im013._reconst_depth.png
[JNet_586_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_586_beads_roi012_im014._heatmap_depth.png
[JNet_586_beads_roi012_im014._original_depth]: /experiments/images/JNet_586_beads_roi012_im014._original_depth.png
[JNet_586_beads_roi012_im014._output_depth]: /experiments/images/JNet_586_beads_roi012_im014._output_depth.png
[JNet_586_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_586_beads_roi012_im014._reconst_depth.png
[JNet_586_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_586_beads_roi013_im015._heatmap_depth.png
[JNet_586_beads_roi013_im015._original_depth]: /experiments/images/JNet_586_beads_roi013_im015._original_depth.png
[JNet_586_beads_roi013_im015._output_depth]: /experiments/images/JNet_586_beads_roi013_im015._output_depth.png
[JNet_586_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_586_beads_roi013_im015._reconst_depth.png
[JNet_586_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_586_beads_roi014_im016._heatmap_depth.png
[JNet_586_beads_roi014_im016._original_depth]: /experiments/images/JNet_586_beads_roi014_im016._original_depth.png
[JNet_586_beads_roi014_im016._output_depth]: /experiments/images/JNet_586_beads_roi014_im016._output_depth.png
[JNet_586_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_586_beads_roi014_im016._reconst_depth.png
[JNet_586_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_586_beads_roi015_im017._heatmap_depth.png
[JNet_586_beads_roi015_im017._original_depth]: /experiments/images/JNet_586_beads_roi015_im017._original_depth.png
[JNet_586_beads_roi015_im017._output_depth]: /experiments/images/JNet_586_beads_roi015_im017._output_depth.png
[JNet_586_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_586_beads_roi015_im017._reconst_depth.png
[JNet_586_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_586_beads_roi016_im018._heatmap_depth.png
[JNet_586_beads_roi016_im018._original_depth]: /experiments/images/JNet_586_beads_roi016_im018._original_depth.png
[JNet_586_beads_roi016_im018._output_depth]: /experiments/images/JNet_586_beads_roi016_im018._output_depth.png
[JNet_586_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_586_beads_roi016_im018._reconst_depth.png
[JNet_586_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_586_beads_roi017_im018._heatmap_depth.png
[JNet_586_beads_roi017_im018._original_depth]: /experiments/images/JNet_586_beads_roi017_im018._original_depth.png
[JNet_586_beads_roi017_im018._output_depth]: /experiments/images/JNet_586_beads_roi017_im018._output_depth.png
[JNet_586_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_586_beads_roi017_im018._reconst_depth.png
[JNet_586_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_586_beads_roi018_im022._heatmap_depth.png
[JNet_586_beads_roi018_im022._original_depth]: /experiments/images/JNet_586_beads_roi018_im022._original_depth.png
[JNet_586_beads_roi018_im022._output_depth]: /experiments/images/JNet_586_beads_roi018_im022._output_depth.png
[JNet_586_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_586_beads_roi018_im022._reconst_depth.png
[JNet_586_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_586_beads_roi019_im023._heatmap_depth.png
[JNet_586_beads_roi019_im023._original_depth]: /experiments/images/JNet_586_beads_roi019_im023._original_depth.png
[JNet_586_beads_roi019_im023._output_depth]: /experiments/images/JNet_586_beads_roi019_im023._output_depth.png
[JNet_586_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_586_beads_roi019_im023._reconst_depth.png
[JNet_586_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_586_beads_roi020_im024._heatmap_depth.png
[JNet_586_beads_roi020_im024._original_depth]: /experiments/images/JNet_586_beads_roi020_im024._original_depth.png
[JNet_586_beads_roi020_im024._output_depth]: /experiments/images/JNet_586_beads_roi020_im024._output_depth.png
[JNet_586_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_586_beads_roi020_im024._reconst_depth.png
[JNet_586_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_586_beads_roi021_im026._heatmap_depth.png
[JNet_586_beads_roi021_im026._original_depth]: /experiments/images/JNet_586_beads_roi021_im026._original_depth.png
[JNet_586_beads_roi021_im026._output_depth]: /experiments/images/JNet_586_beads_roi021_im026._output_depth.png
[JNet_586_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_586_beads_roi021_im026._reconst_depth.png
[JNet_586_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_586_beads_roi022_im027._heatmap_depth.png
[JNet_586_beads_roi022_im027._original_depth]: /experiments/images/JNet_586_beads_roi022_im027._original_depth.png
[JNet_586_beads_roi022_im027._output_depth]: /experiments/images/JNet_586_beads_roi022_im027._output_depth.png
[JNet_586_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_586_beads_roi022_im027._reconst_depth.png
[JNet_586_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_586_beads_roi023_im028._heatmap_depth.png
[JNet_586_beads_roi023_im028._original_depth]: /experiments/images/JNet_586_beads_roi023_im028._original_depth.png
[JNet_586_beads_roi023_im028._output_depth]: /experiments/images/JNet_586_beads_roi023_im028._output_depth.png
[JNet_586_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_586_beads_roi023_im028._reconst_depth.png
[JNet_586_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_586_beads_roi024_im028._heatmap_depth.png
[JNet_586_beads_roi024_im028._original_depth]: /experiments/images/JNet_586_beads_roi024_im028._original_depth.png
[JNet_586_beads_roi024_im028._output_depth]: /experiments/images/JNet_586_beads_roi024_im028._output_depth.png
[JNet_586_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_586_beads_roi024_im028._reconst_depth.png
[JNet_586_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_586_beads_roi025_im028._heatmap_depth.png
[JNet_586_beads_roi025_im028._original_depth]: /experiments/images/JNet_586_beads_roi025_im028._original_depth.png
[JNet_586_beads_roi025_im028._output_depth]: /experiments/images/JNet_586_beads_roi025_im028._output_depth.png
[JNet_586_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_586_beads_roi025_im028._reconst_depth.png
[JNet_586_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_586_beads_roi026_im029._heatmap_depth.png
[JNet_586_beads_roi026_im029._original_depth]: /experiments/images/JNet_586_beads_roi026_im029._original_depth.png
[JNet_586_beads_roi026_im029._output_depth]: /experiments/images/JNet_586_beads_roi026_im029._output_depth.png
[JNet_586_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_586_beads_roi026_im029._reconst_depth.png
[JNet_586_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_586_beads_roi027_im029._heatmap_depth.png
[JNet_586_beads_roi027_im029._original_depth]: /experiments/images/JNet_586_beads_roi027_im029._original_depth.png
[JNet_586_beads_roi027_im029._output_depth]: /experiments/images/JNet_586_beads_roi027_im029._output_depth.png
[JNet_586_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_586_beads_roi027_im029._reconst_depth.png
[JNet_586_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_586_beads_roi028_im030._heatmap_depth.png
[JNet_586_beads_roi028_im030._original_depth]: /experiments/images/JNet_586_beads_roi028_im030._original_depth.png
[JNet_586_beads_roi028_im030._output_depth]: /experiments/images/JNet_586_beads_roi028_im030._output_depth.png
[JNet_586_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_586_beads_roi028_im030._reconst_depth.png
[JNet_586_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_586_beads_roi029_im030._heatmap_depth.png
[JNet_586_beads_roi029_im030._original_depth]: /experiments/images/JNet_586_beads_roi029_im030._original_depth.png
[JNet_586_beads_roi029_im030._output_depth]: /experiments/images/JNet_586_beads_roi029_im030._output_depth.png
[JNet_586_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_586_beads_roi029_im030._reconst_depth.png
[JNet_586_psf_post]: /experiments/images/JNet_586_psf_post.png
[JNet_586_psf_pre]: /experiments/images/JNet_586_psf_pre.png
