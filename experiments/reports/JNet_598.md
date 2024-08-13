



# JNet_598 Report
  
psf loss 1.0 and ewc loss 0.01  
pretrained model : JNet_597_pretrain
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
|wavelength|0.8|microns|
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
|ewc_weight|0.01|
|qloss_weight|1.0|
|ploss_weight|1.0|
|mrfloss_order|1|
|mrfloss_dilation|1|
|mrfloss_weights|{'l_00': 0, 'l_01': 0, 'l_10': 0, 'l_11': 0}|

## Results

### Pretraining
  
Segmentation: mean MSE: 0.009859499521553516, mean BCE: 0.03643091022968292  
Luminance Estimation: mean MSE: 0.963756263256073, mean BCE: nan
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_0_original_plane]|![JNet_597_pretrain_0_novibrate_plane]|![JNet_597_pretrain_0_aligned_plane]|![JNet_597_pretrain_0_outputx_plane]|![JNet_597_pretrain_0_labelx_plane]|![JNet_597_pretrain_0_outputz_plane]|![JNet_597_pretrain_0_labelz_plane]|
  
MSEx: 0.007824408821761608, BCEx: 0.028853897005319595  
MSEz: 0.9708977937698364, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_0_original_depth]|![JNet_597_pretrain_0_novibrate_depth]|![JNet_597_pretrain_0_aligned_depth]|![JNet_597_pretrain_0_outputx_depth]|![JNet_597_pretrain_0_labelx_depth]|![JNet_597_pretrain_0_outputz_depth]|![JNet_597_pretrain_0_labelz_depth]|
  
MSEx: 0.007824408821761608, BCEx: 0.028853897005319595  
MSEz: 0.9708977937698364, BCEz: inf  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_1_original_plane]|![JNet_597_pretrain_1_novibrate_plane]|![JNet_597_pretrain_1_aligned_plane]|![JNet_597_pretrain_1_outputx_plane]|![JNet_597_pretrain_1_labelx_plane]|![JNet_597_pretrain_1_outputz_plane]|![JNet_597_pretrain_1_labelz_plane]|
  
MSEx: 0.009726118296384811, BCEx: 0.03475034236907959  
MSEz: 0.9689496159553528, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_1_original_depth]|![JNet_597_pretrain_1_novibrate_depth]|![JNet_597_pretrain_1_aligned_depth]|![JNet_597_pretrain_1_outputx_depth]|![JNet_597_pretrain_1_labelx_depth]|![JNet_597_pretrain_1_outputz_depth]|![JNet_597_pretrain_1_labelz_depth]|
  
MSEx: 0.009726118296384811, BCEx: 0.03475034236907959  
MSEz: 0.9689496159553528, BCEz: inf  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_2_original_plane]|![JNet_597_pretrain_2_novibrate_plane]|![JNet_597_pretrain_2_aligned_plane]|![JNet_597_pretrain_2_outputx_plane]|![JNet_597_pretrain_2_labelx_plane]|![JNet_597_pretrain_2_outputz_plane]|![JNet_597_pretrain_2_labelz_plane]|
  
MSEx: 0.01229506079107523, BCEx: 0.047250375151634216  
MSEz: 0.9681071043014526, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_2_original_depth]|![JNet_597_pretrain_2_novibrate_depth]|![JNet_597_pretrain_2_aligned_depth]|![JNet_597_pretrain_2_outputx_depth]|![JNet_597_pretrain_2_labelx_depth]|![JNet_597_pretrain_2_outputz_depth]|![JNet_597_pretrain_2_labelz_depth]|
  
MSEx: 0.01229506079107523, BCEx: 0.047250375151634216  
MSEz: 0.9681071043014526, BCEz: nan  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_3_original_plane]|![JNet_597_pretrain_3_novibrate_plane]|![JNet_597_pretrain_3_aligned_plane]|![JNet_597_pretrain_3_outputx_plane]|![JNet_597_pretrain_3_labelx_plane]|![JNet_597_pretrain_3_outputz_plane]|![JNet_597_pretrain_3_labelz_plane]|
  
MSEx: 0.007291038520634174, BCEx: 0.02640996314585209  
MSEz: 0.9753527045249939, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_3_original_depth]|![JNet_597_pretrain_3_novibrate_depth]|![JNet_597_pretrain_3_aligned_depth]|![JNet_597_pretrain_3_outputx_depth]|![JNet_597_pretrain_3_labelx_depth]|![JNet_597_pretrain_3_outputz_depth]|![JNet_597_pretrain_3_labelz_depth]|
  
MSEx: 0.007291038520634174, BCEx: 0.02640996314585209  
MSEz: 0.9753527045249939, BCEz: inf  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_4_original_plane]|![JNet_597_pretrain_4_novibrate_plane]|![JNet_597_pretrain_4_aligned_plane]|![JNet_597_pretrain_4_outputx_plane]|![JNet_597_pretrain_4_labelx_plane]|![JNet_597_pretrain_4_outputz_plane]|![JNet_597_pretrain_4_labelz_plane]|
  
MSEx: 0.012160873040556908, BCEx: 0.044889967888593674  
MSEz: 0.9354737997055054, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_4_original_depth]|![JNet_597_pretrain_4_novibrate_depth]|![JNet_597_pretrain_4_aligned_depth]|![JNet_597_pretrain_4_outputx_depth]|![JNet_597_pretrain_4_labelx_depth]|![JNet_597_pretrain_4_outputz_depth]|![JNet_597_pretrain_4_labelz_depth]|
  
MSEx: 0.012160873040556908, BCEx: 0.044889967888593674  
MSEz: 0.9354737997055054, BCEz: inf  

### pretrain
  
volume mean: 4.576250000000001, volume sd: 0.3271657693356688
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi000_im000._original_depth]|![JNet_597_pretrain_beads_roi000_im000._output_depth]|![JNet_597_pretrain_beads_roi000_im000._reconst_depth]|![JNet_597_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 4.131375000000001, MSE: 0.0009602407808415592, quantized loss: 0.0003289403684902936  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi001_im004._original_depth]|![JNet_597_pretrain_beads_roi001_im004._output_depth]|![JNet_597_pretrain_beads_roi001_im004._reconst_depth]|![JNet_597_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 5.143750000000002, MSE: 0.0009963115444406867, quantized loss: 0.00040353136137127876  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi002_im005._original_depth]|![JNet_597_pretrain_beads_roi002_im005._output_depth]|![JNet_597_pretrain_beads_roi002_im005._reconst_depth]|![JNet_597_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 4.307125000000001, MSE: 0.0009556126897223294, quantized loss: 0.0003567506792023778  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi003_im006._original_depth]|![JNet_597_pretrain_beads_roi003_im006._output_depth]|![JNet_597_pretrain_beads_roi003_im006._reconst_depth]|![JNet_597_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 4.452375000000001, MSE: 0.0009748554439283907, quantized loss: 0.0003654134052339941  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi004_im006._original_depth]|![JNet_597_pretrain_beads_roi004_im006._output_depth]|![JNet_597_pretrain_beads_roi004_im006._reconst_depth]|![JNet_597_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 4.536500000000001, MSE: 0.0009985835058614612, quantized loss: 0.00036970002111047506  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi005_im007._original_depth]|![JNet_597_pretrain_beads_roi005_im007._output_depth]|![JNet_597_pretrain_beads_roi005_im007._reconst_depth]|![JNet_597_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 4.406750000000001, MSE: 0.0009690735605545342, quantized loss: 0.0003687829594127834  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi006_im008._original_depth]|![JNet_597_pretrain_beads_roi006_im008._output_depth]|![JNet_597_pretrain_beads_roi006_im008._reconst_depth]|![JNet_597_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 4.587875000000001, MSE: 0.0009362038690596819, quantized loss: 0.0003931781102437526  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi007_im009._original_depth]|![JNet_597_pretrain_beads_roi007_im009._output_depth]|![JNet_597_pretrain_beads_roi007_im009._reconst_depth]|![JNet_597_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 4.226250000000001, MSE: 0.0009917810093611479, quantized loss: 0.0003480614104773849  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi008_im010._original_depth]|![JNet_597_pretrain_beads_roi008_im010._output_depth]|![JNet_597_pretrain_beads_roi008_im010._reconst_depth]|![JNet_597_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 4.641000000000001, MSE: 0.0009311377652920783, quantized loss: 0.00037685834104195237  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi009_im011._original_depth]|![JNet_597_pretrain_beads_roi009_im011._output_depth]|![JNet_597_pretrain_beads_roi009_im011._reconst_depth]|![JNet_597_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 4.306500000000001, MSE: 0.0009161629131995142, quantized loss: 0.0003530007670633495  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi010_im012._original_depth]|![JNet_597_pretrain_beads_roi010_im012._output_depth]|![JNet_597_pretrain_beads_roi010_im012._reconst_depth]|![JNet_597_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 5.243875000000001, MSE: 0.0009551256080158055, quantized loss: 0.0003900300944224  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi011_im013._original_depth]|![JNet_597_pretrain_beads_roi011_im013._output_depth]|![JNet_597_pretrain_beads_roi011_im013._reconst_depth]|![JNet_597_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 5.227125000000001, MSE: 0.0009419899433851242, quantized loss: 0.0003995627339463681  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi012_im014._original_depth]|![JNet_597_pretrain_beads_roi012_im014._output_depth]|![JNet_597_pretrain_beads_roi012_im014._reconst_depth]|![JNet_597_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 4.521250000000001, MSE: 0.0010675941593945026, quantized loss: 0.00039330654544755816  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi013_im015._original_depth]|![JNet_597_pretrain_beads_roi013_im015._output_depth]|![JNet_597_pretrain_beads_roi013_im015._reconst_depth]|![JNet_597_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 4.424500000000001, MSE: 0.0010047319810837507, quantized loss: 0.0003664142277557403  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi014_im016._original_depth]|![JNet_597_pretrain_beads_roi014_im016._output_depth]|![JNet_597_pretrain_beads_roi014_im016._reconst_depth]|![JNet_597_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 4.301500000000001, MSE: 0.0009326470899395645, quantized loss: 0.0004108362190891057  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi015_im017._original_depth]|![JNet_597_pretrain_beads_roi015_im017._output_depth]|![JNet_597_pretrain_beads_roi015_im017._reconst_depth]|![JNet_597_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 4.211000000000001, MSE: 0.0009755390346981585, quantized loss: 0.00036625665961764753  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi016_im018._original_depth]|![JNet_597_pretrain_beads_roi016_im018._output_depth]|![JNet_597_pretrain_beads_roi016_im018._reconst_depth]|![JNet_597_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 4.697375000000001, MSE: 0.001044120523147285, quantized loss: 0.00036999868461862206  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi017_im018._original_depth]|![JNet_597_pretrain_beads_roi017_im018._output_depth]|![JNet_597_pretrain_beads_roi017_im018._reconst_depth]|![JNet_597_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 4.728125000000001, MSE: 0.0010723553132265806, quantized loss: 0.0003716712526511401  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi018_im022._original_depth]|![JNet_597_pretrain_beads_roi018_im022._output_depth]|![JNet_597_pretrain_beads_roi018_im022._reconst_depth]|![JNet_597_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 4.174875000000001, MSE: 0.0009460900910198689, quantized loss: 0.0003515330608934164  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi019_im023._original_depth]|![JNet_597_pretrain_beads_roi019_im023._output_depth]|![JNet_597_pretrain_beads_roi019_im023._reconst_depth]|![JNet_597_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 3.956750000000001, MSE: 0.0009683142998255789, quantized loss: 0.0003267239371780306  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi020_im024._original_depth]|![JNet_597_pretrain_beads_roi020_im024._output_depth]|![JNet_597_pretrain_beads_roi020_im024._reconst_depth]|![JNet_597_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 4.873250000000001, MSE: 0.0009521874017082155, quantized loss: 0.0003674131876323372  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi021_im026._original_depth]|![JNet_597_pretrain_beads_roi021_im026._output_depth]|![JNet_597_pretrain_beads_roi021_im026._reconst_depth]|![JNet_597_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 4.593000000000001, MSE: 0.0008946407469920814, quantized loss: 0.00036757951602339745  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi022_im027._original_depth]|![JNet_597_pretrain_beads_roi022_im027._output_depth]|![JNet_597_pretrain_beads_roi022_im027._reconst_depth]|![JNet_597_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 4.362125000000001, MSE: 0.0009642113000154495, quantized loss: 0.0003491101961117238  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi023_im028._original_depth]|![JNet_597_pretrain_beads_roi023_im028._output_depth]|![JNet_597_pretrain_beads_roi023_im028._reconst_depth]|![JNet_597_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 5.009375000000001, MSE: 0.0007902628276497126, quantized loss: 0.00043022085446864367  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi024_im028._original_depth]|![JNet_597_pretrain_beads_roi024_im028._output_depth]|![JNet_597_pretrain_beads_roi024_im028._reconst_depth]|![JNet_597_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 4.895000000000001, MSE: 0.0008536102832295001, quantized loss: 0.00039264108636416495  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi025_im028._original_depth]|![JNet_597_pretrain_beads_roi025_im028._output_depth]|![JNet_597_pretrain_beads_roi025_im028._reconst_depth]|![JNet_597_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 4.895000000000001, MSE: 0.0008536102832295001, quantized loss: 0.00039264108636416495  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi026_im029._original_depth]|![JNet_597_pretrain_beads_roi026_im029._output_depth]|![JNet_597_pretrain_beads_roi026_im029._reconst_depth]|![JNet_597_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 4.934875000000001, MSE: 0.000983801786787808, quantized loss: 0.00039150906377471983  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi027_im029._original_depth]|![JNet_597_pretrain_beads_roi027_im029._output_depth]|![JNet_597_pretrain_beads_roi027_im029._reconst_depth]|![JNet_597_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 4.549375000000001, MSE: 0.0009670697618275881, quantized loss: 0.00035917741479352117  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi028_im030._original_depth]|![JNet_597_pretrain_beads_roi028_im030._output_depth]|![JNet_597_pretrain_beads_roi028_im030._reconst_depth]|![JNet_597_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 4.399000000000001, MSE: 0.0009344096761196852, quantized loss: 0.0003456615668255836  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_597_pretrain_beads_roi029_im030._original_depth]|![JNet_597_pretrain_beads_roi029_im030._output_depth]|![JNet_597_pretrain_beads_roi029_im030._reconst_depth]|![JNet_597_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 4.550625000000001, MSE: 0.0009735922794789076, quantized loss: 0.00035123960697092116  

### finetuning
  
volume mean: 3.471225, volume sd: 0.243432529153357
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi000_im000._original_depth]|![JNet_598_beads_roi000_im000._output_depth]|![JNet_598_beads_roi000_im000._reconst_depth]|![JNet_598_beads_roi000_im000._heatmap_depth]|
  
volume: 3.292125000000001, MSE: 0.0012614765437319875, quantized loss: 0.00036632470437325537  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi001_im004._original_depth]|![JNet_598_beads_roi001_im004._output_depth]|![JNet_598_beads_roi001_im004._reconst_depth]|![JNet_598_beads_roi001_im004._heatmap_depth]|
  
volume: 3.866250000000001, MSE: 0.0016507729887962341, quantized loss: 0.0004641650302801281  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi002_im005._original_depth]|![JNet_598_beads_roi002_im005._output_depth]|![JNet_598_beads_roi002_im005._reconst_depth]|![JNet_598_beads_roi002_im005._heatmap_depth]|
  
volume: 3.3211250000000008, MSE: 0.0013797618448734283, quantized loss: 0.0004197449015919119  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi003_im006._original_depth]|![JNet_598_beads_roi003_im006._output_depth]|![JNet_598_beads_roi003_im006._reconst_depth]|![JNet_598_beads_roi003_im006._heatmap_depth]|
  
volume: 3.4145000000000008, MSE: 0.0014104965375736356, quantized loss: 0.000432084605563432  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi004_im006._original_depth]|![JNet_598_beads_roi004_im006._output_depth]|![JNet_598_beads_roi004_im006._reconst_depth]|![JNet_598_beads_roi004_im006._heatmap_depth]|
  
volume: 3.456500000000001, MSE: 0.0014668204821646214, quantized loss: 0.00044843897921964526  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi005_im007._original_depth]|![JNet_598_beads_roi005_im007._output_depth]|![JNet_598_beads_roi005_im007._reconst_depth]|![JNet_598_beads_roi005_im007._heatmap_depth]|
  
volume: 3.360875000000001, MSE: 0.0014321543276309967, quantized loss: 0.0004168619343545288  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi006_im008._original_depth]|![JNet_598_beads_roi006_im008._output_depth]|![JNet_598_beads_roi006_im008._reconst_depth]|![JNet_598_beads_roi006_im008._heatmap_depth]|
  
volume: 3.512000000000001, MSE: 0.0014240300515666604, quantized loss: 0.0005732348072342575  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi007_im009._original_depth]|![JNet_598_beads_roi007_im009._output_depth]|![JNet_598_beads_roi007_im009._reconst_depth]|![JNet_598_beads_roi007_im009._heatmap_depth]|
  
volume: 3.404875000000001, MSE: 0.0013918596087023616, quantized loss: 0.0004980514640919864  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi008_im010._original_depth]|![JNet_598_beads_roi008_im010._output_depth]|![JNet_598_beads_roi008_im010._reconst_depth]|![JNet_598_beads_roi008_im010._heatmap_depth]|
  
volume: 3.636000000000001, MSE: 0.001481201034039259, quantized loss: 0.00041539347148500383  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi009_im011._original_depth]|![JNet_598_beads_roi009_im011._output_depth]|![JNet_598_beads_roi009_im011._reconst_depth]|![JNet_598_beads_roi009_im011._heatmap_depth]|
  
volume: 3.416500000000001, MSE: 0.001291246502660215, quantized loss: 0.0004074005119036883  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi010_im012._original_depth]|![JNet_598_beads_roi010_im012._output_depth]|![JNet_598_beads_roi010_im012._reconst_depth]|![JNet_598_beads_roi010_im012._heatmap_depth]|
  
volume: 3.9145000000000008, MSE: 0.0016521656652912498, quantized loss: 0.00042570664663799107  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi011_im013._original_depth]|![JNet_598_beads_roi011_im013._output_depth]|![JNet_598_beads_roi011_im013._reconst_depth]|![JNet_598_beads_roi011_im013._heatmap_depth]|
  
volume: 3.940375000000001, MSE: 0.0016263999277725816, quantized loss: 0.0004394929565023631  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi012_im014._original_depth]|![JNet_598_beads_roi012_im014._output_depth]|![JNet_598_beads_roi012_im014._reconst_depth]|![JNet_598_beads_roi012_im014._heatmap_depth]|
  
volume: 3.435625000000001, MSE: 0.001439135055989027, quantized loss: 0.0003833597293123603  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi013_im015._original_depth]|![JNet_598_beads_roi013_im015._output_depth]|![JNet_598_beads_roi013_im015._reconst_depth]|![JNet_598_beads_roi013_im015._heatmap_depth]|
  
volume: 3.298375000000001, MSE: 0.0013460068730637431, quantized loss: 0.00038993635098449886  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi014_im016._original_depth]|![JNet_598_beads_roi014_im016._output_depth]|![JNet_598_beads_roi014_im016._reconst_depth]|![JNet_598_beads_roi014_im016._heatmap_depth]|
  
volume: 3.1662500000000007, MSE: 0.0013877862365916371, quantized loss: 0.0004590250027831644  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi015_im017._original_depth]|![JNet_598_beads_roi015_im017._output_depth]|![JNet_598_beads_roi015_im017._reconst_depth]|![JNet_598_beads_roi015_im017._heatmap_depth]|
  
volume: 3.1801250000000008, MSE: 0.0012724364642053843, quantized loss: 0.0004120654775761068  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi016_im018._original_depth]|![JNet_598_beads_roi016_im018._output_depth]|![JNet_598_beads_roi016_im018._reconst_depth]|![JNet_598_beads_roi016_im018._heatmap_depth]|
  
volume: 3.633750000000001, MSE: 0.0014890778111293912, quantized loss: 0.000438092858530581  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi017_im018._original_depth]|![JNet_598_beads_roi017_im018._output_depth]|![JNet_598_beads_roi017_im018._reconst_depth]|![JNet_598_beads_roi017_im018._heatmap_depth]|
  
volume: 3.5713750000000006, MSE: 0.0014454121701419353, quantized loss: 0.00040031649405136704  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi018_im022._original_depth]|![JNet_598_beads_roi018_im022._output_depth]|![JNet_598_beads_roi018_im022._reconst_depth]|![JNet_598_beads_roi018_im022._heatmap_depth]|
  
volume: 2.9995000000000007, MSE: 0.0012255116598680615, quantized loss: 0.00035039224894717336  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi019_im023._original_depth]|![JNet_598_beads_roi019_im023._output_depth]|![JNet_598_beads_roi019_im023._reconst_depth]|![JNet_598_beads_roi019_im023._heatmap_depth]|
  
volume: 2.9206250000000007, MSE: 0.00114809675142169, quantized loss: 0.0003313541819807142  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi020_im024._original_depth]|![JNet_598_beads_roi020_im024._output_depth]|![JNet_598_beads_roi020_im024._reconst_depth]|![JNet_598_beads_roi020_im024._heatmap_depth]|
  
volume: 3.726375000000001, MSE: 0.0014718616148456931, quantized loss: 0.0003756382502615452  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi021_im026._original_depth]|![JNet_598_beads_roi021_im026._output_depth]|![JNet_598_beads_roi021_im026._reconst_depth]|![JNet_598_beads_roi021_im026._heatmap_depth]|
  
volume: 3.5351250000000007, MSE: 0.0014460874954238534, quantized loss: 0.00038034317549318075  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi022_im027._original_depth]|![JNet_598_beads_roi022_im027._output_depth]|![JNet_598_beads_roi022_im027._reconst_depth]|![JNet_598_beads_roi022_im027._heatmap_depth]|
  
volume: 3.417250000000001, MSE: 0.0013942663790658116, quantized loss: 0.00037691197940148413  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi023_im028._original_depth]|![JNet_598_beads_roi023_im028._output_depth]|![JNet_598_beads_roi023_im028._reconst_depth]|![JNet_598_beads_roi023_im028._heatmap_depth]|
  
volume: 3.7600000000000007, MSE: 0.0015693586319684982, quantized loss: 0.00044881278881803155  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi024_im028._original_depth]|![JNet_598_beads_roi024_im028._output_depth]|![JNet_598_beads_roi024_im028._reconst_depth]|![JNet_598_beads_roi024_im028._heatmap_depth]|
  
volume: 3.642500000000001, MSE: 0.0015422012656927109, quantized loss: 0.00039916631067171693  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi025_im028._original_depth]|![JNet_598_beads_roi025_im028._output_depth]|![JNet_598_beads_roi025_im028._reconst_depth]|![JNet_598_beads_roi025_im028._heatmap_depth]|
  
volume: 3.642500000000001, MSE: 0.0015422012656927109, quantized loss: 0.00039916631067171693  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi026_im029._original_depth]|![JNet_598_beads_roi026_im029._output_depth]|![JNet_598_beads_roi026_im029._reconst_depth]|![JNet_598_beads_roi026_im029._heatmap_depth]|
  
volume: 3.663125000000001, MSE: 0.0016066767275333405, quantized loss: 0.0003859812277369201  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi027_im029._original_depth]|![JNet_598_beads_roi027_im029._output_depth]|![JNet_598_beads_roi027_im029._reconst_depth]|![JNet_598_beads_roi027_im029._heatmap_depth]|
  
volume: 3.262000000000001, MSE: 0.0014461892424151301, quantized loss: 0.00036453359643928707  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi028_im030._original_depth]|![JNet_598_beads_roi028_im030._output_depth]|![JNet_598_beads_roi028_im030._reconst_depth]|![JNet_598_beads_roi028_im030._heatmap_depth]|
  
volume: 3.290375000000001, MSE: 0.0012960422318428755, quantized loss: 0.00035457464400678873  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_598_beads_roi029_im030._original_depth]|![JNet_598_beads_roi029_im030._output_depth]|![JNet_598_beads_roi029_im030._reconst_depth]|![JNet_598_beads_roi029_im030._heatmap_depth]|
  
volume: 3.4562500000000007, MSE: 0.0013805994531139731, quantized loss: 0.00036027340684086084  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_598_psf_pre]|![JNet_598_psf_post]|

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
  



[JNet_597_pretrain_0_aligned_depth]: /experiments/images/JNet_597_pretrain_0_aligned_depth.png
[JNet_597_pretrain_0_aligned_plane]: /experiments/images/JNet_597_pretrain_0_aligned_plane.png
[JNet_597_pretrain_0_labelx_depth]: /experiments/images/JNet_597_pretrain_0_labelx_depth.png
[JNet_597_pretrain_0_labelx_plane]: /experiments/images/JNet_597_pretrain_0_labelx_plane.png
[JNet_597_pretrain_0_labelz_depth]: /experiments/images/JNet_597_pretrain_0_labelz_depth.png
[JNet_597_pretrain_0_labelz_plane]: /experiments/images/JNet_597_pretrain_0_labelz_plane.png
[JNet_597_pretrain_0_novibrate_depth]: /experiments/images/JNet_597_pretrain_0_novibrate_depth.png
[JNet_597_pretrain_0_novibrate_plane]: /experiments/images/JNet_597_pretrain_0_novibrate_plane.png
[JNet_597_pretrain_0_original_depth]: /experiments/images/JNet_597_pretrain_0_original_depth.png
[JNet_597_pretrain_0_original_plane]: /experiments/images/JNet_597_pretrain_0_original_plane.png
[JNet_597_pretrain_0_outputx_depth]: /experiments/images/JNet_597_pretrain_0_outputx_depth.png
[JNet_597_pretrain_0_outputx_plane]: /experiments/images/JNet_597_pretrain_0_outputx_plane.png
[JNet_597_pretrain_0_outputz_depth]: /experiments/images/JNet_597_pretrain_0_outputz_depth.png
[JNet_597_pretrain_0_outputz_plane]: /experiments/images/JNet_597_pretrain_0_outputz_plane.png
[JNet_597_pretrain_1_aligned_depth]: /experiments/images/JNet_597_pretrain_1_aligned_depth.png
[JNet_597_pretrain_1_aligned_plane]: /experiments/images/JNet_597_pretrain_1_aligned_plane.png
[JNet_597_pretrain_1_labelx_depth]: /experiments/images/JNet_597_pretrain_1_labelx_depth.png
[JNet_597_pretrain_1_labelx_plane]: /experiments/images/JNet_597_pretrain_1_labelx_plane.png
[JNet_597_pretrain_1_labelz_depth]: /experiments/images/JNet_597_pretrain_1_labelz_depth.png
[JNet_597_pretrain_1_labelz_plane]: /experiments/images/JNet_597_pretrain_1_labelz_plane.png
[JNet_597_pretrain_1_novibrate_depth]: /experiments/images/JNet_597_pretrain_1_novibrate_depth.png
[JNet_597_pretrain_1_novibrate_plane]: /experiments/images/JNet_597_pretrain_1_novibrate_plane.png
[JNet_597_pretrain_1_original_depth]: /experiments/images/JNet_597_pretrain_1_original_depth.png
[JNet_597_pretrain_1_original_plane]: /experiments/images/JNet_597_pretrain_1_original_plane.png
[JNet_597_pretrain_1_outputx_depth]: /experiments/images/JNet_597_pretrain_1_outputx_depth.png
[JNet_597_pretrain_1_outputx_plane]: /experiments/images/JNet_597_pretrain_1_outputx_plane.png
[JNet_597_pretrain_1_outputz_depth]: /experiments/images/JNet_597_pretrain_1_outputz_depth.png
[JNet_597_pretrain_1_outputz_plane]: /experiments/images/JNet_597_pretrain_1_outputz_plane.png
[JNet_597_pretrain_2_aligned_depth]: /experiments/images/JNet_597_pretrain_2_aligned_depth.png
[JNet_597_pretrain_2_aligned_plane]: /experiments/images/JNet_597_pretrain_2_aligned_plane.png
[JNet_597_pretrain_2_labelx_depth]: /experiments/images/JNet_597_pretrain_2_labelx_depth.png
[JNet_597_pretrain_2_labelx_plane]: /experiments/images/JNet_597_pretrain_2_labelx_plane.png
[JNet_597_pretrain_2_labelz_depth]: /experiments/images/JNet_597_pretrain_2_labelz_depth.png
[JNet_597_pretrain_2_labelz_plane]: /experiments/images/JNet_597_pretrain_2_labelz_plane.png
[JNet_597_pretrain_2_novibrate_depth]: /experiments/images/JNet_597_pretrain_2_novibrate_depth.png
[JNet_597_pretrain_2_novibrate_plane]: /experiments/images/JNet_597_pretrain_2_novibrate_plane.png
[JNet_597_pretrain_2_original_depth]: /experiments/images/JNet_597_pretrain_2_original_depth.png
[JNet_597_pretrain_2_original_plane]: /experiments/images/JNet_597_pretrain_2_original_plane.png
[JNet_597_pretrain_2_outputx_depth]: /experiments/images/JNet_597_pretrain_2_outputx_depth.png
[JNet_597_pretrain_2_outputx_plane]: /experiments/images/JNet_597_pretrain_2_outputx_plane.png
[JNet_597_pretrain_2_outputz_depth]: /experiments/images/JNet_597_pretrain_2_outputz_depth.png
[JNet_597_pretrain_2_outputz_plane]: /experiments/images/JNet_597_pretrain_2_outputz_plane.png
[JNet_597_pretrain_3_aligned_depth]: /experiments/images/JNet_597_pretrain_3_aligned_depth.png
[JNet_597_pretrain_3_aligned_plane]: /experiments/images/JNet_597_pretrain_3_aligned_plane.png
[JNet_597_pretrain_3_labelx_depth]: /experiments/images/JNet_597_pretrain_3_labelx_depth.png
[JNet_597_pretrain_3_labelx_plane]: /experiments/images/JNet_597_pretrain_3_labelx_plane.png
[JNet_597_pretrain_3_labelz_depth]: /experiments/images/JNet_597_pretrain_3_labelz_depth.png
[JNet_597_pretrain_3_labelz_plane]: /experiments/images/JNet_597_pretrain_3_labelz_plane.png
[JNet_597_pretrain_3_novibrate_depth]: /experiments/images/JNet_597_pretrain_3_novibrate_depth.png
[JNet_597_pretrain_3_novibrate_plane]: /experiments/images/JNet_597_pretrain_3_novibrate_plane.png
[JNet_597_pretrain_3_original_depth]: /experiments/images/JNet_597_pretrain_3_original_depth.png
[JNet_597_pretrain_3_original_plane]: /experiments/images/JNet_597_pretrain_3_original_plane.png
[JNet_597_pretrain_3_outputx_depth]: /experiments/images/JNet_597_pretrain_3_outputx_depth.png
[JNet_597_pretrain_3_outputx_plane]: /experiments/images/JNet_597_pretrain_3_outputx_plane.png
[JNet_597_pretrain_3_outputz_depth]: /experiments/images/JNet_597_pretrain_3_outputz_depth.png
[JNet_597_pretrain_3_outputz_plane]: /experiments/images/JNet_597_pretrain_3_outputz_plane.png
[JNet_597_pretrain_4_aligned_depth]: /experiments/images/JNet_597_pretrain_4_aligned_depth.png
[JNet_597_pretrain_4_aligned_plane]: /experiments/images/JNet_597_pretrain_4_aligned_plane.png
[JNet_597_pretrain_4_labelx_depth]: /experiments/images/JNet_597_pretrain_4_labelx_depth.png
[JNet_597_pretrain_4_labelx_plane]: /experiments/images/JNet_597_pretrain_4_labelx_plane.png
[JNet_597_pretrain_4_labelz_depth]: /experiments/images/JNet_597_pretrain_4_labelz_depth.png
[JNet_597_pretrain_4_labelz_plane]: /experiments/images/JNet_597_pretrain_4_labelz_plane.png
[JNet_597_pretrain_4_novibrate_depth]: /experiments/images/JNet_597_pretrain_4_novibrate_depth.png
[JNet_597_pretrain_4_novibrate_plane]: /experiments/images/JNet_597_pretrain_4_novibrate_plane.png
[JNet_597_pretrain_4_original_depth]: /experiments/images/JNet_597_pretrain_4_original_depth.png
[JNet_597_pretrain_4_original_plane]: /experiments/images/JNet_597_pretrain_4_original_plane.png
[JNet_597_pretrain_4_outputx_depth]: /experiments/images/JNet_597_pretrain_4_outputx_depth.png
[JNet_597_pretrain_4_outputx_plane]: /experiments/images/JNet_597_pretrain_4_outputx_plane.png
[JNet_597_pretrain_4_outputz_depth]: /experiments/images/JNet_597_pretrain_4_outputz_depth.png
[JNet_597_pretrain_4_outputz_plane]: /experiments/images/JNet_597_pretrain_4_outputz_plane.png
[JNet_597_pretrain_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi000_im000._heatmap_depth.png
[JNet_597_pretrain_beads_roi000_im000._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi000_im000._original_depth.png
[JNet_597_pretrain_beads_roi000_im000._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi000_im000._output_depth.png
[JNet_597_pretrain_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi000_im000._reconst_depth.png
[JNet_597_pretrain_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi001_im004._heatmap_depth.png
[JNet_597_pretrain_beads_roi001_im004._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi001_im004._original_depth.png
[JNet_597_pretrain_beads_roi001_im004._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi001_im004._output_depth.png
[JNet_597_pretrain_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi001_im004._reconst_depth.png
[JNet_597_pretrain_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi002_im005._heatmap_depth.png
[JNet_597_pretrain_beads_roi002_im005._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi002_im005._original_depth.png
[JNet_597_pretrain_beads_roi002_im005._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi002_im005._output_depth.png
[JNet_597_pretrain_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi002_im005._reconst_depth.png
[JNet_597_pretrain_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi003_im006._heatmap_depth.png
[JNet_597_pretrain_beads_roi003_im006._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi003_im006._original_depth.png
[JNet_597_pretrain_beads_roi003_im006._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi003_im006._output_depth.png
[JNet_597_pretrain_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi003_im006._reconst_depth.png
[JNet_597_pretrain_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi004_im006._heatmap_depth.png
[JNet_597_pretrain_beads_roi004_im006._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi004_im006._original_depth.png
[JNet_597_pretrain_beads_roi004_im006._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi004_im006._output_depth.png
[JNet_597_pretrain_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi004_im006._reconst_depth.png
[JNet_597_pretrain_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi005_im007._heatmap_depth.png
[JNet_597_pretrain_beads_roi005_im007._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi005_im007._original_depth.png
[JNet_597_pretrain_beads_roi005_im007._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi005_im007._output_depth.png
[JNet_597_pretrain_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi005_im007._reconst_depth.png
[JNet_597_pretrain_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi006_im008._heatmap_depth.png
[JNet_597_pretrain_beads_roi006_im008._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi006_im008._original_depth.png
[JNet_597_pretrain_beads_roi006_im008._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi006_im008._output_depth.png
[JNet_597_pretrain_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi006_im008._reconst_depth.png
[JNet_597_pretrain_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi007_im009._heatmap_depth.png
[JNet_597_pretrain_beads_roi007_im009._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi007_im009._original_depth.png
[JNet_597_pretrain_beads_roi007_im009._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi007_im009._output_depth.png
[JNet_597_pretrain_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi007_im009._reconst_depth.png
[JNet_597_pretrain_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi008_im010._heatmap_depth.png
[JNet_597_pretrain_beads_roi008_im010._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi008_im010._original_depth.png
[JNet_597_pretrain_beads_roi008_im010._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi008_im010._output_depth.png
[JNet_597_pretrain_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi008_im010._reconst_depth.png
[JNet_597_pretrain_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi009_im011._heatmap_depth.png
[JNet_597_pretrain_beads_roi009_im011._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi009_im011._original_depth.png
[JNet_597_pretrain_beads_roi009_im011._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi009_im011._output_depth.png
[JNet_597_pretrain_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi009_im011._reconst_depth.png
[JNet_597_pretrain_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi010_im012._heatmap_depth.png
[JNet_597_pretrain_beads_roi010_im012._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi010_im012._original_depth.png
[JNet_597_pretrain_beads_roi010_im012._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi010_im012._output_depth.png
[JNet_597_pretrain_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi010_im012._reconst_depth.png
[JNet_597_pretrain_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi011_im013._heatmap_depth.png
[JNet_597_pretrain_beads_roi011_im013._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi011_im013._original_depth.png
[JNet_597_pretrain_beads_roi011_im013._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi011_im013._output_depth.png
[JNet_597_pretrain_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi011_im013._reconst_depth.png
[JNet_597_pretrain_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi012_im014._heatmap_depth.png
[JNet_597_pretrain_beads_roi012_im014._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi012_im014._original_depth.png
[JNet_597_pretrain_beads_roi012_im014._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi012_im014._output_depth.png
[JNet_597_pretrain_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi012_im014._reconst_depth.png
[JNet_597_pretrain_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi013_im015._heatmap_depth.png
[JNet_597_pretrain_beads_roi013_im015._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi013_im015._original_depth.png
[JNet_597_pretrain_beads_roi013_im015._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi013_im015._output_depth.png
[JNet_597_pretrain_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi013_im015._reconst_depth.png
[JNet_597_pretrain_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi014_im016._heatmap_depth.png
[JNet_597_pretrain_beads_roi014_im016._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi014_im016._original_depth.png
[JNet_597_pretrain_beads_roi014_im016._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi014_im016._output_depth.png
[JNet_597_pretrain_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi014_im016._reconst_depth.png
[JNet_597_pretrain_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi015_im017._heatmap_depth.png
[JNet_597_pretrain_beads_roi015_im017._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi015_im017._original_depth.png
[JNet_597_pretrain_beads_roi015_im017._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi015_im017._output_depth.png
[JNet_597_pretrain_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi015_im017._reconst_depth.png
[JNet_597_pretrain_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi016_im018._heatmap_depth.png
[JNet_597_pretrain_beads_roi016_im018._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi016_im018._original_depth.png
[JNet_597_pretrain_beads_roi016_im018._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi016_im018._output_depth.png
[JNet_597_pretrain_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi016_im018._reconst_depth.png
[JNet_597_pretrain_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi017_im018._heatmap_depth.png
[JNet_597_pretrain_beads_roi017_im018._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi017_im018._original_depth.png
[JNet_597_pretrain_beads_roi017_im018._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi017_im018._output_depth.png
[JNet_597_pretrain_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi017_im018._reconst_depth.png
[JNet_597_pretrain_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi018_im022._heatmap_depth.png
[JNet_597_pretrain_beads_roi018_im022._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi018_im022._original_depth.png
[JNet_597_pretrain_beads_roi018_im022._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi018_im022._output_depth.png
[JNet_597_pretrain_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi018_im022._reconst_depth.png
[JNet_597_pretrain_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi019_im023._heatmap_depth.png
[JNet_597_pretrain_beads_roi019_im023._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi019_im023._original_depth.png
[JNet_597_pretrain_beads_roi019_im023._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi019_im023._output_depth.png
[JNet_597_pretrain_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi019_im023._reconst_depth.png
[JNet_597_pretrain_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi020_im024._heatmap_depth.png
[JNet_597_pretrain_beads_roi020_im024._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi020_im024._original_depth.png
[JNet_597_pretrain_beads_roi020_im024._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi020_im024._output_depth.png
[JNet_597_pretrain_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi020_im024._reconst_depth.png
[JNet_597_pretrain_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi021_im026._heatmap_depth.png
[JNet_597_pretrain_beads_roi021_im026._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi021_im026._original_depth.png
[JNet_597_pretrain_beads_roi021_im026._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi021_im026._output_depth.png
[JNet_597_pretrain_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi021_im026._reconst_depth.png
[JNet_597_pretrain_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi022_im027._heatmap_depth.png
[JNet_597_pretrain_beads_roi022_im027._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi022_im027._original_depth.png
[JNet_597_pretrain_beads_roi022_im027._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi022_im027._output_depth.png
[JNet_597_pretrain_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi022_im027._reconst_depth.png
[JNet_597_pretrain_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi023_im028._heatmap_depth.png
[JNet_597_pretrain_beads_roi023_im028._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi023_im028._original_depth.png
[JNet_597_pretrain_beads_roi023_im028._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi023_im028._output_depth.png
[JNet_597_pretrain_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi023_im028._reconst_depth.png
[JNet_597_pretrain_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi024_im028._heatmap_depth.png
[JNet_597_pretrain_beads_roi024_im028._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi024_im028._original_depth.png
[JNet_597_pretrain_beads_roi024_im028._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi024_im028._output_depth.png
[JNet_597_pretrain_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi024_im028._reconst_depth.png
[JNet_597_pretrain_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi025_im028._heatmap_depth.png
[JNet_597_pretrain_beads_roi025_im028._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi025_im028._original_depth.png
[JNet_597_pretrain_beads_roi025_im028._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi025_im028._output_depth.png
[JNet_597_pretrain_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi025_im028._reconst_depth.png
[JNet_597_pretrain_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi026_im029._heatmap_depth.png
[JNet_597_pretrain_beads_roi026_im029._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi026_im029._original_depth.png
[JNet_597_pretrain_beads_roi026_im029._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi026_im029._output_depth.png
[JNet_597_pretrain_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi026_im029._reconst_depth.png
[JNet_597_pretrain_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi027_im029._heatmap_depth.png
[JNet_597_pretrain_beads_roi027_im029._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi027_im029._original_depth.png
[JNet_597_pretrain_beads_roi027_im029._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi027_im029._output_depth.png
[JNet_597_pretrain_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi027_im029._reconst_depth.png
[JNet_597_pretrain_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi028_im030._heatmap_depth.png
[JNet_597_pretrain_beads_roi028_im030._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi028_im030._original_depth.png
[JNet_597_pretrain_beads_roi028_im030._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi028_im030._output_depth.png
[JNet_597_pretrain_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi028_im030._reconst_depth.png
[JNet_597_pretrain_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_597_pretrain_beads_roi029_im030._heatmap_depth.png
[JNet_597_pretrain_beads_roi029_im030._original_depth]: /experiments/images/JNet_597_pretrain_beads_roi029_im030._original_depth.png
[JNet_597_pretrain_beads_roi029_im030._output_depth]: /experiments/images/JNet_597_pretrain_beads_roi029_im030._output_depth.png
[JNet_597_pretrain_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_597_pretrain_beads_roi029_im030._reconst_depth.png
[JNet_598_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_598_beads_roi000_im000._heatmap_depth.png
[JNet_598_beads_roi000_im000._original_depth]: /experiments/images/JNet_598_beads_roi000_im000._original_depth.png
[JNet_598_beads_roi000_im000._output_depth]: /experiments/images/JNet_598_beads_roi000_im000._output_depth.png
[JNet_598_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_598_beads_roi000_im000._reconst_depth.png
[JNet_598_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_598_beads_roi001_im004._heatmap_depth.png
[JNet_598_beads_roi001_im004._original_depth]: /experiments/images/JNet_598_beads_roi001_im004._original_depth.png
[JNet_598_beads_roi001_im004._output_depth]: /experiments/images/JNet_598_beads_roi001_im004._output_depth.png
[JNet_598_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_598_beads_roi001_im004._reconst_depth.png
[JNet_598_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_598_beads_roi002_im005._heatmap_depth.png
[JNet_598_beads_roi002_im005._original_depth]: /experiments/images/JNet_598_beads_roi002_im005._original_depth.png
[JNet_598_beads_roi002_im005._output_depth]: /experiments/images/JNet_598_beads_roi002_im005._output_depth.png
[JNet_598_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_598_beads_roi002_im005._reconst_depth.png
[JNet_598_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_598_beads_roi003_im006._heatmap_depth.png
[JNet_598_beads_roi003_im006._original_depth]: /experiments/images/JNet_598_beads_roi003_im006._original_depth.png
[JNet_598_beads_roi003_im006._output_depth]: /experiments/images/JNet_598_beads_roi003_im006._output_depth.png
[JNet_598_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_598_beads_roi003_im006._reconst_depth.png
[JNet_598_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_598_beads_roi004_im006._heatmap_depth.png
[JNet_598_beads_roi004_im006._original_depth]: /experiments/images/JNet_598_beads_roi004_im006._original_depth.png
[JNet_598_beads_roi004_im006._output_depth]: /experiments/images/JNet_598_beads_roi004_im006._output_depth.png
[JNet_598_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_598_beads_roi004_im006._reconst_depth.png
[JNet_598_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_598_beads_roi005_im007._heatmap_depth.png
[JNet_598_beads_roi005_im007._original_depth]: /experiments/images/JNet_598_beads_roi005_im007._original_depth.png
[JNet_598_beads_roi005_im007._output_depth]: /experiments/images/JNet_598_beads_roi005_im007._output_depth.png
[JNet_598_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_598_beads_roi005_im007._reconst_depth.png
[JNet_598_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_598_beads_roi006_im008._heatmap_depth.png
[JNet_598_beads_roi006_im008._original_depth]: /experiments/images/JNet_598_beads_roi006_im008._original_depth.png
[JNet_598_beads_roi006_im008._output_depth]: /experiments/images/JNet_598_beads_roi006_im008._output_depth.png
[JNet_598_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_598_beads_roi006_im008._reconst_depth.png
[JNet_598_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_598_beads_roi007_im009._heatmap_depth.png
[JNet_598_beads_roi007_im009._original_depth]: /experiments/images/JNet_598_beads_roi007_im009._original_depth.png
[JNet_598_beads_roi007_im009._output_depth]: /experiments/images/JNet_598_beads_roi007_im009._output_depth.png
[JNet_598_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_598_beads_roi007_im009._reconst_depth.png
[JNet_598_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_598_beads_roi008_im010._heatmap_depth.png
[JNet_598_beads_roi008_im010._original_depth]: /experiments/images/JNet_598_beads_roi008_im010._original_depth.png
[JNet_598_beads_roi008_im010._output_depth]: /experiments/images/JNet_598_beads_roi008_im010._output_depth.png
[JNet_598_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_598_beads_roi008_im010._reconst_depth.png
[JNet_598_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_598_beads_roi009_im011._heatmap_depth.png
[JNet_598_beads_roi009_im011._original_depth]: /experiments/images/JNet_598_beads_roi009_im011._original_depth.png
[JNet_598_beads_roi009_im011._output_depth]: /experiments/images/JNet_598_beads_roi009_im011._output_depth.png
[JNet_598_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_598_beads_roi009_im011._reconst_depth.png
[JNet_598_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_598_beads_roi010_im012._heatmap_depth.png
[JNet_598_beads_roi010_im012._original_depth]: /experiments/images/JNet_598_beads_roi010_im012._original_depth.png
[JNet_598_beads_roi010_im012._output_depth]: /experiments/images/JNet_598_beads_roi010_im012._output_depth.png
[JNet_598_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_598_beads_roi010_im012._reconst_depth.png
[JNet_598_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_598_beads_roi011_im013._heatmap_depth.png
[JNet_598_beads_roi011_im013._original_depth]: /experiments/images/JNet_598_beads_roi011_im013._original_depth.png
[JNet_598_beads_roi011_im013._output_depth]: /experiments/images/JNet_598_beads_roi011_im013._output_depth.png
[JNet_598_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_598_beads_roi011_im013._reconst_depth.png
[JNet_598_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_598_beads_roi012_im014._heatmap_depth.png
[JNet_598_beads_roi012_im014._original_depth]: /experiments/images/JNet_598_beads_roi012_im014._original_depth.png
[JNet_598_beads_roi012_im014._output_depth]: /experiments/images/JNet_598_beads_roi012_im014._output_depth.png
[JNet_598_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_598_beads_roi012_im014._reconst_depth.png
[JNet_598_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_598_beads_roi013_im015._heatmap_depth.png
[JNet_598_beads_roi013_im015._original_depth]: /experiments/images/JNet_598_beads_roi013_im015._original_depth.png
[JNet_598_beads_roi013_im015._output_depth]: /experiments/images/JNet_598_beads_roi013_im015._output_depth.png
[JNet_598_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_598_beads_roi013_im015._reconst_depth.png
[JNet_598_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_598_beads_roi014_im016._heatmap_depth.png
[JNet_598_beads_roi014_im016._original_depth]: /experiments/images/JNet_598_beads_roi014_im016._original_depth.png
[JNet_598_beads_roi014_im016._output_depth]: /experiments/images/JNet_598_beads_roi014_im016._output_depth.png
[JNet_598_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_598_beads_roi014_im016._reconst_depth.png
[JNet_598_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_598_beads_roi015_im017._heatmap_depth.png
[JNet_598_beads_roi015_im017._original_depth]: /experiments/images/JNet_598_beads_roi015_im017._original_depth.png
[JNet_598_beads_roi015_im017._output_depth]: /experiments/images/JNet_598_beads_roi015_im017._output_depth.png
[JNet_598_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_598_beads_roi015_im017._reconst_depth.png
[JNet_598_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_598_beads_roi016_im018._heatmap_depth.png
[JNet_598_beads_roi016_im018._original_depth]: /experiments/images/JNet_598_beads_roi016_im018._original_depth.png
[JNet_598_beads_roi016_im018._output_depth]: /experiments/images/JNet_598_beads_roi016_im018._output_depth.png
[JNet_598_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_598_beads_roi016_im018._reconst_depth.png
[JNet_598_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_598_beads_roi017_im018._heatmap_depth.png
[JNet_598_beads_roi017_im018._original_depth]: /experiments/images/JNet_598_beads_roi017_im018._original_depth.png
[JNet_598_beads_roi017_im018._output_depth]: /experiments/images/JNet_598_beads_roi017_im018._output_depth.png
[JNet_598_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_598_beads_roi017_im018._reconst_depth.png
[JNet_598_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_598_beads_roi018_im022._heatmap_depth.png
[JNet_598_beads_roi018_im022._original_depth]: /experiments/images/JNet_598_beads_roi018_im022._original_depth.png
[JNet_598_beads_roi018_im022._output_depth]: /experiments/images/JNet_598_beads_roi018_im022._output_depth.png
[JNet_598_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_598_beads_roi018_im022._reconst_depth.png
[JNet_598_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_598_beads_roi019_im023._heatmap_depth.png
[JNet_598_beads_roi019_im023._original_depth]: /experiments/images/JNet_598_beads_roi019_im023._original_depth.png
[JNet_598_beads_roi019_im023._output_depth]: /experiments/images/JNet_598_beads_roi019_im023._output_depth.png
[JNet_598_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_598_beads_roi019_im023._reconst_depth.png
[JNet_598_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_598_beads_roi020_im024._heatmap_depth.png
[JNet_598_beads_roi020_im024._original_depth]: /experiments/images/JNet_598_beads_roi020_im024._original_depth.png
[JNet_598_beads_roi020_im024._output_depth]: /experiments/images/JNet_598_beads_roi020_im024._output_depth.png
[JNet_598_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_598_beads_roi020_im024._reconst_depth.png
[JNet_598_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_598_beads_roi021_im026._heatmap_depth.png
[JNet_598_beads_roi021_im026._original_depth]: /experiments/images/JNet_598_beads_roi021_im026._original_depth.png
[JNet_598_beads_roi021_im026._output_depth]: /experiments/images/JNet_598_beads_roi021_im026._output_depth.png
[JNet_598_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_598_beads_roi021_im026._reconst_depth.png
[JNet_598_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_598_beads_roi022_im027._heatmap_depth.png
[JNet_598_beads_roi022_im027._original_depth]: /experiments/images/JNet_598_beads_roi022_im027._original_depth.png
[JNet_598_beads_roi022_im027._output_depth]: /experiments/images/JNet_598_beads_roi022_im027._output_depth.png
[JNet_598_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_598_beads_roi022_im027._reconst_depth.png
[JNet_598_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_598_beads_roi023_im028._heatmap_depth.png
[JNet_598_beads_roi023_im028._original_depth]: /experiments/images/JNet_598_beads_roi023_im028._original_depth.png
[JNet_598_beads_roi023_im028._output_depth]: /experiments/images/JNet_598_beads_roi023_im028._output_depth.png
[JNet_598_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_598_beads_roi023_im028._reconst_depth.png
[JNet_598_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_598_beads_roi024_im028._heatmap_depth.png
[JNet_598_beads_roi024_im028._original_depth]: /experiments/images/JNet_598_beads_roi024_im028._original_depth.png
[JNet_598_beads_roi024_im028._output_depth]: /experiments/images/JNet_598_beads_roi024_im028._output_depth.png
[JNet_598_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_598_beads_roi024_im028._reconst_depth.png
[JNet_598_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_598_beads_roi025_im028._heatmap_depth.png
[JNet_598_beads_roi025_im028._original_depth]: /experiments/images/JNet_598_beads_roi025_im028._original_depth.png
[JNet_598_beads_roi025_im028._output_depth]: /experiments/images/JNet_598_beads_roi025_im028._output_depth.png
[JNet_598_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_598_beads_roi025_im028._reconst_depth.png
[JNet_598_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_598_beads_roi026_im029._heatmap_depth.png
[JNet_598_beads_roi026_im029._original_depth]: /experiments/images/JNet_598_beads_roi026_im029._original_depth.png
[JNet_598_beads_roi026_im029._output_depth]: /experiments/images/JNet_598_beads_roi026_im029._output_depth.png
[JNet_598_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_598_beads_roi026_im029._reconst_depth.png
[JNet_598_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_598_beads_roi027_im029._heatmap_depth.png
[JNet_598_beads_roi027_im029._original_depth]: /experiments/images/JNet_598_beads_roi027_im029._original_depth.png
[JNet_598_beads_roi027_im029._output_depth]: /experiments/images/JNet_598_beads_roi027_im029._output_depth.png
[JNet_598_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_598_beads_roi027_im029._reconst_depth.png
[JNet_598_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_598_beads_roi028_im030._heatmap_depth.png
[JNet_598_beads_roi028_im030._original_depth]: /experiments/images/JNet_598_beads_roi028_im030._original_depth.png
[JNet_598_beads_roi028_im030._output_depth]: /experiments/images/JNet_598_beads_roi028_im030._output_depth.png
[JNet_598_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_598_beads_roi028_im030._reconst_depth.png
[JNet_598_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_598_beads_roi029_im030._heatmap_depth.png
[JNet_598_beads_roi029_im030._original_depth]: /experiments/images/JNet_598_beads_roi029_im030._original_depth.png
[JNet_598_beads_roi029_im030._output_depth]: /experiments/images/JNet_598_beads_roi029_im030._output_depth.png
[JNet_598_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_598_beads_roi029_im030._reconst_depth.png
[JNet_598_psf_post]: /experiments/images/JNet_598_psf_post.png
[JNet_598_psf_pre]: /experiments/images/JNet_598_psf_pre.png
