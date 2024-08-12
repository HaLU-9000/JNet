



# JNet_585 Report
  
low psf loss to avoid parameter from collapsing  
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
|ploss_weight|1.0|
|mrfloss_order|1|
|mrfloss_dilation|1|
|mrfloss_weights|{'l_00': 0, 'l_01': 0, 'l_10': 0, 'l_11': 0}|

## Results

### Pretraining
  
Segmentation: mean MSE: 0.006667309906333685, mean BCE: 0.02526364102959633  
Luminance Estimation: mean MSE: 0.9831247329711914, mean BCE: inf
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_0_original_plane]|![JNet_584_pretrain_0_novibrate_plane]|![JNet_584_pretrain_0_aligned_plane]|![JNet_584_pretrain_0_outputx_plane]|![JNet_584_pretrain_0_labelx_plane]|![JNet_584_pretrain_0_outputz_plane]|![JNet_584_pretrain_0_labelz_plane]|
  
MSEx: 0.006033070385456085, BCEx: 0.021983161568641663  
MSEz: 0.984930157661438, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_0_original_depth]|![JNet_584_pretrain_0_novibrate_depth]|![JNet_584_pretrain_0_aligned_depth]|![JNet_584_pretrain_0_outputx_depth]|![JNet_584_pretrain_0_labelx_depth]|![JNet_584_pretrain_0_outputz_depth]|![JNet_584_pretrain_0_labelz_depth]|
  
MSEx: 0.006033070385456085, BCEx: 0.021983161568641663  
MSEz: 0.984930157661438, BCEz: inf  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_1_original_plane]|![JNet_584_pretrain_1_novibrate_plane]|![JNet_584_pretrain_1_aligned_plane]|![JNet_584_pretrain_1_outputx_plane]|![JNet_584_pretrain_1_labelx_plane]|![JNet_584_pretrain_1_outputz_plane]|![JNet_584_pretrain_1_labelz_plane]|
  
MSEx: 0.007489303592592478, BCEx: 0.02845052070915699  
MSEz: 0.9852712750434875, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_1_original_depth]|![JNet_584_pretrain_1_novibrate_depth]|![JNet_584_pretrain_1_aligned_depth]|![JNet_584_pretrain_1_outputx_depth]|![JNet_584_pretrain_1_labelx_depth]|![JNet_584_pretrain_1_outputz_depth]|![JNet_584_pretrain_1_labelz_depth]|
  
MSEx: 0.007489303592592478, BCEx: 0.02845052070915699  
MSEz: 0.9852712750434875, BCEz: inf  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_2_original_plane]|![JNet_584_pretrain_2_novibrate_plane]|![JNet_584_pretrain_2_aligned_plane]|![JNet_584_pretrain_2_outputx_plane]|![JNet_584_pretrain_2_labelx_plane]|![JNet_584_pretrain_2_outputz_plane]|![JNet_584_pretrain_2_labelz_plane]|
  
MSEx: 0.004780733957886696, BCEx: 0.01887725107371807  
MSEz: 0.9924774765968323, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_2_original_depth]|![JNet_584_pretrain_2_novibrate_depth]|![JNet_584_pretrain_2_aligned_depth]|![JNet_584_pretrain_2_outputx_depth]|![JNet_584_pretrain_2_labelx_depth]|![JNet_584_pretrain_2_outputz_depth]|![JNet_584_pretrain_2_labelz_depth]|
  
MSEx: 0.004780733957886696, BCEx: 0.01887725107371807  
MSEz: 0.9924774765968323, BCEz: inf  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_3_original_plane]|![JNet_584_pretrain_3_novibrate_plane]|![JNet_584_pretrain_3_aligned_plane]|![JNet_584_pretrain_3_outputx_plane]|![JNet_584_pretrain_3_labelx_plane]|![JNet_584_pretrain_3_outputz_plane]|![JNet_584_pretrain_3_labelz_plane]|
  
MSEx: 0.006210044492036104, BCEx: 0.022941095754504204  
MSEz: 0.9841076135635376, BCEz: 8.885332107543945  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_3_original_depth]|![JNet_584_pretrain_3_novibrate_depth]|![JNet_584_pretrain_3_aligned_depth]|![JNet_584_pretrain_3_outputx_depth]|![JNet_584_pretrain_3_labelx_depth]|![JNet_584_pretrain_3_outputz_depth]|![JNet_584_pretrain_3_labelz_depth]|
  
MSEx: 0.006210044492036104, BCEx: 0.022941095754504204  
MSEz: 0.9841076135635376, BCEz: 8.885332107543945  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_4_original_plane]|![JNet_584_pretrain_4_novibrate_plane]|![JNet_584_pretrain_4_aligned_plane]|![JNet_584_pretrain_4_outputx_plane]|![JNet_584_pretrain_4_labelx_plane]|![JNet_584_pretrain_4_outputz_plane]|![JNet_584_pretrain_4_labelz_plane]|
  
MSEx: 0.008823398500680923, BCEx: 0.03406617045402527  
MSEz: 0.9688370227813721, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_584_pretrain_4_original_depth]|![JNet_584_pretrain_4_novibrate_depth]|![JNet_584_pretrain_4_aligned_depth]|![JNet_584_pretrain_4_outputx_depth]|![JNet_584_pretrain_4_labelx_depth]|![JNet_584_pretrain_4_outputz_depth]|![JNet_584_pretrain_4_labelz_depth]|
  
MSEx: 0.008823398500680923, BCEx: 0.03406617045402527  
MSEz: 0.9688370227813721, BCEz: inf  

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
  
volume mean: 3.1675466064453133, volume sd: 0.2634175280393984
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi000_im000._original_depth]|![JNet_585_beads_roi000_im000._output_depth]|![JNet_585_beads_roi000_im000._reconst_depth]|![JNet_585_beads_roi000_im000._heatmap_depth]|
  
volume: 3.0079855957031256, MSE: 0.0015050872461870313, quantized loss: 0.00017461880634073168  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi001_im004._original_depth]|![JNet_585_beads_roi001_im004._output_depth]|![JNet_585_beads_roi001_im004._reconst_depth]|![JNet_585_beads_roi001_im004._heatmap_depth]|
  
volume: 3.550872558593751, MSE: 0.0020885278936475515, quantized loss: 0.0002477241796441376  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi002_im005._original_depth]|![JNet_585_beads_roi002_im005._output_depth]|![JNet_585_beads_roi002_im005._reconst_depth]|![JNet_585_beads_roi002_im005._heatmap_depth]|
  
volume: 3.203018310546876, MSE: 0.0016961217625066638, quantized loss: 0.00024194101570174098  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi003_im006._original_depth]|![JNet_585_beads_roi003_im006._output_depth]|![JNet_585_beads_roi003_im006._reconst_depth]|![JNet_585_beads_roi003_im006._heatmap_depth]|
  
volume: 3.1988515625000007, MSE: 0.0016853571869432926, quantized loss: 0.00021257309708744287  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi004_im006._original_depth]|![JNet_585_beads_roi004_im006._output_depth]|![JNet_585_beads_roi004_im006._reconst_depth]|![JNet_585_beads_roi004_im006._heatmap_depth]|
  
volume: 3.3401672363281256, MSE: 0.0017101281555369496, quantized loss: 0.00022299271950032562  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi005_im007._original_depth]|![JNet_585_beads_roi005_im007._output_depth]|![JNet_585_beads_roi005_im007._reconst_depth]|![JNet_585_beads_roi005_im007._heatmap_depth]|
  
volume: 3.0866606445312508, MSE: 0.0017624521860852838, quantized loss: 0.00020515572396107018  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi006_im008._original_depth]|![JNet_585_beads_roi006_im008._output_depth]|![JNet_585_beads_roi006_im008._reconst_depth]|![JNet_585_beads_roi006_im008._heatmap_depth]|
  
volume: 3.4788522949218756, MSE: 0.0017059766687452793, quantized loss: 0.0003850361390504986  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi007_im009._original_depth]|![JNet_585_beads_roi007_im009._output_depth]|![JNet_585_beads_roi007_im009._reconst_depth]|![JNet_585_beads_roi007_im009._heatmap_depth]|
  
volume: 3.316072753906251, MSE: 0.0017316759331151843, quantized loss: 0.00024659637711010873  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi008_im010._original_depth]|![JNet_585_beads_roi008_im010._output_depth]|![JNet_585_beads_roi008_im010._reconst_depth]|![JNet_585_beads_roi008_im010._heatmap_depth]|
  
volume: 3.371710205078126, MSE: 0.0018519129371270537, quantized loss: 0.00022550216817762703  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi009_im011._original_depth]|![JNet_585_beads_roi009_im011._output_depth]|![JNet_585_beads_roi009_im011._reconst_depth]|![JNet_585_beads_roi009_im011._heatmap_depth]|
  
volume: 3.1615219726562507, MSE: 0.0016366662457585335, quantized loss: 0.00021168525563552976  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi010_im012._original_depth]|![JNet_585_beads_roi010_im012._output_depth]|![JNet_585_beads_roi010_im012._reconst_depth]|![JNet_585_beads_roi010_im012._heatmap_depth]|
  
volume: 3.603644531250001, MSE: 0.0020997098181396723, quantized loss: 0.0002586265909485519  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi011_im013._original_depth]|![JNet_585_beads_roi011_im013._output_depth]|![JNet_585_beads_roi011_im013._reconst_depth]|![JNet_585_beads_roi011_im013._heatmap_depth]|
  
volume: 3.580439941406251, MSE: 0.002033685566857457, quantized loss: 0.0002591187076177448  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi012_im014._original_depth]|![JNet_585_beads_roi012_im014._output_depth]|![JNet_585_beads_roi012_im014._reconst_depth]|![JNet_585_beads_roi012_im014._heatmap_depth]|
  
volume: 3.040293701171876, MSE: 0.0016568267019465566, quantized loss: 0.0001773997937561944  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi013_im015._original_depth]|![JNet_585_beads_roi013_im015._output_depth]|![JNet_585_beads_roi013_im015._reconst_depth]|![JNet_585_beads_roi013_im015._heatmap_depth]|
  
volume: 2.900183349609376, MSE: 0.0016828662483021617, quantized loss: 0.0001785215426934883  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi014_im016._original_depth]|![JNet_585_beads_roi014_im016._output_depth]|![JNet_585_beads_roi014_im016._reconst_depth]|![JNet_585_beads_roi014_im016._heatmap_depth]|
  
volume: 2.9453432617187505, MSE: 0.0016884302021935582, quantized loss: 0.00021218042820692062  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi015_im017._original_depth]|![JNet_585_beads_roi015_im017._output_depth]|![JNet_585_beads_roi015_im017._reconst_depth]|![JNet_585_beads_roi015_im017._heatmap_depth]|
  
volume: 3.0627780761718757, MSE: 0.0015886089531704783, quantized loss: 0.00019873572455253452  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi016_im018._original_depth]|![JNet_585_beads_roi016_im018._output_depth]|![JNet_585_beads_roi016_im018._reconst_depth]|![JNet_585_beads_roi016_im018._heatmap_depth]|
  
volume: 3.328860107421876, MSE: 0.001928932499140501, quantized loss: 0.00022101284412201494  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi017_im018._original_depth]|![JNet_585_beads_roi017_im018._output_depth]|![JNet_585_beads_roi017_im018._reconst_depth]|![JNet_585_beads_roi017_im018._heatmap_depth]|
  
volume: 3.2068864746093757, MSE: 0.001905616489239037, quantized loss: 0.00021377477969508618  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi018_im022._original_depth]|![JNet_585_beads_roi018_im022._output_depth]|![JNet_585_beads_roi018_im022._reconst_depth]|![JNet_585_beads_roi018_im022._heatmap_depth]|
  
volume: 2.5715605468750007, MSE: 0.0015643653459846973, quantized loss: 0.0001478441699873656  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi019_im023._original_depth]|![JNet_585_beads_roi019_im023._output_depth]|![JNet_585_beads_roi019_im023._reconst_depth]|![JNet_585_beads_roi019_im023._heatmap_depth]|
  
volume: 2.5784887695312504, MSE: 0.001419779728166759, quantized loss: 0.00014245399506762624  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi020_im024._original_depth]|![JNet_585_beads_roi020_im024._output_depth]|![JNet_585_beads_roi020_im024._reconst_depth]|![JNet_585_beads_roi020_im024._heatmap_depth]|
  
volume: 3.335429199218751, MSE: 0.0019005477661266923, quantized loss: 0.00021140028547961265  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi021_im026._original_depth]|![JNet_585_beads_roi021_im026._output_depth]|![JNet_585_beads_roi021_im026._reconst_depth]|![JNet_585_beads_roi021_im026._heatmap_depth]|
  
volume: 3.190220458984376, MSE: 0.0017655928386375308, quantized loss: 0.00020277398289181292  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi022_im027._original_depth]|![JNet_585_beads_roi022_im027._output_depth]|![JNet_585_beads_roi022_im027._reconst_depth]|![JNet_585_beads_roi022_im027._heatmap_depth]|
  
volume: 3.1146113281250005, MSE: 0.0017059780657291412, quantized loss: 0.000192698382306844  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi023_im028._original_depth]|![JNet_585_beads_roi023_im028._output_depth]|![JNet_585_beads_roi023_im028._reconst_depth]|![JNet_585_beads_roi023_im028._heatmap_depth]|
  
volume: 3.466415039062501, MSE: 0.0018555679125711322, quantized loss: 0.00025676863151602447  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi024_im028._original_depth]|![JNet_585_beads_roi024_im028._output_depth]|![JNet_585_beads_roi024_im028._reconst_depth]|![JNet_585_beads_roi024_im028._heatmap_depth]|
  
volume: 3.2150544433593757, MSE: 0.0019108126871287823, quantized loss: 0.0002061748382402584  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi025_im028._original_depth]|![JNet_585_beads_roi025_im028._output_depth]|![JNet_585_beads_roi025_im028._reconst_depth]|![JNet_585_beads_roi025_im028._heatmap_depth]|
  
volume: 3.2150544433593757, MSE: 0.0019108126871287823, quantized loss: 0.0002061748382402584  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi026_im029._original_depth]|![JNet_585_beads_roi026_im029._output_depth]|![JNet_585_beads_roi026_im029._reconst_depth]|![JNet_585_beads_roi026_im029._heatmap_depth]|
  
volume: 3.3717446289062507, MSE: 0.0019235484069213271, quantized loss: 0.00021669133275281638  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi027_im029._original_depth]|![JNet_585_beads_roi027_im029._output_depth]|![JNet_585_beads_roi027_im029._reconst_depth]|![JNet_585_beads_roi027_im029._heatmap_depth]|
  
volume: 2.7796049804687506, MSE: 0.0018150737741962075, quantized loss: 0.0001614049542695284  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi028_im030._original_depth]|![JNet_585_beads_roi028_im030._output_depth]|![JNet_585_beads_roi028_im030._reconst_depth]|![JNet_585_beads_roi028_im030._heatmap_depth]|
  
volume: 2.8213386230468758, MSE: 0.001663167029619217, quantized loss: 0.00016379996668547392  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_585_beads_roi029_im030._original_depth]|![JNet_585_beads_roi029_im030._output_depth]|![JNet_585_beads_roi029_im030._reconst_depth]|![JNet_585_beads_roi029_im030._heatmap_depth]|
  
volume: 2.982733154296876, MSE: 0.0017736725276336074, quantized loss: 0.0001773292024154216  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_585_psf_pre]|![JNet_585_psf_post]|

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
[JNet_585_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_585_beads_roi000_im000._heatmap_depth.png
[JNet_585_beads_roi000_im000._original_depth]: /experiments/images/JNet_585_beads_roi000_im000._original_depth.png
[JNet_585_beads_roi000_im000._output_depth]: /experiments/images/JNet_585_beads_roi000_im000._output_depth.png
[JNet_585_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_585_beads_roi000_im000._reconst_depth.png
[JNet_585_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_585_beads_roi001_im004._heatmap_depth.png
[JNet_585_beads_roi001_im004._original_depth]: /experiments/images/JNet_585_beads_roi001_im004._original_depth.png
[JNet_585_beads_roi001_im004._output_depth]: /experiments/images/JNet_585_beads_roi001_im004._output_depth.png
[JNet_585_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_585_beads_roi001_im004._reconst_depth.png
[JNet_585_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_585_beads_roi002_im005._heatmap_depth.png
[JNet_585_beads_roi002_im005._original_depth]: /experiments/images/JNet_585_beads_roi002_im005._original_depth.png
[JNet_585_beads_roi002_im005._output_depth]: /experiments/images/JNet_585_beads_roi002_im005._output_depth.png
[JNet_585_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_585_beads_roi002_im005._reconst_depth.png
[JNet_585_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_585_beads_roi003_im006._heatmap_depth.png
[JNet_585_beads_roi003_im006._original_depth]: /experiments/images/JNet_585_beads_roi003_im006._original_depth.png
[JNet_585_beads_roi003_im006._output_depth]: /experiments/images/JNet_585_beads_roi003_im006._output_depth.png
[JNet_585_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_585_beads_roi003_im006._reconst_depth.png
[JNet_585_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_585_beads_roi004_im006._heatmap_depth.png
[JNet_585_beads_roi004_im006._original_depth]: /experiments/images/JNet_585_beads_roi004_im006._original_depth.png
[JNet_585_beads_roi004_im006._output_depth]: /experiments/images/JNet_585_beads_roi004_im006._output_depth.png
[JNet_585_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_585_beads_roi004_im006._reconst_depth.png
[JNet_585_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_585_beads_roi005_im007._heatmap_depth.png
[JNet_585_beads_roi005_im007._original_depth]: /experiments/images/JNet_585_beads_roi005_im007._original_depth.png
[JNet_585_beads_roi005_im007._output_depth]: /experiments/images/JNet_585_beads_roi005_im007._output_depth.png
[JNet_585_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_585_beads_roi005_im007._reconst_depth.png
[JNet_585_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_585_beads_roi006_im008._heatmap_depth.png
[JNet_585_beads_roi006_im008._original_depth]: /experiments/images/JNet_585_beads_roi006_im008._original_depth.png
[JNet_585_beads_roi006_im008._output_depth]: /experiments/images/JNet_585_beads_roi006_im008._output_depth.png
[JNet_585_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_585_beads_roi006_im008._reconst_depth.png
[JNet_585_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_585_beads_roi007_im009._heatmap_depth.png
[JNet_585_beads_roi007_im009._original_depth]: /experiments/images/JNet_585_beads_roi007_im009._original_depth.png
[JNet_585_beads_roi007_im009._output_depth]: /experiments/images/JNet_585_beads_roi007_im009._output_depth.png
[JNet_585_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_585_beads_roi007_im009._reconst_depth.png
[JNet_585_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_585_beads_roi008_im010._heatmap_depth.png
[JNet_585_beads_roi008_im010._original_depth]: /experiments/images/JNet_585_beads_roi008_im010._original_depth.png
[JNet_585_beads_roi008_im010._output_depth]: /experiments/images/JNet_585_beads_roi008_im010._output_depth.png
[JNet_585_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_585_beads_roi008_im010._reconst_depth.png
[JNet_585_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_585_beads_roi009_im011._heatmap_depth.png
[JNet_585_beads_roi009_im011._original_depth]: /experiments/images/JNet_585_beads_roi009_im011._original_depth.png
[JNet_585_beads_roi009_im011._output_depth]: /experiments/images/JNet_585_beads_roi009_im011._output_depth.png
[JNet_585_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_585_beads_roi009_im011._reconst_depth.png
[JNet_585_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_585_beads_roi010_im012._heatmap_depth.png
[JNet_585_beads_roi010_im012._original_depth]: /experiments/images/JNet_585_beads_roi010_im012._original_depth.png
[JNet_585_beads_roi010_im012._output_depth]: /experiments/images/JNet_585_beads_roi010_im012._output_depth.png
[JNet_585_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_585_beads_roi010_im012._reconst_depth.png
[JNet_585_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_585_beads_roi011_im013._heatmap_depth.png
[JNet_585_beads_roi011_im013._original_depth]: /experiments/images/JNet_585_beads_roi011_im013._original_depth.png
[JNet_585_beads_roi011_im013._output_depth]: /experiments/images/JNet_585_beads_roi011_im013._output_depth.png
[JNet_585_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_585_beads_roi011_im013._reconst_depth.png
[JNet_585_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_585_beads_roi012_im014._heatmap_depth.png
[JNet_585_beads_roi012_im014._original_depth]: /experiments/images/JNet_585_beads_roi012_im014._original_depth.png
[JNet_585_beads_roi012_im014._output_depth]: /experiments/images/JNet_585_beads_roi012_im014._output_depth.png
[JNet_585_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_585_beads_roi012_im014._reconst_depth.png
[JNet_585_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_585_beads_roi013_im015._heatmap_depth.png
[JNet_585_beads_roi013_im015._original_depth]: /experiments/images/JNet_585_beads_roi013_im015._original_depth.png
[JNet_585_beads_roi013_im015._output_depth]: /experiments/images/JNet_585_beads_roi013_im015._output_depth.png
[JNet_585_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_585_beads_roi013_im015._reconst_depth.png
[JNet_585_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_585_beads_roi014_im016._heatmap_depth.png
[JNet_585_beads_roi014_im016._original_depth]: /experiments/images/JNet_585_beads_roi014_im016._original_depth.png
[JNet_585_beads_roi014_im016._output_depth]: /experiments/images/JNet_585_beads_roi014_im016._output_depth.png
[JNet_585_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_585_beads_roi014_im016._reconst_depth.png
[JNet_585_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_585_beads_roi015_im017._heatmap_depth.png
[JNet_585_beads_roi015_im017._original_depth]: /experiments/images/JNet_585_beads_roi015_im017._original_depth.png
[JNet_585_beads_roi015_im017._output_depth]: /experiments/images/JNet_585_beads_roi015_im017._output_depth.png
[JNet_585_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_585_beads_roi015_im017._reconst_depth.png
[JNet_585_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_585_beads_roi016_im018._heatmap_depth.png
[JNet_585_beads_roi016_im018._original_depth]: /experiments/images/JNet_585_beads_roi016_im018._original_depth.png
[JNet_585_beads_roi016_im018._output_depth]: /experiments/images/JNet_585_beads_roi016_im018._output_depth.png
[JNet_585_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_585_beads_roi016_im018._reconst_depth.png
[JNet_585_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_585_beads_roi017_im018._heatmap_depth.png
[JNet_585_beads_roi017_im018._original_depth]: /experiments/images/JNet_585_beads_roi017_im018._original_depth.png
[JNet_585_beads_roi017_im018._output_depth]: /experiments/images/JNet_585_beads_roi017_im018._output_depth.png
[JNet_585_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_585_beads_roi017_im018._reconst_depth.png
[JNet_585_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_585_beads_roi018_im022._heatmap_depth.png
[JNet_585_beads_roi018_im022._original_depth]: /experiments/images/JNet_585_beads_roi018_im022._original_depth.png
[JNet_585_beads_roi018_im022._output_depth]: /experiments/images/JNet_585_beads_roi018_im022._output_depth.png
[JNet_585_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_585_beads_roi018_im022._reconst_depth.png
[JNet_585_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_585_beads_roi019_im023._heatmap_depth.png
[JNet_585_beads_roi019_im023._original_depth]: /experiments/images/JNet_585_beads_roi019_im023._original_depth.png
[JNet_585_beads_roi019_im023._output_depth]: /experiments/images/JNet_585_beads_roi019_im023._output_depth.png
[JNet_585_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_585_beads_roi019_im023._reconst_depth.png
[JNet_585_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_585_beads_roi020_im024._heatmap_depth.png
[JNet_585_beads_roi020_im024._original_depth]: /experiments/images/JNet_585_beads_roi020_im024._original_depth.png
[JNet_585_beads_roi020_im024._output_depth]: /experiments/images/JNet_585_beads_roi020_im024._output_depth.png
[JNet_585_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_585_beads_roi020_im024._reconst_depth.png
[JNet_585_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_585_beads_roi021_im026._heatmap_depth.png
[JNet_585_beads_roi021_im026._original_depth]: /experiments/images/JNet_585_beads_roi021_im026._original_depth.png
[JNet_585_beads_roi021_im026._output_depth]: /experiments/images/JNet_585_beads_roi021_im026._output_depth.png
[JNet_585_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_585_beads_roi021_im026._reconst_depth.png
[JNet_585_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_585_beads_roi022_im027._heatmap_depth.png
[JNet_585_beads_roi022_im027._original_depth]: /experiments/images/JNet_585_beads_roi022_im027._original_depth.png
[JNet_585_beads_roi022_im027._output_depth]: /experiments/images/JNet_585_beads_roi022_im027._output_depth.png
[JNet_585_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_585_beads_roi022_im027._reconst_depth.png
[JNet_585_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_585_beads_roi023_im028._heatmap_depth.png
[JNet_585_beads_roi023_im028._original_depth]: /experiments/images/JNet_585_beads_roi023_im028._original_depth.png
[JNet_585_beads_roi023_im028._output_depth]: /experiments/images/JNet_585_beads_roi023_im028._output_depth.png
[JNet_585_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_585_beads_roi023_im028._reconst_depth.png
[JNet_585_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_585_beads_roi024_im028._heatmap_depth.png
[JNet_585_beads_roi024_im028._original_depth]: /experiments/images/JNet_585_beads_roi024_im028._original_depth.png
[JNet_585_beads_roi024_im028._output_depth]: /experiments/images/JNet_585_beads_roi024_im028._output_depth.png
[JNet_585_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_585_beads_roi024_im028._reconst_depth.png
[JNet_585_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_585_beads_roi025_im028._heatmap_depth.png
[JNet_585_beads_roi025_im028._original_depth]: /experiments/images/JNet_585_beads_roi025_im028._original_depth.png
[JNet_585_beads_roi025_im028._output_depth]: /experiments/images/JNet_585_beads_roi025_im028._output_depth.png
[JNet_585_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_585_beads_roi025_im028._reconst_depth.png
[JNet_585_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_585_beads_roi026_im029._heatmap_depth.png
[JNet_585_beads_roi026_im029._original_depth]: /experiments/images/JNet_585_beads_roi026_im029._original_depth.png
[JNet_585_beads_roi026_im029._output_depth]: /experiments/images/JNet_585_beads_roi026_im029._output_depth.png
[JNet_585_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_585_beads_roi026_im029._reconst_depth.png
[JNet_585_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_585_beads_roi027_im029._heatmap_depth.png
[JNet_585_beads_roi027_im029._original_depth]: /experiments/images/JNet_585_beads_roi027_im029._original_depth.png
[JNet_585_beads_roi027_im029._output_depth]: /experiments/images/JNet_585_beads_roi027_im029._output_depth.png
[JNet_585_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_585_beads_roi027_im029._reconst_depth.png
[JNet_585_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_585_beads_roi028_im030._heatmap_depth.png
[JNet_585_beads_roi028_im030._original_depth]: /experiments/images/JNet_585_beads_roi028_im030._original_depth.png
[JNet_585_beads_roi028_im030._output_depth]: /experiments/images/JNet_585_beads_roi028_im030._output_depth.png
[JNet_585_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_585_beads_roi028_im030._reconst_depth.png
[JNet_585_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_585_beads_roi029_im030._heatmap_depth.png
[JNet_585_beads_roi029_im030._original_depth]: /experiments/images/JNet_585_beads_roi029_im030._original_depth.png
[JNet_585_beads_roi029_im030._output_depth]: /experiments/images/JNet_585_beads_roi029_im030._output_depth.png
[JNet_585_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_585_beads_roi029_im030._reconst_depth.png
[JNet_585_psf_post]: /experiments/images/JNet_585_psf_post.png
[JNet_585_psf_pre]: /experiments/images/JNet_585_psf_pre.png
