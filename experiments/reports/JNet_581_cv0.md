



# JNet_581_cv0 Report
  
beads cross validation experiment test  
pretrained model : JNet_580_pretrain
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
|poisson_weight|0.0||
|sig_eps|0.0||
|background|0.0||
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
|ewc|True|
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
|ewc|None|
|params|params|
|es_patience|10|
|reconstruct|True|
|is_instantblur|False|
|is_vibrate|False|
|adjust_luminance|True|
|zloss_weight|1|
|ewc_weight|1|
|qloss_weight|10.0|
|ploss_weight|1.0|
|mrfloss_order|1|
|mrfloss_dilation|1|
|mrfloss_weights|{'l_00': 0, 'l_01': 0, 'l_10': 0, 'l_11': 0}|

## Results

### Pretraining
  
Segmentation: mean MSE: 0.007707285229116678, mean BCE: 0.031152229756116867  
Luminance Estimation: mean MSE: 0.9834938049316406, mean BCE: nan
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_0_original_plane]|![JNet_580_pretrain_0_novibrate_plane]|![JNet_580_pretrain_0_aligned_plane]|![JNet_580_pretrain_0_outputx_plane]|![JNet_580_pretrain_0_labelx_plane]|![JNet_580_pretrain_0_outputz_plane]|![JNet_580_pretrain_0_labelz_plane]|
  
MSEx: 0.015139879658818245, BCEx: 0.05906686559319496  
MSEz: 0.9527636766433716, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_0_original_depth]|![JNet_580_pretrain_0_novibrate_depth]|![JNet_580_pretrain_0_aligned_depth]|![JNet_580_pretrain_0_outputx_depth]|![JNet_580_pretrain_0_labelx_depth]|![JNet_580_pretrain_0_outputz_depth]|![JNet_580_pretrain_0_labelz_depth]|
  
MSEx: 0.015139879658818245, BCEx: 0.05906686559319496  
MSEz: 0.9527636766433716, BCEz: inf  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_1_original_plane]|![JNet_580_pretrain_1_novibrate_plane]|![JNet_580_pretrain_1_aligned_plane]|![JNet_580_pretrain_1_outputx_plane]|![JNet_580_pretrain_1_labelx_plane]|![JNet_580_pretrain_1_outputz_plane]|![JNet_580_pretrain_1_labelz_plane]|
  
MSEx: 0.006364457309246063, BCEx: 0.02556501142680645  
MSEz: 0.9873598217964172, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_1_original_depth]|![JNet_580_pretrain_1_novibrate_depth]|![JNet_580_pretrain_1_aligned_depth]|![JNet_580_pretrain_1_outputx_depth]|![JNet_580_pretrain_1_labelx_depth]|![JNet_580_pretrain_1_outputz_depth]|![JNet_580_pretrain_1_labelz_depth]|
  
MSEx: 0.006364457309246063, BCEx: 0.02556501142680645  
MSEz: 0.9873598217964172, BCEz: nan  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_2_original_plane]|![JNet_580_pretrain_2_novibrate_plane]|![JNet_580_pretrain_2_aligned_plane]|![JNet_580_pretrain_2_outputx_plane]|![JNet_580_pretrain_2_labelx_plane]|![JNet_580_pretrain_2_outputz_plane]|![JNet_580_pretrain_2_labelz_plane]|
  
MSEx: 0.004353754688054323, BCEx: 0.018604053184390068  
MSEz: 0.9947536587715149, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_2_original_depth]|![JNet_580_pretrain_2_novibrate_depth]|![JNet_580_pretrain_2_aligned_depth]|![JNet_580_pretrain_2_outputx_depth]|![JNet_580_pretrain_2_labelx_depth]|![JNet_580_pretrain_2_outputz_depth]|![JNet_580_pretrain_2_labelz_depth]|
  
MSEx: 0.004353754688054323, BCEx: 0.018604053184390068  
MSEz: 0.9947536587715149, BCEz: nan  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_3_original_plane]|![JNet_580_pretrain_3_novibrate_plane]|![JNet_580_pretrain_3_aligned_plane]|![JNet_580_pretrain_3_outputx_plane]|![JNet_580_pretrain_3_labelx_plane]|![JNet_580_pretrain_3_outputz_plane]|![JNet_580_pretrain_3_labelz_plane]|
  
MSEx: 0.006311750505119562, BCEx: 0.025979334488511086  
MSEz: 0.9925763607025146, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_3_original_depth]|![JNet_580_pretrain_3_novibrate_depth]|![JNet_580_pretrain_3_aligned_depth]|![JNet_580_pretrain_3_outputx_depth]|![JNet_580_pretrain_3_labelx_depth]|![JNet_580_pretrain_3_outputz_depth]|![JNet_580_pretrain_3_labelz_depth]|
  
MSEx: 0.006311750505119562, BCEx: 0.025979334488511086  
MSEz: 0.9925763607025146, BCEz: inf  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_4_original_plane]|![JNet_580_pretrain_4_novibrate_plane]|![JNet_580_pretrain_4_aligned_plane]|![JNet_580_pretrain_4_outputx_plane]|![JNet_580_pretrain_4_labelx_plane]|![JNet_580_pretrain_4_outputz_plane]|![JNet_580_pretrain_4_labelz_plane]|
  
MSEx: 0.006366584450006485, BCEx: 0.026545889675617218  
MSEz: 0.9900155067443848, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_4_original_depth]|![JNet_580_pretrain_4_novibrate_depth]|![JNet_580_pretrain_4_aligned_depth]|![JNet_580_pretrain_4_outputx_depth]|![JNet_580_pretrain_4_labelx_depth]|![JNet_580_pretrain_4_outputz_depth]|![JNet_580_pretrain_4_labelz_depth]|
  
MSEx: 0.006366584450006485, BCEx: 0.026545889675617218  
MSEz: 0.9900155067443848, BCEz: nan  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi000_im000._original_depth]|![JNet_580_pretrain_beads_roi000_im000._output_depth]|![JNet_580_pretrain_beads_roi000_im000._reconst_depth]|![JNet_580_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 3.110340576171876, MSE: 0.0011009004665538669, quantized loss: 0.0002472313935868442  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi001_im004._original_depth]|![JNet_580_pretrain_beads_roi001_im004._output_depth]|![JNet_580_pretrain_beads_roi001_im004._reconst_depth]|![JNet_580_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 3.6990412597656257, MSE: 0.0011340993223711848, quantized loss: 0.0002874291385523975  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi002_im005._original_depth]|![JNet_580_pretrain_beads_roi002_im005._output_depth]|![JNet_580_pretrain_beads_roi002_im005._reconst_depth]|![JNet_580_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 3.274010986328126, MSE: 0.0010883179493248463, quantized loss: 0.00026635584072209895  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi003_im006._original_depth]|![JNet_580_pretrain_beads_roi003_im006._output_depth]|![JNet_580_pretrain_beads_roi003_im006._reconst_depth]|![JNet_580_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 3.408282226562501, MSE: 0.0010975570185109973, quantized loss: 0.00029648063355125487  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi004_im006._original_depth]|![JNet_580_pretrain_beads_roi004_im006._output_depth]|![JNet_580_pretrain_beads_roi004_im006._reconst_depth]|![JNet_580_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 3.450879638671876, MSE: 0.0011176582193002105, quantized loss: 0.00029673229437321424  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi005_im007._original_depth]|![JNet_580_pretrain_beads_roi005_im007._output_depth]|![JNet_580_pretrain_beads_roi005_im007._reconst_depth]|![JNet_580_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 3.308382812500001, MSE: 0.0010864249197766185, quantized loss: 0.0002765751560218632  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi006_im008._original_depth]|![JNet_580_pretrain_beads_roi006_im008._output_depth]|![JNet_580_pretrain_beads_roi006_im008._reconst_depth]|![JNet_580_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 3.485151367187501, MSE: 0.0010399831226095557, quantized loss: 0.0003225260879844427  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi007_im009._original_depth]|![JNet_580_pretrain_beads_roi007_im009._output_depth]|![JNet_580_pretrain_beads_roi007_im009._reconst_depth]|![JNet_580_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 3.403878417968751, MSE: 0.0010961453663185239, quantized loss: 0.00030673513538204134  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi008_im010._original_depth]|![JNet_580_pretrain_beads_roi008_im010._output_depth]|![JNet_580_pretrain_beads_roi008_im010._reconst_depth]|![JNet_580_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 3.439821289062501, MSE: 0.0010869295801967382, quantized loss: 0.0002757111797109246  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi009_im011._original_depth]|![JNet_580_pretrain_beads_roi009_im011._output_depth]|![JNet_580_pretrain_beads_roi009_im011._reconst_depth]|![JNet_580_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 3.153017089843751, MSE: 0.001065649907104671, quantized loss: 0.0002540632849559188  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi010_im012._original_depth]|![JNet_580_pretrain_beads_roi010_im012._output_depth]|![JNet_580_pretrain_beads_roi010_im012._reconst_depth]|![JNet_580_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 3.6727307128906257, MSE: 0.0011221092427149415, quantized loss: 0.0002770679129753262  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi011_im013._original_depth]|![JNet_580_pretrain_beads_roi011_im013._output_depth]|![JNet_580_pretrain_beads_roi011_im013._reconst_depth]|![JNet_580_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 3.728245605468751, MSE: 0.001096814638003707, quantized loss: 0.0002850613964255899  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi012_im014._original_depth]|![JNet_580_pretrain_beads_roi012_im014._output_depth]|![JNet_580_pretrain_beads_roi012_im014._reconst_depth]|![JNet_580_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 3.213681640625001, MSE: 0.0012077066348865628, quantized loss: 0.0002585780748631805  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi013_im015._original_depth]|![JNet_580_pretrain_beads_roi013_im015._output_depth]|![JNet_580_pretrain_beads_roi013_im015._reconst_depth]|![JNet_580_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 3.1003991699218756, MSE: 0.001149380928836763, quantized loss: 0.0002502653223928064  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi014_im016._original_depth]|![JNet_580_pretrain_beads_roi014_im016._output_depth]|![JNet_580_pretrain_beads_roi014_im016._reconst_depth]|![JNet_580_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 3.2180185546875006, MSE: 0.0010625412687659264, quantized loss: 0.0002787893172353506  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi015_im017._original_depth]|![JNet_580_pretrain_beads_roi015_im017._output_depth]|![JNet_580_pretrain_beads_roi015_im017._reconst_depth]|![JNet_580_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 3.1706921386718756, MSE: 0.0010830876417458057, quantized loss: 0.0002687327505555004  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi016_im018._original_depth]|![JNet_580_pretrain_beads_roi016_im018._output_depth]|![JNet_580_pretrain_beads_roi016_im018._reconst_depth]|![JNet_580_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 3.4129365234375006, MSE: 0.001197430887259543, quantized loss: 0.00027654619771055877  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi017_im018._original_depth]|![JNet_580_pretrain_beads_roi017_im018._output_depth]|![JNet_580_pretrain_beads_roi017_im018._reconst_depth]|![JNet_580_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 3.417743896484376, MSE: 0.0012498609721660614, quantized loss: 0.00027898154803551733  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi018_im022._original_depth]|![JNet_580_pretrain_beads_roi018_im022._output_depth]|![JNet_580_pretrain_beads_roi018_im022._reconst_depth]|![JNet_580_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 2.9183054199218756, MSE: 0.001082596369087696, quantized loss: 0.0002409886074019596  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi019_im023._original_depth]|![JNet_580_pretrain_beads_roi019_im023._output_depth]|![JNet_580_pretrain_beads_roi019_im023._reconst_depth]|![JNet_580_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 2.8712707519531255, MSE: 0.0011043999111279845, quantized loss: 0.00023800990311428905  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi020_im024._original_depth]|![JNet_580_pretrain_beads_roi020_im024._output_depth]|![JNet_580_pretrain_beads_roi020_im024._reconst_depth]|![JNet_580_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 3.459032714843751, MSE: 0.0011147995246574283, quantized loss: 0.0002569811185821891  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi021_im026._original_depth]|![JNet_580_pretrain_beads_roi021_im026._output_depth]|![JNet_580_pretrain_beads_roi021_im026._reconst_depth]|![JNet_580_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 3.3450114746093758, MSE: 0.0010517948539927602, quantized loss: 0.00025798886781558394  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi022_im027._original_depth]|![JNet_580_pretrain_beads_roi022_im027._output_depth]|![JNet_580_pretrain_beads_roi022_im027._reconst_depth]|![JNet_580_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 3.2117690429687507, MSE: 0.001124676549807191, quantized loss: 0.0002481419942341745  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi023_im028._original_depth]|![JNet_580_pretrain_beads_roi023_im028._output_depth]|![JNet_580_pretrain_beads_roi023_im028._reconst_depth]|![JNet_580_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 3.646301757812501, MSE: 0.0009471185621805489, quantized loss: 0.00029903530958108604  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi024_im028._original_depth]|![JNet_580_pretrain_beads_roi024_im028._output_depth]|![JNet_580_pretrain_beads_roi024_im028._reconst_depth]|![JNet_580_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 3.5362907714843757, MSE: 0.0010074891615658998, quantized loss: 0.000272990990197286  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi025_im028._original_depth]|![JNet_580_pretrain_beads_roi025_im028._output_depth]|![JNet_580_pretrain_beads_roi025_im028._reconst_depth]|![JNet_580_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 3.5362907714843757, MSE: 0.0010074891615658998, quantized loss: 0.000272990990197286  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi026_im029._original_depth]|![JNet_580_pretrain_beads_roi026_im029._output_depth]|![JNet_580_pretrain_beads_roi026_im029._reconst_depth]|![JNet_580_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 3.505439208984376, MSE: 0.0011405035620555282, quantized loss: 0.00026848286506719887  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi027_im029._original_depth]|![JNet_580_pretrain_beads_roi027_im029._output_depth]|![JNet_580_pretrain_beads_roi027_im029._reconst_depth]|![JNet_580_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 3.189786376953126, MSE: 0.0011015509953722358, quantized loss: 0.00025053368881344795  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi028_im030._original_depth]|![JNet_580_pretrain_beads_roi028_im030._output_depth]|![JNet_580_pretrain_beads_roi028_im030._reconst_depth]|![JNet_580_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 3.1012324218750007, MSE: 0.0010827371152117848, quantized loss: 0.00024678368936292827  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_580_pretrain_beads_roi029_im030._original_depth]|![JNet_580_pretrain_beads_roi029_im030._output_depth]|![JNet_580_pretrain_beads_roi029_im030._reconst_depth]|![JNet_580_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 3.256739746093751, MSE: 0.001123832305893302, quantized loss: 0.00025147487758658826  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi000_im000._original_depth]|![JNet_581_cv0_beads_roi000_im000._output_depth]|![JNet_581_cv0_beads_roi000_im000._reconst_depth]|![JNet_581_cv0_beads_roi000_im000._heatmap_depth]|
  
volume: 3.110340576171876, MSE: 0.0011009004665538669, quantized loss: 0.0002472313935868442  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi001_im004._original_depth]|![JNet_581_cv0_beads_roi001_im004._output_depth]|![JNet_581_cv0_beads_roi001_im004._reconst_depth]|![JNet_581_cv0_beads_roi001_im004._heatmap_depth]|
  
volume: 3.6990412597656257, MSE: 0.0011340993223711848, quantized loss: 0.0002874291385523975  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi002_im005._original_depth]|![JNet_581_cv0_beads_roi002_im005._output_depth]|![JNet_581_cv0_beads_roi002_im005._reconst_depth]|![JNet_581_cv0_beads_roi002_im005._heatmap_depth]|
  
volume: 3.274010986328126, MSE: 0.0010883179493248463, quantized loss: 0.00026635584072209895  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi003_im006._original_depth]|![JNet_581_cv0_beads_roi003_im006._output_depth]|![JNet_581_cv0_beads_roi003_im006._reconst_depth]|![JNet_581_cv0_beads_roi003_im006._heatmap_depth]|
  
volume: 3.408282226562501, MSE: 0.0010975570185109973, quantized loss: 0.00029648063355125487  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi004_im006._original_depth]|![JNet_581_cv0_beads_roi004_im006._output_depth]|![JNet_581_cv0_beads_roi004_im006._reconst_depth]|![JNet_581_cv0_beads_roi004_im006._heatmap_depth]|
  
volume: 3.450879638671876, MSE: 0.0011176582193002105, quantized loss: 0.00029673229437321424  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi005_im007._original_depth]|![JNet_581_cv0_beads_roi005_im007._output_depth]|![JNet_581_cv0_beads_roi005_im007._reconst_depth]|![JNet_581_cv0_beads_roi005_im007._heatmap_depth]|
  
volume: 3.308382812500001, MSE: 0.0010864249197766185, quantized loss: 0.0002765751560218632  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi006_im008._original_depth]|![JNet_581_cv0_beads_roi006_im008._output_depth]|![JNet_581_cv0_beads_roi006_im008._reconst_depth]|![JNet_581_cv0_beads_roi006_im008._heatmap_depth]|
  
volume: 3.485151367187501, MSE: 0.0010399831226095557, quantized loss: 0.0003225260879844427  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi007_im009._original_depth]|![JNet_581_cv0_beads_roi007_im009._output_depth]|![JNet_581_cv0_beads_roi007_im009._reconst_depth]|![JNet_581_cv0_beads_roi007_im009._heatmap_depth]|
  
volume: 3.403878417968751, MSE: 0.0010961453663185239, quantized loss: 0.00030673513538204134  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi008_im010._original_depth]|![JNet_581_cv0_beads_roi008_im010._output_depth]|![JNet_581_cv0_beads_roi008_im010._reconst_depth]|![JNet_581_cv0_beads_roi008_im010._heatmap_depth]|
  
volume: 3.439821289062501, MSE: 0.0010869295801967382, quantized loss: 0.0002757111797109246  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi009_im011._original_depth]|![JNet_581_cv0_beads_roi009_im011._output_depth]|![JNet_581_cv0_beads_roi009_im011._reconst_depth]|![JNet_581_cv0_beads_roi009_im011._heatmap_depth]|
  
volume: 3.153017089843751, MSE: 0.001065649907104671, quantized loss: 0.0002540632849559188  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi010_im012._original_depth]|![JNet_581_cv0_beads_roi010_im012._output_depth]|![JNet_581_cv0_beads_roi010_im012._reconst_depth]|![JNet_581_cv0_beads_roi010_im012._heatmap_depth]|
  
volume: 3.6727307128906257, MSE: 0.0011221092427149415, quantized loss: 0.0002770679129753262  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi011_im013._original_depth]|![JNet_581_cv0_beads_roi011_im013._output_depth]|![JNet_581_cv0_beads_roi011_im013._reconst_depth]|![JNet_581_cv0_beads_roi011_im013._heatmap_depth]|
  
volume: 3.728245605468751, MSE: 0.001096814638003707, quantized loss: 0.0002850613964255899  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi012_im014._original_depth]|![JNet_581_cv0_beads_roi012_im014._output_depth]|![JNet_581_cv0_beads_roi012_im014._reconst_depth]|![JNet_581_cv0_beads_roi012_im014._heatmap_depth]|
  
volume: 3.213681640625001, MSE: 0.0012077066348865628, quantized loss: 0.0002585780748631805  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi013_im015._original_depth]|![JNet_581_cv0_beads_roi013_im015._output_depth]|![JNet_581_cv0_beads_roi013_im015._reconst_depth]|![JNet_581_cv0_beads_roi013_im015._heatmap_depth]|
  
volume: 3.1003991699218756, MSE: 0.001149380928836763, quantized loss: 0.0002502653223928064  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi014_im016._original_depth]|![JNet_581_cv0_beads_roi014_im016._output_depth]|![JNet_581_cv0_beads_roi014_im016._reconst_depth]|![JNet_581_cv0_beads_roi014_im016._heatmap_depth]|
  
volume: 3.2180185546875006, MSE: 0.0010625412687659264, quantized loss: 0.0002787893172353506  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi015_im017._original_depth]|![JNet_581_cv0_beads_roi015_im017._output_depth]|![JNet_581_cv0_beads_roi015_im017._reconst_depth]|![JNet_581_cv0_beads_roi015_im017._heatmap_depth]|
  
volume: 3.1706921386718756, MSE: 0.0010830876417458057, quantized loss: 0.0002687327505555004  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi016_im018._original_depth]|![JNet_581_cv0_beads_roi016_im018._output_depth]|![JNet_581_cv0_beads_roi016_im018._reconst_depth]|![JNet_581_cv0_beads_roi016_im018._heatmap_depth]|
  
volume: 3.4129365234375006, MSE: 0.001197430887259543, quantized loss: 0.00027654619771055877  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi017_im018._original_depth]|![JNet_581_cv0_beads_roi017_im018._output_depth]|![JNet_581_cv0_beads_roi017_im018._reconst_depth]|![JNet_581_cv0_beads_roi017_im018._heatmap_depth]|
  
volume: 3.417743896484376, MSE: 0.0012498609721660614, quantized loss: 0.00027898154803551733  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi018_im022._original_depth]|![JNet_581_cv0_beads_roi018_im022._output_depth]|![JNet_581_cv0_beads_roi018_im022._reconst_depth]|![JNet_581_cv0_beads_roi018_im022._heatmap_depth]|
  
volume: 2.9183054199218756, MSE: 0.001082596369087696, quantized loss: 0.0002409886074019596  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi019_im023._original_depth]|![JNet_581_cv0_beads_roi019_im023._output_depth]|![JNet_581_cv0_beads_roi019_im023._reconst_depth]|![JNet_581_cv0_beads_roi019_im023._heatmap_depth]|
  
volume: 2.8712707519531255, MSE: 0.0011043999111279845, quantized loss: 0.00023800990311428905  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi020_im024._original_depth]|![JNet_581_cv0_beads_roi020_im024._output_depth]|![JNet_581_cv0_beads_roi020_im024._reconst_depth]|![JNet_581_cv0_beads_roi020_im024._heatmap_depth]|
  
volume: 3.459032714843751, MSE: 0.0011147995246574283, quantized loss: 0.0002569811185821891  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi021_im026._original_depth]|![JNet_581_cv0_beads_roi021_im026._output_depth]|![JNet_581_cv0_beads_roi021_im026._reconst_depth]|![JNet_581_cv0_beads_roi021_im026._heatmap_depth]|
  
volume: 3.3450114746093758, MSE: 0.0010517948539927602, quantized loss: 0.00025798886781558394  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi022_im027._original_depth]|![JNet_581_cv0_beads_roi022_im027._output_depth]|![JNet_581_cv0_beads_roi022_im027._reconst_depth]|![JNet_581_cv0_beads_roi022_im027._heatmap_depth]|
  
volume: 3.2117690429687507, MSE: 0.001124676549807191, quantized loss: 0.0002481419942341745  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi023_im028._original_depth]|![JNet_581_cv0_beads_roi023_im028._output_depth]|![JNet_581_cv0_beads_roi023_im028._reconst_depth]|![JNet_581_cv0_beads_roi023_im028._heatmap_depth]|
  
volume: 3.646301757812501, MSE: 0.0009471185621805489, quantized loss: 0.00029903530958108604  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi024_im028._original_depth]|![JNet_581_cv0_beads_roi024_im028._output_depth]|![JNet_581_cv0_beads_roi024_im028._reconst_depth]|![JNet_581_cv0_beads_roi024_im028._heatmap_depth]|
  
volume: 3.5362907714843757, MSE: 0.0010074891615658998, quantized loss: 0.000272990990197286  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi025_im028._original_depth]|![JNet_581_cv0_beads_roi025_im028._output_depth]|![JNet_581_cv0_beads_roi025_im028._reconst_depth]|![JNet_581_cv0_beads_roi025_im028._heatmap_depth]|
  
volume: 3.5362907714843757, MSE: 0.0010074891615658998, quantized loss: 0.000272990990197286  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi026_im029._original_depth]|![JNet_581_cv0_beads_roi026_im029._output_depth]|![JNet_581_cv0_beads_roi026_im029._reconst_depth]|![JNet_581_cv0_beads_roi026_im029._heatmap_depth]|
  
volume: 3.505439208984376, MSE: 0.0011405035620555282, quantized loss: 0.00026848286506719887  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi027_im029._original_depth]|![JNet_581_cv0_beads_roi027_im029._output_depth]|![JNet_581_cv0_beads_roi027_im029._reconst_depth]|![JNet_581_cv0_beads_roi027_im029._heatmap_depth]|
  
volume: 3.189786376953126, MSE: 0.0011015509953722358, quantized loss: 0.00025053368881344795  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi028_im030._original_depth]|![JNet_581_cv0_beads_roi028_im030._output_depth]|![JNet_581_cv0_beads_roi028_im030._reconst_depth]|![JNet_581_cv0_beads_roi028_im030._heatmap_depth]|
  
volume: 3.1012324218750007, MSE: 0.0010827371152117848, quantized loss: 0.00024678368936292827  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_581_cv0_beads_roi029_im030._original_depth]|![JNet_581_cv0_beads_roi029_im030._output_depth]|![JNet_581_cv0_beads_roi029_im030._reconst_depth]|![JNet_581_cv0_beads_roi029_im030._heatmap_depth]|
  
volume: 3.256739746093751, MSE: 0.001123832305893302, quantized loss: 0.00025147487758658826  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_581_cv0_psf_pre]|![JNet_581_cv0_psf_post]|

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
  



[JNet_580_pretrain_0_aligned_depth]: /experiments/images/JNet_580_pretrain_0_aligned_depth.png
[JNet_580_pretrain_0_aligned_plane]: /experiments/images/JNet_580_pretrain_0_aligned_plane.png
[JNet_580_pretrain_0_labelx_depth]: /experiments/images/JNet_580_pretrain_0_labelx_depth.png
[JNet_580_pretrain_0_labelx_plane]: /experiments/images/JNet_580_pretrain_0_labelx_plane.png
[JNet_580_pretrain_0_labelz_depth]: /experiments/images/JNet_580_pretrain_0_labelz_depth.png
[JNet_580_pretrain_0_labelz_plane]: /experiments/images/JNet_580_pretrain_0_labelz_plane.png
[JNet_580_pretrain_0_novibrate_depth]: /experiments/images/JNet_580_pretrain_0_novibrate_depth.png
[JNet_580_pretrain_0_novibrate_plane]: /experiments/images/JNet_580_pretrain_0_novibrate_plane.png
[JNet_580_pretrain_0_original_depth]: /experiments/images/JNet_580_pretrain_0_original_depth.png
[JNet_580_pretrain_0_original_plane]: /experiments/images/JNet_580_pretrain_0_original_plane.png
[JNet_580_pretrain_0_outputx_depth]: /experiments/images/JNet_580_pretrain_0_outputx_depth.png
[JNet_580_pretrain_0_outputx_plane]: /experiments/images/JNet_580_pretrain_0_outputx_plane.png
[JNet_580_pretrain_0_outputz_depth]: /experiments/images/JNet_580_pretrain_0_outputz_depth.png
[JNet_580_pretrain_0_outputz_plane]: /experiments/images/JNet_580_pretrain_0_outputz_plane.png
[JNet_580_pretrain_1_aligned_depth]: /experiments/images/JNet_580_pretrain_1_aligned_depth.png
[JNet_580_pretrain_1_aligned_plane]: /experiments/images/JNet_580_pretrain_1_aligned_plane.png
[JNet_580_pretrain_1_labelx_depth]: /experiments/images/JNet_580_pretrain_1_labelx_depth.png
[JNet_580_pretrain_1_labelx_plane]: /experiments/images/JNet_580_pretrain_1_labelx_plane.png
[JNet_580_pretrain_1_labelz_depth]: /experiments/images/JNet_580_pretrain_1_labelz_depth.png
[JNet_580_pretrain_1_labelz_plane]: /experiments/images/JNet_580_pretrain_1_labelz_plane.png
[JNet_580_pretrain_1_novibrate_depth]: /experiments/images/JNet_580_pretrain_1_novibrate_depth.png
[JNet_580_pretrain_1_novibrate_plane]: /experiments/images/JNet_580_pretrain_1_novibrate_plane.png
[JNet_580_pretrain_1_original_depth]: /experiments/images/JNet_580_pretrain_1_original_depth.png
[JNet_580_pretrain_1_original_plane]: /experiments/images/JNet_580_pretrain_1_original_plane.png
[JNet_580_pretrain_1_outputx_depth]: /experiments/images/JNet_580_pretrain_1_outputx_depth.png
[JNet_580_pretrain_1_outputx_plane]: /experiments/images/JNet_580_pretrain_1_outputx_plane.png
[JNet_580_pretrain_1_outputz_depth]: /experiments/images/JNet_580_pretrain_1_outputz_depth.png
[JNet_580_pretrain_1_outputz_plane]: /experiments/images/JNet_580_pretrain_1_outputz_plane.png
[JNet_580_pretrain_2_aligned_depth]: /experiments/images/JNet_580_pretrain_2_aligned_depth.png
[JNet_580_pretrain_2_aligned_plane]: /experiments/images/JNet_580_pretrain_2_aligned_plane.png
[JNet_580_pretrain_2_labelx_depth]: /experiments/images/JNet_580_pretrain_2_labelx_depth.png
[JNet_580_pretrain_2_labelx_plane]: /experiments/images/JNet_580_pretrain_2_labelx_plane.png
[JNet_580_pretrain_2_labelz_depth]: /experiments/images/JNet_580_pretrain_2_labelz_depth.png
[JNet_580_pretrain_2_labelz_plane]: /experiments/images/JNet_580_pretrain_2_labelz_plane.png
[JNet_580_pretrain_2_novibrate_depth]: /experiments/images/JNet_580_pretrain_2_novibrate_depth.png
[JNet_580_pretrain_2_novibrate_plane]: /experiments/images/JNet_580_pretrain_2_novibrate_plane.png
[JNet_580_pretrain_2_original_depth]: /experiments/images/JNet_580_pretrain_2_original_depth.png
[JNet_580_pretrain_2_original_plane]: /experiments/images/JNet_580_pretrain_2_original_plane.png
[JNet_580_pretrain_2_outputx_depth]: /experiments/images/JNet_580_pretrain_2_outputx_depth.png
[JNet_580_pretrain_2_outputx_plane]: /experiments/images/JNet_580_pretrain_2_outputx_plane.png
[JNet_580_pretrain_2_outputz_depth]: /experiments/images/JNet_580_pretrain_2_outputz_depth.png
[JNet_580_pretrain_2_outputz_plane]: /experiments/images/JNet_580_pretrain_2_outputz_plane.png
[JNet_580_pretrain_3_aligned_depth]: /experiments/images/JNet_580_pretrain_3_aligned_depth.png
[JNet_580_pretrain_3_aligned_plane]: /experiments/images/JNet_580_pretrain_3_aligned_plane.png
[JNet_580_pretrain_3_labelx_depth]: /experiments/images/JNet_580_pretrain_3_labelx_depth.png
[JNet_580_pretrain_3_labelx_plane]: /experiments/images/JNet_580_pretrain_3_labelx_plane.png
[JNet_580_pretrain_3_labelz_depth]: /experiments/images/JNet_580_pretrain_3_labelz_depth.png
[JNet_580_pretrain_3_labelz_plane]: /experiments/images/JNet_580_pretrain_3_labelz_plane.png
[JNet_580_pretrain_3_novibrate_depth]: /experiments/images/JNet_580_pretrain_3_novibrate_depth.png
[JNet_580_pretrain_3_novibrate_plane]: /experiments/images/JNet_580_pretrain_3_novibrate_plane.png
[JNet_580_pretrain_3_original_depth]: /experiments/images/JNet_580_pretrain_3_original_depth.png
[JNet_580_pretrain_3_original_plane]: /experiments/images/JNet_580_pretrain_3_original_plane.png
[JNet_580_pretrain_3_outputx_depth]: /experiments/images/JNet_580_pretrain_3_outputx_depth.png
[JNet_580_pretrain_3_outputx_plane]: /experiments/images/JNet_580_pretrain_3_outputx_plane.png
[JNet_580_pretrain_3_outputz_depth]: /experiments/images/JNet_580_pretrain_3_outputz_depth.png
[JNet_580_pretrain_3_outputz_plane]: /experiments/images/JNet_580_pretrain_3_outputz_plane.png
[JNet_580_pretrain_4_aligned_depth]: /experiments/images/JNet_580_pretrain_4_aligned_depth.png
[JNet_580_pretrain_4_aligned_plane]: /experiments/images/JNet_580_pretrain_4_aligned_plane.png
[JNet_580_pretrain_4_labelx_depth]: /experiments/images/JNet_580_pretrain_4_labelx_depth.png
[JNet_580_pretrain_4_labelx_plane]: /experiments/images/JNet_580_pretrain_4_labelx_plane.png
[JNet_580_pretrain_4_labelz_depth]: /experiments/images/JNet_580_pretrain_4_labelz_depth.png
[JNet_580_pretrain_4_labelz_plane]: /experiments/images/JNet_580_pretrain_4_labelz_plane.png
[JNet_580_pretrain_4_novibrate_depth]: /experiments/images/JNet_580_pretrain_4_novibrate_depth.png
[JNet_580_pretrain_4_novibrate_plane]: /experiments/images/JNet_580_pretrain_4_novibrate_plane.png
[JNet_580_pretrain_4_original_depth]: /experiments/images/JNet_580_pretrain_4_original_depth.png
[JNet_580_pretrain_4_original_plane]: /experiments/images/JNet_580_pretrain_4_original_plane.png
[JNet_580_pretrain_4_outputx_depth]: /experiments/images/JNet_580_pretrain_4_outputx_depth.png
[JNet_580_pretrain_4_outputx_plane]: /experiments/images/JNet_580_pretrain_4_outputx_plane.png
[JNet_580_pretrain_4_outputz_depth]: /experiments/images/JNet_580_pretrain_4_outputz_depth.png
[JNet_580_pretrain_4_outputz_plane]: /experiments/images/JNet_580_pretrain_4_outputz_plane.png
[JNet_580_pretrain_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi000_im000._heatmap_depth.png
[JNet_580_pretrain_beads_roi000_im000._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi000_im000._original_depth.png
[JNet_580_pretrain_beads_roi000_im000._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi000_im000._output_depth.png
[JNet_580_pretrain_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi000_im000._reconst_depth.png
[JNet_580_pretrain_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi001_im004._heatmap_depth.png
[JNet_580_pretrain_beads_roi001_im004._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi001_im004._original_depth.png
[JNet_580_pretrain_beads_roi001_im004._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi001_im004._output_depth.png
[JNet_580_pretrain_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi001_im004._reconst_depth.png
[JNet_580_pretrain_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi002_im005._heatmap_depth.png
[JNet_580_pretrain_beads_roi002_im005._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi002_im005._original_depth.png
[JNet_580_pretrain_beads_roi002_im005._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi002_im005._output_depth.png
[JNet_580_pretrain_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi002_im005._reconst_depth.png
[JNet_580_pretrain_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi003_im006._heatmap_depth.png
[JNet_580_pretrain_beads_roi003_im006._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi003_im006._original_depth.png
[JNet_580_pretrain_beads_roi003_im006._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi003_im006._output_depth.png
[JNet_580_pretrain_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi003_im006._reconst_depth.png
[JNet_580_pretrain_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi004_im006._heatmap_depth.png
[JNet_580_pretrain_beads_roi004_im006._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi004_im006._original_depth.png
[JNet_580_pretrain_beads_roi004_im006._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi004_im006._output_depth.png
[JNet_580_pretrain_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi004_im006._reconst_depth.png
[JNet_580_pretrain_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi005_im007._heatmap_depth.png
[JNet_580_pretrain_beads_roi005_im007._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi005_im007._original_depth.png
[JNet_580_pretrain_beads_roi005_im007._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi005_im007._output_depth.png
[JNet_580_pretrain_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi005_im007._reconst_depth.png
[JNet_580_pretrain_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi006_im008._heatmap_depth.png
[JNet_580_pretrain_beads_roi006_im008._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi006_im008._original_depth.png
[JNet_580_pretrain_beads_roi006_im008._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi006_im008._output_depth.png
[JNet_580_pretrain_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi006_im008._reconst_depth.png
[JNet_580_pretrain_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi007_im009._heatmap_depth.png
[JNet_580_pretrain_beads_roi007_im009._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi007_im009._original_depth.png
[JNet_580_pretrain_beads_roi007_im009._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi007_im009._output_depth.png
[JNet_580_pretrain_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi007_im009._reconst_depth.png
[JNet_580_pretrain_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi008_im010._heatmap_depth.png
[JNet_580_pretrain_beads_roi008_im010._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi008_im010._original_depth.png
[JNet_580_pretrain_beads_roi008_im010._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi008_im010._output_depth.png
[JNet_580_pretrain_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi008_im010._reconst_depth.png
[JNet_580_pretrain_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi009_im011._heatmap_depth.png
[JNet_580_pretrain_beads_roi009_im011._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi009_im011._original_depth.png
[JNet_580_pretrain_beads_roi009_im011._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi009_im011._output_depth.png
[JNet_580_pretrain_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi009_im011._reconst_depth.png
[JNet_580_pretrain_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi010_im012._heatmap_depth.png
[JNet_580_pretrain_beads_roi010_im012._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi010_im012._original_depth.png
[JNet_580_pretrain_beads_roi010_im012._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi010_im012._output_depth.png
[JNet_580_pretrain_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi010_im012._reconst_depth.png
[JNet_580_pretrain_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi011_im013._heatmap_depth.png
[JNet_580_pretrain_beads_roi011_im013._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi011_im013._original_depth.png
[JNet_580_pretrain_beads_roi011_im013._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi011_im013._output_depth.png
[JNet_580_pretrain_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi011_im013._reconst_depth.png
[JNet_580_pretrain_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi012_im014._heatmap_depth.png
[JNet_580_pretrain_beads_roi012_im014._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi012_im014._original_depth.png
[JNet_580_pretrain_beads_roi012_im014._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi012_im014._output_depth.png
[JNet_580_pretrain_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi012_im014._reconst_depth.png
[JNet_580_pretrain_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi013_im015._heatmap_depth.png
[JNet_580_pretrain_beads_roi013_im015._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi013_im015._original_depth.png
[JNet_580_pretrain_beads_roi013_im015._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi013_im015._output_depth.png
[JNet_580_pretrain_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi013_im015._reconst_depth.png
[JNet_580_pretrain_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi014_im016._heatmap_depth.png
[JNet_580_pretrain_beads_roi014_im016._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi014_im016._original_depth.png
[JNet_580_pretrain_beads_roi014_im016._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi014_im016._output_depth.png
[JNet_580_pretrain_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi014_im016._reconst_depth.png
[JNet_580_pretrain_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi015_im017._heatmap_depth.png
[JNet_580_pretrain_beads_roi015_im017._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi015_im017._original_depth.png
[JNet_580_pretrain_beads_roi015_im017._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi015_im017._output_depth.png
[JNet_580_pretrain_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi015_im017._reconst_depth.png
[JNet_580_pretrain_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi016_im018._heatmap_depth.png
[JNet_580_pretrain_beads_roi016_im018._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi016_im018._original_depth.png
[JNet_580_pretrain_beads_roi016_im018._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi016_im018._output_depth.png
[JNet_580_pretrain_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi016_im018._reconst_depth.png
[JNet_580_pretrain_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi017_im018._heatmap_depth.png
[JNet_580_pretrain_beads_roi017_im018._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi017_im018._original_depth.png
[JNet_580_pretrain_beads_roi017_im018._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi017_im018._output_depth.png
[JNet_580_pretrain_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi017_im018._reconst_depth.png
[JNet_580_pretrain_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi018_im022._heatmap_depth.png
[JNet_580_pretrain_beads_roi018_im022._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi018_im022._original_depth.png
[JNet_580_pretrain_beads_roi018_im022._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi018_im022._output_depth.png
[JNet_580_pretrain_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi018_im022._reconst_depth.png
[JNet_580_pretrain_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi019_im023._heatmap_depth.png
[JNet_580_pretrain_beads_roi019_im023._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi019_im023._original_depth.png
[JNet_580_pretrain_beads_roi019_im023._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi019_im023._output_depth.png
[JNet_580_pretrain_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi019_im023._reconst_depth.png
[JNet_580_pretrain_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi020_im024._heatmap_depth.png
[JNet_580_pretrain_beads_roi020_im024._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi020_im024._original_depth.png
[JNet_580_pretrain_beads_roi020_im024._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi020_im024._output_depth.png
[JNet_580_pretrain_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi020_im024._reconst_depth.png
[JNet_580_pretrain_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi021_im026._heatmap_depth.png
[JNet_580_pretrain_beads_roi021_im026._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi021_im026._original_depth.png
[JNet_580_pretrain_beads_roi021_im026._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi021_im026._output_depth.png
[JNet_580_pretrain_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi021_im026._reconst_depth.png
[JNet_580_pretrain_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi022_im027._heatmap_depth.png
[JNet_580_pretrain_beads_roi022_im027._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi022_im027._original_depth.png
[JNet_580_pretrain_beads_roi022_im027._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi022_im027._output_depth.png
[JNet_580_pretrain_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi022_im027._reconst_depth.png
[JNet_580_pretrain_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi023_im028._heatmap_depth.png
[JNet_580_pretrain_beads_roi023_im028._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi023_im028._original_depth.png
[JNet_580_pretrain_beads_roi023_im028._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi023_im028._output_depth.png
[JNet_580_pretrain_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi023_im028._reconst_depth.png
[JNet_580_pretrain_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi024_im028._heatmap_depth.png
[JNet_580_pretrain_beads_roi024_im028._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi024_im028._original_depth.png
[JNet_580_pretrain_beads_roi024_im028._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi024_im028._output_depth.png
[JNet_580_pretrain_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi024_im028._reconst_depth.png
[JNet_580_pretrain_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi025_im028._heatmap_depth.png
[JNet_580_pretrain_beads_roi025_im028._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi025_im028._original_depth.png
[JNet_580_pretrain_beads_roi025_im028._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi025_im028._output_depth.png
[JNet_580_pretrain_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi025_im028._reconst_depth.png
[JNet_580_pretrain_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi026_im029._heatmap_depth.png
[JNet_580_pretrain_beads_roi026_im029._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi026_im029._original_depth.png
[JNet_580_pretrain_beads_roi026_im029._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi026_im029._output_depth.png
[JNet_580_pretrain_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi026_im029._reconst_depth.png
[JNet_580_pretrain_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi027_im029._heatmap_depth.png
[JNet_580_pretrain_beads_roi027_im029._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi027_im029._original_depth.png
[JNet_580_pretrain_beads_roi027_im029._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi027_im029._output_depth.png
[JNet_580_pretrain_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi027_im029._reconst_depth.png
[JNet_580_pretrain_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi028_im030._heatmap_depth.png
[JNet_580_pretrain_beads_roi028_im030._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi028_im030._original_depth.png
[JNet_580_pretrain_beads_roi028_im030._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi028_im030._output_depth.png
[JNet_580_pretrain_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi028_im030._reconst_depth.png
[JNet_580_pretrain_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_580_pretrain_beads_roi029_im030._heatmap_depth.png
[JNet_580_pretrain_beads_roi029_im030._original_depth]: /experiments/images/JNet_580_pretrain_beads_roi029_im030._original_depth.png
[JNet_580_pretrain_beads_roi029_im030._output_depth]: /experiments/images/JNet_580_pretrain_beads_roi029_im030._output_depth.png
[JNet_580_pretrain_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_580_pretrain_beads_roi029_im030._reconst_depth.png
[JNet_581_cv0_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi000_im000._heatmap_depth.png
[JNet_581_cv0_beads_roi000_im000._original_depth]: /experiments/images/JNet_581_cv0_beads_roi000_im000._original_depth.png
[JNet_581_cv0_beads_roi000_im000._output_depth]: /experiments/images/JNet_581_cv0_beads_roi000_im000._output_depth.png
[JNet_581_cv0_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi000_im000._reconst_depth.png
[JNet_581_cv0_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi001_im004._heatmap_depth.png
[JNet_581_cv0_beads_roi001_im004._original_depth]: /experiments/images/JNet_581_cv0_beads_roi001_im004._original_depth.png
[JNet_581_cv0_beads_roi001_im004._output_depth]: /experiments/images/JNet_581_cv0_beads_roi001_im004._output_depth.png
[JNet_581_cv0_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi001_im004._reconst_depth.png
[JNet_581_cv0_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi002_im005._heatmap_depth.png
[JNet_581_cv0_beads_roi002_im005._original_depth]: /experiments/images/JNet_581_cv0_beads_roi002_im005._original_depth.png
[JNet_581_cv0_beads_roi002_im005._output_depth]: /experiments/images/JNet_581_cv0_beads_roi002_im005._output_depth.png
[JNet_581_cv0_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi002_im005._reconst_depth.png
[JNet_581_cv0_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi003_im006._heatmap_depth.png
[JNet_581_cv0_beads_roi003_im006._original_depth]: /experiments/images/JNet_581_cv0_beads_roi003_im006._original_depth.png
[JNet_581_cv0_beads_roi003_im006._output_depth]: /experiments/images/JNet_581_cv0_beads_roi003_im006._output_depth.png
[JNet_581_cv0_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi003_im006._reconst_depth.png
[JNet_581_cv0_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi004_im006._heatmap_depth.png
[JNet_581_cv0_beads_roi004_im006._original_depth]: /experiments/images/JNet_581_cv0_beads_roi004_im006._original_depth.png
[JNet_581_cv0_beads_roi004_im006._output_depth]: /experiments/images/JNet_581_cv0_beads_roi004_im006._output_depth.png
[JNet_581_cv0_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi004_im006._reconst_depth.png
[JNet_581_cv0_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi005_im007._heatmap_depth.png
[JNet_581_cv0_beads_roi005_im007._original_depth]: /experiments/images/JNet_581_cv0_beads_roi005_im007._original_depth.png
[JNet_581_cv0_beads_roi005_im007._output_depth]: /experiments/images/JNet_581_cv0_beads_roi005_im007._output_depth.png
[JNet_581_cv0_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi005_im007._reconst_depth.png
[JNet_581_cv0_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi006_im008._heatmap_depth.png
[JNet_581_cv0_beads_roi006_im008._original_depth]: /experiments/images/JNet_581_cv0_beads_roi006_im008._original_depth.png
[JNet_581_cv0_beads_roi006_im008._output_depth]: /experiments/images/JNet_581_cv0_beads_roi006_im008._output_depth.png
[JNet_581_cv0_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi006_im008._reconst_depth.png
[JNet_581_cv0_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi007_im009._heatmap_depth.png
[JNet_581_cv0_beads_roi007_im009._original_depth]: /experiments/images/JNet_581_cv0_beads_roi007_im009._original_depth.png
[JNet_581_cv0_beads_roi007_im009._output_depth]: /experiments/images/JNet_581_cv0_beads_roi007_im009._output_depth.png
[JNet_581_cv0_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi007_im009._reconst_depth.png
[JNet_581_cv0_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi008_im010._heatmap_depth.png
[JNet_581_cv0_beads_roi008_im010._original_depth]: /experiments/images/JNet_581_cv0_beads_roi008_im010._original_depth.png
[JNet_581_cv0_beads_roi008_im010._output_depth]: /experiments/images/JNet_581_cv0_beads_roi008_im010._output_depth.png
[JNet_581_cv0_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi008_im010._reconst_depth.png
[JNet_581_cv0_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi009_im011._heatmap_depth.png
[JNet_581_cv0_beads_roi009_im011._original_depth]: /experiments/images/JNet_581_cv0_beads_roi009_im011._original_depth.png
[JNet_581_cv0_beads_roi009_im011._output_depth]: /experiments/images/JNet_581_cv0_beads_roi009_im011._output_depth.png
[JNet_581_cv0_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi009_im011._reconst_depth.png
[JNet_581_cv0_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi010_im012._heatmap_depth.png
[JNet_581_cv0_beads_roi010_im012._original_depth]: /experiments/images/JNet_581_cv0_beads_roi010_im012._original_depth.png
[JNet_581_cv0_beads_roi010_im012._output_depth]: /experiments/images/JNet_581_cv0_beads_roi010_im012._output_depth.png
[JNet_581_cv0_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi010_im012._reconst_depth.png
[JNet_581_cv0_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi011_im013._heatmap_depth.png
[JNet_581_cv0_beads_roi011_im013._original_depth]: /experiments/images/JNet_581_cv0_beads_roi011_im013._original_depth.png
[JNet_581_cv0_beads_roi011_im013._output_depth]: /experiments/images/JNet_581_cv0_beads_roi011_im013._output_depth.png
[JNet_581_cv0_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi011_im013._reconst_depth.png
[JNet_581_cv0_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi012_im014._heatmap_depth.png
[JNet_581_cv0_beads_roi012_im014._original_depth]: /experiments/images/JNet_581_cv0_beads_roi012_im014._original_depth.png
[JNet_581_cv0_beads_roi012_im014._output_depth]: /experiments/images/JNet_581_cv0_beads_roi012_im014._output_depth.png
[JNet_581_cv0_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi012_im014._reconst_depth.png
[JNet_581_cv0_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi013_im015._heatmap_depth.png
[JNet_581_cv0_beads_roi013_im015._original_depth]: /experiments/images/JNet_581_cv0_beads_roi013_im015._original_depth.png
[JNet_581_cv0_beads_roi013_im015._output_depth]: /experiments/images/JNet_581_cv0_beads_roi013_im015._output_depth.png
[JNet_581_cv0_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi013_im015._reconst_depth.png
[JNet_581_cv0_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi014_im016._heatmap_depth.png
[JNet_581_cv0_beads_roi014_im016._original_depth]: /experiments/images/JNet_581_cv0_beads_roi014_im016._original_depth.png
[JNet_581_cv0_beads_roi014_im016._output_depth]: /experiments/images/JNet_581_cv0_beads_roi014_im016._output_depth.png
[JNet_581_cv0_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi014_im016._reconst_depth.png
[JNet_581_cv0_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi015_im017._heatmap_depth.png
[JNet_581_cv0_beads_roi015_im017._original_depth]: /experiments/images/JNet_581_cv0_beads_roi015_im017._original_depth.png
[JNet_581_cv0_beads_roi015_im017._output_depth]: /experiments/images/JNet_581_cv0_beads_roi015_im017._output_depth.png
[JNet_581_cv0_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi015_im017._reconst_depth.png
[JNet_581_cv0_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi016_im018._heatmap_depth.png
[JNet_581_cv0_beads_roi016_im018._original_depth]: /experiments/images/JNet_581_cv0_beads_roi016_im018._original_depth.png
[JNet_581_cv0_beads_roi016_im018._output_depth]: /experiments/images/JNet_581_cv0_beads_roi016_im018._output_depth.png
[JNet_581_cv0_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi016_im018._reconst_depth.png
[JNet_581_cv0_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi017_im018._heatmap_depth.png
[JNet_581_cv0_beads_roi017_im018._original_depth]: /experiments/images/JNet_581_cv0_beads_roi017_im018._original_depth.png
[JNet_581_cv0_beads_roi017_im018._output_depth]: /experiments/images/JNet_581_cv0_beads_roi017_im018._output_depth.png
[JNet_581_cv0_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi017_im018._reconst_depth.png
[JNet_581_cv0_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi018_im022._heatmap_depth.png
[JNet_581_cv0_beads_roi018_im022._original_depth]: /experiments/images/JNet_581_cv0_beads_roi018_im022._original_depth.png
[JNet_581_cv0_beads_roi018_im022._output_depth]: /experiments/images/JNet_581_cv0_beads_roi018_im022._output_depth.png
[JNet_581_cv0_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi018_im022._reconst_depth.png
[JNet_581_cv0_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi019_im023._heatmap_depth.png
[JNet_581_cv0_beads_roi019_im023._original_depth]: /experiments/images/JNet_581_cv0_beads_roi019_im023._original_depth.png
[JNet_581_cv0_beads_roi019_im023._output_depth]: /experiments/images/JNet_581_cv0_beads_roi019_im023._output_depth.png
[JNet_581_cv0_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi019_im023._reconst_depth.png
[JNet_581_cv0_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi020_im024._heatmap_depth.png
[JNet_581_cv0_beads_roi020_im024._original_depth]: /experiments/images/JNet_581_cv0_beads_roi020_im024._original_depth.png
[JNet_581_cv0_beads_roi020_im024._output_depth]: /experiments/images/JNet_581_cv0_beads_roi020_im024._output_depth.png
[JNet_581_cv0_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi020_im024._reconst_depth.png
[JNet_581_cv0_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi021_im026._heatmap_depth.png
[JNet_581_cv0_beads_roi021_im026._original_depth]: /experiments/images/JNet_581_cv0_beads_roi021_im026._original_depth.png
[JNet_581_cv0_beads_roi021_im026._output_depth]: /experiments/images/JNet_581_cv0_beads_roi021_im026._output_depth.png
[JNet_581_cv0_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi021_im026._reconst_depth.png
[JNet_581_cv0_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi022_im027._heatmap_depth.png
[JNet_581_cv0_beads_roi022_im027._original_depth]: /experiments/images/JNet_581_cv0_beads_roi022_im027._original_depth.png
[JNet_581_cv0_beads_roi022_im027._output_depth]: /experiments/images/JNet_581_cv0_beads_roi022_im027._output_depth.png
[JNet_581_cv0_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi022_im027._reconst_depth.png
[JNet_581_cv0_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi023_im028._heatmap_depth.png
[JNet_581_cv0_beads_roi023_im028._original_depth]: /experiments/images/JNet_581_cv0_beads_roi023_im028._original_depth.png
[JNet_581_cv0_beads_roi023_im028._output_depth]: /experiments/images/JNet_581_cv0_beads_roi023_im028._output_depth.png
[JNet_581_cv0_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi023_im028._reconst_depth.png
[JNet_581_cv0_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi024_im028._heatmap_depth.png
[JNet_581_cv0_beads_roi024_im028._original_depth]: /experiments/images/JNet_581_cv0_beads_roi024_im028._original_depth.png
[JNet_581_cv0_beads_roi024_im028._output_depth]: /experiments/images/JNet_581_cv0_beads_roi024_im028._output_depth.png
[JNet_581_cv0_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi024_im028._reconst_depth.png
[JNet_581_cv0_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi025_im028._heatmap_depth.png
[JNet_581_cv0_beads_roi025_im028._original_depth]: /experiments/images/JNet_581_cv0_beads_roi025_im028._original_depth.png
[JNet_581_cv0_beads_roi025_im028._output_depth]: /experiments/images/JNet_581_cv0_beads_roi025_im028._output_depth.png
[JNet_581_cv0_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi025_im028._reconst_depth.png
[JNet_581_cv0_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi026_im029._heatmap_depth.png
[JNet_581_cv0_beads_roi026_im029._original_depth]: /experiments/images/JNet_581_cv0_beads_roi026_im029._original_depth.png
[JNet_581_cv0_beads_roi026_im029._output_depth]: /experiments/images/JNet_581_cv0_beads_roi026_im029._output_depth.png
[JNet_581_cv0_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi026_im029._reconst_depth.png
[JNet_581_cv0_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi027_im029._heatmap_depth.png
[JNet_581_cv0_beads_roi027_im029._original_depth]: /experiments/images/JNet_581_cv0_beads_roi027_im029._original_depth.png
[JNet_581_cv0_beads_roi027_im029._output_depth]: /experiments/images/JNet_581_cv0_beads_roi027_im029._output_depth.png
[JNet_581_cv0_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi027_im029._reconst_depth.png
[JNet_581_cv0_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi028_im030._heatmap_depth.png
[JNet_581_cv0_beads_roi028_im030._original_depth]: /experiments/images/JNet_581_cv0_beads_roi028_im030._original_depth.png
[JNet_581_cv0_beads_roi028_im030._output_depth]: /experiments/images/JNet_581_cv0_beads_roi028_im030._output_depth.png
[JNet_581_cv0_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi028_im030._reconst_depth.png
[JNet_581_cv0_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_581_cv0_beads_roi029_im030._heatmap_depth.png
[JNet_581_cv0_beads_roi029_im030._original_depth]: /experiments/images/JNet_581_cv0_beads_roi029_im030._original_depth.png
[JNet_581_cv0_beads_roi029_im030._output_depth]: /experiments/images/JNet_581_cv0_beads_roi029_im030._output_depth.png
[JNet_581_cv0_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_581_cv0_beads_roi029_im030._reconst_depth.png
[JNet_581_cv0_psf_post]: /experiments/images/JNet_581_cv0_psf_post.png
[JNet_581_cv0_psf_pre]: /experiments/images/JNet_581_cv0_psf_pre.png
