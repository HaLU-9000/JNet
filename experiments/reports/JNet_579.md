



# JNet_579 Report
  
first beads experiment with new methods and data, low psf loss  
pretrained model : JNet_577_pretrain
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
|wavelength|1.1|microns|
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
|poisson_weight|0.01||
|sig_eps|0.01||
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
|qloss_weight|10.0|
|ploss_weight|1.0|
|mrfloss_order|1|
|mrfloss_dilation|1|
|mrfloss_weights|{'l_00': 0, 'l_01': 0, 'l_10': 0, 'l_11': 0}|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results

### Pretraining
  
Segmentation: mean MSE: 0.010337905026972294, mean BCE: 0.04520108178257942  
Luminance Estimation: mean MSE: 0.9723044633865356, mean BCE: 6.825087070465088
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_0_original_plane]|![JNet_577_pretrain_0_novibrate_plane]|![JNet_577_pretrain_0_aligned_plane]|![JNet_577_pretrain_0_outputx_plane]|![JNet_577_pretrain_0_labelx_plane]|![JNet_577_pretrain_0_outputz_plane]|![JNet_577_pretrain_0_labelz_plane]|
  
MSEx: 0.007159891538321972, BCEx: 0.032453302294015884  
MSEz: 0.9866166710853577, BCEz: 7.214510440826416  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_0_original_depth]|![JNet_577_pretrain_0_novibrate_depth]|![JNet_577_pretrain_0_aligned_depth]|![JNet_577_pretrain_0_outputx_depth]|![JNet_577_pretrain_0_labelx_depth]|![JNet_577_pretrain_0_outputz_depth]|![JNet_577_pretrain_0_labelz_depth]|
  
MSEx: 0.007159891538321972, BCEx: 0.032453302294015884  
MSEz: 0.9866166710853577, BCEz: 7.214510440826416  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_1_original_plane]|![JNet_577_pretrain_1_novibrate_plane]|![JNet_577_pretrain_1_aligned_plane]|![JNet_577_pretrain_1_outputx_plane]|![JNet_577_pretrain_1_labelx_plane]|![JNet_577_pretrain_1_outputz_plane]|![JNet_577_pretrain_1_labelz_plane]|
  
MSEx: 0.007058662828058004, BCEx: 0.033639878034591675  
MSEz: 0.991182267665863, BCEz: 7.360293865203857  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_1_original_depth]|![JNet_577_pretrain_1_novibrate_depth]|![JNet_577_pretrain_1_aligned_depth]|![JNet_577_pretrain_1_outputx_depth]|![JNet_577_pretrain_1_labelx_depth]|![JNet_577_pretrain_1_outputz_depth]|![JNet_577_pretrain_1_labelz_depth]|
  
MSEx: 0.007058662828058004, BCEx: 0.033639878034591675  
MSEz: 0.991182267665863, BCEz: 7.360293865203857  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_2_original_plane]|![JNet_577_pretrain_2_novibrate_plane]|![JNet_577_pretrain_2_aligned_plane]|![JNet_577_pretrain_2_outputx_plane]|![JNet_577_pretrain_2_labelx_plane]|![JNet_577_pretrain_2_outputz_plane]|![JNet_577_pretrain_2_labelz_plane]|
  
MSEx: 0.012985104694962502, BCEx: 0.05460256338119507  
MSEz: 0.9596036672592163, BCEz: 6.706027984619141  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_2_original_depth]|![JNet_577_pretrain_2_novibrate_depth]|![JNet_577_pretrain_2_aligned_depth]|![JNet_577_pretrain_2_outputx_depth]|![JNet_577_pretrain_2_labelx_depth]|![JNet_577_pretrain_2_outputz_depth]|![JNet_577_pretrain_2_labelz_depth]|
  
MSEx: 0.012985104694962502, BCEx: 0.05460256338119507  
MSEz: 0.9596036672592163, BCEz: 6.706027984619141  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_3_original_plane]|![JNet_577_pretrain_3_novibrate_plane]|![JNet_577_pretrain_3_aligned_plane]|![JNet_577_pretrain_3_outputx_plane]|![JNet_577_pretrain_3_labelx_plane]|![JNet_577_pretrain_3_outputz_plane]|![JNet_577_pretrain_3_labelz_plane]|
  
MSEx: 0.015148026868700981, BCEx: 0.06365688145160675  
MSEz: 0.9521401524543762, BCEz: 5.982589244842529  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_3_original_depth]|![JNet_577_pretrain_3_novibrate_depth]|![JNet_577_pretrain_3_aligned_depth]|![JNet_577_pretrain_3_outputx_depth]|![JNet_577_pretrain_3_labelx_depth]|![JNet_577_pretrain_3_outputz_depth]|![JNet_577_pretrain_3_labelz_depth]|
  
MSEx: 0.015148026868700981, BCEx: 0.06365688145160675  
MSEz: 0.9521401524543762, BCEz: 5.982589244842529  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_4_original_plane]|![JNet_577_pretrain_4_novibrate_plane]|![JNet_577_pretrain_4_aligned_plane]|![JNet_577_pretrain_4_outputx_plane]|![JNet_577_pretrain_4_labelx_plane]|![JNet_577_pretrain_4_outputz_plane]|![JNet_577_pretrain_4_labelz_plane]|
  
MSEx: 0.009337838739156723, BCEx: 0.041652776300907135  
MSEz: 0.9719793796539307, BCEz: 6.86201286315918  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_4_original_depth]|![JNet_577_pretrain_4_novibrate_depth]|![JNet_577_pretrain_4_aligned_depth]|![JNet_577_pretrain_4_outputx_depth]|![JNet_577_pretrain_4_labelx_depth]|![JNet_577_pretrain_4_outputz_depth]|![JNet_577_pretrain_4_labelz_depth]|
  
MSEx: 0.009337838739156723, BCEx: 0.041652776300907135  
MSEz: 0.9719793796539307, BCEz: 6.86201286315918  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi000_im000._original_depth]|![JNet_577_pretrain_beads_roi000_im000._output_depth]|![JNet_577_pretrain_beads_roi000_im000._reconst_depth]|![JNet_577_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 2.3598649902343753, MSE: 0.001404265407472849, quantized loss: 0.00020025100093334913  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi001_im004._original_depth]|![JNet_577_pretrain_beads_roi001_im004._output_depth]|![JNet_577_pretrain_beads_roi001_im004._reconst_depth]|![JNet_577_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 2.7663908691406256, MSE: 0.0014693336561322212, quantized loss: 0.00022968591656535864  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi002_im005._original_depth]|![JNet_577_pretrain_beads_roi002_im005._output_depth]|![JNet_577_pretrain_beads_roi002_im005._reconst_depth]|![JNet_577_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 2.4870258789062505, MSE: 0.001385921728797257, quantized loss: 0.0002116434625349939  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi003_im006._original_depth]|![JNet_577_pretrain_beads_roi003_im006._output_depth]|![JNet_577_pretrain_beads_roi003_im006._reconst_depth]|![JNet_577_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 2.4808164062500007, MSE: 0.00140003499109298, quantized loss: 0.00021589089010376483  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi004_im006._original_depth]|![JNet_577_pretrain_beads_roi004_im006._output_depth]|![JNet_577_pretrain_beads_roi004_im006._reconst_depth]|![JNet_577_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 2.5957116699218754, MSE: 0.0014186768094077706, quantized loss: 0.00022389761579688638  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi005_im007._original_depth]|![JNet_577_pretrain_beads_roi005_im007._output_depth]|![JNet_577_pretrain_beads_roi005_im007._reconst_depth]|![JNet_577_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 2.4471579589843757, MSE: 0.0013926824321970344, quantized loss: 0.00020856257469858974  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi006_im008._original_depth]|![JNet_577_pretrain_beads_roi006_im008._output_depth]|![JNet_577_pretrain_beads_roi006_im008._reconst_depth]|![JNet_577_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 2.5683923339843755, MSE: 0.00132855458650738, quantized loss: 0.0002385183615842834  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi007_im009._original_depth]|![JNet_577_pretrain_beads_roi007_im009._output_depth]|![JNet_577_pretrain_beads_roi007_im009._reconst_depth]|![JNet_577_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 2.5956484375000004, MSE: 0.0013968636048957705, quantized loss: 0.0002299387997481972  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi008_im010._original_depth]|![JNet_577_pretrain_beads_roi008_im010._output_depth]|![JNet_577_pretrain_beads_roi008_im010._reconst_depth]|![JNet_577_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 2.5861645507812505, MSE: 0.0013851559488102794, quantized loss: 0.00021405029110610485  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi009_im011._original_depth]|![JNet_577_pretrain_beads_roi009_im011._output_depth]|![JNet_577_pretrain_beads_roi009_im011._reconst_depth]|![JNet_577_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 2.4256123046875007, MSE: 0.0013660675613209605, quantized loss: 0.00020359197515062988  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi010_im012._original_depth]|![JNet_577_pretrain_beads_roi010_im012._output_depth]|![JNet_577_pretrain_beads_roi010_im012._reconst_depth]|![JNet_577_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 2.845332031250001, MSE: 0.0014413491589948535, quantized loss: 0.00023556192172691226  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi011_im013._original_depth]|![JNet_577_pretrain_beads_roi011_im013._output_depth]|![JNet_577_pretrain_beads_roi011_im013._reconst_depth]|![JNet_577_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 2.8340922851562507, MSE: 0.0014216724084690213, quantized loss: 0.00023284662165679038  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi012_im014._original_depth]|![JNet_577_pretrain_beads_roi012_im014._output_depth]|![JNet_577_pretrain_beads_roi012_im014._reconst_depth]|![JNet_577_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 2.4731479492187507, MSE: 0.0015558806480839849, quantized loss: 0.0002060716797132045  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi013_im015._original_depth]|![JNet_577_pretrain_beads_roi013_im015._output_depth]|![JNet_577_pretrain_beads_roi013_im015._reconst_depth]|![JNet_577_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 2.3054753417968756, MSE: 0.001462513580918312, quantized loss: 0.00019672267080750316  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi014_im016._original_depth]|![JNet_577_pretrain_beads_roi014_im016._output_depth]|![JNet_577_pretrain_beads_roi014_im016._reconst_depth]|![JNet_577_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 2.2932824707031254, MSE: 0.0013346198247745633, quantized loss: 0.0002083022554870695  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi015_im017._original_depth]|![JNet_577_pretrain_beads_roi015_im017._output_depth]|![JNet_577_pretrain_beads_roi015_im017._reconst_depth]|![JNet_577_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 2.3576174316406258, MSE: 0.0013726981123909354, quantized loss: 0.0002034721983363852  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi016_im018._original_depth]|![JNet_577_pretrain_beads_roi016_im018._output_depth]|![JNet_577_pretrain_beads_roi016_im018._reconst_depth]|![JNet_577_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 2.6444357910156255, MSE: 0.0015119657618924975, quantized loss: 0.00021956494310870767  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi017_im018._original_depth]|![JNet_577_pretrain_beads_roi017_im018._output_depth]|![JNet_577_pretrain_beads_roi017_im018._reconst_depth]|![JNet_577_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 2.5637832031250007, MSE: 0.0015507112257182598, quantized loss: 0.00021255132742226124  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi018_im022._original_depth]|![JNet_577_pretrain_beads_roi018_im022._output_depth]|![JNet_577_pretrain_beads_roi018_im022._reconst_depth]|![JNet_577_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 2.1116445312500005, MSE: 0.0013562639942392707, quantized loss: 0.00018390154582448304  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi019_im023._original_depth]|![JNet_577_pretrain_beads_roi019_im023._output_depth]|![JNet_577_pretrain_beads_roi019_im023._reconst_depth]|![JNet_577_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 2.0851108398437503, MSE: 0.0013699062401428819, quantized loss: 0.00018189134425483644  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi020_im024._original_depth]|![JNet_577_pretrain_beads_roi020_im024._output_depth]|![JNet_577_pretrain_beads_roi020_im024._reconst_depth]|![JNet_577_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 2.6543344726562506, MSE: 0.001444765250198543, quantized loss: 0.00021708586427848786  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi021_im026._original_depth]|![JNet_577_pretrain_beads_roi021_im026._output_depth]|![JNet_577_pretrain_beads_roi021_im026._reconst_depth]|![JNet_577_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 2.5608222656250006, MSE: 0.0013643779093399644, quantized loss: 0.00021245217067189515  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi022_im027._original_depth]|![JNet_577_pretrain_beads_roi022_im027._output_depth]|![JNet_577_pretrain_beads_roi022_im027._reconst_depth]|![JNet_577_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 2.4805200195312507, MSE: 0.001436517108231783, quantized loss: 0.00020371450227685273  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi023_im028._original_depth]|![JNet_577_pretrain_beads_roi023_im028._output_depth]|![JNet_577_pretrain_beads_roi023_im028._reconst_depth]|![JNet_577_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 2.6727846679687506, MSE: 0.0012618092587217689, quantized loss: 0.00022985525720287114  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi024_im028._original_depth]|![JNet_577_pretrain_beads_roi024_im028._output_depth]|![JNet_577_pretrain_beads_roi024_im028._reconst_depth]|![JNet_577_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 2.6107016601562507, MSE: 0.0013169337762519717, quantized loss: 0.00021999599994160235  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi025_im028._original_depth]|![JNet_577_pretrain_beads_roi025_im028._output_depth]|![JNet_577_pretrain_beads_roi025_im028._reconst_depth]|![JNet_577_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 2.6107016601562507, MSE: 0.0013169337762519717, quantized loss: 0.00021999599994160235  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi026_im029._original_depth]|![JNet_577_pretrain_beads_roi026_im029._output_depth]|![JNet_577_pretrain_beads_roi026_im029._reconst_depth]|![JNet_577_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 2.7132121582031257, MSE: 0.0014740973711013794, quantized loss: 0.00022030209947843105  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi027_im029._original_depth]|![JNet_577_pretrain_beads_roi027_im029._output_depth]|![JNet_577_pretrain_beads_roi027_im029._reconst_depth]|![JNet_577_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 2.3730769042968753, MSE: 0.0014078825479373336, quantized loss: 0.00020174859673716128  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi028_im030._original_depth]|![JNet_577_pretrain_beads_roi028_im030._output_depth]|![JNet_577_pretrain_beads_roi028_im030._reconst_depth]|![JNet_577_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 2.3096613769531253, MSE: 0.0013790602097287774, quantized loss: 0.00019784833420999348  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi029_im030._original_depth]|![JNet_577_pretrain_beads_roi029_im030._output_depth]|![JNet_577_pretrain_beads_roi029_im030._reconst_depth]|![JNet_577_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 2.4600390625000004, MSE: 0.0014401735970750451, quantized loss: 0.00020704904454760253  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi000_im000._original_depth]|![JNet_579_beads_roi000_im000._output_depth]|![JNet_579_beads_roi000_im000._reconst_depth]|![JNet_579_beads_roi000_im000._heatmap_depth]|
  
volume: 2.0095283203125005, MSE: 0.0017165719764307141, quantized loss: 6.549225508933887e-05  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi001_im004._original_depth]|![JNet_579_beads_roi001_im004._output_depth]|![JNet_579_beads_roi001_im004._reconst_depth]|![JNet_579_beads_roi001_im004._heatmap_depth]|
  
volume: 2.3698281250000006, MSE: 0.0021094114053994417, quantized loss: 9.102461626753211e-05  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi002_im005._original_depth]|![JNet_579_beads_roi002_im005._output_depth]|![JNet_579_beads_roi002_im005._reconst_depth]|![JNet_579_beads_roi002_im005._heatmap_depth]|
  
volume: 2.0925373535156253, MSE: 0.0019165613921359181, quantized loss: 7.164602720877156e-05  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi003_im006._original_depth]|![JNet_579_beads_roi003_im006._output_depth]|![JNet_579_beads_roi003_im006._reconst_depth]|![JNet_579_beads_roi003_im006._heatmap_depth]|
  
volume: 2.0767302246093755, MSE: 0.0018871049396693707, quantized loss: 8.219144365284592e-05  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi004_im006._original_depth]|![JNet_579_beads_roi004_im006._output_depth]|![JNet_579_beads_roi004_im006._reconst_depth]|![JNet_579_beads_roi004_im006._heatmap_depth]|
  
volume: 2.1691872558593754, MSE: 0.0019454470602795482, quantized loss: 9.263764513889328e-05  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi005_im007._original_depth]|![JNet_579_beads_roi005_im007._output_depth]|![JNet_579_beads_roi005_im007._reconst_depth]|![JNet_579_beads_roi005_im007._heatmap_depth]|
  
volume: 2.1031064453125006, MSE: 0.0018847080646082759, quantized loss: 8.882011752575636e-05  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi006_im008._original_depth]|![JNet_579_beads_roi006_im008._output_depth]|![JNet_579_beads_roi006_im008._reconst_depth]|![JNet_579_beads_roi006_im008._heatmap_depth]|
  
volume: 2.1870649414062506, MSE: 0.0018851010827347636, quantized loss: 0.0001251012145075947  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi007_im009._original_depth]|![JNet_579_beads_roi007_im009._output_depth]|![JNet_579_beads_roi007_im009._reconst_depth]|![JNet_579_beads_roi007_im009._heatmap_depth]|
  
volume: 2.1662612304687503, MSE: 0.0019840579479932785, quantized loss: 9.805164881981909e-05  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi008_im010._original_depth]|![JNet_579_beads_roi008_im010._output_depth]|![JNet_579_beads_roi008_im010._reconst_depth]|![JNet_579_beads_roi008_im010._heatmap_depth]|
  
volume: 2.2120029296875003, MSE: 0.001965658040717244, quantized loss: 7.864650251576677e-05  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi009_im011._original_depth]|![JNet_579_beads_roi009_im011._output_depth]|![JNet_579_beads_roi009_im011._reconst_depth]|![JNet_579_beads_roi009_im011._heatmap_depth]|
  
volume: 2.0584804687500005, MSE: 0.001794462208636105, quantized loss: 6.73382164677605e-05  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi010_im012._original_depth]|![JNet_579_beads_roi010_im012._output_depth]|![JNet_579_beads_roi010_im012._reconst_depth]|![JNet_579_beads_roi010_im012._heatmap_depth]|
  
volume: 2.3744907226562506, MSE: 0.002187203848734498, quantized loss: 7.535186887253076e-05  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi011_im013._original_depth]|![JNet_579_beads_roi011_im013._output_depth]|![JNet_579_beads_roi011_im013._reconst_depth]|![JNet_579_beads_roi011_im013._heatmap_depth]|
  
volume: 2.3673703613281254, MSE: 0.0020938136149197817, quantized loss: 7.242062565637752e-05  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi012_im014._original_depth]|![JNet_579_beads_roi012_im014._output_depth]|![JNet_579_beads_roi012_im014._reconst_depth]|![JNet_579_beads_roi012_im014._heatmap_depth]|
  
volume: 2.0961210937500003, MSE: 0.0019317492842674255, quantized loss: 6.972615665290505e-05  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi013_im015._original_depth]|![JNet_579_beads_roi013_im015._output_depth]|![JNet_579_beads_roi013_im015._reconst_depth]|![JNet_579_beads_roi013_im015._heatmap_depth]|
  
volume: 1.984143432617188, MSE: 0.0017846588743850589, quantized loss: 6.791178020648658e-05  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi014_im016._original_depth]|![JNet_579_beads_roi014_im016._output_depth]|![JNet_579_beads_roi014_im016._reconst_depth]|![JNet_579_beads_roi014_im016._heatmap_depth]|
  
volume: 1.9132443847656255, MSE: 0.0018726987764239311, quantized loss: 6.713442417094484e-05  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi015_im017._original_depth]|![JNet_579_beads_roi015_im017._output_depth]|![JNet_579_beads_roi015_im017._reconst_depth]|![JNet_579_beads_roi015_im017._heatmap_depth]|
  
volume: 1.9885932617187505, MSE: 0.0017795441672205925, quantized loss: 7.739761349512264e-05  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi016_im018._original_depth]|![JNet_579_beads_roi016_im018._output_depth]|![JNet_579_beads_roi016_im018._reconst_depth]|![JNet_579_beads_roi016_im018._heatmap_depth]|
  
volume: 2.2767441406250004, MSE: 0.0020315824076533318, quantized loss: 0.00010430998372612521  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi017_im018._original_depth]|![JNet_579_beads_roi017_im018._output_depth]|![JNet_579_beads_roi017_im018._reconst_depth]|![JNet_579_beads_roi017_im018._heatmap_depth]|
  
volume: 2.2210910644531254, MSE: 0.0020000573713332415, quantized loss: 8.960610284702852e-05  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi018_im022._original_depth]|![JNet_579_beads_roi018_im022._output_depth]|![JNet_579_beads_roi018_im022._reconst_depth]|![JNet_579_beads_roi018_im022._heatmap_depth]|
  
volume: 1.808847534179688, MSE: 0.0016023070784285665, quantized loss: 6.177020259201527e-05  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi019_im023._original_depth]|![JNet_579_beads_roi019_im023._output_depth]|![JNet_579_beads_roi019_im023._reconst_depth]|![JNet_579_beads_roi019_im023._heatmap_depth]|
  
volume: 1.7739818115234378, MSE: 0.0015479469439014792, quantized loss: 6.27635745331645e-05  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi020_im024._original_depth]|![JNet_579_beads_roi020_im024._output_depth]|![JNet_579_beads_roi020_im024._reconst_depth]|![JNet_579_beads_roi020_im024._heatmap_depth]|
  
volume: 2.2494033203125006, MSE: 0.001968202879652381, quantized loss: 6.991041300352663e-05  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi021_im026._original_depth]|![JNet_579_beads_roi021_im026._output_depth]|![JNet_579_beads_roi021_im026._reconst_depth]|![JNet_579_beads_roi021_im026._heatmap_depth]|
  
volume: 2.1567561035156255, MSE: 0.0018952732207253575, quantized loss: 6.751745968358591e-05  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi022_im027._original_depth]|![JNet_579_beads_roi022_im027._output_depth]|![JNet_579_beads_roi022_im027._reconst_depth]|![JNet_579_beads_roi022_im027._heatmap_depth]|
  
volume: 2.1058186035156257, MSE: 0.001856676652096212, quantized loss: 6.538785237353295e-05  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi023_im028._original_depth]|![JNet_579_beads_roi023_im028._output_depth]|![JNet_579_beads_roi023_im028._reconst_depth]|![JNet_579_beads_roi023_im028._heatmap_depth]|
  
volume: 2.2348823242187503, MSE: 0.0020258997101336718, quantized loss: 7.157631625887007e-05  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi024_im028._original_depth]|![JNet_579_beads_roi024_im028._output_depth]|![JNet_579_beads_roi024_im028._reconst_depth]|![JNet_579_beads_roi024_im028._heatmap_depth]|
  
volume: 2.1894863281250005, MSE: 0.0019503083312883973, quantized loss: 6.828757614130154e-05  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi025_im028._original_depth]|![JNet_579_beads_roi025_im028._output_depth]|![JNet_579_beads_roi025_im028._reconst_depth]|![JNet_579_beads_roi025_im028._heatmap_depth]|
  
volume: 2.1894863281250005, MSE: 0.0019503083312883973, quantized loss: 6.828757614130154e-05  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi026_im029._original_depth]|![JNet_579_beads_roi026_im029._output_depth]|![JNet_579_beads_roi026_im029._reconst_depth]|![JNet_579_beads_roi026_im029._heatmap_depth]|
  
volume: 2.2625720214843756, MSE: 0.0020466295536607504, quantized loss: 6.832232611486688e-05  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi027_im029._original_depth]|![JNet_579_beads_roi027_im029._output_depth]|![JNet_579_beads_roi027_im029._reconst_depth]|![JNet_579_beads_roi027_im029._heatmap_depth]|
  
volume: 1.9954616699218755, MSE: 0.0018378781387582421, quantized loss: 6.562505586771294e-05  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi028_im030._original_depth]|![JNet_579_beads_roi028_im030._output_depth]|![JNet_579_beads_roi028_im030._reconst_depth]|![JNet_579_beads_roi028_im030._heatmap_depth]|
  
volume: 1.958009399414063, MSE: 0.0017272974364459515, quantized loss: 6.573406426468864e-05  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_579_beads_roi029_im030._original_depth]|![JNet_579_beads_roi029_im030._output_depth]|![JNet_579_beads_roi029_im030._reconst_depth]|![JNet_579_beads_roi029_im030._heatmap_depth]|
  
volume: 2.0906435546875004, MSE: 0.0017999769188463688, quantized loss: 6.699660298181698e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_579_psf_pre]|![JNet_579_psf_post]|

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
  



[JNet_577_pretrain_0_aligned_depth]: /experiments/images/JNet_577_pretrain_0_aligned_depth.png
[JNet_577_pretrain_0_aligned_plane]: /experiments/images/JNet_577_pretrain_0_aligned_plane.png
[JNet_577_pretrain_0_labelx_depth]: /experiments/images/JNet_577_pretrain_0_labelx_depth.png
[JNet_577_pretrain_0_labelx_plane]: /experiments/images/JNet_577_pretrain_0_labelx_plane.png
[JNet_577_pretrain_0_labelz_depth]: /experiments/images/JNet_577_pretrain_0_labelz_depth.png
[JNet_577_pretrain_0_labelz_plane]: /experiments/images/JNet_577_pretrain_0_labelz_plane.png
[JNet_577_pretrain_0_novibrate_depth]: /experiments/images/JNet_577_pretrain_0_novibrate_depth.png
[JNet_577_pretrain_0_novibrate_plane]: /experiments/images/JNet_577_pretrain_0_novibrate_plane.png
[JNet_577_pretrain_0_original_depth]: /experiments/images/JNet_577_pretrain_0_original_depth.png
[JNet_577_pretrain_0_original_plane]: /experiments/images/JNet_577_pretrain_0_original_plane.png
[JNet_577_pretrain_0_outputx_depth]: /experiments/images/JNet_577_pretrain_0_outputx_depth.png
[JNet_577_pretrain_0_outputx_plane]: /experiments/images/JNet_577_pretrain_0_outputx_plane.png
[JNet_577_pretrain_0_outputz_depth]: /experiments/images/JNet_577_pretrain_0_outputz_depth.png
[JNet_577_pretrain_0_outputz_plane]: /experiments/images/JNet_577_pretrain_0_outputz_plane.png
[JNet_577_pretrain_1_aligned_depth]: /experiments/images/JNet_577_pretrain_1_aligned_depth.png
[JNet_577_pretrain_1_aligned_plane]: /experiments/images/JNet_577_pretrain_1_aligned_plane.png
[JNet_577_pretrain_1_labelx_depth]: /experiments/images/JNet_577_pretrain_1_labelx_depth.png
[JNet_577_pretrain_1_labelx_plane]: /experiments/images/JNet_577_pretrain_1_labelx_plane.png
[JNet_577_pretrain_1_labelz_depth]: /experiments/images/JNet_577_pretrain_1_labelz_depth.png
[JNet_577_pretrain_1_labelz_plane]: /experiments/images/JNet_577_pretrain_1_labelz_plane.png
[JNet_577_pretrain_1_novibrate_depth]: /experiments/images/JNet_577_pretrain_1_novibrate_depth.png
[JNet_577_pretrain_1_novibrate_plane]: /experiments/images/JNet_577_pretrain_1_novibrate_plane.png
[JNet_577_pretrain_1_original_depth]: /experiments/images/JNet_577_pretrain_1_original_depth.png
[JNet_577_pretrain_1_original_plane]: /experiments/images/JNet_577_pretrain_1_original_plane.png
[JNet_577_pretrain_1_outputx_depth]: /experiments/images/JNet_577_pretrain_1_outputx_depth.png
[JNet_577_pretrain_1_outputx_plane]: /experiments/images/JNet_577_pretrain_1_outputx_plane.png
[JNet_577_pretrain_1_outputz_depth]: /experiments/images/JNet_577_pretrain_1_outputz_depth.png
[JNet_577_pretrain_1_outputz_plane]: /experiments/images/JNet_577_pretrain_1_outputz_plane.png
[JNet_577_pretrain_2_aligned_depth]: /experiments/images/JNet_577_pretrain_2_aligned_depth.png
[JNet_577_pretrain_2_aligned_plane]: /experiments/images/JNet_577_pretrain_2_aligned_plane.png
[JNet_577_pretrain_2_labelx_depth]: /experiments/images/JNet_577_pretrain_2_labelx_depth.png
[JNet_577_pretrain_2_labelx_plane]: /experiments/images/JNet_577_pretrain_2_labelx_plane.png
[JNet_577_pretrain_2_labelz_depth]: /experiments/images/JNet_577_pretrain_2_labelz_depth.png
[JNet_577_pretrain_2_labelz_plane]: /experiments/images/JNet_577_pretrain_2_labelz_plane.png
[JNet_577_pretrain_2_novibrate_depth]: /experiments/images/JNet_577_pretrain_2_novibrate_depth.png
[JNet_577_pretrain_2_novibrate_plane]: /experiments/images/JNet_577_pretrain_2_novibrate_plane.png
[JNet_577_pretrain_2_original_depth]: /experiments/images/JNet_577_pretrain_2_original_depth.png
[JNet_577_pretrain_2_original_plane]: /experiments/images/JNet_577_pretrain_2_original_plane.png
[JNet_577_pretrain_2_outputx_depth]: /experiments/images/JNet_577_pretrain_2_outputx_depth.png
[JNet_577_pretrain_2_outputx_plane]: /experiments/images/JNet_577_pretrain_2_outputx_plane.png
[JNet_577_pretrain_2_outputz_depth]: /experiments/images/JNet_577_pretrain_2_outputz_depth.png
[JNet_577_pretrain_2_outputz_plane]: /experiments/images/JNet_577_pretrain_2_outputz_plane.png
[JNet_577_pretrain_3_aligned_depth]: /experiments/images/JNet_577_pretrain_3_aligned_depth.png
[JNet_577_pretrain_3_aligned_plane]: /experiments/images/JNet_577_pretrain_3_aligned_plane.png
[JNet_577_pretrain_3_labelx_depth]: /experiments/images/JNet_577_pretrain_3_labelx_depth.png
[JNet_577_pretrain_3_labelx_plane]: /experiments/images/JNet_577_pretrain_3_labelx_plane.png
[JNet_577_pretrain_3_labelz_depth]: /experiments/images/JNet_577_pretrain_3_labelz_depth.png
[JNet_577_pretrain_3_labelz_plane]: /experiments/images/JNet_577_pretrain_3_labelz_plane.png
[JNet_577_pretrain_3_novibrate_depth]: /experiments/images/JNet_577_pretrain_3_novibrate_depth.png
[JNet_577_pretrain_3_novibrate_plane]: /experiments/images/JNet_577_pretrain_3_novibrate_plane.png
[JNet_577_pretrain_3_original_depth]: /experiments/images/JNet_577_pretrain_3_original_depth.png
[JNet_577_pretrain_3_original_plane]: /experiments/images/JNet_577_pretrain_3_original_plane.png
[JNet_577_pretrain_3_outputx_depth]: /experiments/images/JNet_577_pretrain_3_outputx_depth.png
[JNet_577_pretrain_3_outputx_plane]: /experiments/images/JNet_577_pretrain_3_outputx_plane.png
[JNet_577_pretrain_3_outputz_depth]: /experiments/images/JNet_577_pretrain_3_outputz_depth.png
[JNet_577_pretrain_3_outputz_plane]: /experiments/images/JNet_577_pretrain_3_outputz_plane.png
[JNet_577_pretrain_4_aligned_depth]: /experiments/images/JNet_577_pretrain_4_aligned_depth.png
[JNet_577_pretrain_4_aligned_plane]: /experiments/images/JNet_577_pretrain_4_aligned_plane.png
[JNet_577_pretrain_4_labelx_depth]: /experiments/images/JNet_577_pretrain_4_labelx_depth.png
[JNet_577_pretrain_4_labelx_plane]: /experiments/images/JNet_577_pretrain_4_labelx_plane.png
[JNet_577_pretrain_4_labelz_depth]: /experiments/images/JNet_577_pretrain_4_labelz_depth.png
[JNet_577_pretrain_4_labelz_plane]: /experiments/images/JNet_577_pretrain_4_labelz_plane.png
[JNet_577_pretrain_4_novibrate_depth]: /experiments/images/JNet_577_pretrain_4_novibrate_depth.png
[JNet_577_pretrain_4_novibrate_plane]: /experiments/images/JNet_577_pretrain_4_novibrate_plane.png
[JNet_577_pretrain_4_original_depth]: /experiments/images/JNet_577_pretrain_4_original_depth.png
[JNet_577_pretrain_4_original_plane]: /experiments/images/JNet_577_pretrain_4_original_plane.png
[JNet_577_pretrain_4_outputx_depth]: /experiments/images/JNet_577_pretrain_4_outputx_depth.png
[JNet_577_pretrain_4_outputx_plane]: /experiments/images/JNet_577_pretrain_4_outputx_plane.png
[JNet_577_pretrain_4_outputz_depth]: /experiments/images/JNet_577_pretrain_4_outputz_depth.png
[JNet_577_pretrain_4_outputz_plane]: /experiments/images/JNet_577_pretrain_4_outputz_plane.png
[JNet_577_pretrain_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi000_im000._heatmap_depth.png
[JNet_577_pretrain_beads_roi000_im000._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi000_im000._original_depth.png
[JNet_577_pretrain_beads_roi000_im000._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi000_im000._output_depth.png
[JNet_577_pretrain_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi000_im000._reconst_depth.png
[JNet_577_pretrain_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi001_im004._heatmap_depth.png
[JNet_577_pretrain_beads_roi001_im004._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi001_im004._original_depth.png
[JNet_577_pretrain_beads_roi001_im004._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi001_im004._output_depth.png
[JNet_577_pretrain_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi001_im004._reconst_depth.png
[JNet_577_pretrain_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi002_im005._heatmap_depth.png
[JNet_577_pretrain_beads_roi002_im005._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi002_im005._original_depth.png
[JNet_577_pretrain_beads_roi002_im005._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi002_im005._output_depth.png
[JNet_577_pretrain_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi002_im005._reconst_depth.png
[JNet_577_pretrain_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi003_im006._heatmap_depth.png
[JNet_577_pretrain_beads_roi003_im006._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi003_im006._original_depth.png
[JNet_577_pretrain_beads_roi003_im006._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi003_im006._output_depth.png
[JNet_577_pretrain_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi003_im006._reconst_depth.png
[JNet_577_pretrain_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi004_im006._heatmap_depth.png
[JNet_577_pretrain_beads_roi004_im006._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi004_im006._original_depth.png
[JNet_577_pretrain_beads_roi004_im006._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi004_im006._output_depth.png
[JNet_577_pretrain_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi004_im006._reconst_depth.png
[JNet_577_pretrain_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi005_im007._heatmap_depth.png
[JNet_577_pretrain_beads_roi005_im007._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi005_im007._original_depth.png
[JNet_577_pretrain_beads_roi005_im007._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi005_im007._output_depth.png
[JNet_577_pretrain_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi005_im007._reconst_depth.png
[JNet_577_pretrain_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi006_im008._heatmap_depth.png
[JNet_577_pretrain_beads_roi006_im008._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi006_im008._original_depth.png
[JNet_577_pretrain_beads_roi006_im008._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi006_im008._output_depth.png
[JNet_577_pretrain_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi006_im008._reconst_depth.png
[JNet_577_pretrain_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi007_im009._heatmap_depth.png
[JNet_577_pretrain_beads_roi007_im009._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi007_im009._original_depth.png
[JNet_577_pretrain_beads_roi007_im009._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi007_im009._output_depth.png
[JNet_577_pretrain_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi007_im009._reconst_depth.png
[JNet_577_pretrain_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi008_im010._heatmap_depth.png
[JNet_577_pretrain_beads_roi008_im010._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi008_im010._original_depth.png
[JNet_577_pretrain_beads_roi008_im010._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi008_im010._output_depth.png
[JNet_577_pretrain_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi008_im010._reconst_depth.png
[JNet_577_pretrain_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi009_im011._heatmap_depth.png
[JNet_577_pretrain_beads_roi009_im011._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi009_im011._original_depth.png
[JNet_577_pretrain_beads_roi009_im011._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi009_im011._output_depth.png
[JNet_577_pretrain_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi009_im011._reconst_depth.png
[JNet_577_pretrain_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi010_im012._heatmap_depth.png
[JNet_577_pretrain_beads_roi010_im012._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi010_im012._original_depth.png
[JNet_577_pretrain_beads_roi010_im012._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi010_im012._output_depth.png
[JNet_577_pretrain_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi010_im012._reconst_depth.png
[JNet_577_pretrain_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi011_im013._heatmap_depth.png
[JNet_577_pretrain_beads_roi011_im013._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi011_im013._original_depth.png
[JNet_577_pretrain_beads_roi011_im013._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi011_im013._output_depth.png
[JNet_577_pretrain_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi011_im013._reconst_depth.png
[JNet_577_pretrain_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi012_im014._heatmap_depth.png
[JNet_577_pretrain_beads_roi012_im014._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi012_im014._original_depth.png
[JNet_577_pretrain_beads_roi012_im014._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi012_im014._output_depth.png
[JNet_577_pretrain_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi012_im014._reconst_depth.png
[JNet_577_pretrain_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi013_im015._heatmap_depth.png
[JNet_577_pretrain_beads_roi013_im015._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi013_im015._original_depth.png
[JNet_577_pretrain_beads_roi013_im015._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi013_im015._output_depth.png
[JNet_577_pretrain_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi013_im015._reconst_depth.png
[JNet_577_pretrain_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi014_im016._heatmap_depth.png
[JNet_577_pretrain_beads_roi014_im016._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi014_im016._original_depth.png
[JNet_577_pretrain_beads_roi014_im016._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi014_im016._output_depth.png
[JNet_577_pretrain_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi014_im016._reconst_depth.png
[JNet_577_pretrain_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi015_im017._heatmap_depth.png
[JNet_577_pretrain_beads_roi015_im017._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi015_im017._original_depth.png
[JNet_577_pretrain_beads_roi015_im017._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi015_im017._output_depth.png
[JNet_577_pretrain_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi015_im017._reconst_depth.png
[JNet_577_pretrain_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi016_im018._heatmap_depth.png
[JNet_577_pretrain_beads_roi016_im018._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi016_im018._original_depth.png
[JNet_577_pretrain_beads_roi016_im018._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi016_im018._output_depth.png
[JNet_577_pretrain_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi016_im018._reconst_depth.png
[JNet_577_pretrain_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi017_im018._heatmap_depth.png
[JNet_577_pretrain_beads_roi017_im018._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi017_im018._original_depth.png
[JNet_577_pretrain_beads_roi017_im018._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi017_im018._output_depth.png
[JNet_577_pretrain_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi017_im018._reconst_depth.png
[JNet_577_pretrain_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi018_im022._heatmap_depth.png
[JNet_577_pretrain_beads_roi018_im022._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi018_im022._original_depth.png
[JNet_577_pretrain_beads_roi018_im022._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi018_im022._output_depth.png
[JNet_577_pretrain_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi018_im022._reconst_depth.png
[JNet_577_pretrain_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi019_im023._heatmap_depth.png
[JNet_577_pretrain_beads_roi019_im023._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi019_im023._original_depth.png
[JNet_577_pretrain_beads_roi019_im023._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi019_im023._output_depth.png
[JNet_577_pretrain_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi019_im023._reconst_depth.png
[JNet_577_pretrain_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi020_im024._heatmap_depth.png
[JNet_577_pretrain_beads_roi020_im024._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi020_im024._original_depth.png
[JNet_577_pretrain_beads_roi020_im024._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi020_im024._output_depth.png
[JNet_577_pretrain_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi020_im024._reconst_depth.png
[JNet_577_pretrain_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi021_im026._heatmap_depth.png
[JNet_577_pretrain_beads_roi021_im026._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi021_im026._original_depth.png
[JNet_577_pretrain_beads_roi021_im026._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi021_im026._output_depth.png
[JNet_577_pretrain_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi021_im026._reconst_depth.png
[JNet_577_pretrain_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi022_im027._heatmap_depth.png
[JNet_577_pretrain_beads_roi022_im027._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi022_im027._original_depth.png
[JNet_577_pretrain_beads_roi022_im027._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi022_im027._output_depth.png
[JNet_577_pretrain_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi022_im027._reconst_depth.png
[JNet_577_pretrain_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi023_im028._heatmap_depth.png
[JNet_577_pretrain_beads_roi023_im028._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi023_im028._original_depth.png
[JNet_577_pretrain_beads_roi023_im028._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi023_im028._output_depth.png
[JNet_577_pretrain_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi023_im028._reconst_depth.png
[JNet_577_pretrain_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi024_im028._heatmap_depth.png
[JNet_577_pretrain_beads_roi024_im028._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi024_im028._original_depth.png
[JNet_577_pretrain_beads_roi024_im028._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi024_im028._output_depth.png
[JNet_577_pretrain_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi024_im028._reconst_depth.png
[JNet_577_pretrain_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi025_im028._heatmap_depth.png
[JNet_577_pretrain_beads_roi025_im028._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi025_im028._original_depth.png
[JNet_577_pretrain_beads_roi025_im028._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi025_im028._output_depth.png
[JNet_577_pretrain_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi025_im028._reconst_depth.png
[JNet_577_pretrain_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi026_im029._heatmap_depth.png
[JNet_577_pretrain_beads_roi026_im029._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi026_im029._original_depth.png
[JNet_577_pretrain_beads_roi026_im029._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi026_im029._output_depth.png
[JNet_577_pretrain_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi026_im029._reconst_depth.png
[JNet_577_pretrain_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi027_im029._heatmap_depth.png
[JNet_577_pretrain_beads_roi027_im029._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi027_im029._original_depth.png
[JNet_577_pretrain_beads_roi027_im029._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi027_im029._output_depth.png
[JNet_577_pretrain_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi027_im029._reconst_depth.png
[JNet_577_pretrain_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi028_im030._heatmap_depth.png
[JNet_577_pretrain_beads_roi028_im030._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi028_im030._original_depth.png
[JNet_577_pretrain_beads_roi028_im030._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi028_im030._output_depth.png
[JNet_577_pretrain_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi028_im030._reconst_depth.png
[JNet_577_pretrain_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_577_pretrain_beads_roi029_im030._heatmap_depth.png
[JNet_577_pretrain_beads_roi029_im030._original_depth]: /experiments/images/JNet_577_pretrain_beads_roi029_im030._original_depth.png
[JNet_577_pretrain_beads_roi029_im030._output_depth]: /experiments/images/JNet_577_pretrain_beads_roi029_im030._output_depth.png
[JNet_577_pretrain_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_577_pretrain_beads_roi029_im030._reconst_depth.png
[JNet_579_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_579_beads_roi000_im000._heatmap_depth.png
[JNet_579_beads_roi000_im000._original_depth]: /experiments/images/JNet_579_beads_roi000_im000._original_depth.png
[JNet_579_beads_roi000_im000._output_depth]: /experiments/images/JNet_579_beads_roi000_im000._output_depth.png
[JNet_579_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_579_beads_roi000_im000._reconst_depth.png
[JNet_579_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_579_beads_roi001_im004._heatmap_depth.png
[JNet_579_beads_roi001_im004._original_depth]: /experiments/images/JNet_579_beads_roi001_im004._original_depth.png
[JNet_579_beads_roi001_im004._output_depth]: /experiments/images/JNet_579_beads_roi001_im004._output_depth.png
[JNet_579_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_579_beads_roi001_im004._reconst_depth.png
[JNet_579_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_579_beads_roi002_im005._heatmap_depth.png
[JNet_579_beads_roi002_im005._original_depth]: /experiments/images/JNet_579_beads_roi002_im005._original_depth.png
[JNet_579_beads_roi002_im005._output_depth]: /experiments/images/JNet_579_beads_roi002_im005._output_depth.png
[JNet_579_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_579_beads_roi002_im005._reconst_depth.png
[JNet_579_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_579_beads_roi003_im006._heatmap_depth.png
[JNet_579_beads_roi003_im006._original_depth]: /experiments/images/JNet_579_beads_roi003_im006._original_depth.png
[JNet_579_beads_roi003_im006._output_depth]: /experiments/images/JNet_579_beads_roi003_im006._output_depth.png
[JNet_579_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_579_beads_roi003_im006._reconst_depth.png
[JNet_579_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_579_beads_roi004_im006._heatmap_depth.png
[JNet_579_beads_roi004_im006._original_depth]: /experiments/images/JNet_579_beads_roi004_im006._original_depth.png
[JNet_579_beads_roi004_im006._output_depth]: /experiments/images/JNet_579_beads_roi004_im006._output_depth.png
[JNet_579_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_579_beads_roi004_im006._reconst_depth.png
[JNet_579_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_579_beads_roi005_im007._heatmap_depth.png
[JNet_579_beads_roi005_im007._original_depth]: /experiments/images/JNet_579_beads_roi005_im007._original_depth.png
[JNet_579_beads_roi005_im007._output_depth]: /experiments/images/JNet_579_beads_roi005_im007._output_depth.png
[JNet_579_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_579_beads_roi005_im007._reconst_depth.png
[JNet_579_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_579_beads_roi006_im008._heatmap_depth.png
[JNet_579_beads_roi006_im008._original_depth]: /experiments/images/JNet_579_beads_roi006_im008._original_depth.png
[JNet_579_beads_roi006_im008._output_depth]: /experiments/images/JNet_579_beads_roi006_im008._output_depth.png
[JNet_579_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_579_beads_roi006_im008._reconst_depth.png
[JNet_579_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_579_beads_roi007_im009._heatmap_depth.png
[JNet_579_beads_roi007_im009._original_depth]: /experiments/images/JNet_579_beads_roi007_im009._original_depth.png
[JNet_579_beads_roi007_im009._output_depth]: /experiments/images/JNet_579_beads_roi007_im009._output_depth.png
[JNet_579_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_579_beads_roi007_im009._reconst_depth.png
[JNet_579_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_579_beads_roi008_im010._heatmap_depth.png
[JNet_579_beads_roi008_im010._original_depth]: /experiments/images/JNet_579_beads_roi008_im010._original_depth.png
[JNet_579_beads_roi008_im010._output_depth]: /experiments/images/JNet_579_beads_roi008_im010._output_depth.png
[JNet_579_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_579_beads_roi008_im010._reconst_depth.png
[JNet_579_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_579_beads_roi009_im011._heatmap_depth.png
[JNet_579_beads_roi009_im011._original_depth]: /experiments/images/JNet_579_beads_roi009_im011._original_depth.png
[JNet_579_beads_roi009_im011._output_depth]: /experiments/images/JNet_579_beads_roi009_im011._output_depth.png
[JNet_579_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_579_beads_roi009_im011._reconst_depth.png
[JNet_579_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_579_beads_roi010_im012._heatmap_depth.png
[JNet_579_beads_roi010_im012._original_depth]: /experiments/images/JNet_579_beads_roi010_im012._original_depth.png
[JNet_579_beads_roi010_im012._output_depth]: /experiments/images/JNet_579_beads_roi010_im012._output_depth.png
[JNet_579_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_579_beads_roi010_im012._reconst_depth.png
[JNet_579_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_579_beads_roi011_im013._heatmap_depth.png
[JNet_579_beads_roi011_im013._original_depth]: /experiments/images/JNet_579_beads_roi011_im013._original_depth.png
[JNet_579_beads_roi011_im013._output_depth]: /experiments/images/JNet_579_beads_roi011_im013._output_depth.png
[JNet_579_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_579_beads_roi011_im013._reconst_depth.png
[JNet_579_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_579_beads_roi012_im014._heatmap_depth.png
[JNet_579_beads_roi012_im014._original_depth]: /experiments/images/JNet_579_beads_roi012_im014._original_depth.png
[JNet_579_beads_roi012_im014._output_depth]: /experiments/images/JNet_579_beads_roi012_im014._output_depth.png
[JNet_579_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_579_beads_roi012_im014._reconst_depth.png
[JNet_579_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_579_beads_roi013_im015._heatmap_depth.png
[JNet_579_beads_roi013_im015._original_depth]: /experiments/images/JNet_579_beads_roi013_im015._original_depth.png
[JNet_579_beads_roi013_im015._output_depth]: /experiments/images/JNet_579_beads_roi013_im015._output_depth.png
[JNet_579_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_579_beads_roi013_im015._reconst_depth.png
[JNet_579_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_579_beads_roi014_im016._heatmap_depth.png
[JNet_579_beads_roi014_im016._original_depth]: /experiments/images/JNet_579_beads_roi014_im016._original_depth.png
[JNet_579_beads_roi014_im016._output_depth]: /experiments/images/JNet_579_beads_roi014_im016._output_depth.png
[JNet_579_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_579_beads_roi014_im016._reconst_depth.png
[JNet_579_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_579_beads_roi015_im017._heatmap_depth.png
[JNet_579_beads_roi015_im017._original_depth]: /experiments/images/JNet_579_beads_roi015_im017._original_depth.png
[JNet_579_beads_roi015_im017._output_depth]: /experiments/images/JNet_579_beads_roi015_im017._output_depth.png
[JNet_579_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_579_beads_roi015_im017._reconst_depth.png
[JNet_579_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_579_beads_roi016_im018._heatmap_depth.png
[JNet_579_beads_roi016_im018._original_depth]: /experiments/images/JNet_579_beads_roi016_im018._original_depth.png
[JNet_579_beads_roi016_im018._output_depth]: /experiments/images/JNet_579_beads_roi016_im018._output_depth.png
[JNet_579_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_579_beads_roi016_im018._reconst_depth.png
[JNet_579_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_579_beads_roi017_im018._heatmap_depth.png
[JNet_579_beads_roi017_im018._original_depth]: /experiments/images/JNet_579_beads_roi017_im018._original_depth.png
[JNet_579_beads_roi017_im018._output_depth]: /experiments/images/JNet_579_beads_roi017_im018._output_depth.png
[JNet_579_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_579_beads_roi017_im018._reconst_depth.png
[JNet_579_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_579_beads_roi018_im022._heatmap_depth.png
[JNet_579_beads_roi018_im022._original_depth]: /experiments/images/JNet_579_beads_roi018_im022._original_depth.png
[JNet_579_beads_roi018_im022._output_depth]: /experiments/images/JNet_579_beads_roi018_im022._output_depth.png
[JNet_579_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_579_beads_roi018_im022._reconst_depth.png
[JNet_579_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_579_beads_roi019_im023._heatmap_depth.png
[JNet_579_beads_roi019_im023._original_depth]: /experiments/images/JNet_579_beads_roi019_im023._original_depth.png
[JNet_579_beads_roi019_im023._output_depth]: /experiments/images/JNet_579_beads_roi019_im023._output_depth.png
[JNet_579_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_579_beads_roi019_im023._reconst_depth.png
[JNet_579_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_579_beads_roi020_im024._heatmap_depth.png
[JNet_579_beads_roi020_im024._original_depth]: /experiments/images/JNet_579_beads_roi020_im024._original_depth.png
[JNet_579_beads_roi020_im024._output_depth]: /experiments/images/JNet_579_beads_roi020_im024._output_depth.png
[JNet_579_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_579_beads_roi020_im024._reconst_depth.png
[JNet_579_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_579_beads_roi021_im026._heatmap_depth.png
[JNet_579_beads_roi021_im026._original_depth]: /experiments/images/JNet_579_beads_roi021_im026._original_depth.png
[JNet_579_beads_roi021_im026._output_depth]: /experiments/images/JNet_579_beads_roi021_im026._output_depth.png
[JNet_579_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_579_beads_roi021_im026._reconst_depth.png
[JNet_579_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_579_beads_roi022_im027._heatmap_depth.png
[JNet_579_beads_roi022_im027._original_depth]: /experiments/images/JNet_579_beads_roi022_im027._original_depth.png
[JNet_579_beads_roi022_im027._output_depth]: /experiments/images/JNet_579_beads_roi022_im027._output_depth.png
[JNet_579_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_579_beads_roi022_im027._reconst_depth.png
[JNet_579_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_579_beads_roi023_im028._heatmap_depth.png
[JNet_579_beads_roi023_im028._original_depth]: /experiments/images/JNet_579_beads_roi023_im028._original_depth.png
[JNet_579_beads_roi023_im028._output_depth]: /experiments/images/JNet_579_beads_roi023_im028._output_depth.png
[JNet_579_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_579_beads_roi023_im028._reconst_depth.png
[JNet_579_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_579_beads_roi024_im028._heatmap_depth.png
[JNet_579_beads_roi024_im028._original_depth]: /experiments/images/JNet_579_beads_roi024_im028._original_depth.png
[JNet_579_beads_roi024_im028._output_depth]: /experiments/images/JNet_579_beads_roi024_im028._output_depth.png
[JNet_579_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_579_beads_roi024_im028._reconst_depth.png
[JNet_579_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_579_beads_roi025_im028._heatmap_depth.png
[JNet_579_beads_roi025_im028._original_depth]: /experiments/images/JNet_579_beads_roi025_im028._original_depth.png
[JNet_579_beads_roi025_im028._output_depth]: /experiments/images/JNet_579_beads_roi025_im028._output_depth.png
[JNet_579_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_579_beads_roi025_im028._reconst_depth.png
[JNet_579_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_579_beads_roi026_im029._heatmap_depth.png
[JNet_579_beads_roi026_im029._original_depth]: /experiments/images/JNet_579_beads_roi026_im029._original_depth.png
[JNet_579_beads_roi026_im029._output_depth]: /experiments/images/JNet_579_beads_roi026_im029._output_depth.png
[JNet_579_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_579_beads_roi026_im029._reconst_depth.png
[JNet_579_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_579_beads_roi027_im029._heatmap_depth.png
[JNet_579_beads_roi027_im029._original_depth]: /experiments/images/JNet_579_beads_roi027_im029._original_depth.png
[JNet_579_beads_roi027_im029._output_depth]: /experiments/images/JNet_579_beads_roi027_im029._output_depth.png
[JNet_579_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_579_beads_roi027_im029._reconst_depth.png
[JNet_579_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_579_beads_roi028_im030._heatmap_depth.png
[JNet_579_beads_roi028_im030._original_depth]: /experiments/images/JNet_579_beads_roi028_im030._original_depth.png
[JNet_579_beads_roi028_im030._output_depth]: /experiments/images/JNet_579_beads_roi028_im030._output_depth.png
[JNet_579_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_579_beads_roi028_im030._reconst_depth.png
[JNet_579_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_579_beads_roi029_im030._heatmap_depth.png
[JNet_579_beads_roi029_im030._original_depth]: /experiments/images/JNet_579_beads_roi029_im030._original_depth.png
[JNet_579_beads_roi029_im030._output_depth]: /experiments/images/JNet_579_beads_roi029_im030._output_depth.png
[JNet_579_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_579_beads_roi029_im030._reconst_depth.png
[JNet_579_psf_post]: /experiments/images/JNet_579_psf_post.png
[JNet_579_psf_pre]: /experiments/images/JNet_579_psf_pre.png
[finetuned]: /experiments/tmp/JNet_579_train.png
[pretrained_model]: /experiments/tmp/JNet_577_pretrain_train.png
