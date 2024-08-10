



# JNet_591 Report
  
psf loss 1.0 and ewc 1.0, adjust_luminance = false  
pretrained model : JNet_589_pretrain
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
|n_epochs|200|
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
|adjust_luminance|False|
|zloss_weight|1|
|ewc_weight|0.1|
|qloss_weight|1.0|
|ploss_weight|0.1|
|mrfloss_order|1|
|mrfloss_dilation|1|
|mrfloss_weights|{'l_00': 0, 'l_01': 0, 'l_10': 0, 'l_11': 0}|

## Results

### Pretraining
  
Segmentation: mean MSE: 0.006385980639606714, mean BCE: 0.025872427970170975  
Luminance Estimation: mean MSE: 0.9864093065261841, mean BCE: inf
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_0_original_plane]|![JNet_589_pretrain_0_novibrate_plane]|![JNet_589_pretrain_0_aligned_plane]|![JNet_589_pretrain_0_outputx_plane]|![JNet_589_pretrain_0_labelx_plane]|![JNet_589_pretrain_0_outputz_plane]|![JNet_589_pretrain_0_labelz_plane]|
  
MSEx: 0.01007230207324028, BCEx: 0.04218867048621178  
MSEz: 0.9833747744560242, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_0_original_depth]|![JNet_589_pretrain_0_novibrate_depth]|![JNet_589_pretrain_0_aligned_depth]|![JNet_589_pretrain_0_outputx_depth]|![JNet_589_pretrain_0_labelx_depth]|![JNet_589_pretrain_0_outputz_depth]|![JNet_589_pretrain_0_labelz_depth]|
  
MSEx: 0.01007230207324028, BCEx: 0.04218867048621178  
MSEz: 0.9833747744560242, BCEz: inf  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_1_original_plane]|![JNet_589_pretrain_1_novibrate_plane]|![JNet_589_pretrain_1_aligned_plane]|![JNet_589_pretrain_1_outputx_plane]|![JNet_589_pretrain_1_labelx_plane]|![JNet_589_pretrain_1_outputz_plane]|![JNet_589_pretrain_1_labelz_plane]|
  
MSEx: 0.004863931331783533, BCEx: 0.01983783394098282  
MSEz: 0.9923388957977295, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_1_original_depth]|![JNet_589_pretrain_1_novibrate_depth]|![JNet_589_pretrain_1_aligned_depth]|![JNet_589_pretrain_1_outputx_depth]|![JNet_589_pretrain_1_labelx_depth]|![JNet_589_pretrain_1_outputz_depth]|![JNet_589_pretrain_1_labelz_depth]|
  
MSEx: 0.004863931331783533, BCEx: 0.01983783394098282  
MSEz: 0.9923388957977295, BCEz: inf  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_2_original_plane]|![JNet_589_pretrain_2_novibrate_plane]|![JNet_589_pretrain_2_aligned_plane]|![JNet_589_pretrain_2_outputx_plane]|![JNet_589_pretrain_2_labelx_plane]|![JNet_589_pretrain_2_outputz_plane]|![JNet_589_pretrain_2_labelz_plane]|
  
MSEx: 0.00481423968449235, BCEx: 0.019628552719950676  
MSEz: 0.9822806119918823, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_2_original_depth]|![JNet_589_pretrain_2_novibrate_depth]|![JNet_589_pretrain_2_aligned_depth]|![JNet_589_pretrain_2_outputx_depth]|![JNet_589_pretrain_2_labelx_depth]|![JNet_589_pretrain_2_outputz_depth]|![JNet_589_pretrain_2_labelz_depth]|
  
MSEx: 0.00481423968449235, BCEx: 0.019628552719950676  
MSEz: 0.9822806119918823, BCEz: inf  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_3_original_plane]|![JNet_589_pretrain_3_novibrate_plane]|![JNet_589_pretrain_3_aligned_plane]|![JNet_589_pretrain_3_outputx_plane]|![JNet_589_pretrain_3_labelx_plane]|![JNet_589_pretrain_3_outputz_plane]|![JNet_589_pretrain_3_labelz_plane]|
  
MSEx: 0.005697683896869421, BCEx: 0.022441871464252472  
MSEz: 0.9846926331520081, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_3_original_depth]|![JNet_589_pretrain_3_novibrate_depth]|![JNet_589_pretrain_3_aligned_depth]|![JNet_589_pretrain_3_outputx_depth]|![JNet_589_pretrain_3_labelx_depth]|![JNet_589_pretrain_3_outputz_depth]|![JNet_589_pretrain_3_labelz_depth]|
  
MSEx: 0.005697683896869421, BCEx: 0.022441871464252472  
MSEz: 0.9846926331520081, BCEz: inf  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_4_original_plane]|![JNet_589_pretrain_4_novibrate_plane]|![JNet_589_pretrain_4_aligned_plane]|![JNet_589_pretrain_4_outputx_plane]|![JNet_589_pretrain_4_labelx_plane]|![JNet_589_pretrain_4_outputz_plane]|![JNet_589_pretrain_4_labelz_plane]|
  
MSEx: 0.006481746211647987, BCEx: 0.02526521310210228  
MSEz: 0.9893592596054077, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_4_original_depth]|![JNet_589_pretrain_4_novibrate_depth]|![JNet_589_pretrain_4_aligned_depth]|![JNet_589_pretrain_4_outputx_depth]|![JNet_589_pretrain_4_labelx_depth]|![JNet_589_pretrain_4_outputz_depth]|![JNet_589_pretrain_4_labelz_depth]|
  
MSEx: 0.006481746211647987, BCEx: 0.02526521310210228  
MSEz: 0.9893592596054077, BCEz: inf  

### pretrain
  
volume mean: 4.231467179361981, volume sd: 0.28480292935214424
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi000_im000._original_depth]|![JNet_589_pretrain_beads_roi000_im000._output_depth]|![JNet_589_pretrain_beads_roi000_im000._reconst_depth]|![JNet_589_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 3.915342773437501, MSE: 0.0010962235974147916, quantized loss: 0.00035057374043390155  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi001_im004._original_depth]|![JNet_589_pretrain_beads_roi001_im004._output_depth]|![JNet_589_pretrain_beads_roi001_im004._reconst_depth]|![JNet_589_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 4.708157226562501, MSE: 0.0011226573260501027, quantized loss: 0.000413618516176939  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi002_im005._original_depth]|![JNet_589_pretrain_beads_roi002_im005._output_depth]|![JNet_589_pretrain_beads_roi002_im005._reconst_depth]|![JNet_589_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 4.091107177734376, MSE: 0.0010918512707576156, quantized loss: 0.0003629819257184863  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi003_im006._original_depth]|![JNet_589_pretrain_beads_roi003_im006._output_depth]|![JNet_589_pretrain_beads_roi003_im006._reconst_depth]|![JNet_589_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 4.202637207031251, MSE: 0.0010933353332802653, quantized loss: 0.00038651737850159407  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi004_im006._original_depth]|![JNet_589_pretrain_beads_roi004_im006._output_depth]|![JNet_589_pretrain_beads_roi004_im006._reconst_depth]|![JNet_589_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 4.311104492187501, MSE: 0.0011133629595860839, quantized loss: 0.0003879757132381201  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi005_im007._original_depth]|![JNet_589_pretrain_beads_roi005_im007._output_depth]|![JNet_589_pretrain_beads_roi005_im007._reconst_depth]|![JNet_589_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 4.077726318359376, MSE: 0.0011001249076798558, quantized loss: 0.000381335848942399  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi006_im008._original_depth]|![JNet_589_pretrain_beads_roi006_im008._output_depth]|![JNet_589_pretrain_beads_roi006_im008._reconst_depth]|![JNet_589_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 4.331257324218751, MSE: 0.0010439100442454219, quantized loss: 0.00041710567893460393  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi007_im009._original_depth]|![JNet_589_pretrain_beads_roi007_im009._output_depth]|![JNet_589_pretrain_beads_roi007_im009._reconst_depth]|![JNet_589_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 4.179348144531251, MSE: 0.001107620308175683, quantized loss: 0.00038277325802482665  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi008_im010._original_depth]|![JNet_589_pretrain_beads_roi008_im010._output_depth]|![JNet_589_pretrain_beads_roi008_im010._reconst_depth]|![JNet_589_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 4.312634765625001, MSE: 0.0010804854100570083, quantized loss: 0.00037798230187036097  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi009_im011._original_depth]|![JNet_589_pretrain_beads_roi009_im011._output_depth]|![JNet_589_pretrain_beads_roi009_im011._reconst_depth]|![JNet_589_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 4.021227050781251, MSE: 0.0010586552089080215, quantized loss: 0.0003581719065550715  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi010_im012._original_depth]|![JNet_589_pretrain_beads_roi010_im012._output_depth]|![JNet_589_pretrain_beads_roi010_im012._reconst_depth]|![JNet_589_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 4.847553710937501, MSE: 0.0011097239330410957, quantized loss: 0.0004188823513686657  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi011_im013._original_depth]|![JNet_589_pretrain_beads_roi011_im013._output_depth]|![JNet_589_pretrain_beads_roi011_im013._reconst_depth]|![JNet_589_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 4.786534667968751, MSE: 0.0010804113699123263, quantized loss: 0.00041909675928764045  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi012_im014._original_depth]|![JNet_589_pretrain_beads_roi012_im014._output_depth]|![JNet_589_pretrain_beads_roi012_im014._reconst_depth]|![JNet_589_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 4.090441894531251, MSE: 0.0012091809185221791, quantized loss: 0.0003817819815594703  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi013_im015._original_depth]|![JNet_589_pretrain_beads_roi013_im015._output_depth]|![JNet_589_pretrain_beads_roi013_im015._reconst_depth]|![JNet_589_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 3.984273681640626, MSE: 0.0011441779788583517, quantized loss: 0.0003797804529312998  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi014_im016._original_depth]|![JNet_589_pretrain_beads_roi014_im016._output_depth]|![JNet_589_pretrain_beads_roi014_im016._reconst_depth]|![JNet_589_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 3.947255126953126, MSE: 0.001063598901964724, quantized loss: 0.00038644325104542077  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi015_im017._original_depth]|![JNet_589_pretrain_beads_roi015_im017._output_depth]|![JNet_589_pretrain_beads_roi015_im017._reconst_depth]|![JNet_589_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 3.924223144531251, MSE: 0.0010910589480772614, quantized loss: 0.0003579160838853568  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi016_im018._original_depth]|![JNet_589_pretrain_beads_roi016_im018._output_depth]|![JNet_589_pretrain_beads_roi016_im018._reconst_depth]|![JNet_589_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 4.376124511718751, MSE: 0.0011886957800015807, quantized loss: 0.00036996902781538665  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi017_im018._original_depth]|![JNet_589_pretrain_beads_roi017_im018._output_depth]|![JNet_589_pretrain_beads_roi017_im018._reconst_depth]|![JNet_589_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 4.303888183593751, MSE: 0.0012334787752479315, quantized loss: 0.00037377807893790305  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi018_im022._original_depth]|![JNet_589_pretrain_beads_roi018_im022._output_depth]|![JNet_589_pretrain_beads_roi018_im022._reconst_depth]|![JNet_589_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 3.750876953125001, MSE: 0.0010769737418740988, quantized loss: 0.0003627376281656325  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi019_im023._original_depth]|![JNet_589_pretrain_beads_roi019_im023._output_depth]|![JNet_589_pretrain_beads_roi019_im023._reconst_depth]|![JNet_589_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 3.679505371093751, MSE: 0.0010905331000685692, quantized loss: 0.0003451873199082911  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi020_im024._original_depth]|![JNet_589_pretrain_beads_roi020_im024._output_depth]|![JNet_589_pretrain_beads_roi020_im024._reconst_depth]|![JNet_589_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 4.498078613281251, MSE: 0.001104299328289926, quantized loss: 0.00038012847653590143  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi021_im026._original_depth]|![JNet_589_pretrain_beads_roi021_im026._output_depth]|![JNet_589_pretrain_beads_roi021_im026._reconst_depth]|![JNet_589_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 4.272832031250001, MSE: 0.0010501776123419404, quantized loss: 0.00037700406392104924  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi022_im027._original_depth]|![JNet_589_pretrain_beads_roi022_im027._output_depth]|![JNet_589_pretrain_beads_roi022_im027._reconst_depth]|![JNet_589_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 4.067499023437501, MSE: 0.0011325584491714835, quantized loss: 0.0003595001471694559  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi023_im028._original_depth]|![JNet_589_pretrain_beads_roi023_im028._output_depth]|![JNet_589_pretrain_beads_roi023_im028._reconst_depth]|![JNet_589_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 4.559042480468751, MSE: 0.0009612541180104017, quantized loss: 0.00045270449481904507  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi024_im028._original_depth]|![JNet_589_pretrain_beads_roi024_im028._output_depth]|![JNet_589_pretrain_beads_roi024_im028._reconst_depth]|![JNet_589_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 4.475713378906251, MSE: 0.001016300288029015, quantized loss: 0.00041699950816109776  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi025_im028._original_depth]|![JNet_589_pretrain_beads_roi025_im028._output_depth]|![JNet_589_pretrain_beads_roi025_im028._reconst_depth]|![JNet_589_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 4.475713378906251, MSE: 0.001016300288029015, quantized loss: 0.00041699950816109776  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi026_im029._original_depth]|![JNet_589_pretrain_beads_roi026_im029._output_depth]|![JNet_589_pretrain_beads_roi026_im029._reconst_depth]|![JNet_589_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 4.5003647460937515, MSE: 0.00114008120726794, quantized loss: 0.00039411854231730103  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi027_im029._original_depth]|![JNet_589_pretrain_beads_roi027_im029._output_depth]|![JNet_589_pretrain_beads_roi027_im029._reconst_depth]|![JNet_589_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 4.082078613281251, MSE: 0.0011179113062098622, quantized loss: 0.0004059431958012283  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi028_im030._original_depth]|![JNet_589_pretrain_beads_roi028_im030._output_depth]|![JNet_589_pretrain_beads_roi028_im030._reconst_depth]|![JNet_589_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 3.977997802734376, MSE: 0.001081458874978125, quantized loss: 0.00036328635178506374  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi029_im030._original_depth]|![JNet_589_pretrain_beads_roi029_im030._output_depth]|![JNet_589_pretrain_beads_roi029_im030._reconst_depth]|![JNet_589_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 4.193475585937501, MSE: 0.0011130132479593158, quantized loss: 0.0003671287267934531  

### finetuning
  
volume mean: 2.2841834187825527, volume sd: 0.2221861996504908
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi000_im000._original_depth]|![JNet_591_beads_roi000_im000._output_depth]|![JNet_591_beads_roi000_im000._reconst_depth]|![JNet_591_beads_roi000_im000._heatmap_depth]|
  
volume: 2.1330615234375006, MSE: 0.0030736371409147978, quantized loss: 0.00019820613670162857  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi001_im004._original_depth]|![JNet_591_beads_roi001_im004._output_depth]|![JNet_591_beads_roi001_im004._reconst_depth]|![JNet_591_beads_roi001_im004._heatmap_depth]|
  
volume: 2.7305258789062505, MSE: 0.0034884968772530556, quantized loss: 0.00028403487522155046  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi002_im005._original_depth]|![JNet_591_beads_roi002_im005._output_depth]|![JNet_591_beads_roi002_im005._reconst_depth]|![JNet_591_beads_roi002_im005._heatmap_depth]|
  
volume: 2.1924963378906255, MSE: 0.0034500397741794586, quantized loss: 0.0002394737530266866  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi003_im006._original_depth]|![JNet_591_beads_roi003_im006._output_depth]|![JNet_591_beads_roi003_im006._reconst_depth]|![JNet_591_beads_roi003_im006._heatmap_depth]|
  
volume: 2.3318898925781255, MSE: 0.003352154977619648, quantized loss: 0.0003112352278549224  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi004_im006._original_depth]|![JNet_591_beads_roi004_im006._output_depth]|![JNet_591_beads_roi004_im006._reconst_depth]|![JNet_591_beads_roi004_im006._heatmap_depth]|
  
volume: 2.4051372070312507, MSE: 0.0033544229809194803, quantized loss: 0.00030195110593922436  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi005_im007._original_depth]|![JNet_591_beads_roi005_im007._output_depth]|![JNet_591_beads_roi005_im007._reconst_depth]|![JNet_591_beads_roi005_im007._heatmap_depth]|
  
volume: 2.1606850585937507, MSE: 0.003082873998209834, quantized loss: 0.0002691877889446914  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi006_im008._original_depth]|![JNet_591_beads_roi006_im008._output_depth]|![JNet_591_beads_roi006_im008._reconst_depth]|![JNet_591_beads_roi006_im008._heatmap_depth]|
  
volume: 2.3371914062500005, MSE: 0.0033496732357889414, quantized loss: 0.00030179155874066055  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi007_im009._original_depth]|![JNet_591_beads_roi007_im009._output_depth]|![JNet_591_beads_roi007_im009._reconst_depth]|![JNet_591_beads_roi007_im009._heatmap_depth]|
  
volume: 2.3626540527343756, MSE: 0.003398285247385502, quantized loss: 0.00029239169089123607  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi008_im010._original_depth]|![JNet_591_beads_roi008_im010._output_depth]|![JNet_591_beads_roi008_im010._reconst_depth]|![JNet_591_beads_roi008_im010._heatmap_depth]|
  
volume: 2.3075500488281255, MSE: 0.0035191241186112165, quantized loss: 0.00026150679332204163  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi009_im011._original_depth]|![JNet_591_beads_roi009_im011._output_depth]|![JNet_591_beads_roi009_im011._reconst_depth]|![JNet_591_beads_roi009_im011._heatmap_depth]|
  
volume: 1.996126586914063, MSE: 0.0032683496829122305, quantized loss: 0.0001906081015476957  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi010_im012._original_depth]|![JNet_591_beads_roi010_im012._output_depth]|![JNet_591_beads_roi010_im012._reconst_depth]|![JNet_591_beads_roi010_im012._heatmap_depth]|
  
volume: 2.6761877441406257, MSE: 0.003594058332964778, quantized loss: 0.000298151804599911  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi011_im013._original_depth]|![JNet_591_beads_roi011_im013._output_depth]|![JNet_591_beads_roi011_im013._reconst_depth]|![JNet_591_beads_roi011_im013._heatmap_depth]|
  
volume: 2.6865200195312506, MSE: 0.0035038769710808992, quantized loss: 0.000257847917964682  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi012_im014._original_depth]|![JNet_591_beads_roi012_im014._output_depth]|![JNet_591_beads_roi012_im014._reconst_depth]|![JNet_591_beads_roi012_im014._heatmap_depth]|
  
volume: 2.474188232421876, MSE: 0.003284425474703312, quantized loss: 0.00025607561110518873  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi013_im015._original_depth]|![JNet_591_beads_roi013_im015._output_depth]|![JNet_591_beads_roi013_im015._reconst_depth]|![JNet_591_beads_roi013_im015._heatmap_depth]|
  
volume: 2.0262302246093755, MSE: 0.0031612105667591095, quantized loss: 0.0002465026918798685  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi014_im016._original_depth]|![JNet_591_beads_roi014_im016._output_depth]|![JNet_591_beads_roi014_im016._reconst_depth]|![JNet_591_beads_roi014_im016._heatmap_depth]|
  
volume: 2.2725417480468755, MSE: 0.003128601470962167, quantized loss: 0.0003433115198276937  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi015_im017._original_depth]|![JNet_591_beads_roi015_im017._output_depth]|![JNet_591_beads_roi015_im017._reconst_depth]|![JNet_591_beads_roi015_im017._heatmap_depth]|
  
volume: 2.1119899902343753, MSE: 0.003094903426244855, quantized loss: 0.0002902524429373443  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi016_im018._original_depth]|![JNet_591_beads_roi016_im018._output_depth]|![JNet_591_beads_roi016_im018._reconst_depth]|![JNet_591_beads_roi016_im018._heatmap_depth]|
  
volume: 2.4625791015625005, MSE: 0.0033977823331952095, quantized loss: 0.00028432856197468936  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi017_im018._original_depth]|![JNet_591_beads_roi017_im018._output_depth]|![JNet_591_beads_roi017_im018._reconst_depth]|![JNet_591_beads_roi017_im018._heatmap_depth]|
  
volume: 2.6512902832031258, MSE: 0.0030736832413822412, quantized loss: 0.0003239138750359416  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi018_im022._original_depth]|![JNet_591_beads_roi018_im022._output_depth]|![JNet_591_beads_roi018_im022._reconst_depth]|![JNet_591_beads_roi018_im022._heatmap_depth]|
  
volume: 1.8660677490234379, MSE: 0.0028929992113262415, quantized loss: 0.000227047799853608  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi019_im023._original_depth]|![JNet_591_beads_roi019_im023._output_depth]|![JNet_591_beads_roi019_im023._reconst_depth]|![JNet_591_beads_roi019_im023._heatmap_depth]|
  
volume: 1.972799682617188, MSE: 0.002750013954937458, quantized loss: 0.0002504151489119977  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi020_im024._original_depth]|![JNet_591_beads_roi020_im024._output_depth]|![JNet_591_beads_roi020_im024._reconst_depth]|![JNet_591_beads_roi020_im024._heatmap_depth]|
  
volume: 2.4757326660156256, MSE: 0.0033008484169840813, quantized loss: 0.00025378153077326715  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi021_im026._original_depth]|![JNet_591_beads_roi021_im026._output_depth]|![JNet_591_beads_roi021_im026._reconst_depth]|![JNet_591_beads_roi021_im026._heatmap_depth]|
  
volume: 2.2923723144531256, MSE: 0.003393483115360141, quantized loss: 0.00021049341012258083  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi022_im027._original_depth]|![JNet_591_beads_roi022_im027._output_depth]|![JNet_591_beads_roi022_im027._reconst_depth]|![JNet_591_beads_roi022_im027._heatmap_depth]|
  
volume: 2.2353466796875003, MSE: 0.0031857562717050314, quantized loss: 0.0002766685211099684  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi023_im028._original_depth]|![JNet_591_beads_roi023_im028._output_depth]|![JNet_591_beads_roi023_im028._reconst_depth]|![JNet_591_beads_roi023_im028._heatmap_depth]|
  
volume: 2.1863627929687506, MSE: 0.003792710369452834, quantized loss: 0.0002937555545940995  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi024_im028._original_depth]|![JNet_591_beads_roi024_im028._output_depth]|![JNet_591_beads_roi024_im028._reconst_depth]|![JNet_591_beads_roi024_im028._heatmap_depth]|
  
volume: 2.2548012695312507, MSE: 0.003617211477831006, quantized loss: 0.00025975462631322443  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi025_im028._original_depth]|![JNet_591_beads_roi025_im028._output_depth]|![JNet_591_beads_roi025_im028._reconst_depth]|![JNet_591_beads_roi025_im028._heatmap_depth]|
  
volume: 2.2548012695312507, MSE: 0.003617211477831006, quantized loss: 0.00025975462631322443  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi026_im029._original_depth]|![JNet_591_beads_roi026_im029._output_depth]|![JNet_591_beads_roi026_im029._reconst_depth]|![JNet_591_beads_roi026_im029._heatmap_depth]|
  
volume: 2.4791657714843756, MSE: 0.003545750631019473, quantized loss: 0.00029355729930102825  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi027_im029._original_depth]|![JNet_591_beads_roi027_im029._output_depth]|![JNet_591_beads_roi027_im029._reconst_depth]|![JNet_591_beads_roi027_im029._heatmap_depth]|
  
volume: 2.1094287109375003, MSE: 0.0033423888962715864, quantized loss: 0.00026833423180505633  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi028_im030._original_depth]|![JNet_591_beads_roi028_im030._output_depth]|![JNet_591_beads_roi028_im030._reconst_depth]|![JNet_591_beads_roi028_im030._heatmap_depth]|
  
volume: 2.0155051269531254, MSE: 0.003167136339470744, quantized loss: 0.0002366172120673582  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_591_beads_roi029_im030._original_depth]|![JNet_591_beads_roi029_im030._output_depth]|![JNet_591_beads_roi029_im030._reconst_depth]|![JNet_591_beads_roi029_im030._heatmap_depth]|
  
volume: 2.0642731933593756, MSE: 0.003192639211192727, quantized loss: 0.00020523270359262824  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_591_psf_pre]|![JNet_591_psf_post]|

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
  



[JNet_589_pretrain_0_aligned_depth]: /experiments/images/JNet_589_pretrain_0_aligned_depth.png
[JNet_589_pretrain_0_aligned_plane]: /experiments/images/JNet_589_pretrain_0_aligned_plane.png
[JNet_589_pretrain_0_labelx_depth]: /experiments/images/JNet_589_pretrain_0_labelx_depth.png
[JNet_589_pretrain_0_labelx_plane]: /experiments/images/JNet_589_pretrain_0_labelx_plane.png
[JNet_589_pretrain_0_labelz_depth]: /experiments/images/JNet_589_pretrain_0_labelz_depth.png
[JNet_589_pretrain_0_labelz_plane]: /experiments/images/JNet_589_pretrain_0_labelz_plane.png
[JNet_589_pretrain_0_novibrate_depth]: /experiments/images/JNet_589_pretrain_0_novibrate_depth.png
[JNet_589_pretrain_0_novibrate_plane]: /experiments/images/JNet_589_pretrain_0_novibrate_plane.png
[JNet_589_pretrain_0_original_depth]: /experiments/images/JNet_589_pretrain_0_original_depth.png
[JNet_589_pretrain_0_original_plane]: /experiments/images/JNet_589_pretrain_0_original_plane.png
[JNet_589_pretrain_0_outputx_depth]: /experiments/images/JNet_589_pretrain_0_outputx_depth.png
[JNet_589_pretrain_0_outputx_plane]: /experiments/images/JNet_589_pretrain_0_outputx_plane.png
[JNet_589_pretrain_0_outputz_depth]: /experiments/images/JNet_589_pretrain_0_outputz_depth.png
[JNet_589_pretrain_0_outputz_plane]: /experiments/images/JNet_589_pretrain_0_outputz_plane.png
[JNet_589_pretrain_1_aligned_depth]: /experiments/images/JNet_589_pretrain_1_aligned_depth.png
[JNet_589_pretrain_1_aligned_plane]: /experiments/images/JNet_589_pretrain_1_aligned_plane.png
[JNet_589_pretrain_1_labelx_depth]: /experiments/images/JNet_589_pretrain_1_labelx_depth.png
[JNet_589_pretrain_1_labelx_plane]: /experiments/images/JNet_589_pretrain_1_labelx_plane.png
[JNet_589_pretrain_1_labelz_depth]: /experiments/images/JNet_589_pretrain_1_labelz_depth.png
[JNet_589_pretrain_1_labelz_plane]: /experiments/images/JNet_589_pretrain_1_labelz_plane.png
[JNet_589_pretrain_1_novibrate_depth]: /experiments/images/JNet_589_pretrain_1_novibrate_depth.png
[JNet_589_pretrain_1_novibrate_plane]: /experiments/images/JNet_589_pretrain_1_novibrate_plane.png
[JNet_589_pretrain_1_original_depth]: /experiments/images/JNet_589_pretrain_1_original_depth.png
[JNet_589_pretrain_1_original_plane]: /experiments/images/JNet_589_pretrain_1_original_plane.png
[JNet_589_pretrain_1_outputx_depth]: /experiments/images/JNet_589_pretrain_1_outputx_depth.png
[JNet_589_pretrain_1_outputx_plane]: /experiments/images/JNet_589_pretrain_1_outputx_plane.png
[JNet_589_pretrain_1_outputz_depth]: /experiments/images/JNet_589_pretrain_1_outputz_depth.png
[JNet_589_pretrain_1_outputz_plane]: /experiments/images/JNet_589_pretrain_1_outputz_plane.png
[JNet_589_pretrain_2_aligned_depth]: /experiments/images/JNet_589_pretrain_2_aligned_depth.png
[JNet_589_pretrain_2_aligned_plane]: /experiments/images/JNet_589_pretrain_2_aligned_plane.png
[JNet_589_pretrain_2_labelx_depth]: /experiments/images/JNet_589_pretrain_2_labelx_depth.png
[JNet_589_pretrain_2_labelx_plane]: /experiments/images/JNet_589_pretrain_2_labelx_plane.png
[JNet_589_pretrain_2_labelz_depth]: /experiments/images/JNet_589_pretrain_2_labelz_depth.png
[JNet_589_pretrain_2_labelz_plane]: /experiments/images/JNet_589_pretrain_2_labelz_plane.png
[JNet_589_pretrain_2_novibrate_depth]: /experiments/images/JNet_589_pretrain_2_novibrate_depth.png
[JNet_589_pretrain_2_novibrate_plane]: /experiments/images/JNet_589_pretrain_2_novibrate_plane.png
[JNet_589_pretrain_2_original_depth]: /experiments/images/JNet_589_pretrain_2_original_depth.png
[JNet_589_pretrain_2_original_plane]: /experiments/images/JNet_589_pretrain_2_original_plane.png
[JNet_589_pretrain_2_outputx_depth]: /experiments/images/JNet_589_pretrain_2_outputx_depth.png
[JNet_589_pretrain_2_outputx_plane]: /experiments/images/JNet_589_pretrain_2_outputx_plane.png
[JNet_589_pretrain_2_outputz_depth]: /experiments/images/JNet_589_pretrain_2_outputz_depth.png
[JNet_589_pretrain_2_outputz_plane]: /experiments/images/JNet_589_pretrain_2_outputz_plane.png
[JNet_589_pretrain_3_aligned_depth]: /experiments/images/JNet_589_pretrain_3_aligned_depth.png
[JNet_589_pretrain_3_aligned_plane]: /experiments/images/JNet_589_pretrain_3_aligned_plane.png
[JNet_589_pretrain_3_labelx_depth]: /experiments/images/JNet_589_pretrain_3_labelx_depth.png
[JNet_589_pretrain_3_labelx_plane]: /experiments/images/JNet_589_pretrain_3_labelx_plane.png
[JNet_589_pretrain_3_labelz_depth]: /experiments/images/JNet_589_pretrain_3_labelz_depth.png
[JNet_589_pretrain_3_labelz_plane]: /experiments/images/JNet_589_pretrain_3_labelz_plane.png
[JNet_589_pretrain_3_novibrate_depth]: /experiments/images/JNet_589_pretrain_3_novibrate_depth.png
[JNet_589_pretrain_3_novibrate_plane]: /experiments/images/JNet_589_pretrain_3_novibrate_plane.png
[JNet_589_pretrain_3_original_depth]: /experiments/images/JNet_589_pretrain_3_original_depth.png
[JNet_589_pretrain_3_original_plane]: /experiments/images/JNet_589_pretrain_3_original_plane.png
[JNet_589_pretrain_3_outputx_depth]: /experiments/images/JNet_589_pretrain_3_outputx_depth.png
[JNet_589_pretrain_3_outputx_plane]: /experiments/images/JNet_589_pretrain_3_outputx_plane.png
[JNet_589_pretrain_3_outputz_depth]: /experiments/images/JNet_589_pretrain_3_outputz_depth.png
[JNet_589_pretrain_3_outputz_plane]: /experiments/images/JNet_589_pretrain_3_outputz_plane.png
[JNet_589_pretrain_4_aligned_depth]: /experiments/images/JNet_589_pretrain_4_aligned_depth.png
[JNet_589_pretrain_4_aligned_plane]: /experiments/images/JNet_589_pretrain_4_aligned_plane.png
[JNet_589_pretrain_4_labelx_depth]: /experiments/images/JNet_589_pretrain_4_labelx_depth.png
[JNet_589_pretrain_4_labelx_plane]: /experiments/images/JNet_589_pretrain_4_labelx_plane.png
[JNet_589_pretrain_4_labelz_depth]: /experiments/images/JNet_589_pretrain_4_labelz_depth.png
[JNet_589_pretrain_4_labelz_plane]: /experiments/images/JNet_589_pretrain_4_labelz_plane.png
[JNet_589_pretrain_4_novibrate_depth]: /experiments/images/JNet_589_pretrain_4_novibrate_depth.png
[JNet_589_pretrain_4_novibrate_plane]: /experiments/images/JNet_589_pretrain_4_novibrate_plane.png
[JNet_589_pretrain_4_original_depth]: /experiments/images/JNet_589_pretrain_4_original_depth.png
[JNet_589_pretrain_4_original_plane]: /experiments/images/JNet_589_pretrain_4_original_plane.png
[JNet_589_pretrain_4_outputx_depth]: /experiments/images/JNet_589_pretrain_4_outputx_depth.png
[JNet_589_pretrain_4_outputx_plane]: /experiments/images/JNet_589_pretrain_4_outputx_plane.png
[JNet_589_pretrain_4_outputz_depth]: /experiments/images/JNet_589_pretrain_4_outputz_depth.png
[JNet_589_pretrain_4_outputz_plane]: /experiments/images/JNet_589_pretrain_4_outputz_plane.png
[JNet_589_pretrain_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi000_im000._heatmap_depth.png
[JNet_589_pretrain_beads_roi000_im000._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi000_im000._original_depth.png
[JNet_589_pretrain_beads_roi000_im000._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi000_im000._output_depth.png
[JNet_589_pretrain_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi000_im000._reconst_depth.png
[JNet_589_pretrain_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi001_im004._heatmap_depth.png
[JNet_589_pretrain_beads_roi001_im004._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi001_im004._original_depth.png
[JNet_589_pretrain_beads_roi001_im004._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi001_im004._output_depth.png
[JNet_589_pretrain_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi001_im004._reconst_depth.png
[JNet_589_pretrain_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi002_im005._heatmap_depth.png
[JNet_589_pretrain_beads_roi002_im005._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi002_im005._original_depth.png
[JNet_589_pretrain_beads_roi002_im005._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi002_im005._output_depth.png
[JNet_589_pretrain_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi002_im005._reconst_depth.png
[JNet_589_pretrain_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi003_im006._heatmap_depth.png
[JNet_589_pretrain_beads_roi003_im006._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi003_im006._original_depth.png
[JNet_589_pretrain_beads_roi003_im006._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi003_im006._output_depth.png
[JNet_589_pretrain_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi003_im006._reconst_depth.png
[JNet_589_pretrain_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi004_im006._heatmap_depth.png
[JNet_589_pretrain_beads_roi004_im006._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi004_im006._original_depth.png
[JNet_589_pretrain_beads_roi004_im006._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi004_im006._output_depth.png
[JNet_589_pretrain_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi004_im006._reconst_depth.png
[JNet_589_pretrain_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi005_im007._heatmap_depth.png
[JNet_589_pretrain_beads_roi005_im007._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi005_im007._original_depth.png
[JNet_589_pretrain_beads_roi005_im007._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi005_im007._output_depth.png
[JNet_589_pretrain_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi005_im007._reconst_depth.png
[JNet_589_pretrain_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi006_im008._heatmap_depth.png
[JNet_589_pretrain_beads_roi006_im008._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi006_im008._original_depth.png
[JNet_589_pretrain_beads_roi006_im008._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi006_im008._output_depth.png
[JNet_589_pretrain_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi006_im008._reconst_depth.png
[JNet_589_pretrain_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi007_im009._heatmap_depth.png
[JNet_589_pretrain_beads_roi007_im009._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi007_im009._original_depth.png
[JNet_589_pretrain_beads_roi007_im009._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi007_im009._output_depth.png
[JNet_589_pretrain_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi007_im009._reconst_depth.png
[JNet_589_pretrain_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi008_im010._heatmap_depth.png
[JNet_589_pretrain_beads_roi008_im010._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi008_im010._original_depth.png
[JNet_589_pretrain_beads_roi008_im010._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi008_im010._output_depth.png
[JNet_589_pretrain_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi008_im010._reconst_depth.png
[JNet_589_pretrain_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi009_im011._heatmap_depth.png
[JNet_589_pretrain_beads_roi009_im011._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi009_im011._original_depth.png
[JNet_589_pretrain_beads_roi009_im011._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi009_im011._output_depth.png
[JNet_589_pretrain_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi009_im011._reconst_depth.png
[JNet_589_pretrain_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi010_im012._heatmap_depth.png
[JNet_589_pretrain_beads_roi010_im012._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi010_im012._original_depth.png
[JNet_589_pretrain_beads_roi010_im012._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi010_im012._output_depth.png
[JNet_589_pretrain_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi010_im012._reconst_depth.png
[JNet_589_pretrain_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi011_im013._heatmap_depth.png
[JNet_589_pretrain_beads_roi011_im013._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi011_im013._original_depth.png
[JNet_589_pretrain_beads_roi011_im013._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi011_im013._output_depth.png
[JNet_589_pretrain_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi011_im013._reconst_depth.png
[JNet_589_pretrain_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi012_im014._heatmap_depth.png
[JNet_589_pretrain_beads_roi012_im014._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi012_im014._original_depth.png
[JNet_589_pretrain_beads_roi012_im014._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi012_im014._output_depth.png
[JNet_589_pretrain_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi012_im014._reconst_depth.png
[JNet_589_pretrain_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi013_im015._heatmap_depth.png
[JNet_589_pretrain_beads_roi013_im015._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi013_im015._original_depth.png
[JNet_589_pretrain_beads_roi013_im015._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi013_im015._output_depth.png
[JNet_589_pretrain_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi013_im015._reconst_depth.png
[JNet_589_pretrain_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi014_im016._heatmap_depth.png
[JNet_589_pretrain_beads_roi014_im016._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi014_im016._original_depth.png
[JNet_589_pretrain_beads_roi014_im016._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi014_im016._output_depth.png
[JNet_589_pretrain_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi014_im016._reconst_depth.png
[JNet_589_pretrain_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi015_im017._heatmap_depth.png
[JNet_589_pretrain_beads_roi015_im017._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi015_im017._original_depth.png
[JNet_589_pretrain_beads_roi015_im017._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi015_im017._output_depth.png
[JNet_589_pretrain_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi015_im017._reconst_depth.png
[JNet_589_pretrain_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi016_im018._heatmap_depth.png
[JNet_589_pretrain_beads_roi016_im018._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi016_im018._original_depth.png
[JNet_589_pretrain_beads_roi016_im018._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi016_im018._output_depth.png
[JNet_589_pretrain_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi016_im018._reconst_depth.png
[JNet_589_pretrain_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi017_im018._heatmap_depth.png
[JNet_589_pretrain_beads_roi017_im018._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi017_im018._original_depth.png
[JNet_589_pretrain_beads_roi017_im018._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi017_im018._output_depth.png
[JNet_589_pretrain_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi017_im018._reconst_depth.png
[JNet_589_pretrain_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi018_im022._heatmap_depth.png
[JNet_589_pretrain_beads_roi018_im022._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi018_im022._original_depth.png
[JNet_589_pretrain_beads_roi018_im022._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi018_im022._output_depth.png
[JNet_589_pretrain_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi018_im022._reconst_depth.png
[JNet_589_pretrain_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi019_im023._heatmap_depth.png
[JNet_589_pretrain_beads_roi019_im023._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi019_im023._original_depth.png
[JNet_589_pretrain_beads_roi019_im023._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi019_im023._output_depth.png
[JNet_589_pretrain_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi019_im023._reconst_depth.png
[JNet_589_pretrain_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi020_im024._heatmap_depth.png
[JNet_589_pretrain_beads_roi020_im024._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi020_im024._original_depth.png
[JNet_589_pretrain_beads_roi020_im024._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi020_im024._output_depth.png
[JNet_589_pretrain_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi020_im024._reconst_depth.png
[JNet_589_pretrain_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi021_im026._heatmap_depth.png
[JNet_589_pretrain_beads_roi021_im026._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi021_im026._original_depth.png
[JNet_589_pretrain_beads_roi021_im026._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi021_im026._output_depth.png
[JNet_589_pretrain_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi021_im026._reconst_depth.png
[JNet_589_pretrain_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi022_im027._heatmap_depth.png
[JNet_589_pretrain_beads_roi022_im027._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi022_im027._original_depth.png
[JNet_589_pretrain_beads_roi022_im027._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi022_im027._output_depth.png
[JNet_589_pretrain_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi022_im027._reconst_depth.png
[JNet_589_pretrain_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi023_im028._heatmap_depth.png
[JNet_589_pretrain_beads_roi023_im028._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi023_im028._original_depth.png
[JNet_589_pretrain_beads_roi023_im028._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi023_im028._output_depth.png
[JNet_589_pretrain_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi023_im028._reconst_depth.png
[JNet_589_pretrain_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi024_im028._heatmap_depth.png
[JNet_589_pretrain_beads_roi024_im028._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi024_im028._original_depth.png
[JNet_589_pretrain_beads_roi024_im028._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi024_im028._output_depth.png
[JNet_589_pretrain_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi024_im028._reconst_depth.png
[JNet_589_pretrain_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi025_im028._heatmap_depth.png
[JNet_589_pretrain_beads_roi025_im028._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi025_im028._original_depth.png
[JNet_589_pretrain_beads_roi025_im028._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi025_im028._output_depth.png
[JNet_589_pretrain_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi025_im028._reconst_depth.png
[JNet_589_pretrain_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi026_im029._heatmap_depth.png
[JNet_589_pretrain_beads_roi026_im029._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi026_im029._original_depth.png
[JNet_589_pretrain_beads_roi026_im029._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi026_im029._output_depth.png
[JNet_589_pretrain_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi026_im029._reconst_depth.png
[JNet_589_pretrain_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi027_im029._heatmap_depth.png
[JNet_589_pretrain_beads_roi027_im029._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi027_im029._original_depth.png
[JNet_589_pretrain_beads_roi027_im029._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi027_im029._output_depth.png
[JNet_589_pretrain_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi027_im029._reconst_depth.png
[JNet_589_pretrain_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi028_im030._heatmap_depth.png
[JNet_589_pretrain_beads_roi028_im030._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi028_im030._original_depth.png
[JNet_589_pretrain_beads_roi028_im030._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi028_im030._output_depth.png
[JNet_589_pretrain_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi028_im030._reconst_depth.png
[JNet_589_pretrain_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_589_pretrain_beads_roi029_im030._heatmap_depth.png
[JNet_589_pretrain_beads_roi029_im030._original_depth]: /experiments/images/JNet_589_pretrain_beads_roi029_im030._original_depth.png
[JNet_589_pretrain_beads_roi029_im030._output_depth]: /experiments/images/JNet_589_pretrain_beads_roi029_im030._output_depth.png
[JNet_589_pretrain_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_589_pretrain_beads_roi029_im030._reconst_depth.png
[JNet_591_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_591_beads_roi000_im000._heatmap_depth.png
[JNet_591_beads_roi000_im000._original_depth]: /experiments/images/JNet_591_beads_roi000_im000._original_depth.png
[JNet_591_beads_roi000_im000._output_depth]: /experiments/images/JNet_591_beads_roi000_im000._output_depth.png
[JNet_591_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_591_beads_roi000_im000._reconst_depth.png
[JNet_591_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_591_beads_roi001_im004._heatmap_depth.png
[JNet_591_beads_roi001_im004._original_depth]: /experiments/images/JNet_591_beads_roi001_im004._original_depth.png
[JNet_591_beads_roi001_im004._output_depth]: /experiments/images/JNet_591_beads_roi001_im004._output_depth.png
[JNet_591_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_591_beads_roi001_im004._reconst_depth.png
[JNet_591_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_591_beads_roi002_im005._heatmap_depth.png
[JNet_591_beads_roi002_im005._original_depth]: /experiments/images/JNet_591_beads_roi002_im005._original_depth.png
[JNet_591_beads_roi002_im005._output_depth]: /experiments/images/JNet_591_beads_roi002_im005._output_depth.png
[JNet_591_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_591_beads_roi002_im005._reconst_depth.png
[JNet_591_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_591_beads_roi003_im006._heatmap_depth.png
[JNet_591_beads_roi003_im006._original_depth]: /experiments/images/JNet_591_beads_roi003_im006._original_depth.png
[JNet_591_beads_roi003_im006._output_depth]: /experiments/images/JNet_591_beads_roi003_im006._output_depth.png
[JNet_591_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_591_beads_roi003_im006._reconst_depth.png
[JNet_591_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_591_beads_roi004_im006._heatmap_depth.png
[JNet_591_beads_roi004_im006._original_depth]: /experiments/images/JNet_591_beads_roi004_im006._original_depth.png
[JNet_591_beads_roi004_im006._output_depth]: /experiments/images/JNet_591_beads_roi004_im006._output_depth.png
[JNet_591_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_591_beads_roi004_im006._reconst_depth.png
[JNet_591_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_591_beads_roi005_im007._heatmap_depth.png
[JNet_591_beads_roi005_im007._original_depth]: /experiments/images/JNet_591_beads_roi005_im007._original_depth.png
[JNet_591_beads_roi005_im007._output_depth]: /experiments/images/JNet_591_beads_roi005_im007._output_depth.png
[JNet_591_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_591_beads_roi005_im007._reconst_depth.png
[JNet_591_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_591_beads_roi006_im008._heatmap_depth.png
[JNet_591_beads_roi006_im008._original_depth]: /experiments/images/JNet_591_beads_roi006_im008._original_depth.png
[JNet_591_beads_roi006_im008._output_depth]: /experiments/images/JNet_591_beads_roi006_im008._output_depth.png
[JNet_591_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_591_beads_roi006_im008._reconst_depth.png
[JNet_591_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_591_beads_roi007_im009._heatmap_depth.png
[JNet_591_beads_roi007_im009._original_depth]: /experiments/images/JNet_591_beads_roi007_im009._original_depth.png
[JNet_591_beads_roi007_im009._output_depth]: /experiments/images/JNet_591_beads_roi007_im009._output_depth.png
[JNet_591_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_591_beads_roi007_im009._reconst_depth.png
[JNet_591_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_591_beads_roi008_im010._heatmap_depth.png
[JNet_591_beads_roi008_im010._original_depth]: /experiments/images/JNet_591_beads_roi008_im010._original_depth.png
[JNet_591_beads_roi008_im010._output_depth]: /experiments/images/JNet_591_beads_roi008_im010._output_depth.png
[JNet_591_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_591_beads_roi008_im010._reconst_depth.png
[JNet_591_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_591_beads_roi009_im011._heatmap_depth.png
[JNet_591_beads_roi009_im011._original_depth]: /experiments/images/JNet_591_beads_roi009_im011._original_depth.png
[JNet_591_beads_roi009_im011._output_depth]: /experiments/images/JNet_591_beads_roi009_im011._output_depth.png
[JNet_591_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_591_beads_roi009_im011._reconst_depth.png
[JNet_591_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_591_beads_roi010_im012._heatmap_depth.png
[JNet_591_beads_roi010_im012._original_depth]: /experiments/images/JNet_591_beads_roi010_im012._original_depth.png
[JNet_591_beads_roi010_im012._output_depth]: /experiments/images/JNet_591_beads_roi010_im012._output_depth.png
[JNet_591_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_591_beads_roi010_im012._reconst_depth.png
[JNet_591_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_591_beads_roi011_im013._heatmap_depth.png
[JNet_591_beads_roi011_im013._original_depth]: /experiments/images/JNet_591_beads_roi011_im013._original_depth.png
[JNet_591_beads_roi011_im013._output_depth]: /experiments/images/JNet_591_beads_roi011_im013._output_depth.png
[JNet_591_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_591_beads_roi011_im013._reconst_depth.png
[JNet_591_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_591_beads_roi012_im014._heatmap_depth.png
[JNet_591_beads_roi012_im014._original_depth]: /experiments/images/JNet_591_beads_roi012_im014._original_depth.png
[JNet_591_beads_roi012_im014._output_depth]: /experiments/images/JNet_591_beads_roi012_im014._output_depth.png
[JNet_591_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_591_beads_roi012_im014._reconst_depth.png
[JNet_591_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_591_beads_roi013_im015._heatmap_depth.png
[JNet_591_beads_roi013_im015._original_depth]: /experiments/images/JNet_591_beads_roi013_im015._original_depth.png
[JNet_591_beads_roi013_im015._output_depth]: /experiments/images/JNet_591_beads_roi013_im015._output_depth.png
[JNet_591_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_591_beads_roi013_im015._reconst_depth.png
[JNet_591_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_591_beads_roi014_im016._heatmap_depth.png
[JNet_591_beads_roi014_im016._original_depth]: /experiments/images/JNet_591_beads_roi014_im016._original_depth.png
[JNet_591_beads_roi014_im016._output_depth]: /experiments/images/JNet_591_beads_roi014_im016._output_depth.png
[JNet_591_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_591_beads_roi014_im016._reconst_depth.png
[JNet_591_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_591_beads_roi015_im017._heatmap_depth.png
[JNet_591_beads_roi015_im017._original_depth]: /experiments/images/JNet_591_beads_roi015_im017._original_depth.png
[JNet_591_beads_roi015_im017._output_depth]: /experiments/images/JNet_591_beads_roi015_im017._output_depth.png
[JNet_591_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_591_beads_roi015_im017._reconst_depth.png
[JNet_591_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_591_beads_roi016_im018._heatmap_depth.png
[JNet_591_beads_roi016_im018._original_depth]: /experiments/images/JNet_591_beads_roi016_im018._original_depth.png
[JNet_591_beads_roi016_im018._output_depth]: /experiments/images/JNet_591_beads_roi016_im018._output_depth.png
[JNet_591_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_591_beads_roi016_im018._reconst_depth.png
[JNet_591_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_591_beads_roi017_im018._heatmap_depth.png
[JNet_591_beads_roi017_im018._original_depth]: /experiments/images/JNet_591_beads_roi017_im018._original_depth.png
[JNet_591_beads_roi017_im018._output_depth]: /experiments/images/JNet_591_beads_roi017_im018._output_depth.png
[JNet_591_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_591_beads_roi017_im018._reconst_depth.png
[JNet_591_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_591_beads_roi018_im022._heatmap_depth.png
[JNet_591_beads_roi018_im022._original_depth]: /experiments/images/JNet_591_beads_roi018_im022._original_depth.png
[JNet_591_beads_roi018_im022._output_depth]: /experiments/images/JNet_591_beads_roi018_im022._output_depth.png
[JNet_591_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_591_beads_roi018_im022._reconst_depth.png
[JNet_591_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_591_beads_roi019_im023._heatmap_depth.png
[JNet_591_beads_roi019_im023._original_depth]: /experiments/images/JNet_591_beads_roi019_im023._original_depth.png
[JNet_591_beads_roi019_im023._output_depth]: /experiments/images/JNet_591_beads_roi019_im023._output_depth.png
[JNet_591_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_591_beads_roi019_im023._reconst_depth.png
[JNet_591_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_591_beads_roi020_im024._heatmap_depth.png
[JNet_591_beads_roi020_im024._original_depth]: /experiments/images/JNet_591_beads_roi020_im024._original_depth.png
[JNet_591_beads_roi020_im024._output_depth]: /experiments/images/JNet_591_beads_roi020_im024._output_depth.png
[JNet_591_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_591_beads_roi020_im024._reconst_depth.png
[JNet_591_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_591_beads_roi021_im026._heatmap_depth.png
[JNet_591_beads_roi021_im026._original_depth]: /experiments/images/JNet_591_beads_roi021_im026._original_depth.png
[JNet_591_beads_roi021_im026._output_depth]: /experiments/images/JNet_591_beads_roi021_im026._output_depth.png
[JNet_591_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_591_beads_roi021_im026._reconst_depth.png
[JNet_591_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_591_beads_roi022_im027._heatmap_depth.png
[JNet_591_beads_roi022_im027._original_depth]: /experiments/images/JNet_591_beads_roi022_im027._original_depth.png
[JNet_591_beads_roi022_im027._output_depth]: /experiments/images/JNet_591_beads_roi022_im027._output_depth.png
[JNet_591_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_591_beads_roi022_im027._reconst_depth.png
[JNet_591_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_591_beads_roi023_im028._heatmap_depth.png
[JNet_591_beads_roi023_im028._original_depth]: /experiments/images/JNet_591_beads_roi023_im028._original_depth.png
[JNet_591_beads_roi023_im028._output_depth]: /experiments/images/JNet_591_beads_roi023_im028._output_depth.png
[JNet_591_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_591_beads_roi023_im028._reconst_depth.png
[JNet_591_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_591_beads_roi024_im028._heatmap_depth.png
[JNet_591_beads_roi024_im028._original_depth]: /experiments/images/JNet_591_beads_roi024_im028._original_depth.png
[JNet_591_beads_roi024_im028._output_depth]: /experiments/images/JNet_591_beads_roi024_im028._output_depth.png
[JNet_591_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_591_beads_roi024_im028._reconst_depth.png
[JNet_591_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_591_beads_roi025_im028._heatmap_depth.png
[JNet_591_beads_roi025_im028._original_depth]: /experiments/images/JNet_591_beads_roi025_im028._original_depth.png
[JNet_591_beads_roi025_im028._output_depth]: /experiments/images/JNet_591_beads_roi025_im028._output_depth.png
[JNet_591_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_591_beads_roi025_im028._reconst_depth.png
[JNet_591_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_591_beads_roi026_im029._heatmap_depth.png
[JNet_591_beads_roi026_im029._original_depth]: /experiments/images/JNet_591_beads_roi026_im029._original_depth.png
[JNet_591_beads_roi026_im029._output_depth]: /experiments/images/JNet_591_beads_roi026_im029._output_depth.png
[JNet_591_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_591_beads_roi026_im029._reconst_depth.png
[JNet_591_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_591_beads_roi027_im029._heatmap_depth.png
[JNet_591_beads_roi027_im029._original_depth]: /experiments/images/JNet_591_beads_roi027_im029._original_depth.png
[JNet_591_beads_roi027_im029._output_depth]: /experiments/images/JNet_591_beads_roi027_im029._output_depth.png
[JNet_591_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_591_beads_roi027_im029._reconst_depth.png
[JNet_591_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_591_beads_roi028_im030._heatmap_depth.png
[JNet_591_beads_roi028_im030._original_depth]: /experiments/images/JNet_591_beads_roi028_im030._original_depth.png
[JNet_591_beads_roi028_im030._output_depth]: /experiments/images/JNet_591_beads_roi028_im030._output_depth.png
[JNet_591_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_591_beads_roi028_im030._reconst_depth.png
[JNet_591_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_591_beads_roi029_im030._heatmap_depth.png
[JNet_591_beads_roi029_im030._original_depth]: /experiments/images/JNet_591_beads_roi029_im030._original_depth.png
[JNet_591_beads_roi029_im030._output_depth]: /experiments/images/JNet_591_beads_roi029_im030._output_depth.png
[JNet_591_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_591_beads_roi029_im030._reconst_depth.png
[JNet_591_psf_post]: /experiments/images/JNet_591_psf_post.png
[JNet_591_psf_pre]: /experiments/images/JNet_591_psf_pre.png
