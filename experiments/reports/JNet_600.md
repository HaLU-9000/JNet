



# JNet_600 Report
  
589_pre ewc0.01, parameter1.0  
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
|device|cuda:0||

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
  
Segmentation: mean MSE: 0.007919438183307648, mean BCE: 0.032210540026426315  
Luminance Estimation: mean MSE: 0.984542727470398, mean BCE: inf
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_0_original_plane]|![JNet_589_pretrain_0_novibrate_plane]|![JNet_589_pretrain_0_aligned_plane]|![JNet_589_pretrain_0_outputx_plane]|![JNet_589_pretrain_0_labelx_plane]|![JNet_589_pretrain_0_outputz_plane]|![JNet_589_pretrain_0_labelz_plane]|
  
MSEx: 0.007999162189662457, BCEx: 0.032499417662620544  
MSEz: 0.9847531914710999, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_0_original_depth]|![JNet_589_pretrain_0_novibrate_depth]|![JNet_589_pretrain_0_aligned_depth]|![JNet_589_pretrain_0_outputx_depth]|![JNet_589_pretrain_0_labelx_depth]|![JNet_589_pretrain_0_outputz_depth]|![JNet_589_pretrain_0_labelz_depth]|
  
MSEx: 0.007999162189662457, BCEx: 0.032499417662620544  
MSEz: 0.9847531914710999, BCEz: inf  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_1_original_plane]|![JNet_589_pretrain_1_novibrate_plane]|![JNet_589_pretrain_1_aligned_plane]|![JNet_589_pretrain_1_outputx_plane]|![JNet_589_pretrain_1_labelx_plane]|![JNet_589_pretrain_1_outputz_plane]|![JNet_589_pretrain_1_labelz_plane]|
  
MSEx: 0.004675444681197405, BCEx: 0.019468748942017555  
MSEz: 0.992560625076294, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_1_original_depth]|![JNet_589_pretrain_1_novibrate_depth]|![JNet_589_pretrain_1_aligned_depth]|![JNet_589_pretrain_1_outputx_depth]|![JNet_589_pretrain_1_labelx_depth]|![JNet_589_pretrain_1_outputz_depth]|![JNet_589_pretrain_1_labelz_depth]|
  
MSEx: 0.004675444681197405, BCEx: 0.019468748942017555  
MSEz: 0.992560625076294, BCEz: inf  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_2_original_plane]|![JNet_589_pretrain_2_novibrate_plane]|![JNet_589_pretrain_2_aligned_plane]|![JNet_589_pretrain_2_outputx_plane]|![JNet_589_pretrain_2_labelx_plane]|![JNet_589_pretrain_2_outputz_plane]|![JNet_589_pretrain_2_labelz_plane]|
  
MSEx: 0.008937245234847069, BCEx: 0.03575515374541283  
MSEz: 0.9790393710136414, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_2_original_depth]|![JNet_589_pretrain_2_novibrate_depth]|![JNet_589_pretrain_2_aligned_depth]|![JNet_589_pretrain_2_outputx_depth]|![JNet_589_pretrain_2_labelx_depth]|![JNet_589_pretrain_2_outputz_depth]|![JNet_589_pretrain_2_labelz_depth]|
  
MSEx: 0.008937245234847069, BCEx: 0.03575515374541283  
MSEz: 0.9790393710136414, BCEz: inf  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_3_original_plane]|![JNet_589_pretrain_3_novibrate_plane]|![JNet_589_pretrain_3_aligned_plane]|![JNet_589_pretrain_3_outputx_plane]|![JNet_589_pretrain_3_labelx_plane]|![JNet_589_pretrain_3_outputz_plane]|![JNet_589_pretrain_3_labelz_plane]|
  
MSEx: 0.00900029856711626, BCEx: 0.03671051189303398  
MSEz: 0.9830915927886963, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_3_original_depth]|![JNet_589_pretrain_3_novibrate_depth]|![JNet_589_pretrain_3_aligned_depth]|![JNet_589_pretrain_3_outputx_depth]|![JNet_589_pretrain_3_labelx_depth]|![JNet_589_pretrain_3_outputz_depth]|![JNet_589_pretrain_3_labelz_depth]|
  
MSEx: 0.00900029856711626, BCEx: 0.03671051189303398  
MSEz: 0.9830915927886963, BCEz: inf  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_4_original_plane]|![JNet_589_pretrain_4_novibrate_plane]|![JNet_589_pretrain_4_aligned_plane]|![JNet_589_pretrain_4_outputx_plane]|![JNet_589_pretrain_4_labelx_plane]|![JNet_589_pretrain_4_outputz_plane]|![JNet_589_pretrain_4_labelz_plane]|
  
MSEx: 0.00898503977805376, BCEx: 0.036618877202272415  
MSEz: 0.9832690358161926, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_4_original_depth]|![JNet_589_pretrain_4_novibrate_depth]|![JNet_589_pretrain_4_aligned_depth]|![JNet_589_pretrain_4_outputx_depth]|![JNet_589_pretrain_4_labelx_depth]|![JNet_589_pretrain_4_outputz_depth]|![JNet_589_pretrain_4_labelz_depth]|
  
MSEx: 0.00898503977805376, BCEx: 0.036618877202272415  
MSEz: 0.9832690358161926, BCEz: inf  

### pretrain
  
volume mean: 4.092204166666668, volume sd: 0.28505492249179915
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi000_im000._original_depth]|![JNet_589_pretrain_beads_roi000_im000._output_depth]|![JNet_589_pretrain_beads_roi000_im000._reconst_depth]|![JNet_589_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 3.820625000000001, MSE: 0.0010962235974147916, quantized loss: 0.00035057374043390155  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi001_im004._original_depth]|![JNet_589_pretrain_beads_roi001_im004._output_depth]|![JNet_589_pretrain_beads_roi001_im004._reconst_depth]|![JNet_589_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 4.550500000000001, MSE: 0.0011226573260501027, quantized loss: 0.000413618516176939  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi002_im005._original_depth]|![JNet_589_pretrain_beads_roi002_im005._output_depth]|![JNet_589_pretrain_beads_roi002_im005._reconst_depth]|![JNet_589_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 3.943625000000001, MSE: 0.0010918512707576156, quantized loss: 0.0003629819257184863  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi003_im006._original_depth]|![JNet_589_pretrain_beads_roi003_im006._output_depth]|![JNet_589_pretrain_beads_roi003_im006._reconst_depth]|![JNet_589_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 4.016375000000001, MSE: 0.0010933353332802653, quantized loss: 0.00038651737850159407  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi004_im006._original_depth]|![JNet_589_pretrain_beads_roi004_im006._output_depth]|![JNet_589_pretrain_beads_roi004_im006._reconst_depth]|![JNet_589_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 4.111000000000001, MSE: 0.0011133629595860839, quantized loss: 0.0003879757132381201  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi005_im007._original_depth]|![JNet_589_pretrain_beads_roi005_im007._output_depth]|![JNet_589_pretrain_beads_roi005_im007._reconst_depth]|![JNet_589_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 3.907125000000001, MSE: 0.0011001249076798558, quantized loss: 0.000381335848942399  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi006_im008._original_depth]|![JNet_589_pretrain_beads_roi006_im008._output_depth]|![JNet_589_pretrain_beads_roi006_im008._reconst_depth]|![JNet_589_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 4.069000000000001, MSE: 0.0010439100442454219, quantized loss: 0.00041710567893460393  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi007_im009._original_depth]|![JNet_589_pretrain_beads_roi007_im009._output_depth]|![JNet_589_pretrain_beads_roi007_im009._reconst_depth]|![JNet_589_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 3.912125000000001, MSE: 0.001107620308175683, quantized loss: 0.00038277325802482665  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi008_im010._original_depth]|![JNet_589_pretrain_beads_roi008_im010._output_depth]|![JNet_589_pretrain_beads_roi008_im010._reconst_depth]|![JNet_589_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 4.145875000000001, MSE: 0.0010804854100570083, quantized loss: 0.00037798230187036097  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi009_im011._original_depth]|![JNet_589_pretrain_beads_roi009_im011._output_depth]|![JNet_589_pretrain_beads_roi009_im011._reconst_depth]|![JNet_589_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 3.902250000000001, MSE: 0.0010586552089080215, quantized loss: 0.0003581719065550715  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi010_im012._original_depth]|![JNet_589_pretrain_beads_roi010_im012._output_depth]|![JNet_589_pretrain_beads_roi010_im012._reconst_depth]|![JNet_589_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 4.725125000000001, MSE: 0.0011097239330410957, quantized loss: 0.0004188823513686657  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi011_im013._original_depth]|![JNet_589_pretrain_beads_roi011_im013._output_depth]|![JNet_589_pretrain_beads_roi011_im013._reconst_depth]|![JNet_589_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 4.646625000000001, MSE: 0.0010804113699123263, quantized loss: 0.00041909675928764045  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi012_im014._original_depth]|![JNet_589_pretrain_beads_roi012_im014._output_depth]|![JNet_589_pretrain_beads_roi012_im014._reconst_depth]|![JNet_589_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 3.977250000000001, MSE: 0.0012091809185221791, quantized loss: 0.0003817819815594703  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi013_im015._original_depth]|![JNet_589_pretrain_beads_roi013_im015._output_depth]|![JNet_589_pretrain_beads_roi013_im015._reconst_depth]|![JNet_589_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 3.837125000000001, MSE: 0.0011441779788583517, quantized loss: 0.0003797804529312998  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi014_im016._original_depth]|![JNet_589_pretrain_beads_roi014_im016._output_depth]|![JNet_589_pretrain_beads_roi014_im016._reconst_depth]|![JNet_589_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 3.751125000000001, MSE: 0.001063598901964724, quantized loss: 0.00038644325104542077  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi015_im017._original_depth]|![JNet_589_pretrain_beads_roi015_im017._output_depth]|![JNet_589_pretrain_beads_roi015_im017._reconst_depth]|![JNet_589_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 3.761000000000001, MSE: 0.0010910589480772614, quantized loss: 0.0003579160838853568  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi016_im018._original_depth]|![JNet_589_pretrain_beads_roi016_im018._output_depth]|![JNet_589_pretrain_beads_roi016_im018._reconst_depth]|![JNet_589_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 4.209750000000001, MSE: 0.0011886957800015807, quantized loss: 0.00036996902781538665  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi017_im018._original_depth]|![JNet_589_pretrain_beads_roi017_im018._output_depth]|![JNet_589_pretrain_beads_roi017_im018._reconst_depth]|![JNet_589_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 4.134125000000001, MSE: 0.0012334787752479315, quantized loss: 0.00037377807893790305  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi018_im022._original_depth]|![JNet_589_pretrain_beads_roi018_im022._output_depth]|![JNet_589_pretrain_beads_roi018_im022._reconst_depth]|![JNet_589_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 3.653750000000001, MSE: 0.0010769737418740988, quantized loss: 0.0003627376281656325  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi019_im023._original_depth]|![JNet_589_pretrain_beads_roi019_im023._output_depth]|![JNet_589_pretrain_beads_roi019_im023._reconst_depth]|![JNet_589_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 3.590250000000001, MSE: 0.0010905331000685692, quantized loss: 0.0003451873199082911  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi020_im024._original_depth]|![JNet_589_pretrain_beads_roi020_im024._output_depth]|![JNet_589_pretrain_beads_roi020_im024._reconst_depth]|![JNet_589_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 4.404125000000001, MSE: 0.001104299328289926, quantized loss: 0.00038012847653590143  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi021_im026._original_depth]|![JNet_589_pretrain_beads_roi021_im026._output_depth]|![JNet_589_pretrain_beads_roi021_im026._reconst_depth]|![JNet_589_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 4.178875000000001, MSE: 0.0010501776123419404, quantized loss: 0.00037700406392104924  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi022_im027._original_depth]|![JNet_589_pretrain_beads_roi022_im027._output_depth]|![JNet_589_pretrain_beads_roi022_im027._reconst_depth]|![JNet_589_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 3.968375000000001, MSE: 0.0011325584491714835, quantized loss: 0.0003595001471694559  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi023_im028._original_depth]|![JNet_589_pretrain_beads_roi023_im028._output_depth]|![JNet_589_pretrain_beads_roi023_im028._reconst_depth]|![JNet_589_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 4.431375000000001, MSE: 0.0009612541180104017, quantized loss: 0.00045270449481904507  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi024_im028._original_depth]|![JNet_589_pretrain_beads_roi024_im028._output_depth]|![JNet_589_pretrain_beads_roi024_im028._reconst_depth]|![JNet_589_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 4.374000000000001, MSE: 0.001016300288029015, quantized loss: 0.00041699950816109776  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi025_im028._original_depth]|![JNet_589_pretrain_beads_roi025_im028._output_depth]|![JNet_589_pretrain_beads_roi025_im028._reconst_depth]|![JNet_589_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 4.374000000000001, MSE: 0.001016300288029015, quantized loss: 0.00041699950816109776  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi026_im029._original_depth]|![JNet_589_pretrain_beads_roi026_im029._output_depth]|![JNet_589_pretrain_beads_roi026_im029._reconst_depth]|![JNet_589_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 4.403000000000001, MSE: 0.00114008120726794, quantized loss: 0.00039411854231730103  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi027_im029._original_depth]|![JNet_589_pretrain_beads_roi027_im029._output_depth]|![JNet_589_pretrain_beads_roi027_im029._reconst_depth]|![JNet_589_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 3.967250000000001, MSE: 0.0011179113062098622, quantized loss: 0.0004059431958012283  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi028_im030._original_depth]|![JNet_589_pretrain_beads_roi028_im030._output_depth]|![JNet_589_pretrain_beads_roi028_im030._reconst_depth]|![JNet_589_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 3.889875000000001, MSE: 0.001081458874978125, quantized loss: 0.00036328635178506374  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_589_pretrain_beads_roi029_im030._original_depth]|![JNet_589_pretrain_beads_roi029_im030._output_depth]|![JNet_589_pretrain_beads_roi029_im030._reconst_depth]|![JNet_589_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 4.110625000000001, MSE: 0.0011130132479593158, quantized loss: 0.0003671287267934531  

### finetuning
  
volume mean: 4.112825000000002, volume sd: 0.3052296550004058
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi000_im000._original_depth]|![JNet_600_beads_roi000_im000._output_depth]|![JNet_600_beads_roi000_im000._reconst_depth]|![JNet_600_beads_roi000_im000._heatmap_depth]|
  
volume: 3.918000000000001, MSE: 0.0016917032189667225, quantized loss: 0.00030092810629867017  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi001_im004._original_depth]|![JNet_600_beads_roi001_im004._output_depth]|![JNet_600_beads_roi001_im004._reconst_depth]|![JNet_600_beads_roi001_im004._heatmap_depth]|
  
volume: 4.6007500000000014, MSE: 0.0019777286797761917, quantized loss: 0.00041285419138148427  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi002_im005._original_depth]|![JNet_600_beads_roi002_im005._output_depth]|![JNet_600_beads_roi002_im005._reconst_depth]|![JNet_600_beads_roi002_im005._heatmap_depth]|
  
volume: 3.968750000000001, MSE: 0.0018571199616417289, quantized loss: 0.00032915204064920545  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi003_im006._original_depth]|![JNet_600_beads_roi003_im006._output_depth]|![JNet_600_beads_roi003_im006._reconst_depth]|![JNet_600_beads_roi003_im006._heatmap_depth]|
  
volume: 4.029250000000001, MSE: 0.0017788293771445751, quantized loss: 0.000389514840207994  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi004_im006._original_depth]|![JNet_600_beads_roi004_im006._output_depth]|![JNet_600_beads_roi004_im006._reconst_depth]|![JNet_600_beads_roi004_im006._heatmap_depth]|
  
volume: 4.134000000000001, MSE: 0.0018676543841138482, quantized loss: 0.000409362226491794  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi005_im007._original_depth]|![JNet_600_beads_roi005_im007._output_depth]|![JNet_600_beads_roi005_im007._reconst_depth]|![JNet_600_beads_roi005_im007._heatmap_depth]|
  
volume: 3.944875000000001, MSE: 0.0018776747165247798, quantized loss: 0.00040212454041466117  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi006_im008._original_depth]|![JNet_600_beads_roi006_im008._output_depth]|![JNet_600_beads_roi006_im008._reconst_depth]|![JNet_600_beads_roi006_im008._heatmap_depth]|
  
volume: 4.152375000000001, MSE: 0.0018468977650627494, quantized loss: 0.00045242017949931324  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi007_im009._original_depth]|![JNet_600_beads_roi007_im009._output_depth]|![JNet_600_beads_roi007_im009._reconst_depth]|![JNet_600_beads_roi007_im009._heatmap_depth]|
  
volume: 4.068000000000001, MSE: 0.001844066078774631, quantized loss: 0.0004492577281780541  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi008_im010._original_depth]|![JNet_600_beads_roi008_im010._output_depth]|![JNet_600_beads_roi008_im010._reconst_depth]|![JNet_600_beads_roi008_im010._heatmap_depth]|
  
volume: 4.241750000000001, MSE: 0.001903981319628656, quantized loss: 0.0003726912254933268  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi009_im011._original_depth]|![JNet_600_beads_roi009_im011._output_depth]|![JNet_600_beads_roi009_im011._reconst_depth]|![JNet_600_beads_roi009_im011._heatmap_depth]|
  
volume: 4.007125000000001, MSE: 0.0017532967031002045, quantized loss: 0.00032386896782554686  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi010_im012._original_depth]|![JNet_600_beads_roi010_im012._output_depth]|![JNet_600_beads_roi010_im012._reconst_depth]|![JNet_600_beads_roi010_im012._heatmap_depth]|
  
volume: 4.696500000000001, MSE: 0.0021057692356407642, quantized loss: 0.0003761352563742548  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi011_im013._original_depth]|![JNet_600_beads_roi011_im013._output_depth]|![JNet_600_beads_roi011_im013._reconst_depth]|![JNet_600_beads_roi011_im013._heatmap_depth]|
  
volume: 4.645500000000001, MSE: 0.002023442415520549, quantized loss: 0.00036584201734513044  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi012_im014._original_depth]|![JNet_600_beads_roi012_im014._output_depth]|![JNet_600_beads_roi012_im014._reconst_depth]|![JNet_600_beads_roi012_im014._heatmap_depth]|
  
volume: 4.018500000000001, MSE: 0.0019236319931223989, quantized loss: 0.0003235215262975544  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi013_im015._original_depth]|![JNet_600_beads_roi013_im015._output_depth]|![JNet_600_beads_roi013_im015._reconst_depth]|![JNet_600_beads_roi013_im015._heatmap_depth]|
  
volume: 3.782375000000001, MSE: 0.0017211507074534893, quantized loss: 0.00033381168032065034  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi014_im016._original_depth]|![JNet_600_beads_roi014_im016._output_depth]|![JNet_600_beads_roi014_im016._reconst_depth]|![JNet_600_beads_roi014_im016._heatmap_depth]|
  
volume: 3.6786250000000007, MSE: 0.0017516766674816608, quantized loss: 0.0004065247776452452  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi015_im017._original_depth]|![JNet_600_beads_roi015_im017._output_depth]|![JNet_600_beads_roi015_im017._reconst_depth]|![JNet_600_beads_roi015_im017._heatmap_depth]|
  
volume: 3.8076250000000007, MSE: 0.0017247451469302177, quantized loss: 0.00035230512730777264  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi016_im018._original_depth]|![JNet_600_beads_roi016_im018._output_depth]|![JNet_600_beads_roi016_im018._reconst_depth]|![JNet_600_beads_roi016_im018._heatmap_depth]|
  
volume: 4.310500000000001, MSE: 0.0019808958750218153, quantized loss: 0.0004068651469424367  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi017_im018._original_depth]|![JNet_600_beads_roi017_im018._output_depth]|![JNet_600_beads_roi017_im018._reconst_depth]|![JNet_600_beads_roi017_im018._heatmap_depth]|
  
volume: 4.162500000000001, MSE: 0.0019692499190568924, quantized loss: 0.0004001895140390843  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi018_im022._original_depth]|![JNet_600_beads_roi018_im022._output_depth]|![JNet_600_beads_roi018_im022._reconst_depth]|![JNet_600_beads_roi018_im022._heatmap_depth]|
  
volume: 3.525500000000001, MSE: 0.0015461173607036471, quantized loss: 0.0002911184274125844  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi019_im023._original_depth]|![JNet_600_beads_roi019_im023._output_depth]|![JNet_600_beads_roi019_im023._reconst_depth]|![JNet_600_beads_roi019_im023._heatmap_depth]|
  
volume: 3.436875000000001, MSE: 0.0015081739984452724, quantized loss: 0.000289322022581473  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi020_im024._original_depth]|![JNet_600_beads_roi020_im024._output_depth]|![JNet_600_beads_roi020_im024._reconst_depth]|![JNet_600_beads_roi020_im024._heatmap_depth]|
  
volume: 4.435375000000001, MSE: 0.00191575288772583, quantized loss: 0.0003295907808933407  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi021_im026._original_depth]|![JNet_600_beads_roi021_im026._output_depth]|![JNet_600_beads_roi021_im026._reconst_depth]|![JNet_600_beads_roi021_im026._heatmap_depth]|
  
volume: 4.251250000000001, MSE: 0.001861124881543219, quantized loss: 0.00032063203980214894  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi022_im027._original_depth]|![JNet_600_beads_roi022_im027._output_depth]|![JNet_600_beads_roi022_im027._reconst_depth]|![JNet_600_beads_roi022_im027._heatmap_depth]|
  
volume: 4.093125000000001, MSE: 0.0018729279981926084, quantized loss: 0.0003163076180499047  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi023_im028._original_depth]|![JNet_600_beads_roi023_im028._output_depth]|![JNet_600_beads_roi023_im028._reconst_depth]|![JNet_600_beads_roi023_im028._heatmap_depth]|
  
volume: 4.440000000000001, MSE: 0.002025582594797015, quantized loss: 0.00036713879671879113  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi024_im028._original_depth]|![JNet_600_beads_roi024_im028._output_depth]|![JNet_600_beads_roi024_im028._reconst_depth]|![JNet_600_beads_roi024_im028._heatmap_depth]|
  
volume: 4.3275000000000015, MSE: 0.0019608938600867987, quantized loss: 0.0003431625082157552  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi025_im028._original_depth]|![JNet_600_beads_roi025_im028._output_depth]|![JNet_600_beads_roi025_im028._reconst_depth]|![JNet_600_beads_roi025_im028._heatmap_depth]|
  
volume: 4.3275000000000015, MSE: 0.0019608938600867987, quantized loss: 0.0003431625082157552  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi026_im029._original_depth]|![JNet_600_beads_roi026_im029._output_depth]|![JNet_600_beads_roi026_im029._reconst_depth]|![JNet_600_beads_roi026_im029._heatmap_depth]|
  
volume: 4.472750000000001, MSE: 0.0019949208945035934, quantized loss: 0.0003282915859017521  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi027_im029._original_depth]|![JNet_600_beads_roi027_im029._output_depth]|![JNet_600_beads_roi027_im029._reconst_depth]|![JNet_600_beads_roi027_im029._heatmap_depth]|
  
volume: 3.939125000000001, MSE: 0.00180625484790653, quantized loss: 0.00032186030875891447  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi028_im030._original_depth]|![JNet_600_beads_roi028_im030._output_depth]|![JNet_600_beads_roi028_im030._reconst_depth]|![JNet_600_beads_roi028_im030._heatmap_depth]|
  
volume: 3.829375000000001, MSE: 0.0016877450980246067, quantized loss: 0.0002999379066750407  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_600_beads_roi029_im030._original_depth]|![JNet_600_beads_roi029_im030._output_depth]|![JNet_600_beads_roi029_im030._reconst_depth]|![JNet_600_beads_roi029_im030._heatmap_depth]|
  
volume: 4.139375000000001, MSE: 0.0017418827628716826, quantized loss: 0.0003149181720800698  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_600_psf_pre]|![JNet_600_psf_post]|

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
[JNet_600_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_600_beads_roi000_im000._heatmap_depth.png
[JNet_600_beads_roi000_im000._original_depth]: /experiments/images/JNet_600_beads_roi000_im000._original_depth.png
[JNet_600_beads_roi000_im000._output_depth]: /experiments/images/JNet_600_beads_roi000_im000._output_depth.png
[JNet_600_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_600_beads_roi000_im000._reconst_depth.png
[JNet_600_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_600_beads_roi001_im004._heatmap_depth.png
[JNet_600_beads_roi001_im004._original_depth]: /experiments/images/JNet_600_beads_roi001_im004._original_depth.png
[JNet_600_beads_roi001_im004._output_depth]: /experiments/images/JNet_600_beads_roi001_im004._output_depth.png
[JNet_600_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_600_beads_roi001_im004._reconst_depth.png
[JNet_600_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_600_beads_roi002_im005._heatmap_depth.png
[JNet_600_beads_roi002_im005._original_depth]: /experiments/images/JNet_600_beads_roi002_im005._original_depth.png
[JNet_600_beads_roi002_im005._output_depth]: /experiments/images/JNet_600_beads_roi002_im005._output_depth.png
[JNet_600_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_600_beads_roi002_im005._reconst_depth.png
[JNet_600_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_600_beads_roi003_im006._heatmap_depth.png
[JNet_600_beads_roi003_im006._original_depth]: /experiments/images/JNet_600_beads_roi003_im006._original_depth.png
[JNet_600_beads_roi003_im006._output_depth]: /experiments/images/JNet_600_beads_roi003_im006._output_depth.png
[JNet_600_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_600_beads_roi003_im006._reconst_depth.png
[JNet_600_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_600_beads_roi004_im006._heatmap_depth.png
[JNet_600_beads_roi004_im006._original_depth]: /experiments/images/JNet_600_beads_roi004_im006._original_depth.png
[JNet_600_beads_roi004_im006._output_depth]: /experiments/images/JNet_600_beads_roi004_im006._output_depth.png
[JNet_600_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_600_beads_roi004_im006._reconst_depth.png
[JNet_600_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_600_beads_roi005_im007._heatmap_depth.png
[JNet_600_beads_roi005_im007._original_depth]: /experiments/images/JNet_600_beads_roi005_im007._original_depth.png
[JNet_600_beads_roi005_im007._output_depth]: /experiments/images/JNet_600_beads_roi005_im007._output_depth.png
[JNet_600_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_600_beads_roi005_im007._reconst_depth.png
[JNet_600_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_600_beads_roi006_im008._heatmap_depth.png
[JNet_600_beads_roi006_im008._original_depth]: /experiments/images/JNet_600_beads_roi006_im008._original_depth.png
[JNet_600_beads_roi006_im008._output_depth]: /experiments/images/JNet_600_beads_roi006_im008._output_depth.png
[JNet_600_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_600_beads_roi006_im008._reconst_depth.png
[JNet_600_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_600_beads_roi007_im009._heatmap_depth.png
[JNet_600_beads_roi007_im009._original_depth]: /experiments/images/JNet_600_beads_roi007_im009._original_depth.png
[JNet_600_beads_roi007_im009._output_depth]: /experiments/images/JNet_600_beads_roi007_im009._output_depth.png
[JNet_600_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_600_beads_roi007_im009._reconst_depth.png
[JNet_600_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_600_beads_roi008_im010._heatmap_depth.png
[JNet_600_beads_roi008_im010._original_depth]: /experiments/images/JNet_600_beads_roi008_im010._original_depth.png
[JNet_600_beads_roi008_im010._output_depth]: /experiments/images/JNet_600_beads_roi008_im010._output_depth.png
[JNet_600_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_600_beads_roi008_im010._reconst_depth.png
[JNet_600_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_600_beads_roi009_im011._heatmap_depth.png
[JNet_600_beads_roi009_im011._original_depth]: /experiments/images/JNet_600_beads_roi009_im011._original_depth.png
[JNet_600_beads_roi009_im011._output_depth]: /experiments/images/JNet_600_beads_roi009_im011._output_depth.png
[JNet_600_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_600_beads_roi009_im011._reconst_depth.png
[JNet_600_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_600_beads_roi010_im012._heatmap_depth.png
[JNet_600_beads_roi010_im012._original_depth]: /experiments/images/JNet_600_beads_roi010_im012._original_depth.png
[JNet_600_beads_roi010_im012._output_depth]: /experiments/images/JNet_600_beads_roi010_im012._output_depth.png
[JNet_600_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_600_beads_roi010_im012._reconst_depth.png
[JNet_600_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_600_beads_roi011_im013._heatmap_depth.png
[JNet_600_beads_roi011_im013._original_depth]: /experiments/images/JNet_600_beads_roi011_im013._original_depth.png
[JNet_600_beads_roi011_im013._output_depth]: /experiments/images/JNet_600_beads_roi011_im013._output_depth.png
[JNet_600_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_600_beads_roi011_im013._reconst_depth.png
[JNet_600_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_600_beads_roi012_im014._heatmap_depth.png
[JNet_600_beads_roi012_im014._original_depth]: /experiments/images/JNet_600_beads_roi012_im014._original_depth.png
[JNet_600_beads_roi012_im014._output_depth]: /experiments/images/JNet_600_beads_roi012_im014._output_depth.png
[JNet_600_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_600_beads_roi012_im014._reconst_depth.png
[JNet_600_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_600_beads_roi013_im015._heatmap_depth.png
[JNet_600_beads_roi013_im015._original_depth]: /experiments/images/JNet_600_beads_roi013_im015._original_depth.png
[JNet_600_beads_roi013_im015._output_depth]: /experiments/images/JNet_600_beads_roi013_im015._output_depth.png
[JNet_600_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_600_beads_roi013_im015._reconst_depth.png
[JNet_600_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_600_beads_roi014_im016._heatmap_depth.png
[JNet_600_beads_roi014_im016._original_depth]: /experiments/images/JNet_600_beads_roi014_im016._original_depth.png
[JNet_600_beads_roi014_im016._output_depth]: /experiments/images/JNet_600_beads_roi014_im016._output_depth.png
[JNet_600_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_600_beads_roi014_im016._reconst_depth.png
[JNet_600_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_600_beads_roi015_im017._heatmap_depth.png
[JNet_600_beads_roi015_im017._original_depth]: /experiments/images/JNet_600_beads_roi015_im017._original_depth.png
[JNet_600_beads_roi015_im017._output_depth]: /experiments/images/JNet_600_beads_roi015_im017._output_depth.png
[JNet_600_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_600_beads_roi015_im017._reconst_depth.png
[JNet_600_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_600_beads_roi016_im018._heatmap_depth.png
[JNet_600_beads_roi016_im018._original_depth]: /experiments/images/JNet_600_beads_roi016_im018._original_depth.png
[JNet_600_beads_roi016_im018._output_depth]: /experiments/images/JNet_600_beads_roi016_im018._output_depth.png
[JNet_600_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_600_beads_roi016_im018._reconst_depth.png
[JNet_600_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_600_beads_roi017_im018._heatmap_depth.png
[JNet_600_beads_roi017_im018._original_depth]: /experiments/images/JNet_600_beads_roi017_im018._original_depth.png
[JNet_600_beads_roi017_im018._output_depth]: /experiments/images/JNet_600_beads_roi017_im018._output_depth.png
[JNet_600_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_600_beads_roi017_im018._reconst_depth.png
[JNet_600_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_600_beads_roi018_im022._heatmap_depth.png
[JNet_600_beads_roi018_im022._original_depth]: /experiments/images/JNet_600_beads_roi018_im022._original_depth.png
[JNet_600_beads_roi018_im022._output_depth]: /experiments/images/JNet_600_beads_roi018_im022._output_depth.png
[JNet_600_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_600_beads_roi018_im022._reconst_depth.png
[JNet_600_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_600_beads_roi019_im023._heatmap_depth.png
[JNet_600_beads_roi019_im023._original_depth]: /experiments/images/JNet_600_beads_roi019_im023._original_depth.png
[JNet_600_beads_roi019_im023._output_depth]: /experiments/images/JNet_600_beads_roi019_im023._output_depth.png
[JNet_600_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_600_beads_roi019_im023._reconst_depth.png
[JNet_600_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_600_beads_roi020_im024._heatmap_depth.png
[JNet_600_beads_roi020_im024._original_depth]: /experiments/images/JNet_600_beads_roi020_im024._original_depth.png
[JNet_600_beads_roi020_im024._output_depth]: /experiments/images/JNet_600_beads_roi020_im024._output_depth.png
[JNet_600_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_600_beads_roi020_im024._reconst_depth.png
[JNet_600_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_600_beads_roi021_im026._heatmap_depth.png
[JNet_600_beads_roi021_im026._original_depth]: /experiments/images/JNet_600_beads_roi021_im026._original_depth.png
[JNet_600_beads_roi021_im026._output_depth]: /experiments/images/JNet_600_beads_roi021_im026._output_depth.png
[JNet_600_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_600_beads_roi021_im026._reconst_depth.png
[JNet_600_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_600_beads_roi022_im027._heatmap_depth.png
[JNet_600_beads_roi022_im027._original_depth]: /experiments/images/JNet_600_beads_roi022_im027._original_depth.png
[JNet_600_beads_roi022_im027._output_depth]: /experiments/images/JNet_600_beads_roi022_im027._output_depth.png
[JNet_600_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_600_beads_roi022_im027._reconst_depth.png
[JNet_600_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_600_beads_roi023_im028._heatmap_depth.png
[JNet_600_beads_roi023_im028._original_depth]: /experiments/images/JNet_600_beads_roi023_im028._original_depth.png
[JNet_600_beads_roi023_im028._output_depth]: /experiments/images/JNet_600_beads_roi023_im028._output_depth.png
[JNet_600_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_600_beads_roi023_im028._reconst_depth.png
[JNet_600_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_600_beads_roi024_im028._heatmap_depth.png
[JNet_600_beads_roi024_im028._original_depth]: /experiments/images/JNet_600_beads_roi024_im028._original_depth.png
[JNet_600_beads_roi024_im028._output_depth]: /experiments/images/JNet_600_beads_roi024_im028._output_depth.png
[JNet_600_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_600_beads_roi024_im028._reconst_depth.png
[JNet_600_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_600_beads_roi025_im028._heatmap_depth.png
[JNet_600_beads_roi025_im028._original_depth]: /experiments/images/JNet_600_beads_roi025_im028._original_depth.png
[JNet_600_beads_roi025_im028._output_depth]: /experiments/images/JNet_600_beads_roi025_im028._output_depth.png
[JNet_600_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_600_beads_roi025_im028._reconst_depth.png
[JNet_600_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_600_beads_roi026_im029._heatmap_depth.png
[JNet_600_beads_roi026_im029._original_depth]: /experiments/images/JNet_600_beads_roi026_im029._original_depth.png
[JNet_600_beads_roi026_im029._output_depth]: /experiments/images/JNet_600_beads_roi026_im029._output_depth.png
[JNet_600_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_600_beads_roi026_im029._reconst_depth.png
[JNet_600_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_600_beads_roi027_im029._heatmap_depth.png
[JNet_600_beads_roi027_im029._original_depth]: /experiments/images/JNet_600_beads_roi027_im029._original_depth.png
[JNet_600_beads_roi027_im029._output_depth]: /experiments/images/JNet_600_beads_roi027_im029._output_depth.png
[JNet_600_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_600_beads_roi027_im029._reconst_depth.png
[JNet_600_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_600_beads_roi028_im030._heatmap_depth.png
[JNet_600_beads_roi028_im030._original_depth]: /experiments/images/JNet_600_beads_roi028_im030._original_depth.png
[JNet_600_beads_roi028_im030._output_depth]: /experiments/images/JNet_600_beads_roi028_im030._output_depth.png
[JNet_600_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_600_beads_roi028_im030._reconst_depth.png
[JNet_600_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_600_beads_roi029_im030._heatmap_depth.png
[JNet_600_beads_roi029_im030._original_depth]: /experiments/images/JNet_600_beads_roi029_im030._original_depth.png
[JNet_600_beads_roi029_im030._output_depth]: /experiments/images/JNet_600_beads_roi029_im030._output_depth.png
[JNet_600_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_600_beads_roi029_im030._reconst_depth.png
[JNet_600_psf_post]: /experiments/images/JNet_600_psf_post.png
[JNet_600_psf_pre]: /experiments/images/JNet_600_psf_pre.png
