



# JNet_576 Report
  
first beads experiment with new methods and data  
pretrained model : JNet_575_pretrain
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
|blur_mode|gaussian|`gaussian` or `gibsonlanni`|
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
|ewc_weight|0.1|
|qloss_weight|1|
|ploss_weight|10.0|
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
  
Segmentation: mean MSE: 0.00912503432482481, mean BCE: 0.037161290645599365  
Luminance Estimation: mean MSE: 0.9766618609428406, mean BCE: inf
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_0_original_plane]|![JNet_575_pretrain_0_novibrate_plane]|![JNet_575_pretrain_0_aligned_plane]|![JNet_575_pretrain_0_outputx_plane]|![JNet_575_pretrain_0_labelx_plane]|![JNet_575_pretrain_0_outputz_plane]|![JNet_575_pretrain_0_labelz_plane]|
  
MSEx: 0.005477619357407093, BCEx: 0.022853467613458633  
MSEz: 0.9834424257278442, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_0_original_depth]|![JNet_575_pretrain_0_novibrate_depth]|![JNet_575_pretrain_0_aligned_depth]|![JNet_575_pretrain_0_outputx_depth]|![JNet_575_pretrain_0_labelx_depth]|![JNet_575_pretrain_0_outputz_depth]|![JNet_575_pretrain_0_labelz_depth]|
  
MSEx: 0.005477619357407093, BCEx: 0.022853467613458633  
MSEz: 0.9834424257278442, BCEz: inf  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_1_original_plane]|![JNet_575_pretrain_1_novibrate_plane]|![JNet_575_pretrain_1_aligned_plane]|![JNet_575_pretrain_1_outputx_plane]|![JNet_575_pretrain_1_labelx_plane]|![JNet_575_pretrain_1_outputz_plane]|![JNet_575_pretrain_1_labelz_plane]|
  
MSEx: 0.011210820637643337, BCEx: 0.045252978801727295  
MSEz: 0.9698587656021118, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_1_original_depth]|![JNet_575_pretrain_1_novibrate_depth]|![JNet_575_pretrain_1_aligned_depth]|![JNet_575_pretrain_1_outputx_depth]|![JNet_575_pretrain_1_labelx_depth]|![JNet_575_pretrain_1_outputz_depth]|![JNet_575_pretrain_1_labelz_depth]|
  
MSEx: 0.011210820637643337, BCEx: 0.045252978801727295  
MSEz: 0.9698587656021118, BCEz: inf  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_2_original_plane]|![JNet_575_pretrain_2_novibrate_plane]|![JNet_575_pretrain_2_aligned_plane]|![JNet_575_pretrain_2_outputx_plane]|![JNet_575_pretrain_2_labelx_plane]|![JNet_575_pretrain_2_outputz_plane]|![JNet_575_pretrain_2_labelz_plane]|
  
MSEx: 0.011892426759004593, BCEx: 0.048970777541399  
MSEz: 0.9756659865379333, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_2_original_depth]|![JNet_575_pretrain_2_novibrate_depth]|![JNet_575_pretrain_2_aligned_depth]|![JNet_575_pretrain_2_outputx_depth]|![JNet_575_pretrain_2_labelx_depth]|![JNet_575_pretrain_2_outputz_depth]|![JNet_575_pretrain_2_labelz_depth]|
  
MSEx: 0.011892426759004593, BCEx: 0.048970777541399  
MSEz: 0.9756659865379333, BCEz: inf  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_3_original_plane]|![JNet_575_pretrain_3_novibrate_plane]|![JNet_575_pretrain_3_aligned_plane]|![JNet_575_pretrain_3_outputx_plane]|![JNet_575_pretrain_3_labelx_plane]|![JNet_575_pretrain_3_outputz_plane]|![JNet_575_pretrain_3_labelz_plane]|
  
MSEx: 0.006730569526553154, BCEx: 0.02641691267490387  
MSEz: 0.9832887649536133, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_3_original_depth]|![JNet_575_pretrain_3_novibrate_depth]|![JNet_575_pretrain_3_aligned_depth]|![JNet_575_pretrain_3_outputx_depth]|![JNet_575_pretrain_3_labelx_depth]|![JNet_575_pretrain_3_outputz_depth]|![JNet_575_pretrain_3_labelz_depth]|
  
MSEx: 0.006730569526553154, BCEx: 0.02641691267490387  
MSEz: 0.9832887649536133, BCEz: inf  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_4_original_plane]|![JNet_575_pretrain_4_novibrate_plane]|![JNet_575_pretrain_4_aligned_plane]|![JNet_575_pretrain_4_outputx_plane]|![JNet_575_pretrain_4_labelx_plane]|![JNet_575_pretrain_4_outputz_plane]|![JNet_575_pretrain_4_labelz_plane]|
  
MSEx: 0.010313733480870724, BCEx: 0.042312320321798325  
MSEz: 0.9710533022880554, BCEz: inf  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_4_original_depth]|![JNet_575_pretrain_4_novibrate_depth]|![JNet_575_pretrain_4_aligned_depth]|![JNet_575_pretrain_4_outputx_depth]|![JNet_575_pretrain_4_labelx_depth]|![JNet_575_pretrain_4_outputz_depth]|![JNet_575_pretrain_4_labelz_depth]|
  
MSEx: 0.010313733480870724, BCEx: 0.042312320321798325  
MSEz: 0.9710533022880554, BCEz: inf  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi000_im000._original_depth]|![JNet_575_pretrain_beads_roi000_im000._output_depth]|![JNet_575_pretrain_beads_roi000_im000._reconst_depth]|![JNet_575_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 1.893470825195313, MSE: 0.0010325725888833404, quantized loss: 0.00015030198846943676  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi001_im004._original_depth]|![JNet_575_pretrain_beads_roi001_im004._output_depth]|![JNet_575_pretrain_beads_roi001_im004._reconst_depth]|![JNet_575_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 2.2564953613281253, MSE: 0.0013550323201343417, quantized loss: 0.00018632049614097923  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi002_im005._original_depth]|![JNet_575_pretrain_beads_roi002_im005._output_depth]|![JNet_575_pretrain_beads_roi002_im005._reconst_depth]|![JNet_575_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 1.9823073730468754, MSE: 0.0011877502547577024, quantized loss: 0.000163745426107198  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi003_im006._original_depth]|![JNet_575_pretrain_beads_roi003_im006._output_depth]|![JNet_575_pretrain_beads_roi003_im006._reconst_depth]|![JNet_575_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 1.9972111816406255, MSE: 0.0011961465934291482, quantized loss: 0.00017177549307234585  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi004_im006._original_depth]|![JNet_575_pretrain_beads_roi004_im006._output_depth]|![JNet_575_pretrain_beads_roi004_im006._reconst_depth]|![JNet_575_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 2.1000959472656255, MSE: 0.0012502108002081513, quantized loss: 0.0001796526921680197  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi005_im007._original_depth]|![JNet_575_pretrain_beads_roi005_im007._output_depth]|![JNet_575_pretrain_beads_roi005_im007._reconst_depth]|![JNet_575_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 2.0152468261718757, MSE: 0.0011665612692013383, quantized loss: 0.00016766483895480633  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi006_im008._original_depth]|![JNet_575_pretrain_beads_roi006_im008._output_depth]|![JNet_575_pretrain_beads_roi006_im008._reconst_depth]|![JNet_575_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 2.0140441894531254, MSE: 0.0012469018111005425, quantized loss: 0.00017804683011490852  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi007_im009._original_depth]|![JNet_575_pretrain_beads_roi007_im009._output_depth]|![JNet_575_pretrain_beads_roi007_im009._reconst_depth]|![JNet_575_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 2.1223901367187503, MSE: 0.0012283507967367768, quantized loss: 0.00017906502762343735  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi008_im010._original_depth]|![JNet_575_pretrain_beads_roi008_im010._output_depth]|![JNet_575_pretrain_beads_roi008_im010._reconst_depth]|![JNet_575_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 2.0935307617187506, MSE: 0.00128651293925941, quantized loss: 0.00017035502241924405  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi009_im011._original_depth]|![JNet_575_pretrain_beads_roi009_im011._output_depth]|![JNet_575_pretrain_beads_roi009_im011._reconst_depth]|![JNet_575_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 1.9051356201171878, MSE: 0.0011380083160474896, quantized loss: 0.0001540996163384989  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi010_im012._original_depth]|![JNet_575_pretrain_beads_roi010_im012._output_depth]|![JNet_575_pretrain_beads_roi010_im012._reconst_depth]|![JNet_575_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 2.3740820312500004, MSE: 0.001348273828625679, quantized loss: 0.00018943498434964567  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi011_im013._original_depth]|![JNet_575_pretrain_beads_roi011_im013._output_depth]|![JNet_575_pretrain_beads_roi011_im013._reconst_depth]|![JNet_575_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 2.2807172851562507, MSE: 0.0013127742568030953, quantized loss: 0.00018174000433646142  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi012_im014._original_depth]|![JNet_575_pretrain_beads_roi012_im014._output_depth]|![JNet_575_pretrain_beads_roi012_im014._reconst_depth]|![JNet_575_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 2.0251408691406256, MSE: 0.001205294276587665, quantized loss: 0.0001589281891938299  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi013_im015._original_depth]|![JNet_575_pretrain_beads_roi013_im015._output_depth]|![JNet_575_pretrain_beads_roi013_im015._reconst_depth]|![JNet_575_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 1.8809245605468754, MSE: 0.0011130735510960221, quantized loss: 0.00015586885274387896  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi014_im016._original_depth]|![JNet_575_pretrain_beads_roi014_im016._output_depth]|![JNet_575_pretrain_beads_roi014_im016._reconst_depth]|![JNet_575_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 1.824226684570313, MSE: 0.0011701056500896811, quantized loss: 0.00016359535220544785  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi015_im017._original_depth]|![JNet_575_pretrain_beads_roi015_im017._output_depth]|![JNet_575_pretrain_beads_roi015_im017._reconst_depth]|![JNet_575_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 1.9193615722656254, MSE: 0.0011242054169997573, quantized loss: 0.00015747729048598558  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi016_im018._original_depth]|![JNet_575_pretrain_beads_roi016_im018._output_depth]|![JNet_575_pretrain_beads_roi016_im018._reconst_depth]|![JNet_575_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 2.2405305175781254, MSE: 0.0012731010792776942, quantized loss: 0.00018117285799235106  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi017_im018._original_depth]|![JNet_575_pretrain_beads_roi017_im018._output_depth]|![JNet_575_pretrain_beads_roi017_im018._reconst_depth]|![JNet_575_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 2.2627238769531255, MSE: 0.0011808165581896901, quantized loss: 0.00018461706349626184  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi018_im022._original_depth]|![JNet_575_pretrain_beads_roi018_im022._output_depth]|![JNet_575_pretrain_beads_roi018_im022._reconst_depth]|![JNet_575_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 1.6892849121093754, MSE: 0.0009387434110976756, quantized loss: 0.00013931530702393502  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi019_im023._original_depth]|![JNet_575_pretrain_beads_roi019_im023._output_depth]|![JNet_575_pretrain_beads_roi019_im023._reconst_depth]|![JNet_575_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 1.6546699218750003, MSE: 0.0009144144132733345, quantized loss: 0.00013704827870242298  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi020_im024._original_depth]|![JNet_575_pretrain_beads_roi020_im024._output_depth]|![JNet_575_pretrain_beads_roi020_im024._reconst_depth]|![JNet_575_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 2.1920646972656255, MSE: 0.001187124173156917, quantized loss: 0.00017032335745170712  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi021_im026._original_depth]|![JNet_575_pretrain_beads_roi021_im026._output_depth]|![JNet_575_pretrain_beads_roi021_im026._reconst_depth]|![JNet_575_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 2.0474641113281256, MSE: 0.001165442867204547, quantized loss: 0.00016265474550891668  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi022_im027._original_depth]|![JNet_575_pretrain_beads_roi022_im027._output_depth]|![JNet_575_pretrain_beads_roi022_im027._reconst_depth]|![JNet_575_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 2.038359252929688, MSE: 0.001157689024694264, quantized loss: 0.00015902461018413305  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi023_im028._original_depth]|![JNet_575_pretrain_beads_roi023_im028._output_depth]|![JNet_575_pretrain_beads_roi023_im028._reconst_depth]|![JNet_575_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 2.0580024414062503, MSE: 0.001363881747238338, quantized loss: 0.00017674325499683619  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi024_im028._original_depth]|![JNet_575_pretrain_beads_roi024_im028._output_depth]|![JNet_575_pretrain_beads_roi024_im028._reconst_depth]|![JNet_575_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 2.0649858398437506, MSE: 0.0012371423654258251, quantized loss: 0.00017053112969733775  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi025_im028._original_depth]|![JNet_575_pretrain_beads_roi025_im028._output_depth]|![JNet_575_pretrain_beads_roi025_im028._reconst_depth]|![JNet_575_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 2.0649858398437506, MSE: 0.0012371423654258251, quantized loss: 0.00017053112969733775  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi026_im029._original_depth]|![JNet_575_pretrain_beads_roi026_im029._output_depth]|![JNet_575_pretrain_beads_roi026_im029._reconst_depth]|![JNet_575_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 2.2142766113281254, MSE: 0.001276315306313336, quantized loss: 0.0001751963864080608  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi027_im029._original_depth]|![JNet_575_pretrain_beads_roi027_im029._output_depth]|![JNet_575_pretrain_beads_roi027_im029._reconst_depth]|![JNet_575_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 1.940147338867188, MSE: 0.001114981365390122, quantized loss: 0.00015746851568110287  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi028_im030._original_depth]|![JNet_575_pretrain_beads_roi028_im030._output_depth]|![JNet_575_pretrain_beads_roi028_im030._reconst_depth]|![JNet_575_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 1.8508925781250005, MSE: 0.0010288060875609517, quantized loss: 0.00014945650764275342  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_575_pretrain_beads_roi029_im030._original_depth]|![JNet_575_pretrain_beads_roi029_im030._output_depth]|![JNet_575_pretrain_beads_roi029_im030._reconst_depth]|![JNet_575_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 1.9918737792968755, MSE: 0.001080249436199665, quantized loss: 0.00015828087634872645  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi000_im000._original_depth]|![JNet_576_beads_roi000_im000._output_depth]|![JNet_576_beads_roi000_im000._reconst_depth]|![JNet_576_beads_roi000_im000._heatmap_depth]|
  
volume: 3.218518798828126, MSE: 0.0015793745405972004, quantized loss: 0.0005923343705944717  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi001_im004._original_depth]|![JNet_576_beads_roi001_im004._output_depth]|![JNet_576_beads_roi001_im004._reconst_depth]|![JNet_576_beads_roi001_im004._heatmap_depth]|
  
volume: 4.310269531250001, MSE: 0.002442845841869712, quantized loss: 0.0009125252836383879  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi002_im005._original_depth]|![JNet_576_beads_roi002_im005._output_depth]|![JNet_576_beads_roi002_im005._reconst_depth]|![JNet_576_beads_roi002_im005._heatmap_depth]|
  
volume: 3.8423891601562508, MSE: 0.0019555401522666216, quantized loss: 0.0008542014984413981  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi003_im006._original_depth]|![JNet_576_beads_roi003_im006._output_depth]|![JNet_576_beads_roi003_im006._reconst_depth]|![JNet_576_beads_roi003_im006._heatmap_depth]|
  
volume: 4.123814453125001, MSE: 0.0026666170451790094, quantized loss: 0.0008135524112731218  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi004_im006._original_depth]|![JNet_576_beads_roi004_im006._output_depth]|![JNet_576_beads_roi004_im006._reconst_depth]|![JNet_576_beads_roi004_im006._heatmap_depth]|
  
volume: 4.314216796875001, MSE: 0.0027251699939370155, quantized loss: 0.0008484131540171802  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi005_im007._original_depth]|![JNet_576_beads_roi005_im007._output_depth]|![JNet_576_beads_roi005_im007._reconst_depth]|![JNet_576_beads_roi005_im007._heatmap_depth]|
  
volume: 4.204318847656251, MSE: 0.002836446277797222, quantized loss: 0.0009362654527649283  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi006_im008._original_depth]|![JNet_576_beads_roi006_im008._output_depth]|![JNet_576_beads_roi006_im008._reconst_depth]|![JNet_576_beads_roi006_im008._heatmap_depth]|
  
volume: 4.291162597656251, MSE: 0.002592714037746191, quantized loss: 0.0008947006426751614  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi007_im009._original_depth]|![JNet_576_beads_roi007_im009._output_depth]|![JNet_576_beads_roi007_im009._reconst_depth]|![JNet_576_beads_roi007_im009._heatmap_depth]|
  
volume: 4.815133300781251, MSE: 0.0028010064270347357, quantized loss: 0.001007212558761239  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi008_im010._original_depth]|![JNet_576_beads_roi008_im010._output_depth]|![JNet_576_beads_roi008_im010._reconst_depth]|![JNet_576_beads_roi008_im010._heatmap_depth]|
  
volume: 4.169574218750001, MSE: 0.0025188883300870657, quantized loss: 0.0009537255391478539  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi009_im011._original_depth]|![JNet_576_beads_roi009_im011._output_depth]|![JNet_576_beads_roi009_im011._reconst_depth]|![JNet_576_beads_roi009_im011._heatmap_depth]|
  
volume: 3.415128173828126, MSE: 0.0017135204980149865, quantized loss: 0.0006654424942098558  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi010_im012._original_depth]|![JNet_576_beads_roi010_im012._output_depth]|![JNet_576_beads_roi010_im012._reconst_depth]|![JNet_576_beads_roi010_im012._heatmap_depth]|
  
volume: 4.122727539062501, MSE: 0.0023438965436071157, quantized loss: 0.0008208954241126776  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi011_im013._original_depth]|![JNet_576_beads_roi011_im013._output_depth]|![JNet_576_beads_roi011_im013._reconst_depth]|![JNet_576_beads_roi011_im013._heatmap_depth]|
  
volume: 4.0184091796875006, MSE: 0.0019936354365199804, quantized loss: 0.0007977014756761491  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi012_im014._original_depth]|![JNet_576_beads_roi012_im014._output_depth]|![JNet_576_beads_roi012_im014._reconst_depth]|![JNet_576_beads_roi012_im014._heatmap_depth]|
  
volume: 3.171746093750001, MSE: 0.0015497440472245216, quantized loss: 0.0005291783018037677  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi013_im015._original_depth]|![JNet_576_beads_roi013_im015._output_depth]|![JNet_576_beads_roi013_im015._reconst_depth]|![JNet_576_beads_roi013_im015._heatmap_depth]|
  
volume: 3.300157958984376, MSE: 0.0018883398734033108, quantized loss: 0.0006509662489406765  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi014_im016._original_depth]|![JNet_576_beads_roi014_im016._output_depth]|![JNet_576_beads_roi014_im016._reconst_depth]|![JNet_576_beads_roi014_im016._heatmap_depth]|
  
volume: 3.705104003906251, MSE: 0.0026138557586818933, quantized loss: 0.0007774663390591741  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi015_im017._original_depth]|![JNet_576_beads_roi015_im017._output_depth]|![JNet_576_beads_roi015_im017._reconst_depth]|![JNet_576_beads_roi015_im017._heatmap_depth]|
  
volume: 3.830668457031251, MSE: 0.0024860103148967028, quantized loss: 0.0008080980624072254  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi016_im018._original_depth]|![JNet_576_beads_roi016_im018._output_depth]|![JNet_576_beads_roi016_im018._reconst_depth]|![JNet_576_beads_roi016_im018._heatmap_depth]|
  
volume: 4.529917480468751, MSE: 0.002709766151383519, quantized loss: 0.0008884540293365717  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi017_im018._original_depth]|![JNet_576_beads_roi017_im018._output_depth]|![JNet_576_beads_roi017_im018._reconst_depth]|![JNet_576_beads_roi017_im018._heatmap_depth]|
  
volume: 4.238645507812501, MSE: 0.0025868299417197704, quantized loss: 0.0007839816389605403  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi018_im022._original_depth]|![JNet_576_beads_roi018_im022._output_depth]|![JNet_576_beads_roi018_im022._reconst_depth]|![JNet_576_beads_roi018_im022._heatmap_depth]|
  
volume: 2.4013676757812505, MSE: 0.0012130351969972253, quantized loss: 0.000231185054872185  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi019_im023._original_depth]|![JNet_576_beads_roi019_im023._output_depth]|![JNet_576_beads_roi019_im023._reconst_depth]|![JNet_576_beads_roi019_im023._heatmap_depth]|
  
volume: 2.3898088378906257, MSE: 0.0011726919328793883, quantized loss: 0.0002269832621095702  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi020_im024._original_depth]|![JNet_576_beads_roi020_im024._output_depth]|![JNet_576_beads_roi020_im024._reconst_depth]|![JNet_576_beads_roi020_im024._heatmap_depth]|
  
volume: 3.231620117187501, MSE: 0.0015136479632928967, quantized loss: 0.00036155607085675  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi021_im026._original_depth]|![JNet_576_beads_roi021_im026._output_depth]|![JNet_576_beads_roi021_im026._reconst_depth]|![JNet_576_beads_roi021_im026._heatmap_depth]|
  
volume: 3.268083984375001, MSE: 0.0015280033694580197, quantized loss: 0.00047253037337213755  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi022_im027._original_depth]|![JNet_576_beads_roi022_im027._output_depth]|![JNet_576_beads_roi022_im027._reconst_depth]|![JNet_576_beads_roi022_im027._heatmap_depth]|
  
volume: 3.1482048339843756, MSE: 0.0015189782716333866, quantized loss: 0.0003925019991584122  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi023_im028._original_depth]|![JNet_576_beads_roi023_im028._output_depth]|![JNet_576_beads_roi023_im028._reconst_depth]|![JNet_576_beads_roi023_im028._heatmap_depth]|
  
volume: 3.549830810546876, MSE: 0.001956045860424638, quantized loss: 0.0007080118521116674  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi024_im028._original_depth]|![JNet_576_beads_roi024_im028._output_depth]|![JNet_576_beads_roi024_im028._reconst_depth]|![JNet_576_beads_roi024_im028._heatmap_depth]|
  
volume: 3.298822021484376, MSE: 0.0017218607245013118, quantized loss: 0.0005385153344832361  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi025_im028._original_depth]|![JNet_576_beads_roi025_im028._output_depth]|![JNet_576_beads_roi025_im028._reconst_depth]|![JNet_576_beads_roi025_im028._heatmap_depth]|
  
volume: 3.298822021484376, MSE: 0.0017218607245013118, quantized loss: 0.0005385153344832361  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi026_im029._original_depth]|![JNet_576_beads_roi026_im029._output_depth]|![JNet_576_beads_roi026_im029._reconst_depth]|![JNet_576_beads_roi026_im029._heatmap_depth]|
  
volume: 3.515982177734376, MSE: 0.0017016688361763954, quantized loss: 0.0005291285924613476  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi027_im029._original_depth]|![JNet_576_beads_roi027_im029._output_depth]|![JNet_576_beads_roi027_im029._reconst_depth]|![JNet_576_beads_roi027_im029._heatmap_depth]|
  
volume: 2.9573376464843757, MSE: 0.0015025128377601504, quantized loss: 0.00038053005118854344  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi028_im030._original_depth]|![JNet_576_beads_roi028_im030._output_depth]|![JNet_576_beads_roi028_im030._reconst_depth]|![JNet_576_beads_roi028_im030._heatmap_depth]|
  
volume: 2.9160878906250005, MSE: 0.0014801969518885016, quantized loss: 0.0004343341279309243  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_576_beads_roi029_im030._original_depth]|![JNet_576_beads_roi029_im030._output_depth]|![JNet_576_beads_roi029_im030._reconst_depth]|![JNet_576_beads_roi029_im030._heatmap_depth]|
  
volume: 3.156450439453126, MSE: 0.0015536485007032752, quantized loss: 0.0004917553160339594  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_576_psf_pre]|![JNet_576_psf_post]|

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
  



[JNet_575_pretrain_0_aligned_depth]: /experiments/images/JNet_575_pretrain_0_aligned_depth.png
[JNet_575_pretrain_0_aligned_plane]: /experiments/images/JNet_575_pretrain_0_aligned_plane.png
[JNet_575_pretrain_0_labelx_depth]: /experiments/images/JNet_575_pretrain_0_labelx_depth.png
[JNet_575_pretrain_0_labelx_plane]: /experiments/images/JNet_575_pretrain_0_labelx_plane.png
[JNet_575_pretrain_0_labelz_depth]: /experiments/images/JNet_575_pretrain_0_labelz_depth.png
[JNet_575_pretrain_0_labelz_plane]: /experiments/images/JNet_575_pretrain_0_labelz_plane.png
[JNet_575_pretrain_0_novibrate_depth]: /experiments/images/JNet_575_pretrain_0_novibrate_depth.png
[JNet_575_pretrain_0_novibrate_plane]: /experiments/images/JNet_575_pretrain_0_novibrate_plane.png
[JNet_575_pretrain_0_original_depth]: /experiments/images/JNet_575_pretrain_0_original_depth.png
[JNet_575_pretrain_0_original_plane]: /experiments/images/JNet_575_pretrain_0_original_plane.png
[JNet_575_pretrain_0_outputx_depth]: /experiments/images/JNet_575_pretrain_0_outputx_depth.png
[JNet_575_pretrain_0_outputx_plane]: /experiments/images/JNet_575_pretrain_0_outputx_plane.png
[JNet_575_pretrain_0_outputz_depth]: /experiments/images/JNet_575_pretrain_0_outputz_depth.png
[JNet_575_pretrain_0_outputz_plane]: /experiments/images/JNet_575_pretrain_0_outputz_plane.png
[JNet_575_pretrain_1_aligned_depth]: /experiments/images/JNet_575_pretrain_1_aligned_depth.png
[JNet_575_pretrain_1_aligned_plane]: /experiments/images/JNet_575_pretrain_1_aligned_plane.png
[JNet_575_pretrain_1_labelx_depth]: /experiments/images/JNet_575_pretrain_1_labelx_depth.png
[JNet_575_pretrain_1_labelx_plane]: /experiments/images/JNet_575_pretrain_1_labelx_plane.png
[JNet_575_pretrain_1_labelz_depth]: /experiments/images/JNet_575_pretrain_1_labelz_depth.png
[JNet_575_pretrain_1_labelz_plane]: /experiments/images/JNet_575_pretrain_1_labelz_plane.png
[JNet_575_pretrain_1_novibrate_depth]: /experiments/images/JNet_575_pretrain_1_novibrate_depth.png
[JNet_575_pretrain_1_novibrate_plane]: /experiments/images/JNet_575_pretrain_1_novibrate_plane.png
[JNet_575_pretrain_1_original_depth]: /experiments/images/JNet_575_pretrain_1_original_depth.png
[JNet_575_pretrain_1_original_plane]: /experiments/images/JNet_575_pretrain_1_original_plane.png
[JNet_575_pretrain_1_outputx_depth]: /experiments/images/JNet_575_pretrain_1_outputx_depth.png
[JNet_575_pretrain_1_outputx_plane]: /experiments/images/JNet_575_pretrain_1_outputx_plane.png
[JNet_575_pretrain_1_outputz_depth]: /experiments/images/JNet_575_pretrain_1_outputz_depth.png
[JNet_575_pretrain_1_outputz_plane]: /experiments/images/JNet_575_pretrain_1_outputz_plane.png
[JNet_575_pretrain_2_aligned_depth]: /experiments/images/JNet_575_pretrain_2_aligned_depth.png
[JNet_575_pretrain_2_aligned_plane]: /experiments/images/JNet_575_pretrain_2_aligned_plane.png
[JNet_575_pretrain_2_labelx_depth]: /experiments/images/JNet_575_pretrain_2_labelx_depth.png
[JNet_575_pretrain_2_labelx_plane]: /experiments/images/JNet_575_pretrain_2_labelx_plane.png
[JNet_575_pretrain_2_labelz_depth]: /experiments/images/JNet_575_pretrain_2_labelz_depth.png
[JNet_575_pretrain_2_labelz_plane]: /experiments/images/JNet_575_pretrain_2_labelz_plane.png
[JNet_575_pretrain_2_novibrate_depth]: /experiments/images/JNet_575_pretrain_2_novibrate_depth.png
[JNet_575_pretrain_2_novibrate_plane]: /experiments/images/JNet_575_pretrain_2_novibrate_plane.png
[JNet_575_pretrain_2_original_depth]: /experiments/images/JNet_575_pretrain_2_original_depth.png
[JNet_575_pretrain_2_original_plane]: /experiments/images/JNet_575_pretrain_2_original_plane.png
[JNet_575_pretrain_2_outputx_depth]: /experiments/images/JNet_575_pretrain_2_outputx_depth.png
[JNet_575_pretrain_2_outputx_plane]: /experiments/images/JNet_575_pretrain_2_outputx_plane.png
[JNet_575_pretrain_2_outputz_depth]: /experiments/images/JNet_575_pretrain_2_outputz_depth.png
[JNet_575_pretrain_2_outputz_plane]: /experiments/images/JNet_575_pretrain_2_outputz_plane.png
[JNet_575_pretrain_3_aligned_depth]: /experiments/images/JNet_575_pretrain_3_aligned_depth.png
[JNet_575_pretrain_3_aligned_plane]: /experiments/images/JNet_575_pretrain_3_aligned_plane.png
[JNet_575_pretrain_3_labelx_depth]: /experiments/images/JNet_575_pretrain_3_labelx_depth.png
[JNet_575_pretrain_3_labelx_plane]: /experiments/images/JNet_575_pretrain_3_labelx_plane.png
[JNet_575_pretrain_3_labelz_depth]: /experiments/images/JNet_575_pretrain_3_labelz_depth.png
[JNet_575_pretrain_3_labelz_plane]: /experiments/images/JNet_575_pretrain_3_labelz_plane.png
[JNet_575_pretrain_3_novibrate_depth]: /experiments/images/JNet_575_pretrain_3_novibrate_depth.png
[JNet_575_pretrain_3_novibrate_plane]: /experiments/images/JNet_575_pretrain_3_novibrate_plane.png
[JNet_575_pretrain_3_original_depth]: /experiments/images/JNet_575_pretrain_3_original_depth.png
[JNet_575_pretrain_3_original_plane]: /experiments/images/JNet_575_pretrain_3_original_plane.png
[JNet_575_pretrain_3_outputx_depth]: /experiments/images/JNet_575_pretrain_3_outputx_depth.png
[JNet_575_pretrain_3_outputx_plane]: /experiments/images/JNet_575_pretrain_3_outputx_plane.png
[JNet_575_pretrain_3_outputz_depth]: /experiments/images/JNet_575_pretrain_3_outputz_depth.png
[JNet_575_pretrain_3_outputz_plane]: /experiments/images/JNet_575_pretrain_3_outputz_plane.png
[JNet_575_pretrain_4_aligned_depth]: /experiments/images/JNet_575_pretrain_4_aligned_depth.png
[JNet_575_pretrain_4_aligned_plane]: /experiments/images/JNet_575_pretrain_4_aligned_plane.png
[JNet_575_pretrain_4_labelx_depth]: /experiments/images/JNet_575_pretrain_4_labelx_depth.png
[JNet_575_pretrain_4_labelx_plane]: /experiments/images/JNet_575_pretrain_4_labelx_plane.png
[JNet_575_pretrain_4_labelz_depth]: /experiments/images/JNet_575_pretrain_4_labelz_depth.png
[JNet_575_pretrain_4_labelz_plane]: /experiments/images/JNet_575_pretrain_4_labelz_plane.png
[JNet_575_pretrain_4_novibrate_depth]: /experiments/images/JNet_575_pretrain_4_novibrate_depth.png
[JNet_575_pretrain_4_novibrate_plane]: /experiments/images/JNet_575_pretrain_4_novibrate_plane.png
[JNet_575_pretrain_4_original_depth]: /experiments/images/JNet_575_pretrain_4_original_depth.png
[JNet_575_pretrain_4_original_plane]: /experiments/images/JNet_575_pretrain_4_original_plane.png
[JNet_575_pretrain_4_outputx_depth]: /experiments/images/JNet_575_pretrain_4_outputx_depth.png
[JNet_575_pretrain_4_outputx_plane]: /experiments/images/JNet_575_pretrain_4_outputx_plane.png
[JNet_575_pretrain_4_outputz_depth]: /experiments/images/JNet_575_pretrain_4_outputz_depth.png
[JNet_575_pretrain_4_outputz_plane]: /experiments/images/JNet_575_pretrain_4_outputz_plane.png
[JNet_575_pretrain_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi000_im000._heatmap_depth.png
[JNet_575_pretrain_beads_roi000_im000._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi000_im000._original_depth.png
[JNet_575_pretrain_beads_roi000_im000._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi000_im000._output_depth.png
[JNet_575_pretrain_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi000_im000._reconst_depth.png
[JNet_575_pretrain_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi001_im004._heatmap_depth.png
[JNet_575_pretrain_beads_roi001_im004._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi001_im004._original_depth.png
[JNet_575_pretrain_beads_roi001_im004._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi001_im004._output_depth.png
[JNet_575_pretrain_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi001_im004._reconst_depth.png
[JNet_575_pretrain_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi002_im005._heatmap_depth.png
[JNet_575_pretrain_beads_roi002_im005._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi002_im005._original_depth.png
[JNet_575_pretrain_beads_roi002_im005._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi002_im005._output_depth.png
[JNet_575_pretrain_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi002_im005._reconst_depth.png
[JNet_575_pretrain_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi003_im006._heatmap_depth.png
[JNet_575_pretrain_beads_roi003_im006._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi003_im006._original_depth.png
[JNet_575_pretrain_beads_roi003_im006._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi003_im006._output_depth.png
[JNet_575_pretrain_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi003_im006._reconst_depth.png
[JNet_575_pretrain_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi004_im006._heatmap_depth.png
[JNet_575_pretrain_beads_roi004_im006._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi004_im006._original_depth.png
[JNet_575_pretrain_beads_roi004_im006._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi004_im006._output_depth.png
[JNet_575_pretrain_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi004_im006._reconst_depth.png
[JNet_575_pretrain_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi005_im007._heatmap_depth.png
[JNet_575_pretrain_beads_roi005_im007._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi005_im007._original_depth.png
[JNet_575_pretrain_beads_roi005_im007._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi005_im007._output_depth.png
[JNet_575_pretrain_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi005_im007._reconst_depth.png
[JNet_575_pretrain_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi006_im008._heatmap_depth.png
[JNet_575_pretrain_beads_roi006_im008._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi006_im008._original_depth.png
[JNet_575_pretrain_beads_roi006_im008._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi006_im008._output_depth.png
[JNet_575_pretrain_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi006_im008._reconst_depth.png
[JNet_575_pretrain_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi007_im009._heatmap_depth.png
[JNet_575_pretrain_beads_roi007_im009._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi007_im009._original_depth.png
[JNet_575_pretrain_beads_roi007_im009._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi007_im009._output_depth.png
[JNet_575_pretrain_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi007_im009._reconst_depth.png
[JNet_575_pretrain_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi008_im010._heatmap_depth.png
[JNet_575_pretrain_beads_roi008_im010._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi008_im010._original_depth.png
[JNet_575_pretrain_beads_roi008_im010._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi008_im010._output_depth.png
[JNet_575_pretrain_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi008_im010._reconst_depth.png
[JNet_575_pretrain_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi009_im011._heatmap_depth.png
[JNet_575_pretrain_beads_roi009_im011._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi009_im011._original_depth.png
[JNet_575_pretrain_beads_roi009_im011._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi009_im011._output_depth.png
[JNet_575_pretrain_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi009_im011._reconst_depth.png
[JNet_575_pretrain_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi010_im012._heatmap_depth.png
[JNet_575_pretrain_beads_roi010_im012._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi010_im012._original_depth.png
[JNet_575_pretrain_beads_roi010_im012._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi010_im012._output_depth.png
[JNet_575_pretrain_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi010_im012._reconst_depth.png
[JNet_575_pretrain_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi011_im013._heatmap_depth.png
[JNet_575_pretrain_beads_roi011_im013._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi011_im013._original_depth.png
[JNet_575_pretrain_beads_roi011_im013._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi011_im013._output_depth.png
[JNet_575_pretrain_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi011_im013._reconst_depth.png
[JNet_575_pretrain_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi012_im014._heatmap_depth.png
[JNet_575_pretrain_beads_roi012_im014._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi012_im014._original_depth.png
[JNet_575_pretrain_beads_roi012_im014._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi012_im014._output_depth.png
[JNet_575_pretrain_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi012_im014._reconst_depth.png
[JNet_575_pretrain_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi013_im015._heatmap_depth.png
[JNet_575_pretrain_beads_roi013_im015._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi013_im015._original_depth.png
[JNet_575_pretrain_beads_roi013_im015._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi013_im015._output_depth.png
[JNet_575_pretrain_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi013_im015._reconst_depth.png
[JNet_575_pretrain_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi014_im016._heatmap_depth.png
[JNet_575_pretrain_beads_roi014_im016._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi014_im016._original_depth.png
[JNet_575_pretrain_beads_roi014_im016._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi014_im016._output_depth.png
[JNet_575_pretrain_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi014_im016._reconst_depth.png
[JNet_575_pretrain_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi015_im017._heatmap_depth.png
[JNet_575_pretrain_beads_roi015_im017._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi015_im017._original_depth.png
[JNet_575_pretrain_beads_roi015_im017._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi015_im017._output_depth.png
[JNet_575_pretrain_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi015_im017._reconst_depth.png
[JNet_575_pretrain_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi016_im018._heatmap_depth.png
[JNet_575_pretrain_beads_roi016_im018._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi016_im018._original_depth.png
[JNet_575_pretrain_beads_roi016_im018._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi016_im018._output_depth.png
[JNet_575_pretrain_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi016_im018._reconst_depth.png
[JNet_575_pretrain_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi017_im018._heatmap_depth.png
[JNet_575_pretrain_beads_roi017_im018._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi017_im018._original_depth.png
[JNet_575_pretrain_beads_roi017_im018._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi017_im018._output_depth.png
[JNet_575_pretrain_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi017_im018._reconst_depth.png
[JNet_575_pretrain_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi018_im022._heatmap_depth.png
[JNet_575_pretrain_beads_roi018_im022._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi018_im022._original_depth.png
[JNet_575_pretrain_beads_roi018_im022._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi018_im022._output_depth.png
[JNet_575_pretrain_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi018_im022._reconst_depth.png
[JNet_575_pretrain_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi019_im023._heatmap_depth.png
[JNet_575_pretrain_beads_roi019_im023._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi019_im023._original_depth.png
[JNet_575_pretrain_beads_roi019_im023._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi019_im023._output_depth.png
[JNet_575_pretrain_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi019_im023._reconst_depth.png
[JNet_575_pretrain_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi020_im024._heatmap_depth.png
[JNet_575_pretrain_beads_roi020_im024._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi020_im024._original_depth.png
[JNet_575_pretrain_beads_roi020_im024._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi020_im024._output_depth.png
[JNet_575_pretrain_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi020_im024._reconst_depth.png
[JNet_575_pretrain_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi021_im026._heatmap_depth.png
[JNet_575_pretrain_beads_roi021_im026._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi021_im026._original_depth.png
[JNet_575_pretrain_beads_roi021_im026._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi021_im026._output_depth.png
[JNet_575_pretrain_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi021_im026._reconst_depth.png
[JNet_575_pretrain_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi022_im027._heatmap_depth.png
[JNet_575_pretrain_beads_roi022_im027._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi022_im027._original_depth.png
[JNet_575_pretrain_beads_roi022_im027._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi022_im027._output_depth.png
[JNet_575_pretrain_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi022_im027._reconst_depth.png
[JNet_575_pretrain_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi023_im028._heatmap_depth.png
[JNet_575_pretrain_beads_roi023_im028._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi023_im028._original_depth.png
[JNet_575_pretrain_beads_roi023_im028._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi023_im028._output_depth.png
[JNet_575_pretrain_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi023_im028._reconst_depth.png
[JNet_575_pretrain_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi024_im028._heatmap_depth.png
[JNet_575_pretrain_beads_roi024_im028._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi024_im028._original_depth.png
[JNet_575_pretrain_beads_roi024_im028._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi024_im028._output_depth.png
[JNet_575_pretrain_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi024_im028._reconst_depth.png
[JNet_575_pretrain_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi025_im028._heatmap_depth.png
[JNet_575_pretrain_beads_roi025_im028._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi025_im028._original_depth.png
[JNet_575_pretrain_beads_roi025_im028._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi025_im028._output_depth.png
[JNet_575_pretrain_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi025_im028._reconst_depth.png
[JNet_575_pretrain_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi026_im029._heatmap_depth.png
[JNet_575_pretrain_beads_roi026_im029._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi026_im029._original_depth.png
[JNet_575_pretrain_beads_roi026_im029._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi026_im029._output_depth.png
[JNet_575_pretrain_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi026_im029._reconst_depth.png
[JNet_575_pretrain_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi027_im029._heatmap_depth.png
[JNet_575_pretrain_beads_roi027_im029._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi027_im029._original_depth.png
[JNet_575_pretrain_beads_roi027_im029._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi027_im029._output_depth.png
[JNet_575_pretrain_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi027_im029._reconst_depth.png
[JNet_575_pretrain_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi028_im030._heatmap_depth.png
[JNet_575_pretrain_beads_roi028_im030._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi028_im030._original_depth.png
[JNet_575_pretrain_beads_roi028_im030._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi028_im030._output_depth.png
[JNet_575_pretrain_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi028_im030._reconst_depth.png
[JNet_575_pretrain_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_575_pretrain_beads_roi029_im030._heatmap_depth.png
[JNet_575_pretrain_beads_roi029_im030._original_depth]: /experiments/images/JNet_575_pretrain_beads_roi029_im030._original_depth.png
[JNet_575_pretrain_beads_roi029_im030._output_depth]: /experiments/images/JNet_575_pretrain_beads_roi029_im030._output_depth.png
[JNet_575_pretrain_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_575_pretrain_beads_roi029_im030._reconst_depth.png
[JNet_576_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_576_beads_roi000_im000._heatmap_depth.png
[JNet_576_beads_roi000_im000._original_depth]: /experiments/images/JNet_576_beads_roi000_im000._original_depth.png
[JNet_576_beads_roi000_im000._output_depth]: /experiments/images/JNet_576_beads_roi000_im000._output_depth.png
[JNet_576_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_576_beads_roi000_im000._reconst_depth.png
[JNet_576_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_576_beads_roi001_im004._heatmap_depth.png
[JNet_576_beads_roi001_im004._original_depth]: /experiments/images/JNet_576_beads_roi001_im004._original_depth.png
[JNet_576_beads_roi001_im004._output_depth]: /experiments/images/JNet_576_beads_roi001_im004._output_depth.png
[JNet_576_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_576_beads_roi001_im004._reconst_depth.png
[JNet_576_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_576_beads_roi002_im005._heatmap_depth.png
[JNet_576_beads_roi002_im005._original_depth]: /experiments/images/JNet_576_beads_roi002_im005._original_depth.png
[JNet_576_beads_roi002_im005._output_depth]: /experiments/images/JNet_576_beads_roi002_im005._output_depth.png
[JNet_576_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_576_beads_roi002_im005._reconst_depth.png
[JNet_576_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_576_beads_roi003_im006._heatmap_depth.png
[JNet_576_beads_roi003_im006._original_depth]: /experiments/images/JNet_576_beads_roi003_im006._original_depth.png
[JNet_576_beads_roi003_im006._output_depth]: /experiments/images/JNet_576_beads_roi003_im006._output_depth.png
[JNet_576_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_576_beads_roi003_im006._reconst_depth.png
[JNet_576_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_576_beads_roi004_im006._heatmap_depth.png
[JNet_576_beads_roi004_im006._original_depth]: /experiments/images/JNet_576_beads_roi004_im006._original_depth.png
[JNet_576_beads_roi004_im006._output_depth]: /experiments/images/JNet_576_beads_roi004_im006._output_depth.png
[JNet_576_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_576_beads_roi004_im006._reconst_depth.png
[JNet_576_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_576_beads_roi005_im007._heatmap_depth.png
[JNet_576_beads_roi005_im007._original_depth]: /experiments/images/JNet_576_beads_roi005_im007._original_depth.png
[JNet_576_beads_roi005_im007._output_depth]: /experiments/images/JNet_576_beads_roi005_im007._output_depth.png
[JNet_576_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_576_beads_roi005_im007._reconst_depth.png
[JNet_576_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_576_beads_roi006_im008._heatmap_depth.png
[JNet_576_beads_roi006_im008._original_depth]: /experiments/images/JNet_576_beads_roi006_im008._original_depth.png
[JNet_576_beads_roi006_im008._output_depth]: /experiments/images/JNet_576_beads_roi006_im008._output_depth.png
[JNet_576_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_576_beads_roi006_im008._reconst_depth.png
[JNet_576_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_576_beads_roi007_im009._heatmap_depth.png
[JNet_576_beads_roi007_im009._original_depth]: /experiments/images/JNet_576_beads_roi007_im009._original_depth.png
[JNet_576_beads_roi007_im009._output_depth]: /experiments/images/JNet_576_beads_roi007_im009._output_depth.png
[JNet_576_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_576_beads_roi007_im009._reconst_depth.png
[JNet_576_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_576_beads_roi008_im010._heatmap_depth.png
[JNet_576_beads_roi008_im010._original_depth]: /experiments/images/JNet_576_beads_roi008_im010._original_depth.png
[JNet_576_beads_roi008_im010._output_depth]: /experiments/images/JNet_576_beads_roi008_im010._output_depth.png
[JNet_576_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_576_beads_roi008_im010._reconst_depth.png
[JNet_576_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_576_beads_roi009_im011._heatmap_depth.png
[JNet_576_beads_roi009_im011._original_depth]: /experiments/images/JNet_576_beads_roi009_im011._original_depth.png
[JNet_576_beads_roi009_im011._output_depth]: /experiments/images/JNet_576_beads_roi009_im011._output_depth.png
[JNet_576_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_576_beads_roi009_im011._reconst_depth.png
[JNet_576_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_576_beads_roi010_im012._heatmap_depth.png
[JNet_576_beads_roi010_im012._original_depth]: /experiments/images/JNet_576_beads_roi010_im012._original_depth.png
[JNet_576_beads_roi010_im012._output_depth]: /experiments/images/JNet_576_beads_roi010_im012._output_depth.png
[JNet_576_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_576_beads_roi010_im012._reconst_depth.png
[JNet_576_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_576_beads_roi011_im013._heatmap_depth.png
[JNet_576_beads_roi011_im013._original_depth]: /experiments/images/JNet_576_beads_roi011_im013._original_depth.png
[JNet_576_beads_roi011_im013._output_depth]: /experiments/images/JNet_576_beads_roi011_im013._output_depth.png
[JNet_576_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_576_beads_roi011_im013._reconst_depth.png
[JNet_576_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_576_beads_roi012_im014._heatmap_depth.png
[JNet_576_beads_roi012_im014._original_depth]: /experiments/images/JNet_576_beads_roi012_im014._original_depth.png
[JNet_576_beads_roi012_im014._output_depth]: /experiments/images/JNet_576_beads_roi012_im014._output_depth.png
[JNet_576_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_576_beads_roi012_im014._reconst_depth.png
[JNet_576_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_576_beads_roi013_im015._heatmap_depth.png
[JNet_576_beads_roi013_im015._original_depth]: /experiments/images/JNet_576_beads_roi013_im015._original_depth.png
[JNet_576_beads_roi013_im015._output_depth]: /experiments/images/JNet_576_beads_roi013_im015._output_depth.png
[JNet_576_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_576_beads_roi013_im015._reconst_depth.png
[JNet_576_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_576_beads_roi014_im016._heatmap_depth.png
[JNet_576_beads_roi014_im016._original_depth]: /experiments/images/JNet_576_beads_roi014_im016._original_depth.png
[JNet_576_beads_roi014_im016._output_depth]: /experiments/images/JNet_576_beads_roi014_im016._output_depth.png
[JNet_576_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_576_beads_roi014_im016._reconst_depth.png
[JNet_576_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_576_beads_roi015_im017._heatmap_depth.png
[JNet_576_beads_roi015_im017._original_depth]: /experiments/images/JNet_576_beads_roi015_im017._original_depth.png
[JNet_576_beads_roi015_im017._output_depth]: /experiments/images/JNet_576_beads_roi015_im017._output_depth.png
[JNet_576_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_576_beads_roi015_im017._reconst_depth.png
[JNet_576_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_576_beads_roi016_im018._heatmap_depth.png
[JNet_576_beads_roi016_im018._original_depth]: /experiments/images/JNet_576_beads_roi016_im018._original_depth.png
[JNet_576_beads_roi016_im018._output_depth]: /experiments/images/JNet_576_beads_roi016_im018._output_depth.png
[JNet_576_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_576_beads_roi016_im018._reconst_depth.png
[JNet_576_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_576_beads_roi017_im018._heatmap_depth.png
[JNet_576_beads_roi017_im018._original_depth]: /experiments/images/JNet_576_beads_roi017_im018._original_depth.png
[JNet_576_beads_roi017_im018._output_depth]: /experiments/images/JNet_576_beads_roi017_im018._output_depth.png
[JNet_576_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_576_beads_roi017_im018._reconst_depth.png
[JNet_576_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_576_beads_roi018_im022._heatmap_depth.png
[JNet_576_beads_roi018_im022._original_depth]: /experiments/images/JNet_576_beads_roi018_im022._original_depth.png
[JNet_576_beads_roi018_im022._output_depth]: /experiments/images/JNet_576_beads_roi018_im022._output_depth.png
[JNet_576_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_576_beads_roi018_im022._reconst_depth.png
[JNet_576_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_576_beads_roi019_im023._heatmap_depth.png
[JNet_576_beads_roi019_im023._original_depth]: /experiments/images/JNet_576_beads_roi019_im023._original_depth.png
[JNet_576_beads_roi019_im023._output_depth]: /experiments/images/JNet_576_beads_roi019_im023._output_depth.png
[JNet_576_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_576_beads_roi019_im023._reconst_depth.png
[JNet_576_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_576_beads_roi020_im024._heatmap_depth.png
[JNet_576_beads_roi020_im024._original_depth]: /experiments/images/JNet_576_beads_roi020_im024._original_depth.png
[JNet_576_beads_roi020_im024._output_depth]: /experiments/images/JNet_576_beads_roi020_im024._output_depth.png
[JNet_576_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_576_beads_roi020_im024._reconst_depth.png
[JNet_576_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_576_beads_roi021_im026._heatmap_depth.png
[JNet_576_beads_roi021_im026._original_depth]: /experiments/images/JNet_576_beads_roi021_im026._original_depth.png
[JNet_576_beads_roi021_im026._output_depth]: /experiments/images/JNet_576_beads_roi021_im026._output_depth.png
[JNet_576_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_576_beads_roi021_im026._reconst_depth.png
[JNet_576_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_576_beads_roi022_im027._heatmap_depth.png
[JNet_576_beads_roi022_im027._original_depth]: /experiments/images/JNet_576_beads_roi022_im027._original_depth.png
[JNet_576_beads_roi022_im027._output_depth]: /experiments/images/JNet_576_beads_roi022_im027._output_depth.png
[JNet_576_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_576_beads_roi022_im027._reconst_depth.png
[JNet_576_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_576_beads_roi023_im028._heatmap_depth.png
[JNet_576_beads_roi023_im028._original_depth]: /experiments/images/JNet_576_beads_roi023_im028._original_depth.png
[JNet_576_beads_roi023_im028._output_depth]: /experiments/images/JNet_576_beads_roi023_im028._output_depth.png
[JNet_576_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_576_beads_roi023_im028._reconst_depth.png
[JNet_576_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_576_beads_roi024_im028._heatmap_depth.png
[JNet_576_beads_roi024_im028._original_depth]: /experiments/images/JNet_576_beads_roi024_im028._original_depth.png
[JNet_576_beads_roi024_im028._output_depth]: /experiments/images/JNet_576_beads_roi024_im028._output_depth.png
[JNet_576_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_576_beads_roi024_im028._reconst_depth.png
[JNet_576_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_576_beads_roi025_im028._heatmap_depth.png
[JNet_576_beads_roi025_im028._original_depth]: /experiments/images/JNet_576_beads_roi025_im028._original_depth.png
[JNet_576_beads_roi025_im028._output_depth]: /experiments/images/JNet_576_beads_roi025_im028._output_depth.png
[JNet_576_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_576_beads_roi025_im028._reconst_depth.png
[JNet_576_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_576_beads_roi026_im029._heatmap_depth.png
[JNet_576_beads_roi026_im029._original_depth]: /experiments/images/JNet_576_beads_roi026_im029._original_depth.png
[JNet_576_beads_roi026_im029._output_depth]: /experiments/images/JNet_576_beads_roi026_im029._output_depth.png
[JNet_576_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_576_beads_roi026_im029._reconst_depth.png
[JNet_576_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_576_beads_roi027_im029._heatmap_depth.png
[JNet_576_beads_roi027_im029._original_depth]: /experiments/images/JNet_576_beads_roi027_im029._original_depth.png
[JNet_576_beads_roi027_im029._output_depth]: /experiments/images/JNet_576_beads_roi027_im029._output_depth.png
[JNet_576_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_576_beads_roi027_im029._reconst_depth.png
[JNet_576_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_576_beads_roi028_im030._heatmap_depth.png
[JNet_576_beads_roi028_im030._original_depth]: /experiments/images/JNet_576_beads_roi028_im030._original_depth.png
[JNet_576_beads_roi028_im030._output_depth]: /experiments/images/JNet_576_beads_roi028_im030._output_depth.png
[JNet_576_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_576_beads_roi028_im030._reconst_depth.png
[JNet_576_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_576_beads_roi029_im030._heatmap_depth.png
[JNet_576_beads_roi029_im030._original_depth]: /experiments/images/JNet_576_beads_roi029_im030._original_depth.png
[JNet_576_beads_roi029_im030._output_depth]: /experiments/images/JNet_576_beads_roi029_im030._output_depth.png
[JNet_576_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_576_beads_roi029_im030._reconst_depth.png
[JNet_576_psf_post]: /experiments/images/JNet_576_psf_post.png
[JNet_576_psf_pre]: /experiments/images/JNet_576_psf_pre.png
[finetuned]: /experiments/tmp/JNet_576_train.png
[pretrained_model]: /experiments/tmp/JNet_575_pretrain_train.png
