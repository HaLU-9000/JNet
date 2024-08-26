



# JNet_603 Report
  
592_pre, ewc1.0, p 1.0 q 1.0 adjust=false, 588 revenge  
pretrained model : JNet_592_pretrain
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
|adjust_luminance|False|
|zloss_weight|1|
|ewc_weight|1.0|
|qloss_weight|1.0|
|ploss_weight|1.0|
|mrfloss_order|1|
|mrfloss_dilation|1|
|mrfloss_weights|{'l_00': 0, 'l_01': 0, 'l_10': 0, 'l_11': 0}|

## Results

### Pretraining
  
Segmentation: mean MSE: 0.008186662569642067, mean BCE: 0.03194045275449753  
Luminance Estimation: mean MSE: 0.9816209673881531, mean BCE: nan
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_0_original_plane]|![JNet_592_pretrain_0_novibrate_plane]|![JNet_592_pretrain_0_aligned_plane]|![JNet_592_pretrain_0_outputx_plane]|![JNet_592_pretrain_0_labelx_plane]|![JNet_592_pretrain_0_outputz_plane]|![JNet_592_pretrain_0_labelz_plane]|
  
MSEx: 0.006835775449872017, BCEx: 0.026583008468151093  
MSEz: 0.9859671592712402, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_0_original_depth]|![JNet_592_pretrain_0_novibrate_depth]|![JNet_592_pretrain_0_aligned_depth]|![JNet_592_pretrain_0_outputx_depth]|![JNet_592_pretrain_0_labelx_depth]|![JNet_592_pretrain_0_outputz_depth]|![JNet_592_pretrain_0_labelz_depth]|
  
MSEx: 0.006835775449872017, BCEx: 0.026583008468151093  
MSEz: 0.9859671592712402, BCEz: nan  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_1_original_plane]|![JNet_592_pretrain_1_novibrate_plane]|![JNet_592_pretrain_1_aligned_plane]|![JNet_592_pretrain_1_outputx_plane]|![JNet_592_pretrain_1_labelx_plane]|![JNet_592_pretrain_1_outputz_plane]|![JNet_592_pretrain_1_labelz_plane]|
  
MSEx: 0.008289165794849396, BCEx: 0.033317554742097855  
MSEz: 0.9908297657966614, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_1_original_depth]|![JNet_592_pretrain_1_novibrate_depth]|![JNet_592_pretrain_1_aligned_depth]|![JNet_592_pretrain_1_outputx_depth]|![JNet_592_pretrain_1_labelx_depth]|![JNet_592_pretrain_1_outputz_depth]|![JNet_592_pretrain_1_labelz_depth]|
  
MSEx: 0.008289165794849396, BCEx: 0.033317554742097855  
MSEz: 0.9908297657966614, BCEz: nan  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_2_original_plane]|![JNet_592_pretrain_2_novibrate_plane]|![JNet_592_pretrain_2_aligned_plane]|![JNet_592_pretrain_2_outputx_plane]|![JNet_592_pretrain_2_labelx_plane]|![JNet_592_pretrain_2_outputz_plane]|![JNet_592_pretrain_2_labelz_plane]|
  
MSEx: 0.007379851769655943, BCEx: 0.028408177196979523  
MSEz: 0.9903133511543274, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_2_original_depth]|![JNet_592_pretrain_2_novibrate_depth]|![JNet_592_pretrain_2_aligned_depth]|![JNet_592_pretrain_2_outputx_depth]|![JNet_592_pretrain_2_labelx_depth]|![JNet_592_pretrain_2_outputz_depth]|![JNet_592_pretrain_2_labelz_depth]|
  
MSEx: 0.007379851769655943, BCEx: 0.028408177196979523  
MSEz: 0.9903133511543274, BCEz: nan  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_3_original_plane]|![JNet_592_pretrain_3_novibrate_plane]|![JNet_592_pretrain_3_aligned_plane]|![JNet_592_pretrain_3_outputx_plane]|![JNet_592_pretrain_3_labelx_plane]|![JNet_592_pretrain_3_outputz_plane]|![JNet_592_pretrain_3_labelz_plane]|
  
MSEx: 0.007738557178527117, BCEx: 0.03036637045443058  
MSEz: 0.9878877997398376, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_3_original_depth]|![JNet_592_pretrain_3_novibrate_depth]|![JNet_592_pretrain_3_aligned_depth]|![JNet_592_pretrain_3_outputx_depth]|![JNet_592_pretrain_3_labelx_depth]|![JNet_592_pretrain_3_outputz_depth]|![JNet_592_pretrain_3_labelz_depth]|
  
MSEx: 0.007738557178527117, BCEx: 0.03036637045443058  
MSEz: 0.9878877997398376, BCEz: nan  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_4_original_plane]|![JNet_592_pretrain_4_novibrate_plane]|![JNet_592_pretrain_4_aligned_plane]|![JNet_592_pretrain_4_outputx_plane]|![JNet_592_pretrain_4_labelx_plane]|![JNet_592_pretrain_4_outputz_plane]|![JNet_592_pretrain_4_labelz_plane]|
  
MSEx: 0.010689963586628437, BCEx: 0.04102715104818344  
MSEz: 0.9531069993972778, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_4_original_depth]|![JNet_592_pretrain_4_novibrate_depth]|![JNet_592_pretrain_4_aligned_depth]|![JNet_592_pretrain_4_outputx_depth]|![JNet_592_pretrain_4_labelx_depth]|![JNet_592_pretrain_4_outputz_depth]|![JNet_592_pretrain_4_labelz_depth]|
  
MSEx: 0.010689963586628437, BCEx: 0.04102715104818344  
MSEz: 0.9531069993972778, BCEz: nan  

### pretrain
  
volume mean: 3.9415750000000007, volume sd: 0.2557161381949394
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi000_im000._original_depth]|![JNet_592_pretrain_beads_roi000_im000._output_depth]|![JNet_592_pretrain_beads_roi000_im000._reconst_depth]|![JNet_592_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 3.6252500000000007, MSE: 0.0010889930417761207, quantized loss: 0.0003128394018858671  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi001_im004._original_depth]|![JNet_592_pretrain_beads_roi001_im004._output_depth]|![JNet_592_pretrain_beads_roi001_im004._reconst_depth]|![JNet_592_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 4.308000000000001, MSE: 0.0011491893092170358, quantized loss: 0.0003615067107602954  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi002_im005._original_depth]|![JNet_592_pretrain_beads_roi002_im005._output_depth]|![JNet_592_pretrain_beads_roi002_im005._reconst_depth]|![JNet_592_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 3.7586250000000008, MSE: 0.0010788943618535995, quantized loss: 0.0003170713025610894  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi003_im006._original_depth]|![JNet_592_pretrain_beads_roi003_im006._output_depth]|![JNet_592_pretrain_beads_roi003_im006._reconst_depth]|![JNet_592_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 3.865250000000001, MSE: 0.0011012180475518107, quantized loss: 0.0003514688287395984  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi004_im006._original_depth]|![JNet_592_pretrain_beads_roi004_im006._output_depth]|![JNet_592_pretrain_beads_roi004_im006._reconst_depth]|![JNet_592_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 3.9145000000000008, MSE: 0.0011299648322165012, quantized loss: 0.0003540611651260406  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi005_im007._original_depth]|![JNet_592_pretrain_beads_roi005_im007._output_depth]|![JNet_592_pretrain_beads_roi005_im007._reconst_depth]|![JNet_592_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 3.820250000000001, MSE: 0.0011069749016314745, quantized loss: 0.00034225385752506554  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi006_im008._original_depth]|![JNet_592_pretrain_beads_roi006_im008._output_depth]|![JNet_592_pretrain_beads_roi006_im008._reconst_depth]|![JNet_592_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 3.983250000000001, MSE: 0.001046577817760408, quantized loss: 0.00038989598397165537  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi007_im009._original_depth]|![JNet_592_pretrain_beads_roi007_im009._output_depth]|![JNet_592_pretrain_beads_roi007_im009._reconst_depth]|![JNet_592_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 3.7382500000000007, MSE: 0.0011017926735803485, quantized loss: 0.0003299988165963441  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi008_im010._original_depth]|![JNet_592_pretrain_beads_roi008_im010._output_depth]|![JNet_592_pretrain_beads_roi008_im010._reconst_depth]|![JNet_592_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 3.979125000000001, MSE: 0.0010695351520553231, quantized loss: 0.00035743237822316587  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi009_im011._original_depth]|![JNet_592_pretrain_beads_roi009_im011._output_depth]|![JNet_592_pretrain_beads_roi009_im011._reconst_depth]|![JNet_592_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 3.682125000000001, MSE: 0.001044733915477991, quantized loss: 0.0003282020043116063  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi010_im012._original_depth]|![JNet_592_pretrain_beads_roi010_im012._output_depth]|![JNet_592_pretrain_beads_roi010_im012._reconst_depth]|![JNet_592_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 4.461125000000001, MSE: 0.0011387414997443557, quantized loss: 0.00037982664071023464  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi011_im013._original_depth]|![JNet_592_pretrain_beads_roi011_im013._output_depth]|![JNet_592_pretrain_beads_roi011_im013._reconst_depth]|![JNet_592_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 4.408875000000001, MSE: 0.0010998956859111786, quantized loss: 0.00036728245322592556  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi012_im014._original_depth]|![JNet_592_pretrain_beads_roi012_im014._output_depth]|![JNet_592_pretrain_beads_roi012_im014._reconst_depth]|![JNet_592_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 3.884750000000001, MSE: 0.0012117716250941157, quantized loss: 0.00034339146804995835  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi013_im015._original_depth]|![JNet_592_pretrain_beads_roi013_im015._output_depth]|![JNet_592_pretrain_beads_roi013_im015._reconst_depth]|![JNet_592_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 3.761375000000001, MSE: 0.0011589424684643745, quantized loss: 0.0003348365717101842  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi014_im016._original_depth]|![JNet_592_pretrain_beads_roi014_im016._output_depth]|![JNet_592_pretrain_beads_roi014_im016._reconst_depth]|![JNet_592_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 3.761000000000001, MSE: 0.001048172591254115, quantized loss: 0.0003831070789601654  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi015_im017._original_depth]|![JNet_592_pretrain_beads_roi015_im017._output_depth]|![JNet_592_pretrain_beads_roi015_im017._reconst_depth]|![JNet_592_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 3.641125000000001, MSE: 0.0010753074893727899, quantized loss: 0.0003450574295129627  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi016_im018._original_depth]|![JNet_592_pretrain_beads_roi016_im018._output_depth]|![JNet_592_pretrain_beads_roi016_im018._reconst_depth]|![JNet_592_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 4.010250000000001, MSE: 0.00122215470764786, quantized loss: 0.00034620356746017933  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi017_im018._original_depth]|![JNet_592_pretrain_beads_roi017_im018._output_depth]|![JNet_592_pretrain_beads_roi017_im018._reconst_depth]|![JNet_592_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 3.928375000000001, MSE: 0.0012472454691305757, quantized loss: 0.0003350062179379165  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi018_im022._original_depth]|![JNet_592_pretrain_beads_roi018_im022._output_depth]|![JNet_592_pretrain_beads_roi018_im022._reconst_depth]|![JNet_592_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 3.5827500000000008, MSE: 0.0010792647954076529, quantized loss: 0.0003358496178407222  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi019_im023._original_depth]|![JNet_592_pretrain_beads_roi019_im023._output_depth]|![JNet_592_pretrain_beads_roi019_im023._reconst_depth]|![JNet_592_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 3.478625000000001, MSE: 0.00109770055860281, quantized loss: 0.00032613513758406043  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi020_im024._original_depth]|![JNet_592_pretrain_beads_roi020_im024._output_depth]|![JNet_592_pretrain_beads_roi020_im024._reconst_depth]|![JNet_592_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 4.172125000000001, MSE: 0.0011242501204833388, quantized loss: 0.0003496900317259133  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi021_im026._original_depth]|![JNet_592_pretrain_beads_roi021_im026._output_depth]|![JNet_592_pretrain_beads_roi021_im026._reconst_depth]|![JNet_592_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 3.975750000000001, MSE: 0.0010281483409926295, quantized loss: 0.0003438495332375169  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi022_im027._original_depth]|![JNet_592_pretrain_beads_roi022_im027._output_depth]|![JNet_592_pretrain_beads_roi022_im027._reconst_depth]|![JNet_592_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 3.797250000000001, MSE: 0.0010948582785204053, quantized loss: 0.00033353554317727685  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi023_im028._original_depth]|![JNet_592_pretrain_beads_roi023_im028._output_depth]|![JNet_592_pretrain_beads_roi023_im028._reconst_depth]|![JNet_592_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 4.354000000000001, MSE: 0.0009226580150425434, quantized loss: 0.0004076574696227908  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi024_im028._original_depth]|![JNet_592_pretrain_beads_roi024_im028._output_depth]|![JNet_592_pretrain_beads_roi024_im028._reconst_depth]|![JNet_592_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 4.272375000000001, MSE: 0.0009931474924087524, quantized loss: 0.0003746850125025958  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi025_im028._original_depth]|![JNet_592_pretrain_beads_roi025_im028._output_depth]|![JNet_592_pretrain_beads_roi025_im028._reconst_depth]|![JNet_592_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 4.272375000000001, MSE: 0.0009931474924087524, quantized loss: 0.0003746850125025958  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi026_im029._original_depth]|![JNet_592_pretrain_beads_roi026_im029._output_depth]|![JNet_592_pretrain_beads_roi026_im029._reconst_depth]|![JNet_592_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 4.203000000000001, MSE: 0.001128699746914208, quantized loss: 0.0003621990035753697  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi027_im029._original_depth]|![JNet_592_pretrain_beads_roi027_im029._output_depth]|![JNet_592_pretrain_beads_roi027_im029._reconst_depth]|![JNet_592_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 3.934000000000001, MSE: 0.0011243581539019942, quantized loss: 0.0003573574358597398  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi028_im030._original_depth]|![JNet_592_pretrain_beads_roi028_im030._output_depth]|![JNet_592_pretrain_beads_roi028_im030._reconst_depth]|![JNet_592_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 3.760875000000001, MSE: 0.0010836278088390827, quantized loss: 0.00033172505209222436  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi029_im030._original_depth]|![JNet_592_pretrain_beads_roi029_im030._output_depth]|![JNet_592_pretrain_beads_roi029_im030._reconst_depth]|![JNet_592_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 3.912625000000001, MSE: 0.001145395333878696, quantized loss: 0.00033403458655811846  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
### finetuning

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi000_im000._original_depth]|![JNet_603_beads_roi000_im000._output_depth]|![JNet_603_beads_roi000_im000._reconst_depth]|![JNet_603_beads_roi000_im000._heatmap_depth]|
  
volume: 3.652000000000001, MSE: 0.0009575168951414526, quantized loss: 0.00036274289595894516  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi001_im004._original_depth]|![JNet_603_beads_roi001_im004._output_depth]|![JNet_603_beads_roi001_im004._reconst_depth]|![JNet_603_beads_roi001_im004._heatmap_depth]|
  
volume: 4.367000000000001, MSE: 0.0013522319495677948, quantized loss: 0.0004160084936302155  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi002_im005._original_depth]|![JNet_603_beads_roi002_im005._output_depth]|![JNet_603_beads_roi002_im005._reconst_depth]|![JNet_603_beads_roi002_im005._heatmap_depth]|
  
volume: 3.845625000000001, MSE: 0.001159410341642797, quantized loss: 0.0003834444214589894  
  
volume mean: 0.3954875000000001, volume sd: 1.190298474626393
### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi003_im006._original_depth]|![JNet_603_beads_roi003_im006._output_depth]|![JNet_603_beads_roi003_im006._reconst_depth]|![JNet_603_beads_roi003_im006._heatmap_depth]|
  
volume: 4.406750000000001, MSE: 0.001184226362966001, quantized loss: 0.00045837584184482694  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi004_im006._original_depth]|![JNet_603_beads_roi004_im006._output_depth]|![JNet_603_beads_roi004_im006._reconst_depth]|![JNet_603_beads_roi004_im006._heatmap_depth]|
  
volume: 4.485750000000001, MSE: 0.0012258189963176847, quantized loss: 0.0004647108435165137  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi005_im007._original_depth]|![JNet_603_beads_roi005_im007._output_depth]|![JNet_603_beads_roi005_im007._reconst_depth]|![JNet_603_beads_roi005_im007._heatmap_depth]|
  
volume: 4.241125000000001, MSE: 0.001135101425461471, quantized loss: 0.0004792609252035618  
  
volume mean: 0.8332750000000002, volume sd: 1.6722713870910229
### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi006_im008._original_depth]|![JNet_603_beads_roi006_im008._output_depth]|![JNet_603_beads_roi006_im008._reconst_depth]|![JNet_603_beads_roi006_im008._heatmap_depth]|
  
volume: 3.5692500000000007, MSE: 0.0016631986945867538, quantized loss: 0.00028269155882298946  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi007_im009._original_depth]|![JNet_603_beads_roi007_im009._output_depth]|![JNet_603_beads_roi007_im009._reconst_depth]|![JNet_603_beads_roi007_im009._heatmap_depth]|
  
volume: 3.3621250000000007, MSE: 0.0015993887791410089, quantized loss: 0.00022630803869105875  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi008_im010._original_depth]|![JNet_603_beads_roi008_im010._output_depth]|![JNet_603_beads_roi008_im010._reconst_depth]|![JNet_603_beads_roi008_im010._heatmap_depth]|
  
volume: 3.428875000000001, MSE: 0.0017681472236290574, quantized loss: 0.0002066770102828741  
  
volume mean: 1.1786166666666669, volume sd: 1.8152287364284307
### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi009_im011._original_depth]|![JNet_603_beads_roi009_im011._output_depth]|![JNet_603_beads_roi009_im011._reconst_depth]|![JNet_603_beads_roi009_im011._heatmap_depth]|
  
volume: 4.487125000000001, MSE: 0.0015676262555643916, quantized loss: 0.0006836211541667581  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi010_im012._original_depth]|![JNet_603_beads_roi010_im012._output_depth]|![JNet_603_beads_roi010_im012._reconst_depth]|![JNet_603_beads_roi010_im012._heatmap_depth]|
  
volume: 5.1190000000000015, MSE: 0.0019098203629255295, quantized loss: 0.0007179668173193932  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi011_im013._original_depth]|![JNet_603_beads_roi011_im013._output_depth]|![JNet_603_beads_roi011_im013._reconst_depth]|![JNet_603_beads_roi011_im013._heatmap_depth]|
  
volume: 5.165375000000001, MSE: 0.0018704729154706001, quantized loss: 0.0007030479609966278  
  
volume mean: 1.6710000000000003, volume sd: 2.079888611049881
### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi012_im014._original_depth]|![JNet_603_beads_roi012_im014._output_depth]|![JNet_603_beads_roi012_im014._reconst_depth]|![JNet_603_beads_roi012_im014._heatmap_depth]|
  
volume: 3.872875000000001, MSE: 0.002157684648409486, quantized loss: 0.0011572085786610842  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi013_im015._original_depth]|![JNet_603_beads_roi013_im015._output_depth]|![JNet_603_beads_roi013_im015._reconst_depth]|![JNet_603_beads_roi013_im015._heatmap_depth]|
  
volume: 3.694000000000001, MSE: 0.001424127141945064, quantized loss: 0.0006659427308477461  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi014_im016._original_depth]|![JNet_603_beads_roi014_im016._output_depth]|![JNet_603_beads_roi014_im016._reconst_depth]|![JNet_603_beads_roi014_im016._heatmap_depth]|
  
volume: 3.574625000000001, MSE: 0.0015528261428698897, quantized loss: 0.0006932523683644831  
  
volume mean: 2.042383333333334, volume sd: 2.080289877870924
### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi015_im017._original_depth]|![JNet_603_beads_roi015_im017._output_depth]|![JNet_603_beads_roi015_im017._reconst_depth]|![JNet_603_beads_roi015_im017._heatmap_depth]|
  
volume: 3.688750000000001, MSE: 0.0035137515515089035, quantized loss: 0.0024416835512965918  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi016_im018._original_depth]|![JNet_603_beads_roi016_im018._output_depth]|![JNet_603_beads_roi016_im018._reconst_depth]|![JNet_603_beads_roi016_im018._heatmap_depth]|
  
volume: 4.222000000000001, MSE: 0.0036097727715969086, quantized loss: 0.00240061292424798  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi017_im018._original_depth]|![JNet_603_beads_roi017_im018._output_depth]|![JNet_603_beads_roi017_im018._reconst_depth]|![JNet_603_beads_roi017_im018._heatmap_depth]|
  
volume: 4.0371250000000005, MSE: 0.0036700544878840446, quantized loss: 0.002301166532561183  
  
volume mean: 2.440645833333334, volume sd: 2.0330276765012676
### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi018_im022._original_depth]|![JNet_603_beads_roi018_im022._output_depth]|![JNet_603_beads_roi018_im022._reconst_depth]|![JNet_603_beads_roi018_im022._heatmap_depth]|
  
volume: 3.507375000000001, MSE: 0.001871221698820591, quantized loss: 0.0010068234987556934  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi019_im023._original_depth]|![JNet_603_beads_roi019_im023._output_depth]|![JNet_603_beads_roi019_im023._reconst_depth]|![JNet_603_beads_roi019_im023._heatmap_depth]|
  
volume: 3.4050000000000007, MSE: 0.0018315381603315473, quantized loss: 0.0010004718787968159  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi020_im024._original_depth]|![JNet_603_beads_roi020_im024._output_depth]|![JNet_603_beads_roi020_im024._reconst_depth]|![JNet_603_beads_roi020_im024._heatmap_depth]|
  
volume: 4.312375000000001, MSE: 0.0021349384915083647, quantized loss: 0.001040346920490265  
  
volume mean: 2.814804166666667, volume sd: 1.8929449379767513
### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi021_im026._original_depth]|![JNet_603_beads_roi021_im026._output_depth]|![JNet_603_beads_roi021_im026._reconst_depth]|![JNet_603_beads_roi021_im026._heatmap_depth]|
  
volume: 4.208125000000001, MSE: 0.0016992324963212013, quantized loss: 0.000942987680900842  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi022_im027._original_depth]|![JNet_603_beads_roi022_im027._output_depth]|![JNet_603_beads_roi022_im027._reconst_depth]|![JNet_603_beads_roi022_im027._heatmap_depth]|
  
volume: 3.918625000000001, MSE: 0.0017837929772213101, quantized loss: 0.0009092853288166225  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi023_im028._original_depth]|![JNet_603_beads_roi023_im028._output_depth]|![JNet_603_beads_roi023_im028._reconst_depth]|![JNet_603_beads_roi023_im028._heatmap_depth]|
  
volume: 4.434625000000001, MSE: 0.001778505276888609, quantized loss: 0.0009525044006295502  
  
volume mean: 3.233516666666667, volume sd: 1.6758264313010531
### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi024_im028._original_depth]|![JNet_603_beads_roi024_im028._output_depth]|![JNet_603_beads_roi024_im028._reconst_depth]|![JNet_603_beads_roi024_im028._heatmap_depth]|
  
volume: 4.507875000000001, MSE: 0.002519867615774274, quantized loss: 0.0009426975157111883  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi025_im028._original_depth]|![JNet_603_beads_roi025_im028._output_depth]|![JNet_603_beads_roi025_im028._reconst_depth]|![JNet_603_beads_roi025_im028._heatmap_depth]|
  
volume: 4.507875000000001, MSE: 0.002519867615774274, quantized loss: 0.0009426975157111883  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi026_im029._original_depth]|![JNet_603_beads_roi026_im029._output_depth]|![JNet_603_beads_roi026_im029._reconst_depth]|![JNet_603_beads_roi026_im029._heatmap_depth]|
  
volume: 4.449125000000001, MSE: 0.0025704021099954844, quantized loss: 0.000919973012059927  
  
volume mean: 3.6823458333333337, volume sd: 1.311070682719588
### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi027_im029._original_depth]|![JNet_603_beads_roi027_im029._output_depth]|![JNet_603_beads_roi027_im029._reconst_depth]|![JNet_603_beads_roi027_im029._heatmap_depth]|
  
volume: 3.1721250000000007, MSE: 0.0010541094234213233, quantized loss: 0.0004219531547278166  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi028_im030._original_depth]|![JNet_603_beads_roi028_im030._output_depth]|![JNet_603_beads_roi028_im030._reconst_depth]|![JNet_603_beads_roi028_im030._heatmap_depth]|
  
volume: 3.1708750000000006, MSE: 0.0009546351502649486, quantized loss: 0.0003927464422304183  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_603_beads_roi029_im030._original_depth]|![JNet_603_beads_roi029_im030._output_depth]|![JNet_603_beads_roi029_im030._reconst_depth]|![JNet_603_beads_roi029_im030._heatmap_depth]|
  
volume: 3.2886250000000006, MSE: 0.0010091366712003946, quantized loss: 0.0004146022256463766  
  
volume mean: 4.0034, volume sd: 0.5314398057008277  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_603_psf_pre]|![JNet_603_psf_post]|

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
  



[JNet_592_pretrain_0_aligned_depth]: /experiments/images/JNet_592_pretrain_0_aligned_depth.png
[JNet_592_pretrain_0_aligned_plane]: /experiments/images/JNet_592_pretrain_0_aligned_plane.png
[JNet_592_pretrain_0_labelx_depth]: /experiments/images/JNet_592_pretrain_0_labelx_depth.png
[JNet_592_pretrain_0_labelx_plane]: /experiments/images/JNet_592_pretrain_0_labelx_plane.png
[JNet_592_pretrain_0_labelz_depth]: /experiments/images/JNet_592_pretrain_0_labelz_depth.png
[JNet_592_pretrain_0_labelz_plane]: /experiments/images/JNet_592_pretrain_0_labelz_plane.png
[JNet_592_pretrain_0_novibrate_depth]: /experiments/images/JNet_592_pretrain_0_novibrate_depth.png
[JNet_592_pretrain_0_novibrate_plane]: /experiments/images/JNet_592_pretrain_0_novibrate_plane.png
[JNet_592_pretrain_0_original_depth]: /experiments/images/JNet_592_pretrain_0_original_depth.png
[JNet_592_pretrain_0_original_plane]: /experiments/images/JNet_592_pretrain_0_original_plane.png
[JNet_592_pretrain_0_outputx_depth]: /experiments/images/JNet_592_pretrain_0_outputx_depth.png
[JNet_592_pretrain_0_outputx_plane]: /experiments/images/JNet_592_pretrain_0_outputx_plane.png
[JNet_592_pretrain_0_outputz_depth]: /experiments/images/JNet_592_pretrain_0_outputz_depth.png
[JNet_592_pretrain_0_outputz_plane]: /experiments/images/JNet_592_pretrain_0_outputz_plane.png
[JNet_592_pretrain_1_aligned_depth]: /experiments/images/JNet_592_pretrain_1_aligned_depth.png
[JNet_592_pretrain_1_aligned_plane]: /experiments/images/JNet_592_pretrain_1_aligned_plane.png
[JNet_592_pretrain_1_labelx_depth]: /experiments/images/JNet_592_pretrain_1_labelx_depth.png
[JNet_592_pretrain_1_labelx_plane]: /experiments/images/JNet_592_pretrain_1_labelx_plane.png
[JNet_592_pretrain_1_labelz_depth]: /experiments/images/JNet_592_pretrain_1_labelz_depth.png
[JNet_592_pretrain_1_labelz_plane]: /experiments/images/JNet_592_pretrain_1_labelz_plane.png
[JNet_592_pretrain_1_novibrate_depth]: /experiments/images/JNet_592_pretrain_1_novibrate_depth.png
[JNet_592_pretrain_1_novibrate_plane]: /experiments/images/JNet_592_pretrain_1_novibrate_plane.png
[JNet_592_pretrain_1_original_depth]: /experiments/images/JNet_592_pretrain_1_original_depth.png
[JNet_592_pretrain_1_original_plane]: /experiments/images/JNet_592_pretrain_1_original_plane.png
[JNet_592_pretrain_1_outputx_depth]: /experiments/images/JNet_592_pretrain_1_outputx_depth.png
[JNet_592_pretrain_1_outputx_plane]: /experiments/images/JNet_592_pretrain_1_outputx_plane.png
[JNet_592_pretrain_1_outputz_depth]: /experiments/images/JNet_592_pretrain_1_outputz_depth.png
[JNet_592_pretrain_1_outputz_plane]: /experiments/images/JNet_592_pretrain_1_outputz_plane.png
[JNet_592_pretrain_2_aligned_depth]: /experiments/images/JNet_592_pretrain_2_aligned_depth.png
[JNet_592_pretrain_2_aligned_plane]: /experiments/images/JNet_592_pretrain_2_aligned_plane.png
[JNet_592_pretrain_2_labelx_depth]: /experiments/images/JNet_592_pretrain_2_labelx_depth.png
[JNet_592_pretrain_2_labelx_plane]: /experiments/images/JNet_592_pretrain_2_labelx_plane.png
[JNet_592_pretrain_2_labelz_depth]: /experiments/images/JNet_592_pretrain_2_labelz_depth.png
[JNet_592_pretrain_2_labelz_plane]: /experiments/images/JNet_592_pretrain_2_labelz_plane.png
[JNet_592_pretrain_2_novibrate_depth]: /experiments/images/JNet_592_pretrain_2_novibrate_depth.png
[JNet_592_pretrain_2_novibrate_plane]: /experiments/images/JNet_592_pretrain_2_novibrate_plane.png
[JNet_592_pretrain_2_original_depth]: /experiments/images/JNet_592_pretrain_2_original_depth.png
[JNet_592_pretrain_2_original_plane]: /experiments/images/JNet_592_pretrain_2_original_plane.png
[JNet_592_pretrain_2_outputx_depth]: /experiments/images/JNet_592_pretrain_2_outputx_depth.png
[JNet_592_pretrain_2_outputx_plane]: /experiments/images/JNet_592_pretrain_2_outputx_plane.png
[JNet_592_pretrain_2_outputz_depth]: /experiments/images/JNet_592_pretrain_2_outputz_depth.png
[JNet_592_pretrain_2_outputz_plane]: /experiments/images/JNet_592_pretrain_2_outputz_plane.png
[JNet_592_pretrain_3_aligned_depth]: /experiments/images/JNet_592_pretrain_3_aligned_depth.png
[JNet_592_pretrain_3_aligned_plane]: /experiments/images/JNet_592_pretrain_3_aligned_plane.png
[JNet_592_pretrain_3_labelx_depth]: /experiments/images/JNet_592_pretrain_3_labelx_depth.png
[JNet_592_pretrain_3_labelx_plane]: /experiments/images/JNet_592_pretrain_3_labelx_plane.png
[JNet_592_pretrain_3_labelz_depth]: /experiments/images/JNet_592_pretrain_3_labelz_depth.png
[JNet_592_pretrain_3_labelz_plane]: /experiments/images/JNet_592_pretrain_3_labelz_plane.png
[JNet_592_pretrain_3_novibrate_depth]: /experiments/images/JNet_592_pretrain_3_novibrate_depth.png
[JNet_592_pretrain_3_novibrate_plane]: /experiments/images/JNet_592_pretrain_3_novibrate_plane.png
[JNet_592_pretrain_3_original_depth]: /experiments/images/JNet_592_pretrain_3_original_depth.png
[JNet_592_pretrain_3_original_plane]: /experiments/images/JNet_592_pretrain_3_original_plane.png
[JNet_592_pretrain_3_outputx_depth]: /experiments/images/JNet_592_pretrain_3_outputx_depth.png
[JNet_592_pretrain_3_outputx_plane]: /experiments/images/JNet_592_pretrain_3_outputx_plane.png
[JNet_592_pretrain_3_outputz_depth]: /experiments/images/JNet_592_pretrain_3_outputz_depth.png
[JNet_592_pretrain_3_outputz_plane]: /experiments/images/JNet_592_pretrain_3_outputz_plane.png
[JNet_592_pretrain_4_aligned_depth]: /experiments/images/JNet_592_pretrain_4_aligned_depth.png
[JNet_592_pretrain_4_aligned_plane]: /experiments/images/JNet_592_pretrain_4_aligned_plane.png
[JNet_592_pretrain_4_labelx_depth]: /experiments/images/JNet_592_pretrain_4_labelx_depth.png
[JNet_592_pretrain_4_labelx_plane]: /experiments/images/JNet_592_pretrain_4_labelx_plane.png
[JNet_592_pretrain_4_labelz_depth]: /experiments/images/JNet_592_pretrain_4_labelz_depth.png
[JNet_592_pretrain_4_labelz_plane]: /experiments/images/JNet_592_pretrain_4_labelz_plane.png
[JNet_592_pretrain_4_novibrate_depth]: /experiments/images/JNet_592_pretrain_4_novibrate_depth.png
[JNet_592_pretrain_4_novibrate_plane]: /experiments/images/JNet_592_pretrain_4_novibrate_plane.png
[JNet_592_pretrain_4_original_depth]: /experiments/images/JNet_592_pretrain_4_original_depth.png
[JNet_592_pretrain_4_original_plane]: /experiments/images/JNet_592_pretrain_4_original_plane.png
[JNet_592_pretrain_4_outputx_depth]: /experiments/images/JNet_592_pretrain_4_outputx_depth.png
[JNet_592_pretrain_4_outputx_plane]: /experiments/images/JNet_592_pretrain_4_outputx_plane.png
[JNet_592_pretrain_4_outputz_depth]: /experiments/images/JNet_592_pretrain_4_outputz_depth.png
[JNet_592_pretrain_4_outputz_plane]: /experiments/images/JNet_592_pretrain_4_outputz_plane.png
[JNet_592_pretrain_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi000_im000._heatmap_depth.png
[JNet_592_pretrain_beads_roi000_im000._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi000_im000._original_depth.png
[JNet_592_pretrain_beads_roi000_im000._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi000_im000._output_depth.png
[JNet_592_pretrain_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi000_im000._reconst_depth.png
[JNet_592_pretrain_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi001_im004._heatmap_depth.png
[JNet_592_pretrain_beads_roi001_im004._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi001_im004._original_depth.png
[JNet_592_pretrain_beads_roi001_im004._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi001_im004._output_depth.png
[JNet_592_pretrain_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi001_im004._reconst_depth.png
[JNet_592_pretrain_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi002_im005._heatmap_depth.png
[JNet_592_pretrain_beads_roi002_im005._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi002_im005._original_depth.png
[JNet_592_pretrain_beads_roi002_im005._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi002_im005._output_depth.png
[JNet_592_pretrain_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi002_im005._reconst_depth.png
[JNet_592_pretrain_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi003_im006._heatmap_depth.png
[JNet_592_pretrain_beads_roi003_im006._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi003_im006._original_depth.png
[JNet_592_pretrain_beads_roi003_im006._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi003_im006._output_depth.png
[JNet_592_pretrain_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi003_im006._reconst_depth.png
[JNet_592_pretrain_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi004_im006._heatmap_depth.png
[JNet_592_pretrain_beads_roi004_im006._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi004_im006._original_depth.png
[JNet_592_pretrain_beads_roi004_im006._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi004_im006._output_depth.png
[JNet_592_pretrain_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi004_im006._reconst_depth.png
[JNet_592_pretrain_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi005_im007._heatmap_depth.png
[JNet_592_pretrain_beads_roi005_im007._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi005_im007._original_depth.png
[JNet_592_pretrain_beads_roi005_im007._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi005_im007._output_depth.png
[JNet_592_pretrain_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi005_im007._reconst_depth.png
[JNet_592_pretrain_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi006_im008._heatmap_depth.png
[JNet_592_pretrain_beads_roi006_im008._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi006_im008._original_depth.png
[JNet_592_pretrain_beads_roi006_im008._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi006_im008._output_depth.png
[JNet_592_pretrain_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi006_im008._reconst_depth.png
[JNet_592_pretrain_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi007_im009._heatmap_depth.png
[JNet_592_pretrain_beads_roi007_im009._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi007_im009._original_depth.png
[JNet_592_pretrain_beads_roi007_im009._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi007_im009._output_depth.png
[JNet_592_pretrain_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi007_im009._reconst_depth.png
[JNet_592_pretrain_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi008_im010._heatmap_depth.png
[JNet_592_pretrain_beads_roi008_im010._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi008_im010._original_depth.png
[JNet_592_pretrain_beads_roi008_im010._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi008_im010._output_depth.png
[JNet_592_pretrain_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi008_im010._reconst_depth.png
[JNet_592_pretrain_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi009_im011._heatmap_depth.png
[JNet_592_pretrain_beads_roi009_im011._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi009_im011._original_depth.png
[JNet_592_pretrain_beads_roi009_im011._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi009_im011._output_depth.png
[JNet_592_pretrain_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi009_im011._reconst_depth.png
[JNet_592_pretrain_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi010_im012._heatmap_depth.png
[JNet_592_pretrain_beads_roi010_im012._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi010_im012._original_depth.png
[JNet_592_pretrain_beads_roi010_im012._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi010_im012._output_depth.png
[JNet_592_pretrain_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi010_im012._reconst_depth.png
[JNet_592_pretrain_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi011_im013._heatmap_depth.png
[JNet_592_pretrain_beads_roi011_im013._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi011_im013._original_depth.png
[JNet_592_pretrain_beads_roi011_im013._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi011_im013._output_depth.png
[JNet_592_pretrain_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi011_im013._reconst_depth.png
[JNet_592_pretrain_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi012_im014._heatmap_depth.png
[JNet_592_pretrain_beads_roi012_im014._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi012_im014._original_depth.png
[JNet_592_pretrain_beads_roi012_im014._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi012_im014._output_depth.png
[JNet_592_pretrain_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi012_im014._reconst_depth.png
[JNet_592_pretrain_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi013_im015._heatmap_depth.png
[JNet_592_pretrain_beads_roi013_im015._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi013_im015._original_depth.png
[JNet_592_pretrain_beads_roi013_im015._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi013_im015._output_depth.png
[JNet_592_pretrain_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi013_im015._reconst_depth.png
[JNet_592_pretrain_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi014_im016._heatmap_depth.png
[JNet_592_pretrain_beads_roi014_im016._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi014_im016._original_depth.png
[JNet_592_pretrain_beads_roi014_im016._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi014_im016._output_depth.png
[JNet_592_pretrain_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi014_im016._reconst_depth.png
[JNet_592_pretrain_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi015_im017._heatmap_depth.png
[JNet_592_pretrain_beads_roi015_im017._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi015_im017._original_depth.png
[JNet_592_pretrain_beads_roi015_im017._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi015_im017._output_depth.png
[JNet_592_pretrain_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi015_im017._reconst_depth.png
[JNet_592_pretrain_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi016_im018._heatmap_depth.png
[JNet_592_pretrain_beads_roi016_im018._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi016_im018._original_depth.png
[JNet_592_pretrain_beads_roi016_im018._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi016_im018._output_depth.png
[JNet_592_pretrain_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi016_im018._reconst_depth.png
[JNet_592_pretrain_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi017_im018._heatmap_depth.png
[JNet_592_pretrain_beads_roi017_im018._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi017_im018._original_depth.png
[JNet_592_pretrain_beads_roi017_im018._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi017_im018._output_depth.png
[JNet_592_pretrain_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi017_im018._reconst_depth.png
[JNet_592_pretrain_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi018_im022._heatmap_depth.png
[JNet_592_pretrain_beads_roi018_im022._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi018_im022._original_depth.png
[JNet_592_pretrain_beads_roi018_im022._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi018_im022._output_depth.png
[JNet_592_pretrain_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi018_im022._reconst_depth.png
[JNet_592_pretrain_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi019_im023._heatmap_depth.png
[JNet_592_pretrain_beads_roi019_im023._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi019_im023._original_depth.png
[JNet_592_pretrain_beads_roi019_im023._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi019_im023._output_depth.png
[JNet_592_pretrain_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi019_im023._reconst_depth.png
[JNet_592_pretrain_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi020_im024._heatmap_depth.png
[JNet_592_pretrain_beads_roi020_im024._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi020_im024._original_depth.png
[JNet_592_pretrain_beads_roi020_im024._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi020_im024._output_depth.png
[JNet_592_pretrain_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi020_im024._reconst_depth.png
[JNet_592_pretrain_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi021_im026._heatmap_depth.png
[JNet_592_pretrain_beads_roi021_im026._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi021_im026._original_depth.png
[JNet_592_pretrain_beads_roi021_im026._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi021_im026._output_depth.png
[JNet_592_pretrain_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi021_im026._reconst_depth.png
[JNet_592_pretrain_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi022_im027._heatmap_depth.png
[JNet_592_pretrain_beads_roi022_im027._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi022_im027._original_depth.png
[JNet_592_pretrain_beads_roi022_im027._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi022_im027._output_depth.png
[JNet_592_pretrain_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi022_im027._reconst_depth.png
[JNet_592_pretrain_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi023_im028._heatmap_depth.png
[JNet_592_pretrain_beads_roi023_im028._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi023_im028._original_depth.png
[JNet_592_pretrain_beads_roi023_im028._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi023_im028._output_depth.png
[JNet_592_pretrain_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi023_im028._reconst_depth.png
[JNet_592_pretrain_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi024_im028._heatmap_depth.png
[JNet_592_pretrain_beads_roi024_im028._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi024_im028._original_depth.png
[JNet_592_pretrain_beads_roi024_im028._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi024_im028._output_depth.png
[JNet_592_pretrain_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi024_im028._reconst_depth.png
[JNet_592_pretrain_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi025_im028._heatmap_depth.png
[JNet_592_pretrain_beads_roi025_im028._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi025_im028._original_depth.png
[JNet_592_pretrain_beads_roi025_im028._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi025_im028._output_depth.png
[JNet_592_pretrain_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi025_im028._reconst_depth.png
[JNet_592_pretrain_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi026_im029._heatmap_depth.png
[JNet_592_pretrain_beads_roi026_im029._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi026_im029._original_depth.png
[JNet_592_pretrain_beads_roi026_im029._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi026_im029._output_depth.png
[JNet_592_pretrain_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi026_im029._reconst_depth.png
[JNet_592_pretrain_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi027_im029._heatmap_depth.png
[JNet_592_pretrain_beads_roi027_im029._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi027_im029._original_depth.png
[JNet_592_pretrain_beads_roi027_im029._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi027_im029._output_depth.png
[JNet_592_pretrain_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi027_im029._reconst_depth.png
[JNet_592_pretrain_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi028_im030._heatmap_depth.png
[JNet_592_pretrain_beads_roi028_im030._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi028_im030._original_depth.png
[JNet_592_pretrain_beads_roi028_im030._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi028_im030._output_depth.png
[JNet_592_pretrain_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi028_im030._reconst_depth.png
[JNet_592_pretrain_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_592_pretrain_beads_roi029_im030._heatmap_depth.png
[JNet_592_pretrain_beads_roi029_im030._original_depth]: /experiments/images/JNet_592_pretrain_beads_roi029_im030._original_depth.png
[JNet_592_pretrain_beads_roi029_im030._output_depth]: /experiments/images/JNet_592_pretrain_beads_roi029_im030._output_depth.png
[JNet_592_pretrain_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_592_pretrain_beads_roi029_im030._reconst_depth.png
[JNet_603_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_603_beads_roi000_im000._heatmap_depth.png
[JNet_603_beads_roi000_im000._original_depth]: /experiments/images/JNet_603_beads_roi000_im000._original_depth.png
[JNet_603_beads_roi000_im000._output_depth]: /experiments/images/JNet_603_beads_roi000_im000._output_depth.png
[JNet_603_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_603_beads_roi000_im000._reconst_depth.png
[JNet_603_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_603_beads_roi001_im004._heatmap_depth.png
[JNet_603_beads_roi001_im004._original_depth]: /experiments/images/JNet_603_beads_roi001_im004._original_depth.png
[JNet_603_beads_roi001_im004._output_depth]: /experiments/images/JNet_603_beads_roi001_im004._output_depth.png
[JNet_603_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_603_beads_roi001_im004._reconst_depth.png
[JNet_603_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_603_beads_roi002_im005._heatmap_depth.png
[JNet_603_beads_roi002_im005._original_depth]: /experiments/images/JNet_603_beads_roi002_im005._original_depth.png
[JNet_603_beads_roi002_im005._output_depth]: /experiments/images/JNet_603_beads_roi002_im005._output_depth.png
[JNet_603_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_603_beads_roi002_im005._reconst_depth.png
[JNet_603_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_603_beads_roi003_im006._heatmap_depth.png
[JNet_603_beads_roi003_im006._original_depth]: /experiments/images/JNet_603_beads_roi003_im006._original_depth.png
[JNet_603_beads_roi003_im006._output_depth]: /experiments/images/JNet_603_beads_roi003_im006._output_depth.png
[JNet_603_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_603_beads_roi003_im006._reconst_depth.png
[JNet_603_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_603_beads_roi004_im006._heatmap_depth.png
[JNet_603_beads_roi004_im006._original_depth]: /experiments/images/JNet_603_beads_roi004_im006._original_depth.png
[JNet_603_beads_roi004_im006._output_depth]: /experiments/images/JNet_603_beads_roi004_im006._output_depth.png
[JNet_603_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_603_beads_roi004_im006._reconst_depth.png
[JNet_603_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_603_beads_roi005_im007._heatmap_depth.png
[JNet_603_beads_roi005_im007._original_depth]: /experiments/images/JNet_603_beads_roi005_im007._original_depth.png
[JNet_603_beads_roi005_im007._output_depth]: /experiments/images/JNet_603_beads_roi005_im007._output_depth.png
[JNet_603_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_603_beads_roi005_im007._reconst_depth.png
[JNet_603_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_603_beads_roi006_im008._heatmap_depth.png
[JNet_603_beads_roi006_im008._original_depth]: /experiments/images/JNet_603_beads_roi006_im008._original_depth.png
[JNet_603_beads_roi006_im008._output_depth]: /experiments/images/JNet_603_beads_roi006_im008._output_depth.png
[JNet_603_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_603_beads_roi006_im008._reconst_depth.png
[JNet_603_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_603_beads_roi007_im009._heatmap_depth.png
[JNet_603_beads_roi007_im009._original_depth]: /experiments/images/JNet_603_beads_roi007_im009._original_depth.png
[JNet_603_beads_roi007_im009._output_depth]: /experiments/images/JNet_603_beads_roi007_im009._output_depth.png
[JNet_603_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_603_beads_roi007_im009._reconst_depth.png
[JNet_603_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_603_beads_roi008_im010._heatmap_depth.png
[JNet_603_beads_roi008_im010._original_depth]: /experiments/images/JNet_603_beads_roi008_im010._original_depth.png
[JNet_603_beads_roi008_im010._output_depth]: /experiments/images/JNet_603_beads_roi008_im010._output_depth.png
[JNet_603_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_603_beads_roi008_im010._reconst_depth.png
[JNet_603_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_603_beads_roi009_im011._heatmap_depth.png
[JNet_603_beads_roi009_im011._original_depth]: /experiments/images/JNet_603_beads_roi009_im011._original_depth.png
[JNet_603_beads_roi009_im011._output_depth]: /experiments/images/JNet_603_beads_roi009_im011._output_depth.png
[JNet_603_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_603_beads_roi009_im011._reconst_depth.png
[JNet_603_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_603_beads_roi010_im012._heatmap_depth.png
[JNet_603_beads_roi010_im012._original_depth]: /experiments/images/JNet_603_beads_roi010_im012._original_depth.png
[JNet_603_beads_roi010_im012._output_depth]: /experiments/images/JNet_603_beads_roi010_im012._output_depth.png
[JNet_603_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_603_beads_roi010_im012._reconst_depth.png
[JNet_603_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_603_beads_roi011_im013._heatmap_depth.png
[JNet_603_beads_roi011_im013._original_depth]: /experiments/images/JNet_603_beads_roi011_im013._original_depth.png
[JNet_603_beads_roi011_im013._output_depth]: /experiments/images/JNet_603_beads_roi011_im013._output_depth.png
[JNet_603_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_603_beads_roi011_im013._reconst_depth.png
[JNet_603_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_603_beads_roi012_im014._heatmap_depth.png
[JNet_603_beads_roi012_im014._original_depth]: /experiments/images/JNet_603_beads_roi012_im014._original_depth.png
[JNet_603_beads_roi012_im014._output_depth]: /experiments/images/JNet_603_beads_roi012_im014._output_depth.png
[JNet_603_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_603_beads_roi012_im014._reconst_depth.png
[JNet_603_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_603_beads_roi013_im015._heatmap_depth.png
[JNet_603_beads_roi013_im015._original_depth]: /experiments/images/JNet_603_beads_roi013_im015._original_depth.png
[JNet_603_beads_roi013_im015._output_depth]: /experiments/images/JNet_603_beads_roi013_im015._output_depth.png
[JNet_603_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_603_beads_roi013_im015._reconst_depth.png
[JNet_603_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_603_beads_roi014_im016._heatmap_depth.png
[JNet_603_beads_roi014_im016._original_depth]: /experiments/images/JNet_603_beads_roi014_im016._original_depth.png
[JNet_603_beads_roi014_im016._output_depth]: /experiments/images/JNet_603_beads_roi014_im016._output_depth.png
[JNet_603_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_603_beads_roi014_im016._reconst_depth.png
[JNet_603_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_603_beads_roi015_im017._heatmap_depth.png
[JNet_603_beads_roi015_im017._original_depth]: /experiments/images/JNet_603_beads_roi015_im017._original_depth.png
[JNet_603_beads_roi015_im017._output_depth]: /experiments/images/JNet_603_beads_roi015_im017._output_depth.png
[JNet_603_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_603_beads_roi015_im017._reconst_depth.png
[JNet_603_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_603_beads_roi016_im018._heatmap_depth.png
[JNet_603_beads_roi016_im018._original_depth]: /experiments/images/JNet_603_beads_roi016_im018._original_depth.png
[JNet_603_beads_roi016_im018._output_depth]: /experiments/images/JNet_603_beads_roi016_im018._output_depth.png
[JNet_603_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_603_beads_roi016_im018._reconst_depth.png
[JNet_603_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_603_beads_roi017_im018._heatmap_depth.png
[JNet_603_beads_roi017_im018._original_depth]: /experiments/images/JNet_603_beads_roi017_im018._original_depth.png
[JNet_603_beads_roi017_im018._output_depth]: /experiments/images/JNet_603_beads_roi017_im018._output_depth.png
[JNet_603_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_603_beads_roi017_im018._reconst_depth.png
[JNet_603_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_603_beads_roi018_im022._heatmap_depth.png
[JNet_603_beads_roi018_im022._original_depth]: /experiments/images/JNet_603_beads_roi018_im022._original_depth.png
[JNet_603_beads_roi018_im022._output_depth]: /experiments/images/JNet_603_beads_roi018_im022._output_depth.png
[JNet_603_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_603_beads_roi018_im022._reconst_depth.png
[JNet_603_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_603_beads_roi019_im023._heatmap_depth.png
[JNet_603_beads_roi019_im023._original_depth]: /experiments/images/JNet_603_beads_roi019_im023._original_depth.png
[JNet_603_beads_roi019_im023._output_depth]: /experiments/images/JNet_603_beads_roi019_im023._output_depth.png
[JNet_603_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_603_beads_roi019_im023._reconst_depth.png
[JNet_603_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_603_beads_roi020_im024._heatmap_depth.png
[JNet_603_beads_roi020_im024._original_depth]: /experiments/images/JNet_603_beads_roi020_im024._original_depth.png
[JNet_603_beads_roi020_im024._output_depth]: /experiments/images/JNet_603_beads_roi020_im024._output_depth.png
[JNet_603_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_603_beads_roi020_im024._reconst_depth.png
[JNet_603_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_603_beads_roi021_im026._heatmap_depth.png
[JNet_603_beads_roi021_im026._original_depth]: /experiments/images/JNet_603_beads_roi021_im026._original_depth.png
[JNet_603_beads_roi021_im026._output_depth]: /experiments/images/JNet_603_beads_roi021_im026._output_depth.png
[JNet_603_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_603_beads_roi021_im026._reconst_depth.png
[JNet_603_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_603_beads_roi022_im027._heatmap_depth.png
[JNet_603_beads_roi022_im027._original_depth]: /experiments/images/JNet_603_beads_roi022_im027._original_depth.png
[JNet_603_beads_roi022_im027._output_depth]: /experiments/images/JNet_603_beads_roi022_im027._output_depth.png
[JNet_603_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_603_beads_roi022_im027._reconst_depth.png
[JNet_603_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_603_beads_roi023_im028._heatmap_depth.png
[JNet_603_beads_roi023_im028._original_depth]: /experiments/images/JNet_603_beads_roi023_im028._original_depth.png
[JNet_603_beads_roi023_im028._output_depth]: /experiments/images/JNet_603_beads_roi023_im028._output_depth.png
[JNet_603_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_603_beads_roi023_im028._reconst_depth.png
[JNet_603_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_603_beads_roi024_im028._heatmap_depth.png
[JNet_603_beads_roi024_im028._original_depth]: /experiments/images/JNet_603_beads_roi024_im028._original_depth.png
[JNet_603_beads_roi024_im028._output_depth]: /experiments/images/JNet_603_beads_roi024_im028._output_depth.png
[JNet_603_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_603_beads_roi024_im028._reconst_depth.png
[JNet_603_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_603_beads_roi025_im028._heatmap_depth.png
[JNet_603_beads_roi025_im028._original_depth]: /experiments/images/JNet_603_beads_roi025_im028._original_depth.png
[JNet_603_beads_roi025_im028._output_depth]: /experiments/images/JNet_603_beads_roi025_im028._output_depth.png
[JNet_603_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_603_beads_roi025_im028._reconst_depth.png
[JNet_603_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_603_beads_roi026_im029._heatmap_depth.png
[JNet_603_beads_roi026_im029._original_depth]: /experiments/images/JNet_603_beads_roi026_im029._original_depth.png
[JNet_603_beads_roi026_im029._output_depth]: /experiments/images/JNet_603_beads_roi026_im029._output_depth.png
[JNet_603_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_603_beads_roi026_im029._reconst_depth.png
[JNet_603_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_603_beads_roi027_im029._heatmap_depth.png
[JNet_603_beads_roi027_im029._original_depth]: /experiments/images/JNet_603_beads_roi027_im029._original_depth.png
[JNet_603_beads_roi027_im029._output_depth]: /experiments/images/JNet_603_beads_roi027_im029._output_depth.png
[JNet_603_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_603_beads_roi027_im029._reconst_depth.png
[JNet_603_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_603_beads_roi028_im030._heatmap_depth.png
[JNet_603_beads_roi028_im030._original_depth]: /experiments/images/JNet_603_beads_roi028_im030._original_depth.png
[JNet_603_beads_roi028_im030._output_depth]: /experiments/images/JNet_603_beads_roi028_im030._output_depth.png
[JNet_603_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_603_beads_roi028_im030._reconst_depth.png
[JNet_603_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_603_beads_roi029_im030._heatmap_depth.png
[JNet_603_beads_roi029_im030._original_depth]: /experiments/images/JNet_603_beads_roi029_im030._original_depth.png
[JNet_603_beads_roi029_im030._output_depth]: /experiments/images/JNet_603_beads_roi029_im030._output_depth.png
[JNet_603_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_603_beads_roi029_im030._reconst_depth.png
[JNet_603_psf_post]: /experiments/images/JNet_603_psf_post.png
[JNet_603_psf_pre]: /experiments/images/JNet_603_psf_pre.png
