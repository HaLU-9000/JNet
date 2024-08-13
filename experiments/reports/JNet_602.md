



# JNet_602 Report
  
592 pre ewc0.1, parameter1.0  
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
  
Segmentation: mean MSE: 0.007944582030177116, mean BCE: 0.030213389545679092  
Luminance Estimation: mean MSE: 0.9842451214790344, mean BCE: nan
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_0_original_plane]|![JNet_592_pretrain_0_novibrate_plane]|![JNet_592_pretrain_0_aligned_plane]|![JNet_592_pretrain_0_outputx_plane]|![JNet_592_pretrain_0_labelx_plane]|![JNet_592_pretrain_0_outputz_plane]|![JNet_592_pretrain_0_labelz_plane]|
  
MSEx: 0.007774516474455595, BCEx: 0.029148368164896965  
MSEz: 0.9838436841964722, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_0_original_depth]|![JNet_592_pretrain_0_novibrate_depth]|![JNet_592_pretrain_0_aligned_depth]|![JNet_592_pretrain_0_outputx_depth]|![JNet_592_pretrain_0_labelx_depth]|![JNet_592_pretrain_0_outputz_depth]|![JNet_592_pretrain_0_labelz_depth]|
  
MSEx: 0.007774516474455595, BCEx: 0.029148368164896965  
MSEz: 0.9838436841964722, BCEz: nan  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_1_original_plane]|![JNet_592_pretrain_1_novibrate_plane]|![JNet_592_pretrain_1_aligned_plane]|![JNet_592_pretrain_1_outputx_plane]|![JNet_592_pretrain_1_labelx_plane]|![JNet_592_pretrain_1_outputz_plane]|![JNet_592_pretrain_1_labelz_plane]|
  
MSEx: 0.009517773985862732, BCEx: 0.03711167722940445  
MSEz: 0.9827543497085571, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_1_original_depth]|![JNet_592_pretrain_1_novibrate_depth]|![JNet_592_pretrain_1_aligned_depth]|![JNet_592_pretrain_1_outputx_depth]|![JNet_592_pretrain_1_labelx_depth]|![JNet_592_pretrain_1_outputz_depth]|![JNet_592_pretrain_1_labelz_depth]|
  
MSEx: 0.009517773985862732, BCEx: 0.03711167722940445  
MSEz: 0.9827543497085571, BCEz: nan  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_2_original_plane]|![JNet_592_pretrain_2_novibrate_plane]|![JNet_592_pretrain_2_aligned_plane]|![JNet_592_pretrain_2_outputx_plane]|![JNet_592_pretrain_2_labelx_plane]|![JNet_592_pretrain_2_outputz_plane]|![JNet_592_pretrain_2_labelz_plane]|
  
MSEx: 0.008878514170646667, BCEx: 0.035202011466026306  
MSEz: 0.9860235452651978, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_2_original_depth]|![JNet_592_pretrain_2_novibrate_depth]|![JNet_592_pretrain_2_aligned_depth]|![JNet_592_pretrain_2_outputx_depth]|![JNet_592_pretrain_2_labelx_depth]|![JNet_592_pretrain_2_outputz_depth]|![JNet_592_pretrain_2_labelz_depth]|
  
MSEx: 0.008878514170646667, BCEx: 0.035202011466026306  
MSEz: 0.9860235452651978, BCEz: nan  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_3_original_plane]|![JNet_592_pretrain_3_novibrate_plane]|![JNet_592_pretrain_3_aligned_plane]|![JNet_592_pretrain_3_outputx_plane]|![JNet_592_pretrain_3_labelx_plane]|![JNet_592_pretrain_3_outputz_plane]|![JNet_592_pretrain_3_labelz_plane]|
  
MSEx: 0.006439365446567535, BCEx: 0.02468518167734146  
MSEz: 0.9877615571022034, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_3_original_depth]|![JNet_592_pretrain_3_novibrate_depth]|![JNet_592_pretrain_3_aligned_depth]|![JNet_592_pretrain_3_outputx_depth]|![JNet_592_pretrain_3_labelx_depth]|![JNet_592_pretrain_3_outputz_depth]|![JNet_592_pretrain_3_labelz_depth]|
  
MSEx: 0.006439365446567535, BCEx: 0.02468518167734146  
MSEz: 0.9877615571022034, BCEz: nan  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_4_original_plane]|![JNet_592_pretrain_4_novibrate_plane]|![JNet_592_pretrain_4_aligned_plane]|![JNet_592_pretrain_4_outputx_plane]|![JNet_592_pretrain_4_labelx_plane]|![JNet_592_pretrain_4_outputz_plane]|![JNet_592_pretrain_4_labelz_plane]|
  
MSEx: 0.007112741935998201, BCEx: 0.024919699877500534  
MSEz: 0.9808423519134521, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_4_original_depth]|![JNet_592_pretrain_4_novibrate_depth]|![JNet_592_pretrain_4_aligned_depth]|![JNet_592_pretrain_4_outputx_depth]|![JNet_592_pretrain_4_labelx_depth]|![JNet_592_pretrain_4_outputz_depth]|![JNet_592_pretrain_4_labelz_depth]|
  
MSEx: 0.007112741935998201, BCEx: 0.024919699877500534  
MSEz: 0.9808423519134521, BCEz: nan  

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

### finetuning
  
volume mean: 3.2889875000000006, volume sd: 0.205062224677251
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi000_im000._original_depth]|![JNet_602_beads_roi000_im000._output_depth]|![JNet_602_beads_roi000_im000._reconst_depth]|![JNet_602_beads_roi000_im000._heatmap_depth]|
  
volume: 3.1245000000000007, MSE: 0.0009685683180578053, quantized loss: 0.00042971796938218176  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi001_im004._original_depth]|![JNet_602_beads_roi001_im004._output_depth]|![JNet_602_beads_roi001_im004._reconst_depth]|![JNet_602_beads_roi001_im004._heatmap_depth]|
  
volume: 3.5198750000000008, MSE: 0.001323835807852447, quantized loss: 0.0005538803525269032  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi002_im005._original_depth]|![JNet_602_beads_roi002_im005._output_depth]|![JNet_602_beads_roi002_im005._reconst_depth]|![JNet_602_beads_roi002_im005._heatmap_depth]|
  
volume: 3.2335000000000007, MSE: 0.0011043257545679808, quantized loss: 0.00048691011033952236  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi003_im006._original_depth]|![JNet_602_beads_roi003_im006._output_depth]|![JNet_602_beads_roi003_im006._reconst_depth]|![JNet_602_beads_roi003_im006._heatmap_depth]|
  
volume: 3.287125000000001, MSE: 0.0010713414521887898, quantized loss: 0.00053533969912678  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi004_im006._original_depth]|![JNet_602_beads_roi004_im006._output_depth]|![JNet_602_beads_roi004_im006._reconst_depth]|![JNet_602_beads_roi004_im006._heatmap_depth]|
  
volume: 3.342000000000001, MSE: 0.0011350595159456134, quantized loss: 0.0005424809060059488  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi005_im007._original_depth]|![JNet_602_beads_roi005_im007._output_depth]|![JNet_602_beads_roi005_im007._reconst_depth]|![JNet_602_beads_roi005_im007._heatmap_depth]|
  
volume: 3.1687500000000006, MSE: 0.0010486759711056948, quantized loss: 0.0005174684338271618  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi006_im008._original_depth]|![JNet_602_beads_roi006_im008._output_depth]|![JNet_602_beads_roi006_im008._reconst_depth]|![JNet_602_beads_roi006_im008._heatmap_depth]|
  
volume: 3.275375000000001, MSE: 0.0010579227237030864, quantized loss: 0.0006030803197063506  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi007_im009._original_depth]|![JNet_602_beads_roi007_im009._output_depth]|![JNet_602_beads_roi007_im009._reconst_depth]|![JNet_602_beads_roi007_im009._heatmap_depth]|
  
volume: 3.285625000000001, MSE: 0.0010291252983734012, quantized loss: 0.0005643940530717373  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi008_im010._original_depth]|![JNet_602_beads_roi008_im010._output_depth]|![JNet_602_beads_roi008_im010._reconst_depth]|![JNet_602_beads_roi008_im010._heatmap_depth]|
  
volume: 3.358375000000001, MSE: 0.0011558341793715954, quantized loss: 0.0005222272593528032  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi009_im011._original_depth]|![JNet_602_beads_roi009_im011._output_depth]|![JNet_602_beads_roi009_im011._reconst_depth]|![JNet_602_beads_roi009_im011._heatmap_depth]|
  
volume: 3.1963750000000006, MSE: 0.00100235384888947, quantized loss: 0.0004753550747409463  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi010_im012._original_depth]|![JNet_602_beads_roi010_im012._output_depth]|![JNet_602_beads_roi010_im012._reconst_depth]|![JNet_602_beads_roi010_im012._heatmap_depth]|
  
volume: 3.686750000000001, MSE: 0.0013517980696633458, quantized loss: 0.0005685345968231559  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi011_im013._original_depth]|![JNet_602_beads_roi011_im013._output_depth]|![JNet_602_beads_roi011_im013._reconst_depth]|![JNet_602_beads_roi011_im013._heatmap_depth]|
  
volume: 3.618250000000001, MSE: 0.0013247084571048617, quantized loss: 0.0005398442735895514  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi012_im014._original_depth]|![JNet_602_beads_roi012_im014._output_depth]|![JNet_602_beads_roi012_im014._reconst_depth]|![JNet_602_beads_roi012_im014._heatmap_depth]|
  
volume: 3.3050000000000006, MSE: 0.0011407387210056186, quantized loss: 0.00044915027683600783  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi013_im015._original_depth]|![JNet_602_beads_roi013_im015._output_depth]|![JNet_602_beads_roi013_im015._reconst_depth]|![JNet_602_beads_roi013_im015._heatmap_depth]|
  
volume: 3.0711250000000008, MSE: 0.0010304749011993408, quantized loss: 0.00044252353836782277  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi014_im016._original_depth]|![JNet_602_beads_roi014_im016._output_depth]|![JNet_602_beads_roi014_im016._reconst_depth]|![JNet_602_beads_roi014_im016._heatmap_depth]|
  
volume: 3.053375000000001, MSE: 0.0010262540308758616, quantized loss: 0.0005456855287775397  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi015_im017._original_depth]|![JNet_602_beads_roi015_im017._output_depth]|![JNet_602_beads_roi015_im017._reconst_depth]|![JNet_602_beads_roi015_im017._heatmap_depth]|
  
volume: 3.053000000000001, MSE: 0.0009584356448613107, quantized loss: 0.00047544107655994594  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi016_im018._original_depth]|![JNet_602_beads_roi016_im018._output_depth]|![JNet_602_beads_roi016_im018._reconst_depth]|![JNet_602_beads_roi016_im018._heatmap_depth]|
  
volume: 3.378375000000001, MSE: 0.0012345793657004833, quantized loss: 0.0005481929983943701  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi017_im018._original_depth]|![JNet_602_beads_roi017_im018._output_depth]|![JNet_602_beads_roi017_im018._reconst_depth]|![JNet_602_beads_roi017_im018._heatmap_depth]|
  
volume: 3.221000000000001, MSE: 0.001245663850568235, quantized loss: 0.0005161400767974555  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi018_im022._original_depth]|![JNet_602_beads_roi018_im022._output_depth]|![JNet_602_beads_roi018_im022._reconst_depth]|![JNet_602_beads_roi018_im022._heatmap_depth]|
  
volume: 2.8787500000000006, MSE: 0.0009311243775300682, quantized loss: 0.0003986289957538247  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi019_im023._original_depth]|![JNet_602_beads_roi019_im023._output_depth]|![JNet_602_beads_roi019_im023._reconst_depth]|![JNet_602_beads_roi019_im023._heatmap_depth]|
  
volume: 2.838250000000001, MSE: 0.0009056268609128892, quantized loss: 0.00038136818329803646  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi020_im024._original_depth]|![JNet_602_beads_roi020_im024._output_depth]|![JNet_602_beads_roi020_im024._reconst_depth]|![JNet_602_beads_roi020_im024._heatmap_depth]|
  
volume: 3.490500000000001, MSE: 0.001251435955055058, quantized loss: 0.00046258349902927876  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi021_im026._original_depth]|![JNet_602_beads_roi021_im026._output_depth]|![JNet_602_beads_roi021_im026._reconst_depth]|![JNet_602_beads_roi021_im026._heatmap_depth]|
  
volume: 3.341125000000001, MSE: 0.0011428608559072018, quantized loss: 0.0004778861766681075  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi022_im027._original_depth]|![JNet_602_beads_roi022_im027._output_depth]|![JNet_602_beads_roi022_im027._reconst_depth]|![JNet_602_beads_roi022_im027._heatmap_depth]|
  
volume: 3.271500000000001, MSE: 0.0010558959329500794, quantized loss: 0.0004295650578569621  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi023_im028._original_depth]|![JNet_602_beads_roi023_im028._output_depth]|![JNet_602_beads_roi023_im028._reconst_depth]|![JNet_602_beads_roi023_im028._heatmap_depth]|
  
volume: 3.6161250000000007, MSE: 0.001116934814490378, quantized loss: 0.0005802478990517557  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi024_im028._original_depth]|![JNet_602_beads_roi024_im028._output_depth]|![JNet_602_beads_roi024_im028._reconst_depth]|![JNet_602_beads_roi024_im028._heatmap_depth]|
  
volume: 3.4918750000000007, MSE: 0.001198648358695209, quantized loss: 0.0005258450983092189  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi025_im028._original_depth]|![JNet_602_beads_roi025_im028._output_depth]|![JNet_602_beads_roi025_im028._reconst_depth]|![JNet_602_beads_roi025_im028._heatmap_depth]|
  
volume: 3.4918750000000007, MSE: 0.001198648358695209, quantized loss: 0.0005258450983092189  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi026_im029._original_depth]|![JNet_602_beads_roi026_im029._output_depth]|![JNet_602_beads_roi026_im029._reconst_depth]|![JNet_602_beads_roi026_im029._heatmap_depth]|
  
volume: 3.535000000000001, MSE: 0.0012653314042836428, quantized loss: 0.00048278263420797884  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi027_im029._original_depth]|![JNet_602_beads_roi027_im029._output_depth]|![JNet_602_beads_roi027_im029._reconst_depth]|![JNet_602_beads_roi027_im029._heatmap_depth]|
  
volume: 3.1556250000000006, MSE: 0.0011584096355363727, quantized loss: 0.00043260896927677095  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi028_im030._original_depth]|![JNet_602_beads_roi028_im030._output_depth]|![JNet_602_beads_roi028_im030._reconst_depth]|![JNet_602_beads_roi028_im030._heatmap_depth]|
  
volume: 3.131875000000001, MSE: 0.0010410647373646498, quantized loss: 0.0004380134923849255  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_602_beads_roi029_im030._original_depth]|![JNet_602_beads_roi029_im030._output_depth]|![JNet_602_beads_roi029_im030._reconst_depth]|![JNet_602_beads_roi029_im030._heatmap_depth]|
  
volume: 3.2487500000000007, MSE: 0.001118000946007669, quantized loss: 0.0004350616654846817  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_602_psf_pre]|![JNet_602_psf_post]|

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
[JNet_602_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_602_beads_roi000_im000._heatmap_depth.png
[JNet_602_beads_roi000_im000._original_depth]: /experiments/images/JNet_602_beads_roi000_im000._original_depth.png
[JNet_602_beads_roi000_im000._output_depth]: /experiments/images/JNet_602_beads_roi000_im000._output_depth.png
[JNet_602_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_602_beads_roi000_im000._reconst_depth.png
[JNet_602_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_602_beads_roi001_im004._heatmap_depth.png
[JNet_602_beads_roi001_im004._original_depth]: /experiments/images/JNet_602_beads_roi001_im004._original_depth.png
[JNet_602_beads_roi001_im004._output_depth]: /experiments/images/JNet_602_beads_roi001_im004._output_depth.png
[JNet_602_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_602_beads_roi001_im004._reconst_depth.png
[JNet_602_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_602_beads_roi002_im005._heatmap_depth.png
[JNet_602_beads_roi002_im005._original_depth]: /experiments/images/JNet_602_beads_roi002_im005._original_depth.png
[JNet_602_beads_roi002_im005._output_depth]: /experiments/images/JNet_602_beads_roi002_im005._output_depth.png
[JNet_602_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_602_beads_roi002_im005._reconst_depth.png
[JNet_602_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_602_beads_roi003_im006._heatmap_depth.png
[JNet_602_beads_roi003_im006._original_depth]: /experiments/images/JNet_602_beads_roi003_im006._original_depth.png
[JNet_602_beads_roi003_im006._output_depth]: /experiments/images/JNet_602_beads_roi003_im006._output_depth.png
[JNet_602_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_602_beads_roi003_im006._reconst_depth.png
[JNet_602_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_602_beads_roi004_im006._heatmap_depth.png
[JNet_602_beads_roi004_im006._original_depth]: /experiments/images/JNet_602_beads_roi004_im006._original_depth.png
[JNet_602_beads_roi004_im006._output_depth]: /experiments/images/JNet_602_beads_roi004_im006._output_depth.png
[JNet_602_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_602_beads_roi004_im006._reconst_depth.png
[JNet_602_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_602_beads_roi005_im007._heatmap_depth.png
[JNet_602_beads_roi005_im007._original_depth]: /experiments/images/JNet_602_beads_roi005_im007._original_depth.png
[JNet_602_beads_roi005_im007._output_depth]: /experiments/images/JNet_602_beads_roi005_im007._output_depth.png
[JNet_602_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_602_beads_roi005_im007._reconst_depth.png
[JNet_602_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_602_beads_roi006_im008._heatmap_depth.png
[JNet_602_beads_roi006_im008._original_depth]: /experiments/images/JNet_602_beads_roi006_im008._original_depth.png
[JNet_602_beads_roi006_im008._output_depth]: /experiments/images/JNet_602_beads_roi006_im008._output_depth.png
[JNet_602_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_602_beads_roi006_im008._reconst_depth.png
[JNet_602_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_602_beads_roi007_im009._heatmap_depth.png
[JNet_602_beads_roi007_im009._original_depth]: /experiments/images/JNet_602_beads_roi007_im009._original_depth.png
[JNet_602_beads_roi007_im009._output_depth]: /experiments/images/JNet_602_beads_roi007_im009._output_depth.png
[JNet_602_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_602_beads_roi007_im009._reconst_depth.png
[JNet_602_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_602_beads_roi008_im010._heatmap_depth.png
[JNet_602_beads_roi008_im010._original_depth]: /experiments/images/JNet_602_beads_roi008_im010._original_depth.png
[JNet_602_beads_roi008_im010._output_depth]: /experiments/images/JNet_602_beads_roi008_im010._output_depth.png
[JNet_602_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_602_beads_roi008_im010._reconst_depth.png
[JNet_602_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_602_beads_roi009_im011._heatmap_depth.png
[JNet_602_beads_roi009_im011._original_depth]: /experiments/images/JNet_602_beads_roi009_im011._original_depth.png
[JNet_602_beads_roi009_im011._output_depth]: /experiments/images/JNet_602_beads_roi009_im011._output_depth.png
[JNet_602_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_602_beads_roi009_im011._reconst_depth.png
[JNet_602_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_602_beads_roi010_im012._heatmap_depth.png
[JNet_602_beads_roi010_im012._original_depth]: /experiments/images/JNet_602_beads_roi010_im012._original_depth.png
[JNet_602_beads_roi010_im012._output_depth]: /experiments/images/JNet_602_beads_roi010_im012._output_depth.png
[JNet_602_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_602_beads_roi010_im012._reconst_depth.png
[JNet_602_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_602_beads_roi011_im013._heatmap_depth.png
[JNet_602_beads_roi011_im013._original_depth]: /experiments/images/JNet_602_beads_roi011_im013._original_depth.png
[JNet_602_beads_roi011_im013._output_depth]: /experiments/images/JNet_602_beads_roi011_im013._output_depth.png
[JNet_602_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_602_beads_roi011_im013._reconst_depth.png
[JNet_602_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_602_beads_roi012_im014._heatmap_depth.png
[JNet_602_beads_roi012_im014._original_depth]: /experiments/images/JNet_602_beads_roi012_im014._original_depth.png
[JNet_602_beads_roi012_im014._output_depth]: /experiments/images/JNet_602_beads_roi012_im014._output_depth.png
[JNet_602_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_602_beads_roi012_im014._reconst_depth.png
[JNet_602_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_602_beads_roi013_im015._heatmap_depth.png
[JNet_602_beads_roi013_im015._original_depth]: /experiments/images/JNet_602_beads_roi013_im015._original_depth.png
[JNet_602_beads_roi013_im015._output_depth]: /experiments/images/JNet_602_beads_roi013_im015._output_depth.png
[JNet_602_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_602_beads_roi013_im015._reconst_depth.png
[JNet_602_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_602_beads_roi014_im016._heatmap_depth.png
[JNet_602_beads_roi014_im016._original_depth]: /experiments/images/JNet_602_beads_roi014_im016._original_depth.png
[JNet_602_beads_roi014_im016._output_depth]: /experiments/images/JNet_602_beads_roi014_im016._output_depth.png
[JNet_602_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_602_beads_roi014_im016._reconst_depth.png
[JNet_602_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_602_beads_roi015_im017._heatmap_depth.png
[JNet_602_beads_roi015_im017._original_depth]: /experiments/images/JNet_602_beads_roi015_im017._original_depth.png
[JNet_602_beads_roi015_im017._output_depth]: /experiments/images/JNet_602_beads_roi015_im017._output_depth.png
[JNet_602_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_602_beads_roi015_im017._reconst_depth.png
[JNet_602_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_602_beads_roi016_im018._heatmap_depth.png
[JNet_602_beads_roi016_im018._original_depth]: /experiments/images/JNet_602_beads_roi016_im018._original_depth.png
[JNet_602_beads_roi016_im018._output_depth]: /experiments/images/JNet_602_beads_roi016_im018._output_depth.png
[JNet_602_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_602_beads_roi016_im018._reconst_depth.png
[JNet_602_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_602_beads_roi017_im018._heatmap_depth.png
[JNet_602_beads_roi017_im018._original_depth]: /experiments/images/JNet_602_beads_roi017_im018._original_depth.png
[JNet_602_beads_roi017_im018._output_depth]: /experiments/images/JNet_602_beads_roi017_im018._output_depth.png
[JNet_602_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_602_beads_roi017_im018._reconst_depth.png
[JNet_602_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_602_beads_roi018_im022._heatmap_depth.png
[JNet_602_beads_roi018_im022._original_depth]: /experiments/images/JNet_602_beads_roi018_im022._original_depth.png
[JNet_602_beads_roi018_im022._output_depth]: /experiments/images/JNet_602_beads_roi018_im022._output_depth.png
[JNet_602_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_602_beads_roi018_im022._reconst_depth.png
[JNet_602_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_602_beads_roi019_im023._heatmap_depth.png
[JNet_602_beads_roi019_im023._original_depth]: /experiments/images/JNet_602_beads_roi019_im023._original_depth.png
[JNet_602_beads_roi019_im023._output_depth]: /experiments/images/JNet_602_beads_roi019_im023._output_depth.png
[JNet_602_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_602_beads_roi019_im023._reconst_depth.png
[JNet_602_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_602_beads_roi020_im024._heatmap_depth.png
[JNet_602_beads_roi020_im024._original_depth]: /experiments/images/JNet_602_beads_roi020_im024._original_depth.png
[JNet_602_beads_roi020_im024._output_depth]: /experiments/images/JNet_602_beads_roi020_im024._output_depth.png
[JNet_602_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_602_beads_roi020_im024._reconst_depth.png
[JNet_602_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_602_beads_roi021_im026._heatmap_depth.png
[JNet_602_beads_roi021_im026._original_depth]: /experiments/images/JNet_602_beads_roi021_im026._original_depth.png
[JNet_602_beads_roi021_im026._output_depth]: /experiments/images/JNet_602_beads_roi021_im026._output_depth.png
[JNet_602_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_602_beads_roi021_im026._reconst_depth.png
[JNet_602_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_602_beads_roi022_im027._heatmap_depth.png
[JNet_602_beads_roi022_im027._original_depth]: /experiments/images/JNet_602_beads_roi022_im027._original_depth.png
[JNet_602_beads_roi022_im027._output_depth]: /experiments/images/JNet_602_beads_roi022_im027._output_depth.png
[JNet_602_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_602_beads_roi022_im027._reconst_depth.png
[JNet_602_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_602_beads_roi023_im028._heatmap_depth.png
[JNet_602_beads_roi023_im028._original_depth]: /experiments/images/JNet_602_beads_roi023_im028._original_depth.png
[JNet_602_beads_roi023_im028._output_depth]: /experiments/images/JNet_602_beads_roi023_im028._output_depth.png
[JNet_602_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_602_beads_roi023_im028._reconst_depth.png
[JNet_602_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_602_beads_roi024_im028._heatmap_depth.png
[JNet_602_beads_roi024_im028._original_depth]: /experiments/images/JNet_602_beads_roi024_im028._original_depth.png
[JNet_602_beads_roi024_im028._output_depth]: /experiments/images/JNet_602_beads_roi024_im028._output_depth.png
[JNet_602_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_602_beads_roi024_im028._reconst_depth.png
[JNet_602_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_602_beads_roi025_im028._heatmap_depth.png
[JNet_602_beads_roi025_im028._original_depth]: /experiments/images/JNet_602_beads_roi025_im028._original_depth.png
[JNet_602_beads_roi025_im028._output_depth]: /experiments/images/JNet_602_beads_roi025_im028._output_depth.png
[JNet_602_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_602_beads_roi025_im028._reconst_depth.png
[JNet_602_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_602_beads_roi026_im029._heatmap_depth.png
[JNet_602_beads_roi026_im029._original_depth]: /experiments/images/JNet_602_beads_roi026_im029._original_depth.png
[JNet_602_beads_roi026_im029._output_depth]: /experiments/images/JNet_602_beads_roi026_im029._output_depth.png
[JNet_602_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_602_beads_roi026_im029._reconst_depth.png
[JNet_602_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_602_beads_roi027_im029._heatmap_depth.png
[JNet_602_beads_roi027_im029._original_depth]: /experiments/images/JNet_602_beads_roi027_im029._original_depth.png
[JNet_602_beads_roi027_im029._output_depth]: /experiments/images/JNet_602_beads_roi027_im029._output_depth.png
[JNet_602_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_602_beads_roi027_im029._reconst_depth.png
[JNet_602_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_602_beads_roi028_im030._heatmap_depth.png
[JNet_602_beads_roi028_im030._original_depth]: /experiments/images/JNet_602_beads_roi028_im030._original_depth.png
[JNet_602_beads_roi028_im030._output_depth]: /experiments/images/JNet_602_beads_roi028_im030._output_depth.png
[JNet_602_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_602_beads_roi028_im030._reconst_depth.png
[JNet_602_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_602_beads_roi029_im030._heatmap_depth.png
[JNet_602_beads_roi029_im030._original_depth]: /experiments/images/JNet_602_beads_roi029_im030._original_depth.png
[JNet_602_beads_roi029_im030._output_depth]: /experiments/images/JNet_602_beads_roi029_im030._output_depth.png
[JNet_602_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_602_beads_roi029_im030._reconst_depth.png
[JNet_602_psf_post]: /experiments/images/JNet_602_psf_post.png
[JNet_602_psf_pre]: /experiments/images/JNet_602_psf_pre.png
