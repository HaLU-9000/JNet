



# JNet_593 Report
  
psf loss 0.1 and ewc 0.1, adjust_luminance = false  
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
  
Segmentation: mean MSE: 0.008550615049898624, mean BCE: 0.031759120523929596  
Luminance Estimation: mean MSE: 0.9698507189750671, mean BCE: nan
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_0_original_plane]|![JNet_592_pretrain_0_novibrate_plane]|![JNet_592_pretrain_0_aligned_plane]|![JNet_592_pretrain_0_outputx_plane]|![JNet_592_pretrain_0_labelx_plane]|![JNet_592_pretrain_0_outputz_plane]|![JNet_592_pretrain_0_labelz_plane]|
  
MSEx: 0.004031396005302668, BCEx: 0.014457346871495247  
MSEz: 0.9848049283027649, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_0_original_depth]|![JNet_592_pretrain_0_novibrate_depth]|![JNet_592_pretrain_0_aligned_depth]|![JNet_592_pretrain_0_outputx_depth]|![JNet_592_pretrain_0_labelx_depth]|![JNet_592_pretrain_0_outputz_depth]|![JNet_592_pretrain_0_labelz_depth]|
  
MSEx: 0.004031396005302668, BCEx: 0.014457346871495247  
MSEz: 0.9848049283027649, BCEz: nan  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_1_original_plane]|![JNet_592_pretrain_1_novibrate_plane]|![JNet_592_pretrain_1_aligned_plane]|![JNet_592_pretrain_1_outputx_plane]|![JNet_592_pretrain_1_labelx_plane]|![JNet_592_pretrain_1_outputz_plane]|![JNet_592_pretrain_1_labelz_plane]|
  
MSEx: 0.008595339022576809, BCEx: 0.03053484484553337  
MSEz: 0.9551911354064941, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_1_original_depth]|![JNet_592_pretrain_1_novibrate_depth]|![JNet_592_pretrain_1_aligned_depth]|![JNet_592_pretrain_1_outputx_depth]|![JNet_592_pretrain_1_labelx_depth]|![JNet_592_pretrain_1_outputz_depth]|![JNet_592_pretrain_1_labelz_depth]|
  
MSEx: 0.008595339022576809, BCEx: 0.03053484484553337  
MSEz: 0.9551911354064941, BCEz: nan  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_2_original_plane]|![JNet_592_pretrain_2_novibrate_plane]|![JNet_592_pretrain_2_aligned_plane]|![JNet_592_pretrain_2_outputx_plane]|![JNet_592_pretrain_2_labelx_plane]|![JNet_592_pretrain_2_outputz_plane]|![JNet_592_pretrain_2_labelz_plane]|
  
MSEx: 0.0085286945104599, BCEx: 0.03221075236797333  
MSEz: 0.9695190191268921, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_2_original_depth]|![JNet_592_pretrain_2_novibrate_depth]|![JNet_592_pretrain_2_aligned_depth]|![JNet_592_pretrain_2_outputx_depth]|![JNet_592_pretrain_2_labelx_depth]|![JNet_592_pretrain_2_outputz_depth]|![JNet_592_pretrain_2_labelz_depth]|
  
MSEx: 0.0085286945104599, BCEx: 0.03221075236797333  
MSEz: 0.9695190191268921, BCEz: nan  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_3_original_plane]|![JNet_592_pretrain_3_novibrate_plane]|![JNet_592_pretrain_3_aligned_plane]|![JNet_592_pretrain_3_outputx_plane]|![JNet_592_pretrain_3_labelx_plane]|![JNet_592_pretrain_3_outputz_plane]|![JNet_592_pretrain_3_labelz_plane]|
  
MSEx: 0.008774391375482082, BCEx: 0.032752007246017456  
MSEz: 0.9707377552986145, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_3_original_depth]|![JNet_592_pretrain_3_novibrate_depth]|![JNet_592_pretrain_3_aligned_depth]|![JNet_592_pretrain_3_outputx_depth]|![JNet_592_pretrain_3_labelx_depth]|![JNet_592_pretrain_3_outputz_depth]|![JNet_592_pretrain_3_labelz_depth]|
  
MSEx: 0.008774391375482082, BCEx: 0.032752007246017456  
MSEz: 0.9707377552986145, BCEz: nan  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_4_original_plane]|![JNet_592_pretrain_4_novibrate_plane]|![JNet_592_pretrain_4_aligned_plane]|![JNet_592_pretrain_4_outputx_plane]|![JNet_592_pretrain_4_labelx_plane]|![JNet_592_pretrain_4_outputz_plane]|![JNet_592_pretrain_4_labelz_plane]|
  
MSEx: 0.012823252938687801, BCEx: 0.04884065315127373  
MSEz: 0.9690011739730835, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_4_original_depth]|![JNet_592_pretrain_4_novibrate_depth]|![JNet_592_pretrain_4_aligned_depth]|![JNet_592_pretrain_4_outputx_depth]|![JNet_592_pretrain_4_labelx_depth]|![JNet_592_pretrain_4_outputz_depth]|![JNet_592_pretrain_4_labelz_depth]|
  
MSEx: 0.012823252938687801, BCEx: 0.04884065315127373  
MSEz: 0.9690011739730835, BCEz: nan  

### pretrain
  
volume mean: 4.284562426757813, volume sd: 0.2573759425720666
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi000_im000._original_depth]|![JNet_592_pretrain_beads_roi000_im000._output_depth]|![JNet_592_pretrain_beads_roi000_im000._reconst_depth]|![JNet_592_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 3.9323112792968757, MSE: 0.0010889930417761207, quantized loss: 0.0003128394018858671  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi001_im004._original_depth]|![JNet_592_pretrain_beads_roi001_im004._output_depth]|![JNet_592_pretrain_beads_roi001_im004._reconst_depth]|![JNet_592_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 4.660659179687501, MSE: 0.0011491893092170358, quantized loss: 0.0003615067107602954  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi002_im005._original_depth]|![JNet_592_pretrain_beads_roi002_im005._output_depth]|![JNet_592_pretrain_beads_roi002_im005._reconst_depth]|![JNet_592_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 4.1042114257812505, MSE: 0.0010788943618535995, quantized loss: 0.0003170713025610894  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi003_im006._original_depth]|![JNet_592_pretrain_beads_roi003_im006._output_depth]|![JNet_592_pretrain_beads_roi003_im006._reconst_depth]|![JNet_592_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 4.240589355468751, MSE: 0.0011012180475518107, quantized loss: 0.0003514688287395984  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi004_im006._original_depth]|![JNet_592_pretrain_beads_roi004_im006._output_depth]|![JNet_592_pretrain_beads_roi004_im006._reconst_depth]|![JNet_592_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 4.294565429687501, MSE: 0.0011299648322165012, quantized loss: 0.0003540611651260406  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi005_im007._original_depth]|![JNet_592_pretrain_beads_roi005_im007._output_depth]|![JNet_592_pretrain_beads_roi005_im007._reconst_depth]|![JNet_592_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 4.165670898437501, MSE: 0.0011069749016314745, quantized loss: 0.00034225385752506554  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi006_im008._original_depth]|![JNet_592_pretrain_beads_roi006_im008._output_depth]|![JNet_592_pretrain_beads_roi006_im008._reconst_depth]|![JNet_592_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 4.418497070312501, MSE: 0.001046577817760408, quantized loss: 0.00038989598397165537  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi007_im009._original_depth]|![JNet_592_pretrain_beads_roi007_im009._output_depth]|![JNet_592_pretrain_beads_roi007_im009._reconst_depth]|![JNet_592_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 4.151570312500001, MSE: 0.0011017926735803485, quantized loss: 0.0003299988165963441  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi008_im010._original_depth]|![JNet_592_pretrain_beads_roi008_im010._output_depth]|![JNet_592_pretrain_beads_roi008_im010._reconst_depth]|![JNet_592_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 4.348276855468751, MSE: 0.0010695351520553231, quantized loss: 0.00035743237822316587  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi009_im011._original_depth]|![JNet_592_pretrain_beads_roi009_im011._output_depth]|![JNet_592_pretrain_beads_roi009_im011._reconst_depth]|![JNet_592_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 4.003686279296876, MSE: 0.001044733915477991, quantized loss: 0.0003282020043116063  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi010_im012._original_depth]|![JNet_592_pretrain_beads_roi010_im012._output_depth]|![JNet_592_pretrain_beads_roi010_im012._reconst_depth]|![JNet_592_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 4.7894096679687514, MSE: 0.0011387414997443557, quantized loss: 0.00037982664071023464  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi011_im013._original_depth]|![JNet_592_pretrain_beads_roi011_im013._output_depth]|![JNet_592_pretrain_beads_roi011_im013._reconst_depth]|![JNet_592_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 4.737977539062501, MSE: 0.0010998956859111786, quantized loss: 0.00036728245322592556  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi012_im014._original_depth]|![JNet_592_pretrain_beads_roi012_im014._output_depth]|![JNet_592_pretrain_beads_roi012_im014._reconst_depth]|![JNet_592_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 4.213318847656251, MSE: 0.0012117716250941157, quantized loss: 0.00034339146804995835  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi013_im015._original_depth]|![JNet_592_pretrain_beads_roi013_im015._output_depth]|![JNet_592_pretrain_beads_roi013_im015._reconst_depth]|![JNet_592_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 4.092673828125001, MSE: 0.0011589424684643745, quantized loss: 0.0003348365717101842  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi014_im016._original_depth]|![JNet_592_pretrain_beads_roi014_im016._output_depth]|![JNet_592_pretrain_beads_roi014_im016._reconst_depth]|![JNet_592_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 4.147944824218751, MSE: 0.001048172591254115, quantized loss: 0.0003831070789601654  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi015_im017._original_depth]|![JNet_592_pretrain_beads_roi015_im017._output_depth]|![JNet_592_pretrain_beads_roi015_im017._reconst_depth]|![JNet_592_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 3.998151855468751, MSE: 0.0010753074893727899, quantized loss: 0.0003450574295129627  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi016_im018._original_depth]|![JNet_592_pretrain_beads_roi016_im018._output_depth]|![JNet_592_pretrain_beads_roi016_im018._reconst_depth]|![JNet_592_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 4.370160644531251, MSE: 0.00122215470764786, quantized loss: 0.00034620356746017933  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi017_im018._original_depth]|![JNet_592_pretrain_beads_roi017_im018._output_depth]|![JNet_592_pretrain_beads_roi017_im018._reconst_depth]|![JNet_592_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 4.287719238281251, MSE: 0.0012472454691305757, quantized loss: 0.0003350062179379165  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi018_im022._original_depth]|![JNet_592_pretrain_beads_roi018_im022._output_depth]|![JNet_592_pretrain_beads_roi018_im022._reconst_depth]|![JNet_592_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 3.8888891601562507, MSE: 0.0010792647954076529, quantized loss: 0.0003358496178407222  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi019_im023._original_depth]|![JNet_592_pretrain_beads_roi019_im023._output_depth]|![JNet_592_pretrain_beads_roi019_im023._reconst_depth]|![JNet_592_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 3.7884753417968757, MSE: 0.00109770055860281, quantized loss: 0.00032613513758406043  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi020_im024._original_depth]|![JNet_592_pretrain_beads_roi020_im024._output_depth]|![JNet_592_pretrain_beads_roi020_im024._reconst_depth]|![JNet_592_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 4.495268066406251, MSE: 0.0011242501204833388, quantized loss: 0.0003496900317259133  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi021_im026._original_depth]|![JNet_592_pretrain_beads_roi021_im026._output_depth]|![JNet_592_pretrain_beads_roi021_im026._reconst_depth]|![JNet_592_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 4.309168945312501, MSE: 0.0010281483409926295, quantized loss: 0.0003438495332375169  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi022_im027._original_depth]|![JNet_592_pretrain_beads_roi022_im027._output_depth]|![JNet_592_pretrain_beads_roi022_im027._reconst_depth]|![JNet_592_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 4.121901855468751, MSE: 0.0010948582785204053, quantized loss: 0.00033353554317727685  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi023_im028._original_depth]|![JNet_592_pretrain_beads_roi023_im028._output_depth]|![JNet_592_pretrain_beads_roi023_im028._reconst_depth]|![JNet_592_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 4.691045410156251, MSE: 0.0009226580150425434, quantized loss: 0.0004076574696227908  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi024_im028._original_depth]|![JNet_592_pretrain_beads_roi024_im028._output_depth]|![JNet_592_pretrain_beads_roi024_im028._reconst_depth]|![JNet_592_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 4.594440429687501, MSE: 0.0009931474924087524, quantized loss: 0.0003746850125025958  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi025_im028._original_depth]|![JNet_592_pretrain_beads_roi025_im028._output_depth]|![JNet_592_pretrain_beads_roi025_im028._reconst_depth]|![JNet_592_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 4.594440429687501, MSE: 0.0009931474924087524, quantized loss: 0.0003746850125025958  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi026_im029._original_depth]|![JNet_592_pretrain_beads_roi026_im029._output_depth]|![JNet_592_pretrain_beads_roi026_im029._reconst_depth]|![JNet_592_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 4.534850097656251, MSE: 0.001128699746914208, quantized loss: 0.0003621990035753697  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi027_im029._original_depth]|![JNet_592_pretrain_beads_roi027_im029._output_depth]|![JNet_592_pretrain_beads_roi027_im029._reconst_depth]|![JNet_592_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 4.261646484375001, MSE: 0.0011243581539019942, quantized loss: 0.0003573574358597398  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi028_im030._original_depth]|![JNet_592_pretrain_beads_roi028_im030._output_depth]|![JNet_592_pretrain_beads_roi028_im030._reconst_depth]|![JNet_592_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 4.068653808593751, MSE: 0.0010836278088390827, quantized loss: 0.00033172505209222436  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_592_pretrain_beads_roi029_im030._original_depth]|![JNet_592_pretrain_beads_roi029_im030._output_depth]|![JNet_592_pretrain_beads_roi029_im030._reconst_depth]|![JNet_592_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 4.230698242187501, MSE: 0.001145395333878696, quantized loss: 0.00033403458655811846  

### finetuning
  
volume mean: 5.143838102213542, volume sd: 0.2487191565189581
### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi000_im000._original_depth]|![JNet_593_beads_roi000_im000._output_depth]|![JNet_593_beads_roi000_im000._reconst_depth]|![JNet_593_beads_roi000_im000._heatmap_depth]|
  
volume: 4.934069335937501, MSE: 0.0010712180519476533, quantized loss: 0.0003753382188733667  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi001_im004._original_depth]|![JNet_593_beads_roi001_im004._output_depth]|![JNet_593_beads_roi001_im004._reconst_depth]|![JNet_593_beads_roi001_im004._heatmap_depth]|
  
volume: 5.616886718750001, MSE: 0.001447469461709261, quantized loss: 0.00045579028665088117  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi002_im005._original_depth]|![JNet_593_beads_roi002_im005._output_depth]|![JNet_593_beads_roi002_im005._reconst_depth]|![JNet_593_beads_roi002_im005._heatmap_depth]|
  
volume: 5.170708007812501, MSE: 0.0012290551094338298, quantized loss: 0.00040877709398046136  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi003_im006._original_depth]|![JNet_593_beads_roi003_im006._output_depth]|![JNet_593_beads_roi003_im006._reconst_depth]|![JNet_593_beads_roi003_im006._heatmap_depth]|
  
volume: 5.150454101562501, MSE: 0.0012601219350472093, quantized loss: 0.00041080688242800534  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi004_im006._original_depth]|![JNet_593_beads_roi004_im006._output_depth]|![JNet_593_beads_roi004_im006._reconst_depth]|![JNet_593_beads_roi004_im006._heatmap_depth]|
  
volume: 5.247751464843751, MSE: 0.001329264254309237, quantized loss: 0.00042666110675781965  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi005_im007._original_depth]|![JNet_593_beads_roi005_im007._output_depth]|![JNet_593_beads_roi005_im007._reconst_depth]|![JNet_593_beads_roi005_im007._heatmap_depth]|
  
volume: 5.148532714843752, MSE: 0.0012768401065841317, quantized loss: 0.00042524735908955336  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi006_im008._original_depth]|![JNet_593_beads_roi006_im008._output_depth]|![JNet_593_beads_roi006_im008._reconst_depth]|![JNet_593_beads_roi006_im008._heatmap_depth]|
  
volume: 5.346138183593752, MSE: 0.0012306177522987127, quantized loss: 0.0005215356941334903  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi007_im009._original_depth]|![JNet_593_beads_roi007_im009._output_depth]|![JNet_593_beads_roi007_im009._reconst_depth]|![JNet_593_beads_roi007_im009._heatmap_depth]|
  
volume: 5.2599604492187515, MSE: 0.0011998707195743918, quantized loss: 0.0004441401979420334  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi008_im010._original_depth]|![JNet_593_beads_roi008_im010._output_depth]|![JNet_593_beads_roi008_im010._reconst_depth]|![JNet_593_beads_roi008_im010._heatmap_depth]|
  
volume: 5.227567382812501, MSE: 0.0013467654353007674, quantized loss: 0.00041258393321186304  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi009_im011._original_depth]|![JNet_593_beads_roi009_im011._output_depth]|![JNet_593_beads_roi009_im011._reconst_depth]|![JNet_593_beads_roi009_im011._heatmap_depth]|
  
volume: 5.036693359375001, MSE: 0.0011425259290263057, quantized loss: 0.0003953904379159212  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi010_im012._original_depth]|![JNet_593_beads_roi010_im012._output_depth]|![JNet_593_beads_roi010_im012._reconst_depth]|![JNet_593_beads_roi010_im012._heatmap_depth]|
  
volume: 5.599746093750001, MSE: 0.0015305783599615097, quantized loss: 0.00045556199620477855  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi011_im013._original_depth]|![JNet_593_beads_roi011_im013._output_depth]|![JNet_593_beads_roi011_im013._reconst_depth]|![JNet_593_beads_roi011_im013._heatmap_depth]|
  
volume: 5.568319335937502, MSE: 0.0014243394834920764, quantized loss: 0.00045211490942165256  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi012_im014._original_depth]|![JNet_593_beads_roi012_im014._output_depth]|![JNet_593_beads_roi012_im014._reconst_depth]|![JNet_593_beads_roi012_im014._heatmap_depth]|
  
volume: 5.100245117187502, MSE: 0.0012753214687108994, quantized loss: 0.0003880271688103676  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi013_im015._original_depth]|![JNet_593_beads_roi013_im015._output_depth]|![JNet_593_beads_roi013_im015._reconst_depth]|![JNet_593_beads_roi013_im015._heatmap_depth]|
  
volume: 4.957895996093751, MSE: 0.0012054967228323221, quantized loss: 0.0003875406982842833  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi014_im016._original_depth]|![JNet_593_beads_roi014_im016._output_depth]|![JNet_593_beads_roi014_im016._reconst_depth]|![JNet_593_beads_roi014_im016._heatmap_depth]|
  
volume: 4.994443359375001, MSE: 0.0012396785896271467, quantized loss: 0.000425962294684723  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi015_im017._original_depth]|![JNet_593_beads_roi015_im017._output_depth]|![JNet_593_beads_roi015_im017._reconst_depth]|![JNet_593_beads_roi015_im017._heatmap_depth]|
  
volume: 4.930644531250001, MSE: 0.0011444068513810635, quantized loss: 0.0003879473952110857  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi016_im018._original_depth]|![JNet_593_beads_roi016_im018._output_depth]|![JNet_593_beads_roi016_im018._reconst_depth]|![JNet_593_beads_roi016_im018._heatmap_depth]|
  
volume: 5.405389160156251, MSE: 0.0013868125388398767, quantized loss: 0.0004440954071469605  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi017_im018._original_depth]|![JNet_593_beads_roi017_im018._output_depth]|![JNet_593_beads_roi017_im018._reconst_depth]|![JNet_593_beads_roi017_im018._heatmap_depth]|
  
volume: 5.2827397460937515, MSE: 0.0013570976443588734, quantized loss: 0.00043953690328635275  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi018_im022._original_depth]|![JNet_593_beads_roi018_im022._output_depth]|![JNet_593_beads_roi018_im022._reconst_depth]|![JNet_593_beads_roi018_im022._heatmap_depth]|
  
volume: 4.633979980468751, MSE: 0.0010735181858763099, quantized loss: 0.000363058497896418  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi019_im023._original_depth]|![JNet_593_beads_roi019_im023._output_depth]|![JNet_593_beads_roi019_im023._reconst_depth]|![JNet_593_beads_roi019_im023._heatmap_depth]|
  
volume: 4.561951171875001, MSE: 0.0010218617971986532, quantized loss: 0.0003548610839061439  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi020_im024._original_depth]|![JNet_593_beads_roi020_im024._output_depth]|![JNet_593_beads_roi020_im024._reconst_depth]|![JNet_593_beads_roi020_im024._heatmap_depth]|
  
volume: 5.210588867187501, MSE: 0.0013392200926318765, quantized loss: 0.00040014556725509465  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi021_im026._original_depth]|![JNet_593_beads_roi021_im026._output_depth]|![JNet_593_beads_roi021_im026._reconst_depth]|![JNet_593_beads_roi021_im026._heatmap_depth]|
  
volume: 5.078545410156251, MSE: 0.001259022275917232, quantized loss: 0.00039163147448562086  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi022_im027._original_depth]|![JNet_593_beads_roi022_im027._output_depth]|![JNet_593_beads_roi022_im027._reconst_depth]|![JNet_593_beads_roi022_im027._heatmap_depth]|
  
volume: 4.891276367187501, MSE: 0.0012299000518396497, quantized loss: 0.00037034705746918917  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi023_im028._original_depth]|![JNet_593_beads_roi023_im028._output_depth]|![JNet_593_beads_roi023_im028._reconst_depth]|![JNet_593_beads_roi023_im028._heatmap_depth]|
  
volume: 5.339249023437501, MSE: 0.0013692621141672134, quantized loss: 0.0004530038859229535  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi024_im028._original_depth]|![JNet_593_beads_roi024_im028._output_depth]|![JNet_593_beads_roi024_im028._reconst_depth]|![JNet_593_beads_roi024_im028._heatmap_depth]|
  
volume: 5.252881835937501, MSE: 0.0013608969748020172, quantized loss: 0.0004230838385410607  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi025_im028._original_depth]|![JNet_593_beads_roi025_im028._output_depth]|![JNet_593_beads_roi025_im028._reconst_depth]|![JNet_593_beads_roi025_im028._heatmap_depth]|
  
volume: 5.252881835937501, MSE: 0.0013608969748020172, quantized loss: 0.0004230838385410607  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi026_im029._original_depth]|![JNet_593_beads_roi026_im029._output_depth]|![JNet_593_beads_roi026_im029._reconst_depth]|![JNet_593_beads_roi026_im029._heatmap_depth]|
  
volume: 5.339205078125001, MSE: 0.0014162897132337093, quantized loss: 0.00041180578409694135  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi027_im029._original_depth]|![JNet_593_beads_roi027_im029._output_depth]|![JNet_593_beads_roi027_im029._reconst_depth]|![JNet_593_beads_roi027_im029._heatmap_depth]|
  
volume: 4.909291015625001, MSE: 0.001317028421908617, quantized loss: 0.0003958142187912017  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi028_im030._original_depth]|![JNet_593_beads_roi028_im030._output_depth]|![JNet_593_beads_roi028_im030._reconst_depth]|![JNet_593_beads_roi028_im030._heatmap_depth]|
  
volume: 4.866664550781251, MSE: 0.001161152613349259, quantized loss: 0.0003866964252665639  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_593_beads_roi029_im030._original_depth]|![JNet_593_beads_roi029_im030._output_depth]|![JNet_593_beads_roi029_im030._reconst_depth]|![JNet_593_beads_roi029_im030._heatmap_depth]|
  
volume: 5.000442871093751, MSE: 0.001238113734871149, quantized loss: 0.0003862647572532296  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_593_psf_pre]|![JNet_593_psf_post]|

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
[JNet_593_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_593_beads_roi000_im000._heatmap_depth.png
[JNet_593_beads_roi000_im000._original_depth]: /experiments/images/JNet_593_beads_roi000_im000._original_depth.png
[JNet_593_beads_roi000_im000._output_depth]: /experiments/images/JNet_593_beads_roi000_im000._output_depth.png
[JNet_593_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_593_beads_roi000_im000._reconst_depth.png
[JNet_593_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_593_beads_roi001_im004._heatmap_depth.png
[JNet_593_beads_roi001_im004._original_depth]: /experiments/images/JNet_593_beads_roi001_im004._original_depth.png
[JNet_593_beads_roi001_im004._output_depth]: /experiments/images/JNet_593_beads_roi001_im004._output_depth.png
[JNet_593_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_593_beads_roi001_im004._reconst_depth.png
[JNet_593_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_593_beads_roi002_im005._heatmap_depth.png
[JNet_593_beads_roi002_im005._original_depth]: /experiments/images/JNet_593_beads_roi002_im005._original_depth.png
[JNet_593_beads_roi002_im005._output_depth]: /experiments/images/JNet_593_beads_roi002_im005._output_depth.png
[JNet_593_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_593_beads_roi002_im005._reconst_depth.png
[JNet_593_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_593_beads_roi003_im006._heatmap_depth.png
[JNet_593_beads_roi003_im006._original_depth]: /experiments/images/JNet_593_beads_roi003_im006._original_depth.png
[JNet_593_beads_roi003_im006._output_depth]: /experiments/images/JNet_593_beads_roi003_im006._output_depth.png
[JNet_593_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_593_beads_roi003_im006._reconst_depth.png
[JNet_593_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_593_beads_roi004_im006._heatmap_depth.png
[JNet_593_beads_roi004_im006._original_depth]: /experiments/images/JNet_593_beads_roi004_im006._original_depth.png
[JNet_593_beads_roi004_im006._output_depth]: /experiments/images/JNet_593_beads_roi004_im006._output_depth.png
[JNet_593_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_593_beads_roi004_im006._reconst_depth.png
[JNet_593_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_593_beads_roi005_im007._heatmap_depth.png
[JNet_593_beads_roi005_im007._original_depth]: /experiments/images/JNet_593_beads_roi005_im007._original_depth.png
[JNet_593_beads_roi005_im007._output_depth]: /experiments/images/JNet_593_beads_roi005_im007._output_depth.png
[JNet_593_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_593_beads_roi005_im007._reconst_depth.png
[JNet_593_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_593_beads_roi006_im008._heatmap_depth.png
[JNet_593_beads_roi006_im008._original_depth]: /experiments/images/JNet_593_beads_roi006_im008._original_depth.png
[JNet_593_beads_roi006_im008._output_depth]: /experiments/images/JNet_593_beads_roi006_im008._output_depth.png
[JNet_593_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_593_beads_roi006_im008._reconst_depth.png
[JNet_593_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_593_beads_roi007_im009._heatmap_depth.png
[JNet_593_beads_roi007_im009._original_depth]: /experiments/images/JNet_593_beads_roi007_im009._original_depth.png
[JNet_593_beads_roi007_im009._output_depth]: /experiments/images/JNet_593_beads_roi007_im009._output_depth.png
[JNet_593_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_593_beads_roi007_im009._reconst_depth.png
[JNet_593_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_593_beads_roi008_im010._heatmap_depth.png
[JNet_593_beads_roi008_im010._original_depth]: /experiments/images/JNet_593_beads_roi008_im010._original_depth.png
[JNet_593_beads_roi008_im010._output_depth]: /experiments/images/JNet_593_beads_roi008_im010._output_depth.png
[JNet_593_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_593_beads_roi008_im010._reconst_depth.png
[JNet_593_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_593_beads_roi009_im011._heatmap_depth.png
[JNet_593_beads_roi009_im011._original_depth]: /experiments/images/JNet_593_beads_roi009_im011._original_depth.png
[JNet_593_beads_roi009_im011._output_depth]: /experiments/images/JNet_593_beads_roi009_im011._output_depth.png
[JNet_593_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_593_beads_roi009_im011._reconst_depth.png
[JNet_593_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_593_beads_roi010_im012._heatmap_depth.png
[JNet_593_beads_roi010_im012._original_depth]: /experiments/images/JNet_593_beads_roi010_im012._original_depth.png
[JNet_593_beads_roi010_im012._output_depth]: /experiments/images/JNet_593_beads_roi010_im012._output_depth.png
[JNet_593_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_593_beads_roi010_im012._reconst_depth.png
[JNet_593_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_593_beads_roi011_im013._heatmap_depth.png
[JNet_593_beads_roi011_im013._original_depth]: /experiments/images/JNet_593_beads_roi011_im013._original_depth.png
[JNet_593_beads_roi011_im013._output_depth]: /experiments/images/JNet_593_beads_roi011_im013._output_depth.png
[JNet_593_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_593_beads_roi011_im013._reconst_depth.png
[JNet_593_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_593_beads_roi012_im014._heatmap_depth.png
[JNet_593_beads_roi012_im014._original_depth]: /experiments/images/JNet_593_beads_roi012_im014._original_depth.png
[JNet_593_beads_roi012_im014._output_depth]: /experiments/images/JNet_593_beads_roi012_im014._output_depth.png
[JNet_593_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_593_beads_roi012_im014._reconst_depth.png
[JNet_593_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_593_beads_roi013_im015._heatmap_depth.png
[JNet_593_beads_roi013_im015._original_depth]: /experiments/images/JNet_593_beads_roi013_im015._original_depth.png
[JNet_593_beads_roi013_im015._output_depth]: /experiments/images/JNet_593_beads_roi013_im015._output_depth.png
[JNet_593_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_593_beads_roi013_im015._reconst_depth.png
[JNet_593_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_593_beads_roi014_im016._heatmap_depth.png
[JNet_593_beads_roi014_im016._original_depth]: /experiments/images/JNet_593_beads_roi014_im016._original_depth.png
[JNet_593_beads_roi014_im016._output_depth]: /experiments/images/JNet_593_beads_roi014_im016._output_depth.png
[JNet_593_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_593_beads_roi014_im016._reconst_depth.png
[JNet_593_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_593_beads_roi015_im017._heatmap_depth.png
[JNet_593_beads_roi015_im017._original_depth]: /experiments/images/JNet_593_beads_roi015_im017._original_depth.png
[JNet_593_beads_roi015_im017._output_depth]: /experiments/images/JNet_593_beads_roi015_im017._output_depth.png
[JNet_593_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_593_beads_roi015_im017._reconst_depth.png
[JNet_593_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_593_beads_roi016_im018._heatmap_depth.png
[JNet_593_beads_roi016_im018._original_depth]: /experiments/images/JNet_593_beads_roi016_im018._original_depth.png
[JNet_593_beads_roi016_im018._output_depth]: /experiments/images/JNet_593_beads_roi016_im018._output_depth.png
[JNet_593_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_593_beads_roi016_im018._reconst_depth.png
[JNet_593_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_593_beads_roi017_im018._heatmap_depth.png
[JNet_593_beads_roi017_im018._original_depth]: /experiments/images/JNet_593_beads_roi017_im018._original_depth.png
[JNet_593_beads_roi017_im018._output_depth]: /experiments/images/JNet_593_beads_roi017_im018._output_depth.png
[JNet_593_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_593_beads_roi017_im018._reconst_depth.png
[JNet_593_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_593_beads_roi018_im022._heatmap_depth.png
[JNet_593_beads_roi018_im022._original_depth]: /experiments/images/JNet_593_beads_roi018_im022._original_depth.png
[JNet_593_beads_roi018_im022._output_depth]: /experiments/images/JNet_593_beads_roi018_im022._output_depth.png
[JNet_593_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_593_beads_roi018_im022._reconst_depth.png
[JNet_593_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_593_beads_roi019_im023._heatmap_depth.png
[JNet_593_beads_roi019_im023._original_depth]: /experiments/images/JNet_593_beads_roi019_im023._original_depth.png
[JNet_593_beads_roi019_im023._output_depth]: /experiments/images/JNet_593_beads_roi019_im023._output_depth.png
[JNet_593_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_593_beads_roi019_im023._reconst_depth.png
[JNet_593_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_593_beads_roi020_im024._heatmap_depth.png
[JNet_593_beads_roi020_im024._original_depth]: /experiments/images/JNet_593_beads_roi020_im024._original_depth.png
[JNet_593_beads_roi020_im024._output_depth]: /experiments/images/JNet_593_beads_roi020_im024._output_depth.png
[JNet_593_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_593_beads_roi020_im024._reconst_depth.png
[JNet_593_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_593_beads_roi021_im026._heatmap_depth.png
[JNet_593_beads_roi021_im026._original_depth]: /experiments/images/JNet_593_beads_roi021_im026._original_depth.png
[JNet_593_beads_roi021_im026._output_depth]: /experiments/images/JNet_593_beads_roi021_im026._output_depth.png
[JNet_593_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_593_beads_roi021_im026._reconst_depth.png
[JNet_593_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_593_beads_roi022_im027._heatmap_depth.png
[JNet_593_beads_roi022_im027._original_depth]: /experiments/images/JNet_593_beads_roi022_im027._original_depth.png
[JNet_593_beads_roi022_im027._output_depth]: /experiments/images/JNet_593_beads_roi022_im027._output_depth.png
[JNet_593_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_593_beads_roi022_im027._reconst_depth.png
[JNet_593_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_593_beads_roi023_im028._heatmap_depth.png
[JNet_593_beads_roi023_im028._original_depth]: /experiments/images/JNet_593_beads_roi023_im028._original_depth.png
[JNet_593_beads_roi023_im028._output_depth]: /experiments/images/JNet_593_beads_roi023_im028._output_depth.png
[JNet_593_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_593_beads_roi023_im028._reconst_depth.png
[JNet_593_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_593_beads_roi024_im028._heatmap_depth.png
[JNet_593_beads_roi024_im028._original_depth]: /experiments/images/JNet_593_beads_roi024_im028._original_depth.png
[JNet_593_beads_roi024_im028._output_depth]: /experiments/images/JNet_593_beads_roi024_im028._output_depth.png
[JNet_593_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_593_beads_roi024_im028._reconst_depth.png
[JNet_593_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_593_beads_roi025_im028._heatmap_depth.png
[JNet_593_beads_roi025_im028._original_depth]: /experiments/images/JNet_593_beads_roi025_im028._original_depth.png
[JNet_593_beads_roi025_im028._output_depth]: /experiments/images/JNet_593_beads_roi025_im028._output_depth.png
[JNet_593_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_593_beads_roi025_im028._reconst_depth.png
[JNet_593_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_593_beads_roi026_im029._heatmap_depth.png
[JNet_593_beads_roi026_im029._original_depth]: /experiments/images/JNet_593_beads_roi026_im029._original_depth.png
[JNet_593_beads_roi026_im029._output_depth]: /experiments/images/JNet_593_beads_roi026_im029._output_depth.png
[JNet_593_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_593_beads_roi026_im029._reconst_depth.png
[JNet_593_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_593_beads_roi027_im029._heatmap_depth.png
[JNet_593_beads_roi027_im029._original_depth]: /experiments/images/JNet_593_beads_roi027_im029._original_depth.png
[JNet_593_beads_roi027_im029._output_depth]: /experiments/images/JNet_593_beads_roi027_im029._output_depth.png
[JNet_593_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_593_beads_roi027_im029._reconst_depth.png
[JNet_593_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_593_beads_roi028_im030._heatmap_depth.png
[JNet_593_beads_roi028_im030._original_depth]: /experiments/images/JNet_593_beads_roi028_im030._original_depth.png
[JNet_593_beads_roi028_im030._output_depth]: /experiments/images/JNet_593_beads_roi028_im030._output_depth.png
[JNet_593_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_593_beads_roi028_im030._reconst_depth.png
[JNet_593_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_593_beads_roi029_im030._heatmap_depth.png
[JNet_593_beads_roi029_im030._original_depth]: /experiments/images/JNet_593_beads_roi029_im030._original_depth.png
[JNet_593_beads_roi029_im030._output_depth]: /experiments/images/JNet_593_beads_roi029_im030._output_depth.png
[JNet_593_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_593_beads_roi029_im030._reconst_depth.png
[JNet_593_psf_post]: /experiments/images/JNet_593_psf_post.png
[JNet_593_psf_pre]: /experiments/images/JNet_593_psf_pre.png
