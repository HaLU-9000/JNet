



# JNet_470 Report
  
new data generation with more objects. axon deconv  
pretrained model : JNet_469_pretrain
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
|res_lateral|0.16|microns|
|res_axial|1.0|microns|
|pZ|0|microns, particle distance from coverslip|
|bet_z|30.0||
|bet_xy|3.0||
|sig_eps|0.01||
|background|0.01||
|scale|10||
|mid|20|num of NeurIPSF middle channel|
|loss_fn|nn.MSELoss()|loss func for NeurIPSF|
|lr|0.01|lr for pre-training NeurIPSF|
|num_iter_psf_pretrain|1000|epoch for pre-training of NeurIPSF|
|device|cuda||

## Datasets and other training details

### simulation_data_generation

|Parameter|Value|
| :--- | :--- |
|dataset_name|_var_num_realisticdataset|
|train_num|16|
|valid_num|4|
|image_size|[1200, 500, 500]|
|train_object_num_min|2000|
|train_object_num_max|18000|
|valid_object_num_min|6000|
|valid_object_num_max|10000|

### pretrain_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_var_num_realisticdata|
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
|folderpath|_var_num_realisticdata|
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
|seed|1204|

### pretrain_loop

|Parameter|Value|
| :--- | :--- |
|batch_size|1|
|n_epochs|200|
|lr|0.001|
|loss_fn|nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=params['device']))|
|path|model|
|savefig_path|train|
|partial|params['partial']|
|ewc|None|
|params|params|
|es_patience|10|
|reconstruct|False|
|is_instantblur|True|
|is_vibrate|True|
|loss_weight|1|
|qloss_weight|0|
|ploss_weight|0|

### train_loop

|Parameter|Value|
| :--- | :--- |
|batch_size|1|
|n_epochs|200|
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
|is_vibrate|True|
|loss_weight|1|
|ewc_weight|100000|
|qloss_weight|1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.014057539403438568, mean BCE: 0.059262972325086594
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_0_original_plane]|![JNet_469_pretrain_0_output_plane]|![JNet_469_pretrain_0_label_plane]|
  
MSE: 0.014804385602474213, BCE: 0.06079857051372528  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_0_original_depth]|![JNet_469_pretrain_0_output_depth]|![JNet_469_pretrain_0_label_depth]|
  
MSE: 0.014804385602474213, BCE: 0.06079857051372528  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_1_original_plane]|![JNet_469_pretrain_1_output_plane]|![JNet_469_pretrain_1_label_plane]|
  
MSE: 0.012289234437048435, BCE: 0.05409935489296913  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_1_original_depth]|![JNet_469_pretrain_1_output_depth]|![JNet_469_pretrain_1_label_depth]|
  
MSE: 0.012289234437048435, BCE: 0.05409935489296913  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_2_original_plane]|![JNet_469_pretrain_2_output_plane]|![JNet_469_pretrain_2_label_plane]|
  
MSE: 0.015124142169952393, BCE: 0.06463007628917694  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_2_original_depth]|![JNet_469_pretrain_2_output_depth]|![JNet_469_pretrain_2_label_depth]|
  
MSE: 0.015124142169952393, BCE: 0.06463007628917694  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_3_original_plane]|![JNet_469_pretrain_3_output_plane]|![JNet_469_pretrain_3_label_plane]|
  
MSE: 0.012341792695224285, BCE: 0.053482118993997574  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_3_original_depth]|![JNet_469_pretrain_3_output_depth]|![JNet_469_pretrain_3_label_depth]|
  
MSE: 0.012341792695224285, BCE: 0.053482118993997574  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_4_original_plane]|![JNet_469_pretrain_4_output_plane]|![JNet_469_pretrain_4_label_plane]|
  
MSE: 0.015728140249848366, BCE: 0.06330475211143494  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_4_original_depth]|![JNet_469_pretrain_4_output_depth]|![JNet_469_pretrain_4_label_depth]|
  
MSE: 0.015728140249848366, BCE: 0.06330475211143494  
  
mean MSE: 0.021558642387390137, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_0_original_plane]|![JNet_470_0_output_plane]|![JNet_470_0_label_plane]|
  
MSE: 0.027724947780370712, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_0_original_depth]|![JNet_470_0_output_depth]|![JNet_470_0_label_depth]|
  
MSE: 0.027724947780370712, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_1_original_plane]|![JNet_470_1_output_plane]|![JNet_470_1_label_plane]|
  
MSE: 0.026876434683799744, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_1_original_depth]|![JNet_470_1_output_depth]|![JNet_470_1_label_depth]|
  
MSE: 0.026876434683799744, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_2_original_plane]|![JNet_470_2_output_plane]|![JNet_470_2_label_plane]|
  
MSE: 0.015941884368658066, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_2_original_depth]|![JNet_470_2_output_depth]|![JNet_470_2_label_depth]|
  
MSE: 0.015941884368658066, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_3_original_plane]|![JNet_470_3_output_plane]|![JNet_470_3_label_plane]|
  
MSE: 0.015538305044174194, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_3_original_depth]|![JNet_470_3_output_depth]|![JNet_470_3_label_depth]|
  
MSE: 0.015538305044174194, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_4_original_plane]|![JNet_470_4_output_plane]|![JNet_470_4_label_plane]|
  
MSE: 0.021711641922593117, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_4_original_depth]|![JNet_470_4_output_depth]|![JNet_470_4_label_depth]|
  
MSE: 0.021711641922593117, BCE: nan  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi000_im000._original_depth]|![JNet_469_pretrain_beads_roi000_im000._output_depth]|![JNet_469_pretrain_beads_roi000_im000._reconst_depth]|![JNet_469_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 750.2831360000001, MSE: 0.0011550920316949487, quantized loss: 0.0026492357719689608  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi001_im004._original_depth]|![JNet_469_pretrain_beads_roi001_im004._output_depth]|![JNet_469_pretrain_beads_roi001_im004._reconst_depth]|![JNet_469_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 843.3008640000002, MSE: 0.0012432447401806712, quantized loss: 0.002918704180046916  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi002_im005._original_depth]|![JNet_469_pretrain_beads_roi002_im005._output_depth]|![JNet_469_pretrain_beads_roi002_im005._reconst_depth]|![JNet_469_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 783.9946880000001, MSE: 0.0011396988993510604, quantized loss: 0.002764116507023573  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi003_im006._original_depth]|![JNet_469_pretrain_beads_roi003_im006._output_depth]|![JNet_469_pretrain_beads_roi003_im006._reconst_depth]|![JNet_469_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 782.8827520000001, MSE: 0.001157391699962318, quantized loss: 0.0027254170272499323  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi004_im006._original_depth]|![JNet_469_pretrain_beads_roi004_im006._output_depth]|![JNet_469_pretrain_beads_roi004_im006._reconst_depth]|![JNet_469_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 804.8228480000001, MSE: 0.0011790214339271188, quantized loss: 0.0027681482024490833  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi005_im007._original_depth]|![JNet_469_pretrain_beads_roi005_im007._output_depth]|![JNet_469_pretrain_beads_roi005_im007._reconst_depth]|![JNet_469_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 792.9351680000001, MSE: 0.0011742005590349436, quantized loss: 0.002781071001663804  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi006_im008._original_depth]|![JNet_469_pretrain_beads_roi006_im008._output_depth]|![JNet_469_pretrain_beads_roi006_im008._reconst_depth]|![JNet_469_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 802.6817280000001, MSE: 0.001065785763785243, quantized loss: 0.0029032158199697733  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi007_im009._original_depth]|![JNet_469_pretrain_beads_roi007_im009._output_depth]|![JNet_469_pretrain_beads_roi007_im009._reconst_depth]|![JNet_469_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 819.1414400000001, MSE: 0.001135848113335669, quantized loss: 0.0029667806811630726  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi008_im010._original_depth]|![JNet_469_pretrain_beads_roi008_im010._output_depth]|![JNet_469_pretrain_beads_roi008_im010._reconst_depth]|![JNet_469_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 824.7531520000001, MSE: 0.0011743841459974647, quantized loss: 0.0028548038098961115  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi009_im011._original_depth]|![JNet_469_pretrain_beads_roi009_im011._output_depth]|![JNet_469_pretrain_beads_roi009_im011._reconst_depth]|![JNet_469_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 769.8781440000001, MSE: 0.001107916352339089, quantized loss: 0.0027254014275968075  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi010_im012._original_depth]|![JNet_469_pretrain_beads_roi010_im012._output_depth]|![JNet_469_pretrain_beads_roi010_im012._reconst_depth]|![JNet_469_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 858.6085760000002, MSE: 0.0012626483803614974, quantized loss: 0.0030339043587446213  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi011_im013._original_depth]|![JNet_469_pretrain_beads_roi011_im013._output_depth]|![JNet_469_pretrain_beads_roi011_im013._reconst_depth]|![JNet_469_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 838.4322560000002, MSE: 0.001199158257804811, quantized loss: 0.002921337028965354  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi012_im014._original_depth]|![JNet_469_pretrain_beads_roi012_im014._output_depth]|![JNet_469_pretrain_beads_roi012_im014._reconst_depth]|![JNet_469_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 781.9468160000001, MSE: 0.0013002714840695262, quantized loss: 0.0026893496979027987  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi013_im015._original_depth]|![JNet_469_pretrain_beads_roi013_im015._output_depth]|![JNet_469_pretrain_beads_roi013_im015._reconst_depth]|![JNet_469_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 751.8110720000001, MSE: 0.001246912288479507, quantized loss: 0.002543593989685178  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi014_im016._original_depth]|![JNet_469_pretrain_beads_roi014_im016._output_depth]|![JNet_469_pretrain_beads_roi014_im016._reconst_depth]|![JNet_469_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 758.6160000000001, MSE: 0.0010736059630289674, quantized loss: 0.00270954892039299  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi015_im017._original_depth]|![JNet_469_pretrain_beads_roi015_im017._output_depth]|![JNet_469_pretrain_beads_roi015_im017._reconst_depth]|![JNet_469_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 769.2415360000001, MSE: 0.0011362632503733039, quantized loss: 0.002682807855308056  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi016_im018._original_depth]|![JNet_469_pretrain_beads_roi016_im018._output_depth]|![JNet_469_pretrain_beads_roi016_im018._reconst_depth]|![JNet_469_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 830.3155200000001, MSE: 0.0013059144839644432, quantized loss: 0.00283693871460855  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi017_im018._original_depth]|![JNet_469_pretrain_beads_roi017_im018._output_depth]|![JNet_469_pretrain_beads_roi017_im018._reconst_depth]|![JNet_469_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 822.9349120000002, MSE: 0.0013561192899942398, quantized loss: 0.002767205471172929  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi018_im022._original_depth]|![JNet_469_pretrain_beads_roi018_im022._output_depth]|![JNet_469_pretrain_beads_roi018_im022._reconst_depth]|![JNet_469_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 700.9958400000002, MSE: 0.0011678197188302875, quantized loss: 0.0024526838678866625  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi019_im023._original_depth]|![JNet_469_pretrain_beads_roi019_im023._output_depth]|![JNet_469_pretrain_beads_roi019_im023._reconst_depth]|![JNet_469_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 692.4280320000001, MSE: 0.001172143965959549, quantized loss: 0.002375928685069084  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi020_im024._original_depth]|![JNet_469_pretrain_beads_roi020_im024._output_depth]|![JNet_469_pretrain_beads_roi020_im024._reconst_depth]|![JNet_469_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 801.9821440000002, MSE: 0.0012505438644438982, quantized loss: 0.0027473934460431337  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi021_im026._original_depth]|![JNet_469_pretrain_beads_roi021_im026._output_depth]|![JNet_469_pretrain_beads_roi021_im026._reconst_depth]|![JNet_469_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 791.8119040000001, MSE: 0.0011575723765417933, quantized loss: 0.0027792900800704956  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi022_im027._original_depth]|![JNet_469_pretrain_beads_roi022_im027._output_depth]|![JNet_469_pretrain_beads_roi022_im027._reconst_depth]|![JNet_469_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 830.2775680000001, MSE: 0.0013957554474473, quantized loss: 0.003031452652066946  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi023_im028._original_depth]|![JNet_469_pretrain_beads_roi023_im028._output_depth]|![JNet_469_pretrain_beads_roi023_im028._reconst_depth]|![JNet_469_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 824.3673600000002, MSE: 0.001019078423269093, quantized loss: 0.0029731641989201307  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi024_im028._original_depth]|![JNet_469_pretrain_beads_roi024_im028._output_depth]|![JNet_469_pretrain_beads_roi024_im028._reconst_depth]|![JNet_469_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 799.4286720000001, MSE: 0.0011314492439851165, quantized loss: 0.002798470901325345  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi025_im028._original_depth]|![JNet_469_pretrain_beads_roi025_im028._output_depth]|![JNet_469_pretrain_beads_roi025_im028._reconst_depth]|![JNet_469_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 799.4286720000001, MSE: 0.0011314492439851165, quantized loss: 0.002798470901325345  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi026_im029._original_depth]|![JNet_469_pretrain_beads_roi026_im029._output_depth]|![JNet_469_pretrain_beads_roi026_im029._reconst_depth]|![JNet_469_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 814.5281920000001, MSE: 0.0012528299121186137, quantized loss: 0.0028668579179793596  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi027_im029._original_depth]|![JNet_469_pretrain_beads_roi027_im029._output_depth]|![JNet_469_pretrain_beads_roi027_im029._reconst_depth]|![JNet_469_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 752.4505600000001, MSE: 0.0012480765581130981, quantized loss: 0.002595743630081415  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi028_im030._original_depth]|![JNet_469_pretrain_beads_roi028_im030._output_depth]|![JNet_469_pretrain_beads_roi028_im030._reconst_depth]|![JNet_469_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 737.5308160000001, MSE: 0.0011814298341050744, quantized loss: 0.00251800287514925  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_roi029_im030._original_depth]|![JNet_469_pretrain_beads_roi029_im030._output_depth]|![JNet_469_pretrain_beads_roi029_im030._reconst_depth]|![JNet_469_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 762.9344000000001, MSE: 0.0012451042421162128, quantized loss: 0.0025807435158640146  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi000_im000._original_depth]|![JNet_470_beads_roi000_im000._output_depth]|![JNet_470_beads_roi000_im000._reconst_depth]|![JNet_470_beads_roi000_im000._heatmap_depth]|
  
volume: 514.6358080000001, MSE: 0.00019679874822031707, quantized loss: 1.3273132708491175e-06  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi001_im004._original_depth]|![JNet_470_beads_roi001_im004._output_depth]|![JNet_470_beads_roi001_im004._reconst_depth]|![JNet_470_beads_roi001_im004._heatmap_depth]|
  
volume: 580.3813120000001, MSE: 0.00033892225474119186, quantized loss: 1.2661764685617527e-06  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi002_im005._original_depth]|![JNet_470_beads_roi002_im005._output_depth]|![JNet_470_beads_roi002_im005._reconst_depth]|![JNet_470_beads_roi002_im005._heatmap_depth]|
  
volume: 534.0046080000001, MSE: 0.00028083438519388437, quantized loss: 1.19237893159152e-06  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi003_im006._original_depth]|![JNet_470_beads_roi003_im006._output_depth]|![JNet_470_beads_roi003_im006._reconst_depth]|![JNet_470_beads_roi003_im006._heatmap_depth]|
  
volume: 530.6785920000001, MSE: 0.0003327477315906435, quantized loss: 1.0109135928360047e-06  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi004_im006._original_depth]|![JNet_470_beads_roi004_im006._output_depth]|![JNet_470_beads_roi004_im006._reconst_depth]|![JNet_470_beads_roi004_im006._heatmap_depth]|
  
volume: 543.7707520000001, MSE: 0.00034374860115349293, quantized loss: 1.2697071269940352e-06  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi005_im007._original_depth]|![JNet_470_beads_roi005_im007._output_depth]|![JNet_470_beads_roi005_im007._reconst_depth]|![JNet_470_beads_roi005_im007._heatmap_depth]|
  
volume: 533.8222720000001, MSE: 0.000323060987284407, quantized loss: 1.214250687553431e-06  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi006_im008._original_depth]|![JNet_470_beads_roi006_im008._output_depth]|![JNet_470_beads_roi006_im008._reconst_depth]|![JNet_470_beads_roi006_im008._heatmap_depth]|
  
volume: 548.884352, MSE: 0.00030519734718836844, quantized loss: 1.5256599681379157e-06  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi007_im009._original_depth]|![JNet_470_beads_roi007_im009._output_depth]|![JNet_470_beads_roi007_im009._reconst_depth]|![JNet_470_beads_roi007_im009._heatmap_depth]|
  
volume: 558.1443200000001, MSE: 0.00033598506706766784, quantized loss: 1.3585155329565168e-06  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi008_im010._original_depth]|![JNet_470_beads_roi008_im010._output_depth]|![JNet_470_beads_roi008_im010._reconst_depth]|![JNet_470_beads_roi008_im010._heatmap_depth]|
  
volume: 562.1724800000001, MSE: 0.0003285465354565531, quantized loss: 1.1395023875593324e-06  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi009_im011._original_depth]|![JNet_470_beads_roi009_im011._output_depth]|![JNet_470_beads_roi009_im011._reconst_depth]|![JNet_470_beads_roi009_im011._heatmap_depth]|
  
volume: 524.8814400000001, MSE: 0.00021922198357060552, quantized loss: 1.2572567129609524e-06  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi010_im012._original_depth]|![JNet_470_beads_roi010_im012._output_depth]|![JNet_470_beads_roi010_im012._reconst_depth]|![JNet_470_beads_roi010_im012._heatmap_depth]|
  
volume: 586.3496960000001, MSE: 0.0002853016776498407, quantized loss: 1.1165441264893161e-06  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi011_im013._original_depth]|![JNet_470_beads_roi011_im013._output_depth]|![JNet_470_beads_roi011_im013._reconst_depth]|![JNet_470_beads_roi011_im013._heatmap_depth]|
  
volume: 580.9144320000001, MSE: 0.00025957319303415716, quantized loss: 1.4319290357889258e-06  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi012_im014._original_depth]|![JNet_470_beads_roi012_im014._output_depth]|![JNet_470_beads_roi012_im014._reconst_depth]|![JNet_470_beads_roi012_im014._heatmap_depth]|
  
volume: 539.4364800000001, MSE: 0.0002655394491739571, quantized loss: 9.600076964488835e-07  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi013_im015._original_depth]|![JNet_470_beads_roi013_im015._output_depth]|![JNet_470_beads_roi013_im015._reconst_depth]|![JNet_470_beads_roi013_im015._heatmap_depth]|
  
volume: 511.6046720000001, MSE: 0.0002791580918710679, quantized loss: 1.2879419273303938e-06  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi014_im016._original_depth]|![JNet_470_beads_roi014_im016._output_depth]|![JNet_470_beads_roi014_im016._reconst_depth]|![JNet_470_beads_roi014_im016._heatmap_depth]|
  
volume: 522.4734400000001, MSE: 0.0003245649568270892, quantized loss: 1.604203021088324e-06  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi015_im017._original_depth]|![JNet_470_beads_roi015_im017._output_depth]|![JNet_470_beads_roi015_im017._reconst_depth]|![JNet_470_beads_roi015_im017._heatmap_depth]|
  
volume: 515.5274240000001, MSE: 0.00029666832415387034, quantized loss: 1.0287234317729599e-06  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi016_im018._original_depth]|![JNet_470_beads_roi016_im018._output_depth]|![JNet_470_beads_roi016_im018._reconst_depth]|![JNet_470_beads_roi016_im018._heatmap_depth]|
  
volume: 560.9707520000001, MSE: 0.00039701478090137243, quantized loss: 1.3029015235588304e-06  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi017_im018._original_depth]|![JNet_470_beads_roi017_im018._output_depth]|![JNet_470_beads_roi017_im018._reconst_depth]|![JNet_470_beads_roi017_im018._heatmap_depth]|
  
volume: 558.1102720000001, MSE: 0.0003821555292233825, quantized loss: 1.7110274939113879e-06  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi018_im022._original_depth]|![JNet_470_beads_roi018_im022._output_depth]|![JNet_470_beads_roi018_im022._reconst_depth]|![JNet_470_beads_roi018_im022._heatmap_depth]|
  
volume: 484.6018880000001, MSE: 0.00019624778360594064, quantized loss: 1.0040697588920011e-06  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi019_im023._original_depth]|![JNet_470_beads_roi019_im023._output_depth]|![JNet_470_beads_roi019_im023._reconst_depth]|![JNet_470_beads_roi019_im023._heatmap_depth]|
  
volume: 471.2412800000001, MSE: 0.00019471650011837482, quantized loss: 8.772107094046078e-07  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi020_im024._original_depth]|![JNet_470_beads_roi020_im024._output_depth]|![JNet_470_beads_roi020_im024._reconst_depth]|![JNet_470_beads_roi020_im024._heatmap_depth]|
  
volume: 563.3852800000001, MSE: 0.00021089035726618022, quantized loss: 1.06439551927906e-06  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi021_im026._original_depth]|![JNet_470_beads_roi021_im026._output_depth]|![JNet_470_beads_roi021_im026._reconst_depth]|![JNet_470_beads_roi021_im026._heatmap_depth]|
  
volume: 553.0240000000001, MSE: 0.00020684611808974296, quantized loss: 9.912558880387223e-07  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi022_im027._original_depth]|![JNet_470_beads_roi022_im027._output_depth]|![JNet_470_beads_roi022_im027._reconst_depth]|![JNet_470_beads_roi022_im027._heatmap_depth]|
  
volume: 541.7608960000001, MSE: 0.00022040243493393064, quantized loss: 1.4106681192060933e-06  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi023_im028._original_depth]|![JNet_470_beads_roi023_im028._output_depth]|![JNet_470_beads_roi023_im028._reconst_depth]|![JNet_470_beads_roi023_im028._heatmap_depth]|
  
volume: 579.0087040000001, MSE: 0.00019712239736691117, quantized loss: 1.270072289116797e-06  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi024_im028._original_depth]|![JNet_470_beads_roi024_im028._output_depth]|![JNet_470_beads_roi024_im028._reconst_depth]|![JNet_470_beads_roi024_im028._heatmap_depth]|
  
volume: 563.3936000000001, MSE: 0.00021157025184947997, quantized loss: 1.1837095144073828e-06  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi025_im028._original_depth]|![JNet_470_beads_roi025_im028._output_depth]|![JNet_470_beads_roi025_im028._reconst_depth]|![JNet_470_beads_roi025_im028._heatmap_depth]|
  
volume: 563.3936000000001, MSE: 0.00021157025184947997, quantized loss: 1.1837095144073828e-06  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi026_im029._original_depth]|![JNet_470_beads_roi026_im029._output_depth]|![JNet_470_beads_roi026_im029._reconst_depth]|![JNet_470_beads_roi026_im029._heatmap_depth]|
  
volume: 567.0817280000001, MSE: 0.0002343415835639462, quantized loss: 1.2259213235665811e-06  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi027_im029._original_depth]|![JNet_470_beads_roi027_im029._output_depth]|![JNet_470_beads_roi027_im029._reconst_depth]|![JNet_470_beads_roi027_im029._heatmap_depth]|
  
volume: 521.3697280000001, MSE: 0.00023552525090053678, quantized loss: 1.1452406170064933e-06  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi028_im030._original_depth]|![JNet_470_beads_roi028_im030._output_depth]|![JNet_470_beads_roi028_im030._reconst_depth]|![JNet_470_beads_roi028_im030._heatmap_depth]|
  
volume: 512.2096960000001, MSE: 0.00020328648679424077, quantized loss: 8.805523066257592e-07  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_roi029_im030._original_depth]|![JNet_470_beads_roi029_im030._output_depth]|![JNet_470_beads_roi029_im030._reconst_depth]|![JNet_470_beads_roi029_im030._heatmap_depth]|
  
volume: 532.137408, MSE: 0.00021725043188780546, quantized loss: 1.0296196251147194e-06  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_470_psf_pre]|![JNet_470_psf_post]|

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
  (post): ModuleList(  
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
  (post0): JNetBlockN(  
    (conv): Conv3d(16, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
  )  
  (image): ImagingProcess(  
    (emission): Emission()  
    (blur): Blur(  
      (neuripsf): NeuralImplicitPSF(  
        (layers): Sequential(  
          (0): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
          (1): Linear(in_features=2, out_features=20, bias=True)  
          (2): Sigmoid()  
          (3): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
          (4): Linear(in_features=20, out_features=1, bias=True)  
          (5): Sigmoid()  
        )  
      )  
    )  
    (noise): Noise()  
    (preprocess): PreProcess()  
  )  
  (upsample): JNetUpsample(  
    (upsample): Upsample(scale_factor=(10.0, 1.0, 1.0), mode='trilinear')  
  )  
  (vq): VectorQuantizer()  
)  
```  
  



[JNet_469_pretrain_0_label_depth]: /experiments/images/JNet_469_pretrain_0_label_depth.png
[JNet_469_pretrain_0_label_plane]: /experiments/images/JNet_469_pretrain_0_label_plane.png
[JNet_469_pretrain_0_original_depth]: /experiments/images/JNet_469_pretrain_0_original_depth.png
[JNet_469_pretrain_0_original_plane]: /experiments/images/JNet_469_pretrain_0_original_plane.png
[JNet_469_pretrain_0_output_depth]: /experiments/images/JNet_469_pretrain_0_output_depth.png
[JNet_469_pretrain_0_output_plane]: /experiments/images/JNet_469_pretrain_0_output_plane.png
[JNet_469_pretrain_1_label_depth]: /experiments/images/JNet_469_pretrain_1_label_depth.png
[JNet_469_pretrain_1_label_plane]: /experiments/images/JNet_469_pretrain_1_label_plane.png
[JNet_469_pretrain_1_original_depth]: /experiments/images/JNet_469_pretrain_1_original_depth.png
[JNet_469_pretrain_1_original_plane]: /experiments/images/JNet_469_pretrain_1_original_plane.png
[JNet_469_pretrain_1_output_depth]: /experiments/images/JNet_469_pretrain_1_output_depth.png
[JNet_469_pretrain_1_output_plane]: /experiments/images/JNet_469_pretrain_1_output_plane.png
[JNet_469_pretrain_2_label_depth]: /experiments/images/JNet_469_pretrain_2_label_depth.png
[JNet_469_pretrain_2_label_plane]: /experiments/images/JNet_469_pretrain_2_label_plane.png
[JNet_469_pretrain_2_original_depth]: /experiments/images/JNet_469_pretrain_2_original_depth.png
[JNet_469_pretrain_2_original_plane]: /experiments/images/JNet_469_pretrain_2_original_plane.png
[JNet_469_pretrain_2_output_depth]: /experiments/images/JNet_469_pretrain_2_output_depth.png
[JNet_469_pretrain_2_output_plane]: /experiments/images/JNet_469_pretrain_2_output_plane.png
[JNet_469_pretrain_3_label_depth]: /experiments/images/JNet_469_pretrain_3_label_depth.png
[JNet_469_pretrain_3_label_plane]: /experiments/images/JNet_469_pretrain_3_label_plane.png
[JNet_469_pretrain_3_original_depth]: /experiments/images/JNet_469_pretrain_3_original_depth.png
[JNet_469_pretrain_3_original_plane]: /experiments/images/JNet_469_pretrain_3_original_plane.png
[JNet_469_pretrain_3_output_depth]: /experiments/images/JNet_469_pretrain_3_output_depth.png
[JNet_469_pretrain_3_output_plane]: /experiments/images/JNet_469_pretrain_3_output_plane.png
[JNet_469_pretrain_4_label_depth]: /experiments/images/JNet_469_pretrain_4_label_depth.png
[JNet_469_pretrain_4_label_plane]: /experiments/images/JNet_469_pretrain_4_label_plane.png
[JNet_469_pretrain_4_original_depth]: /experiments/images/JNet_469_pretrain_4_original_depth.png
[JNet_469_pretrain_4_original_plane]: /experiments/images/JNet_469_pretrain_4_original_plane.png
[JNet_469_pretrain_4_output_depth]: /experiments/images/JNet_469_pretrain_4_output_depth.png
[JNet_469_pretrain_4_output_plane]: /experiments/images/JNet_469_pretrain_4_output_plane.png
[JNet_469_pretrain_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi000_im000._heatmap_depth.png
[JNet_469_pretrain_beads_roi000_im000._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi000_im000._original_depth.png
[JNet_469_pretrain_beads_roi000_im000._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi000_im000._output_depth.png
[JNet_469_pretrain_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi000_im000._reconst_depth.png
[JNet_469_pretrain_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi001_im004._heatmap_depth.png
[JNet_469_pretrain_beads_roi001_im004._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi001_im004._original_depth.png
[JNet_469_pretrain_beads_roi001_im004._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi001_im004._output_depth.png
[JNet_469_pretrain_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi001_im004._reconst_depth.png
[JNet_469_pretrain_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi002_im005._heatmap_depth.png
[JNet_469_pretrain_beads_roi002_im005._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi002_im005._original_depth.png
[JNet_469_pretrain_beads_roi002_im005._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi002_im005._output_depth.png
[JNet_469_pretrain_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi002_im005._reconst_depth.png
[JNet_469_pretrain_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi003_im006._heatmap_depth.png
[JNet_469_pretrain_beads_roi003_im006._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi003_im006._original_depth.png
[JNet_469_pretrain_beads_roi003_im006._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi003_im006._output_depth.png
[JNet_469_pretrain_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi003_im006._reconst_depth.png
[JNet_469_pretrain_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi004_im006._heatmap_depth.png
[JNet_469_pretrain_beads_roi004_im006._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi004_im006._original_depth.png
[JNet_469_pretrain_beads_roi004_im006._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi004_im006._output_depth.png
[JNet_469_pretrain_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi004_im006._reconst_depth.png
[JNet_469_pretrain_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi005_im007._heatmap_depth.png
[JNet_469_pretrain_beads_roi005_im007._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi005_im007._original_depth.png
[JNet_469_pretrain_beads_roi005_im007._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi005_im007._output_depth.png
[JNet_469_pretrain_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi005_im007._reconst_depth.png
[JNet_469_pretrain_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi006_im008._heatmap_depth.png
[JNet_469_pretrain_beads_roi006_im008._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi006_im008._original_depth.png
[JNet_469_pretrain_beads_roi006_im008._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi006_im008._output_depth.png
[JNet_469_pretrain_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi006_im008._reconst_depth.png
[JNet_469_pretrain_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi007_im009._heatmap_depth.png
[JNet_469_pretrain_beads_roi007_im009._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi007_im009._original_depth.png
[JNet_469_pretrain_beads_roi007_im009._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi007_im009._output_depth.png
[JNet_469_pretrain_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi007_im009._reconst_depth.png
[JNet_469_pretrain_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi008_im010._heatmap_depth.png
[JNet_469_pretrain_beads_roi008_im010._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi008_im010._original_depth.png
[JNet_469_pretrain_beads_roi008_im010._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi008_im010._output_depth.png
[JNet_469_pretrain_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi008_im010._reconst_depth.png
[JNet_469_pretrain_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi009_im011._heatmap_depth.png
[JNet_469_pretrain_beads_roi009_im011._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi009_im011._original_depth.png
[JNet_469_pretrain_beads_roi009_im011._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi009_im011._output_depth.png
[JNet_469_pretrain_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi009_im011._reconst_depth.png
[JNet_469_pretrain_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi010_im012._heatmap_depth.png
[JNet_469_pretrain_beads_roi010_im012._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi010_im012._original_depth.png
[JNet_469_pretrain_beads_roi010_im012._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi010_im012._output_depth.png
[JNet_469_pretrain_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi010_im012._reconst_depth.png
[JNet_469_pretrain_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi011_im013._heatmap_depth.png
[JNet_469_pretrain_beads_roi011_im013._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi011_im013._original_depth.png
[JNet_469_pretrain_beads_roi011_im013._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi011_im013._output_depth.png
[JNet_469_pretrain_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi011_im013._reconst_depth.png
[JNet_469_pretrain_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi012_im014._heatmap_depth.png
[JNet_469_pretrain_beads_roi012_im014._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi012_im014._original_depth.png
[JNet_469_pretrain_beads_roi012_im014._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi012_im014._output_depth.png
[JNet_469_pretrain_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi012_im014._reconst_depth.png
[JNet_469_pretrain_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi013_im015._heatmap_depth.png
[JNet_469_pretrain_beads_roi013_im015._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi013_im015._original_depth.png
[JNet_469_pretrain_beads_roi013_im015._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi013_im015._output_depth.png
[JNet_469_pretrain_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi013_im015._reconst_depth.png
[JNet_469_pretrain_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi014_im016._heatmap_depth.png
[JNet_469_pretrain_beads_roi014_im016._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi014_im016._original_depth.png
[JNet_469_pretrain_beads_roi014_im016._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi014_im016._output_depth.png
[JNet_469_pretrain_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi014_im016._reconst_depth.png
[JNet_469_pretrain_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi015_im017._heatmap_depth.png
[JNet_469_pretrain_beads_roi015_im017._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi015_im017._original_depth.png
[JNet_469_pretrain_beads_roi015_im017._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi015_im017._output_depth.png
[JNet_469_pretrain_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi015_im017._reconst_depth.png
[JNet_469_pretrain_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi016_im018._heatmap_depth.png
[JNet_469_pretrain_beads_roi016_im018._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi016_im018._original_depth.png
[JNet_469_pretrain_beads_roi016_im018._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi016_im018._output_depth.png
[JNet_469_pretrain_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi016_im018._reconst_depth.png
[JNet_469_pretrain_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi017_im018._heatmap_depth.png
[JNet_469_pretrain_beads_roi017_im018._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi017_im018._original_depth.png
[JNet_469_pretrain_beads_roi017_im018._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi017_im018._output_depth.png
[JNet_469_pretrain_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi017_im018._reconst_depth.png
[JNet_469_pretrain_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi018_im022._heatmap_depth.png
[JNet_469_pretrain_beads_roi018_im022._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi018_im022._original_depth.png
[JNet_469_pretrain_beads_roi018_im022._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi018_im022._output_depth.png
[JNet_469_pretrain_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi018_im022._reconst_depth.png
[JNet_469_pretrain_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi019_im023._heatmap_depth.png
[JNet_469_pretrain_beads_roi019_im023._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi019_im023._original_depth.png
[JNet_469_pretrain_beads_roi019_im023._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi019_im023._output_depth.png
[JNet_469_pretrain_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi019_im023._reconst_depth.png
[JNet_469_pretrain_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi020_im024._heatmap_depth.png
[JNet_469_pretrain_beads_roi020_im024._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi020_im024._original_depth.png
[JNet_469_pretrain_beads_roi020_im024._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi020_im024._output_depth.png
[JNet_469_pretrain_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi020_im024._reconst_depth.png
[JNet_469_pretrain_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi021_im026._heatmap_depth.png
[JNet_469_pretrain_beads_roi021_im026._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi021_im026._original_depth.png
[JNet_469_pretrain_beads_roi021_im026._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi021_im026._output_depth.png
[JNet_469_pretrain_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi021_im026._reconst_depth.png
[JNet_469_pretrain_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi022_im027._heatmap_depth.png
[JNet_469_pretrain_beads_roi022_im027._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi022_im027._original_depth.png
[JNet_469_pretrain_beads_roi022_im027._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi022_im027._output_depth.png
[JNet_469_pretrain_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi022_im027._reconst_depth.png
[JNet_469_pretrain_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi023_im028._heatmap_depth.png
[JNet_469_pretrain_beads_roi023_im028._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi023_im028._original_depth.png
[JNet_469_pretrain_beads_roi023_im028._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi023_im028._output_depth.png
[JNet_469_pretrain_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi023_im028._reconst_depth.png
[JNet_469_pretrain_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi024_im028._heatmap_depth.png
[JNet_469_pretrain_beads_roi024_im028._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi024_im028._original_depth.png
[JNet_469_pretrain_beads_roi024_im028._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi024_im028._output_depth.png
[JNet_469_pretrain_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi024_im028._reconst_depth.png
[JNet_469_pretrain_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi025_im028._heatmap_depth.png
[JNet_469_pretrain_beads_roi025_im028._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi025_im028._original_depth.png
[JNet_469_pretrain_beads_roi025_im028._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi025_im028._output_depth.png
[JNet_469_pretrain_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi025_im028._reconst_depth.png
[JNet_469_pretrain_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi026_im029._heatmap_depth.png
[JNet_469_pretrain_beads_roi026_im029._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi026_im029._original_depth.png
[JNet_469_pretrain_beads_roi026_im029._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi026_im029._output_depth.png
[JNet_469_pretrain_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi026_im029._reconst_depth.png
[JNet_469_pretrain_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi027_im029._heatmap_depth.png
[JNet_469_pretrain_beads_roi027_im029._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi027_im029._original_depth.png
[JNet_469_pretrain_beads_roi027_im029._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi027_im029._output_depth.png
[JNet_469_pretrain_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi027_im029._reconst_depth.png
[JNet_469_pretrain_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi028_im030._heatmap_depth.png
[JNet_469_pretrain_beads_roi028_im030._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi028_im030._original_depth.png
[JNet_469_pretrain_beads_roi028_im030._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi028_im030._output_depth.png
[JNet_469_pretrain_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi028_im030._reconst_depth.png
[JNet_469_pretrain_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_roi029_im030._heatmap_depth.png
[JNet_469_pretrain_beads_roi029_im030._original_depth]: /experiments/images/JNet_469_pretrain_beads_roi029_im030._original_depth.png
[JNet_469_pretrain_beads_roi029_im030._output_depth]: /experiments/images/JNet_469_pretrain_beads_roi029_im030._output_depth.png
[JNet_469_pretrain_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_469_pretrain_beads_roi029_im030._reconst_depth.png
[JNet_470_0_label_depth]: /experiments/images/JNet_470_0_label_depth.png
[JNet_470_0_label_plane]: /experiments/images/JNet_470_0_label_plane.png
[JNet_470_0_original_depth]: /experiments/images/JNet_470_0_original_depth.png
[JNet_470_0_original_plane]: /experiments/images/JNet_470_0_original_plane.png
[JNet_470_0_output_depth]: /experiments/images/JNet_470_0_output_depth.png
[JNet_470_0_output_plane]: /experiments/images/JNet_470_0_output_plane.png
[JNet_470_1_label_depth]: /experiments/images/JNet_470_1_label_depth.png
[JNet_470_1_label_plane]: /experiments/images/JNet_470_1_label_plane.png
[JNet_470_1_original_depth]: /experiments/images/JNet_470_1_original_depth.png
[JNet_470_1_original_plane]: /experiments/images/JNet_470_1_original_plane.png
[JNet_470_1_output_depth]: /experiments/images/JNet_470_1_output_depth.png
[JNet_470_1_output_plane]: /experiments/images/JNet_470_1_output_plane.png
[JNet_470_2_label_depth]: /experiments/images/JNet_470_2_label_depth.png
[JNet_470_2_label_plane]: /experiments/images/JNet_470_2_label_plane.png
[JNet_470_2_original_depth]: /experiments/images/JNet_470_2_original_depth.png
[JNet_470_2_original_plane]: /experiments/images/JNet_470_2_original_plane.png
[JNet_470_2_output_depth]: /experiments/images/JNet_470_2_output_depth.png
[JNet_470_2_output_plane]: /experiments/images/JNet_470_2_output_plane.png
[JNet_470_3_label_depth]: /experiments/images/JNet_470_3_label_depth.png
[JNet_470_3_label_plane]: /experiments/images/JNet_470_3_label_plane.png
[JNet_470_3_original_depth]: /experiments/images/JNet_470_3_original_depth.png
[JNet_470_3_original_plane]: /experiments/images/JNet_470_3_original_plane.png
[JNet_470_3_output_depth]: /experiments/images/JNet_470_3_output_depth.png
[JNet_470_3_output_plane]: /experiments/images/JNet_470_3_output_plane.png
[JNet_470_4_label_depth]: /experiments/images/JNet_470_4_label_depth.png
[JNet_470_4_label_plane]: /experiments/images/JNet_470_4_label_plane.png
[JNet_470_4_original_depth]: /experiments/images/JNet_470_4_original_depth.png
[JNet_470_4_original_plane]: /experiments/images/JNet_470_4_original_plane.png
[JNet_470_4_output_depth]: /experiments/images/JNet_470_4_output_depth.png
[JNet_470_4_output_plane]: /experiments/images/JNet_470_4_output_plane.png
[JNet_470_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_470_beads_roi000_im000._heatmap_depth.png
[JNet_470_beads_roi000_im000._original_depth]: /experiments/images/JNet_470_beads_roi000_im000._original_depth.png
[JNet_470_beads_roi000_im000._output_depth]: /experiments/images/JNet_470_beads_roi000_im000._output_depth.png
[JNet_470_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_470_beads_roi000_im000._reconst_depth.png
[JNet_470_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_470_beads_roi001_im004._heatmap_depth.png
[JNet_470_beads_roi001_im004._original_depth]: /experiments/images/JNet_470_beads_roi001_im004._original_depth.png
[JNet_470_beads_roi001_im004._output_depth]: /experiments/images/JNet_470_beads_roi001_im004._output_depth.png
[JNet_470_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_470_beads_roi001_im004._reconst_depth.png
[JNet_470_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_470_beads_roi002_im005._heatmap_depth.png
[JNet_470_beads_roi002_im005._original_depth]: /experiments/images/JNet_470_beads_roi002_im005._original_depth.png
[JNet_470_beads_roi002_im005._output_depth]: /experiments/images/JNet_470_beads_roi002_im005._output_depth.png
[JNet_470_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_470_beads_roi002_im005._reconst_depth.png
[JNet_470_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_470_beads_roi003_im006._heatmap_depth.png
[JNet_470_beads_roi003_im006._original_depth]: /experiments/images/JNet_470_beads_roi003_im006._original_depth.png
[JNet_470_beads_roi003_im006._output_depth]: /experiments/images/JNet_470_beads_roi003_im006._output_depth.png
[JNet_470_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_470_beads_roi003_im006._reconst_depth.png
[JNet_470_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_470_beads_roi004_im006._heatmap_depth.png
[JNet_470_beads_roi004_im006._original_depth]: /experiments/images/JNet_470_beads_roi004_im006._original_depth.png
[JNet_470_beads_roi004_im006._output_depth]: /experiments/images/JNet_470_beads_roi004_im006._output_depth.png
[JNet_470_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_470_beads_roi004_im006._reconst_depth.png
[JNet_470_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_470_beads_roi005_im007._heatmap_depth.png
[JNet_470_beads_roi005_im007._original_depth]: /experiments/images/JNet_470_beads_roi005_im007._original_depth.png
[JNet_470_beads_roi005_im007._output_depth]: /experiments/images/JNet_470_beads_roi005_im007._output_depth.png
[JNet_470_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_470_beads_roi005_im007._reconst_depth.png
[JNet_470_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_470_beads_roi006_im008._heatmap_depth.png
[JNet_470_beads_roi006_im008._original_depth]: /experiments/images/JNet_470_beads_roi006_im008._original_depth.png
[JNet_470_beads_roi006_im008._output_depth]: /experiments/images/JNet_470_beads_roi006_im008._output_depth.png
[JNet_470_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_470_beads_roi006_im008._reconst_depth.png
[JNet_470_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_470_beads_roi007_im009._heatmap_depth.png
[JNet_470_beads_roi007_im009._original_depth]: /experiments/images/JNet_470_beads_roi007_im009._original_depth.png
[JNet_470_beads_roi007_im009._output_depth]: /experiments/images/JNet_470_beads_roi007_im009._output_depth.png
[JNet_470_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_470_beads_roi007_im009._reconst_depth.png
[JNet_470_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_470_beads_roi008_im010._heatmap_depth.png
[JNet_470_beads_roi008_im010._original_depth]: /experiments/images/JNet_470_beads_roi008_im010._original_depth.png
[JNet_470_beads_roi008_im010._output_depth]: /experiments/images/JNet_470_beads_roi008_im010._output_depth.png
[JNet_470_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_470_beads_roi008_im010._reconst_depth.png
[JNet_470_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_470_beads_roi009_im011._heatmap_depth.png
[JNet_470_beads_roi009_im011._original_depth]: /experiments/images/JNet_470_beads_roi009_im011._original_depth.png
[JNet_470_beads_roi009_im011._output_depth]: /experiments/images/JNet_470_beads_roi009_im011._output_depth.png
[JNet_470_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_470_beads_roi009_im011._reconst_depth.png
[JNet_470_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_470_beads_roi010_im012._heatmap_depth.png
[JNet_470_beads_roi010_im012._original_depth]: /experiments/images/JNet_470_beads_roi010_im012._original_depth.png
[JNet_470_beads_roi010_im012._output_depth]: /experiments/images/JNet_470_beads_roi010_im012._output_depth.png
[JNet_470_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_470_beads_roi010_im012._reconst_depth.png
[JNet_470_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_470_beads_roi011_im013._heatmap_depth.png
[JNet_470_beads_roi011_im013._original_depth]: /experiments/images/JNet_470_beads_roi011_im013._original_depth.png
[JNet_470_beads_roi011_im013._output_depth]: /experiments/images/JNet_470_beads_roi011_im013._output_depth.png
[JNet_470_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_470_beads_roi011_im013._reconst_depth.png
[JNet_470_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_470_beads_roi012_im014._heatmap_depth.png
[JNet_470_beads_roi012_im014._original_depth]: /experiments/images/JNet_470_beads_roi012_im014._original_depth.png
[JNet_470_beads_roi012_im014._output_depth]: /experiments/images/JNet_470_beads_roi012_im014._output_depth.png
[JNet_470_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_470_beads_roi012_im014._reconst_depth.png
[JNet_470_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_470_beads_roi013_im015._heatmap_depth.png
[JNet_470_beads_roi013_im015._original_depth]: /experiments/images/JNet_470_beads_roi013_im015._original_depth.png
[JNet_470_beads_roi013_im015._output_depth]: /experiments/images/JNet_470_beads_roi013_im015._output_depth.png
[JNet_470_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_470_beads_roi013_im015._reconst_depth.png
[JNet_470_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_470_beads_roi014_im016._heatmap_depth.png
[JNet_470_beads_roi014_im016._original_depth]: /experiments/images/JNet_470_beads_roi014_im016._original_depth.png
[JNet_470_beads_roi014_im016._output_depth]: /experiments/images/JNet_470_beads_roi014_im016._output_depth.png
[JNet_470_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_470_beads_roi014_im016._reconst_depth.png
[JNet_470_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_470_beads_roi015_im017._heatmap_depth.png
[JNet_470_beads_roi015_im017._original_depth]: /experiments/images/JNet_470_beads_roi015_im017._original_depth.png
[JNet_470_beads_roi015_im017._output_depth]: /experiments/images/JNet_470_beads_roi015_im017._output_depth.png
[JNet_470_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_470_beads_roi015_im017._reconst_depth.png
[JNet_470_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_470_beads_roi016_im018._heatmap_depth.png
[JNet_470_beads_roi016_im018._original_depth]: /experiments/images/JNet_470_beads_roi016_im018._original_depth.png
[JNet_470_beads_roi016_im018._output_depth]: /experiments/images/JNet_470_beads_roi016_im018._output_depth.png
[JNet_470_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_470_beads_roi016_im018._reconst_depth.png
[JNet_470_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_470_beads_roi017_im018._heatmap_depth.png
[JNet_470_beads_roi017_im018._original_depth]: /experiments/images/JNet_470_beads_roi017_im018._original_depth.png
[JNet_470_beads_roi017_im018._output_depth]: /experiments/images/JNet_470_beads_roi017_im018._output_depth.png
[JNet_470_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_470_beads_roi017_im018._reconst_depth.png
[JNet_470_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_470_beads_roi018_im022._heatmap_depth.png
[JNet_470_beads_roi018_im022._original_depth]: /experiments/images/JNet_470_beads_roi018_im022._original_depth.png
[JNet_470_beads_roi018_im022._output_depth]: /experiments/images/JNet_470_beads_roi018_im022._output_depth.png
[JNet_470_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_470_beads_roi018_im022._reconst_depth.png
[JNet_470_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_470_beads_roi019_im023._heatmap_depth.png
[JNet_470_beads_roi019_im023._original_depth]: /experiments/images/JNet_470_beads_roi019_im023._original_depth.png
[JNet_470_beads_roi019_im023._output_depth]: /experiments/images/JNet_470_beads_roi019_im023._output_depth.png
[JNet_470_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_470_beads_roi019_im023._reconst_depth.png
[JNet_470_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_470_beads_roi020_im024._heatmap_depth.png
[JNet_470_beads_roi020_im024._original_depth]: /experiments/images/JNet_470_beads_roi020_im024._original_depth.png
[JNet_470_beads_roi020_im024._output_depth]: /experiments/images/JNet_470_beads_roi020_im024._output_depth.png
[JNet_470_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_470_beads_roi020_im024._reconst_depth.png
[JNet_470_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_470_beads_roi021_im026._heatmap_depth.png
[JNet_470_beads_roi021_im026._original_depth]: /experiments/images/JNet_470_beads_roi021_im026._original_depth.png
[JNet_470_beads_roi021_im026._output_depth]: /experiments/images/JNet_470_beads_roi021_im026._output_depth.png
[JNet_470_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_470_beads_roi021_im026._reconst_depth.png
[JNet_470_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_470_beads_roi022_im027._heatmap_depth.png
[JNet_470_beads_roi022_im027._original_depth]: /experiments/images/JNet_470_beads_roi022_im027._original_depth.png
[JNet_470_beads_roi022_im027._output_depth]: /experiments/images/JNet_470_beads_roi022_im027._output_depth.png
[JNet_470_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_470_beads_roi022_im027._reconst_depth.png
[JNet_470_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_470_beads_roi023_im028._heatmap_depth.png
[JNet_470_beads_roi023_im028._original_depth]: /experiments/images/JNet_470_beads_roi023_im028._original_depth.png
[JNet_470_beads_roi023_im028._output_depth]: /experiments/images/JNet_470_beads_roi023_im028._output_depth.png
[JNet_470_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_470_beads_roi023_im028._reconst_depth.png
[JNet_470_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_470_beads_roi024_im028._heatmap_depth.png
[JNet_470_beads_roi024_im028._original_depth]: /experiments/images/JNet_470_beads_roi024_im028._original_depth.png
[JNet_470_beads_roi024_im028._output_depth]: /experiments/images/JNet_470_beads_roi024_im028._output_depth.png
[JNet_470_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_470_beads_roi024_im028._reconst_depth.png
[JNet_470_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_470_beads_roi025_im028._heatmap_depth.png
[JNet_470_beads_roi025_im028._original_depth]: /experiments/images/JNet_470_beads_roi025_im028._original_depth.png
[JNet_470_beads_roi025_im028._output_depth]: /experiments/images/JNet_470_beads_roi025_im028._output_depth.png
[JNet_470_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_470_beads_roi025_im028._reconst_depth.png
[JNet_470_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_470_beads_roi026_im029._heatmap_depth.png
[JNet_470_beads_roi026_im029._original_depth]: /experiments/images/JNet_470_beads_roi026_im029._original_depth.png
[JNet_470_beads_roi026_im029._output_depth]: /experiments/images/JNet_470_beads_roi026_im029._output_depth.png
[JNet_470_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_470_beads_roi026_im029._reconst_depth.png
[JNet_470_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_470_beads_roi027_im029._heatmap_depth.png
[JNet_470_beads_roi027_im029._original_depth]: /experiments/images/JNet_470_beads_roi027_im029._original_depth.png
[JNet_470_beads_roi027_im029._output_depth]: /experiments/images/JNet_470_beads_roi027_im029._output_depth.png
[JNet_470_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_470_beads_roi027_im029._reconst_depth.png
[JNet_470_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_470_beads_roi028_im030._heatmap_depth.png
[JNet_470_beads_roi028_im030._original_depth]: /experiments/images/JNet_470_beads_roi028_im030._original_depth.png
[JNet_470_beads_roi028_im030._output_depth]: /experiments/images/JNet_470_beads_roi028_im030._output_depth.png
[JNet_470_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_470_beads_roi028_im030._reconst_depth.png
[JNet_470_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_470_beads_roi029_im030._heatmap_depth.png
[JNet_470_beads_roi029_im030._original_depth]: /experiments/images/JNet_470_beads_roi029_im030._original_depth.png
[JNet_470_beads_roi029_im030._output_depth]: /experiments/images/JNet_470_beads_roi029_im030._output_depth.png
[JNet_470_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_470_beads_roi029_im030._reconst_depth.png
[JNet_470_psf_post]: /experiments/images/JNet_470_psf_post.png
[JNet_470_psf_pre]: /experiments/images/JNet_470_psf_pre.png
[finetuned]: /experiments/tmp/JNet_470_train.png
[pretrained_model]: /experiments/tmp/JNet_469_pretrain_train.png
