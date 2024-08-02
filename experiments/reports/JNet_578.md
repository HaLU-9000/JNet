



# JNet_578 Report
  
first beads experiment with new methods and data  
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
  
Segmentation: mean MSE: 0.008913272991776466, mean BCE: 0.038607507944107056  
Luminance Estimation: mean MSE: 0.9824510812759399, mean BCE: nan
### 0

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_0_original_plane]|![JNet_577_pretrain_0_novibrate_plane]|![JNet_577_pretrain_0_aligned_plane]|![JNet_577_pretrain_0_outputx_plane]|![JNet_577_pretrain_0_labelx_plane]|![JNet_577_pretrain_0_outputz_plane]|![JNet_577_pretrain_0_labelz_plane]|
  
MSEx: 0.009791842661798, BCEx: 0.042973216623067856  
MSEz: 0.9800573587417603, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_0_original_depth]|![JNet_577_pretrain_0_novibrate_depth]|![JNet_577_pretrain_0_aligned_depth]|![JNet_577_pretrain_0_outputx_depth]|![JNet_577_pretrain_0_labelx_depth]|![JNet_577_pretrain_0_outputz_depth]|![JNet_577_pretrain_0_labelz_depth]|
  
MSEx: 0.009791842661798, BCEx: 0.042973216623067856  
MSEz: 0.9800573587417603, BCEz: nan  

### 1

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_1_original_plane]|![JNet_577_pretrain_1_novibrate_plane]|![JNet_577_pretrain_1_aligned_plane]|![JNet_577_pretrain_1_outputx_plane]|![JNet_577_pretrain_1_labelx_plane]|![JNet_577_pretrain_1_outputz_plane]|![JNet_577_pretrain_1_labelz_plane]|
  
MSEx: 0.007571682333946228, BCEx: 0.03260055556893349  
MSEz: 0.9828972816467285, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_1_original_depth]|![JNet_577_pretrain_1_novibrate_depth]|![JNet_577_pretrain_1_aligned_depth]|![JNet_577_pretrain_1_outputx_depth]|![JNet_577_pretrain_1_labelx_depth]|![JNet_577_pretrain_1_outputz_depth]|![JNet_577_pretrain_1_labelz_depth]|
  
MSEx: 0.007571682333946228, BCEx: 0.03260055556893349  
MSEz: 0.9828972816467285, BCEz: nan  

### 2

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_2_original_plane]|![JNet_577_pretrain_2_novibrate_plane]|![JNet_577_pretrain_2_aligned_plane]|![JNet_577_pretrain_2_outputx_plane]|![JNet_577_pretrain_2_labelx_plane]|![JNet_577_pretrain_2_outputz_plane]|![JNet_577_pretrain_2_labelz_plane]|
  
MSEx: 0.009151830337941647, BCEx: 0.0409136526286602  
MSEz: 0.9771819710731506, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_2_original_depth]|![JNet_577_pretrain_2_novibrate_depth]|![JNet_577_pretrain_2_aligned_depth]|![JNet_577_pretrain_2_outputx_depth]|![JNet_577_pretrain_2_labelx_depth]|![JNet_577_pretrain_2_outputz_depth]|![JNet_577_pretrain_2_labelz_depth]|
  
MSEx: 0.009151830337941647, BCEx: 0.0409136526286602  
MSEz: 0.9771819710731506, BCEz: nan  

### 3

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_3_original_plane]|![JNet_577_pretrain_3_novibrate_plane]|![JNet_577_pretrain_3_aligned_plane]|![JNet_577_pretrain_3_outputx_plane]|![JNet_577_pretrain_3_labelx_plane]|![JNet_577_pretrain_3_outputz_plane]|![JNet_577_pretrain_3_labelz_plane]|
  
MSEx: 0.009757406078279018, BCEx: 0.04061046615242958  
MSEz: 0.984761118888855, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_3_original_depth]|![JNet_577_pretrain_3_novibrate_depth]|![JNet_577_pretrain_3_aligned_depth]|![JNet_577_pretrain_3_outputx_depth]|![JNet_577_pretrain_3_labelx_depth]|![JNet_577_pretrain_3_outputz_depth]|![JNet_577_pretrain_3_labelz_depth]|
  
MSEx: 0.009757406078279018, BCEx: 0.04061046615242958  
MSEz: 0.984761118888855, BCEz: nan  

### 4

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_4_original_plane]|![JNet_577_pretrain_4_novibrate_plane]|![JNet_577_pretrain_4_aligned_plane]|![JNet_577_pretrain_4_outputx_plane]|![JNet_577_pretrain_4_labelx_plane]|![JNet_577_pretrain_4_outputz_plane]|![JNet_577_pretrain_4_labelz_plane]|
  
MSEx: 0.008293603546917439, BCEx: 0.035939641296863556  
MSEz: 0.987357497215271, BCEz: nan  

|original|novibrate|aligned|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_4_original_depth]|![JNet_577_pretrain_4_novibrate_depth]|![JNet_577_pretrain_4_aligned_depth]|![JNet_577_pretrain_4_outputx_depth]|![JNet_577_pretrain_4_labelx_depth]|![JNet_577_pretrain_4_outputz_depth]|![JNet_577_pretrain_4_labelz_depth]|
  
MSEx: 0.008293603546917439, BCEx: 0.035939641296863556  
MSEz: 0.987357497215271, BCEz: nan  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi000_im000._original_depth]|![JNet_577_pretrain_beads_roi000_im000._output_depth]|![JNet_577_pretrain_beads_roi000_im000._reconst_depth]|![JNet_577_pretrain_beads_roi000_im000._heatmap_depth]|
  
volume: 2.6542407226562506, MSE: 0.0012986300280317664, quantized loss: 0.00022575769980903715  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi001_im004._original_depth]|![JNet_577_pretrain_beads_roi001_im004._output_depth]|![JNet_577_pretrain_beads_roi001_im004._reconst_depth]|![JNet_577_pretrain_beads_roi001_im004._heatmap_depth]|
  
volume: 3.203835449218751, MSE: 0.00135069212410599, quantized loss: 0.0002785121032502502  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi002_im005._original_depth]|![JNet_577_pretrain_beads_roi002_im005._output_depth]|![JNet_577_pretrain_beads_roi002_im005._reconst_depth]|![JNet_577_pretrain_beads_roi002_im005._heatmap_depth]|
  
volume: 2.804748535156251, MSE: 0.0012902783928439021, quantized loss: 0.00024243614461738616  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi003_im006._original_depth]|![JNet_577_pretrain_beads_roi003_im006._output_depth]|![JNet_577_pretrain_beads_roi003_im006._reconst_depth]|![JNet_577_pretrain_beads_roi003_im006._heatmap_depth]|
  
volume: 2.947888671875001, MSE: 0.0012916548876091838, quantized loss: 0.0002703992067836225  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi004_im006._original_depth]|![JNet_577_pretrain_beads_roi004_im006._output_depth]|![JNet_577_pretrain_beads_roi004_im006._reconst_depth]|![JNet_577_pretrain_beads_roi004_im006._heatmap_depth]|
  
volume: 3.0286721191406256, MSE: 0.0013166022254154086, quantized loss: 0.00027897668769583106  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi005_im007._original_depth]|![JNet_577_pretrain_beads_roi005_im007._output_depth]|![JNet_577_pretrain_beads_roi005_im007._reconst_depth]|![JNet_577_pretrain_beads_roi005_im007._heatmap_depth]|
  
volume: 2.9012324218750005, MSE: 0.0012847663601860404, quantized loss: 0.0002610212250147015  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi006_im008._original_depth]|![JNet_577_pretrain_beads_roi006_im008._output_depth]|![JNet_577_pretrain_beads_roi006_im008._reconst_depth]|![JNet_577_pretrain_beads_roi006_im008._heatmap_depth]|
  
volume: 3.057560791015626, MSE: 0.0012159826001152396, quantized loss: 0.00031493313144892454  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi007_im009._original_depth]|![JNet_577_pretrain_beads_roi007_im009._output_depth]|![JNet_577_pretrain_beads_roi007_im009._reconst_depth]|![JNet_577_pretrain_beads_roi007_im009._heatmap_depth]|
  
volume: 3.0466166992187507, MSE: 0.0012879966525360942, quantized loss: 0.00031204489641822875  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi008_im010._original_depth]|![JNet_577_pretrain_beads_roi008_im010._output_depth]|![JNet_577_pretrain_beads_roi008_im010._reconst_depth]|![JNet_577_pretrain_beads_roi008_im010._heatmap_depth]|
  
volume: 2.9840446777343757, MSE: 0.0012732439208775759, quantized loss: 0.0002541680878493935  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi009_im011._original_depth]|![JNet_577_pretrain_beads_roi009_im011._output_depth]|![JNet_577_pretrain_beads_roi009_im011._reconst_depth]|![JNet_577_pretrain_beads_roi009_im011._heatmap_depth]|
  
volume: 2.7404482421875005, MSE: 0.001263818354345858, quantized loss: 0.00023775442969053984  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi010_im012._original_depth]|![JNet_577_pretrain_beads_roi010_im012._output_depth]|![JNet_577_pretrain_beads_roi010_im012._reconst_depth]|![JNet_577_pretrain_beads_roi010_im012._heatmap_depth]|
  
volume: 3.2161259765625005, MSE: 0.001339834532700479, quantized loss: 0.00027305528055876493  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi011_im013._original_depth]|![JNet_577_pretrain_beads_roi011_im013._output_depth]|![JNet_577_pretrain_beads_roi011_im013._reconst_depth]|![JNet_577_pretrain_beads_roi011_im013._heatmap_depth]|
  
volume: 3.1790253906250006, MSE: 0.0013183815171942115, quantized loss: 0.0002669435634743422  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi012_im014._original_depth]|![JNet_577_pretrain_beads_roi012_im014._output_depth]|![JNet_577_pretrain_beads_roi012_im014._reconst_depth]|![JNet_577_pretrain_beads_roi012_im014._heatmap_depth]|
  
volume: 2.7468349609375005, MSE: 0.0014539837138727307, quantized loss: 0.00022616157366428524  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi013_im015._original_depth]|![JNet_577_pretrain_beads_roi013_im015._output_depth]|![JNet_577_pretrain_beads_roi013_im015._reconst_depth]|![JNet_577_pretrain_beads_roi013_im015._heatmap_depth]|
  
volume: 2.745975585937501, MSE: 0.001379328896291554, quantized loss: 0.00024052648223005235  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi014_im016._original_depth]|![JNet_577_pretrain_beads_roi014_im016._output_depth]|![JNet_577_pretrain_beads_roi014_im016._reconst_depth]|![JNet_577_pretrain_beads_roi014_im016._heatmap_depth]|
  
volume: 2.7868305664062505, MSE: 0.0012087876675650477, quantized loss: 0.0002631492097862065  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi015_im017._original_depth]|![JNet_577_pretrain_beads_roi015_im017._output_depth]|![JNet_577_pretrain_beads_roi015_im017._reconst_depth]|![JNet_577_pretrain_beads_roi015_im017._heatmap_depth]|
  
volume: 2.7632587890625007, MSE: 0.0012748746667057276, quantized loss: 0.00024661706993356347  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi016_im018._original_depth]|![JNet_577_pretrain_beads_roi016_im018._output_depth]|![JNet_577_pretrain_beads_roi016_im018._reconst_depth]|![JNet_577_pretrain_beads_roi016_im018._heatmap_depth]|
  
volume: 3.0878632812500006, MSE: 0.0014058449305593967, quantized loss: 0.00027891193167306483  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi017_im018._original_depth]|![JNet_577_pretrain_beads_roi017_im018._output_depth]|![JNet_577_pretrain_beads_roi017_im018._reconst_depth]|![JNet_577_pretrain_beads_roi017_im018._heatmap_depth]|
  
volume: 2.9472265625000005, MSE: 0.0014637409476563334, quantized loss: 0.0002618294092826545  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi018_im022._original_depth]|![JNet_577_pretrain_beads_roi018_im022._output_depth]|![JNet_577_pretrain_beads_roi018_im022._reconst_depth]|![JNet_577_pretrain_beads_roi018_im022._heatmap_depth]|
  
volume: 2.513649658203126, MSE: 0.0012711796443909407, quantized loss: 0.0002152841043425724  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi019_im023._original_depth]|![JNet_577_pretrain_beads_roi019_im023._output_depth]|![JNet_577_pretrain_beads_roi019_im023._reconst_depth]|![JNet_577_pretrain_beads_roi019_im023._heatmap_depth]|
  
volume: 2.4940764160156257, MSE: 0.0012951098615303636, quantized loss: 0.00021573090634774417  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi020_im024._original_depth]|![JNet_577_pretrain_beads_roi020_im024._output_depth]|![JNet_577_pretrain_beads_roi020_im024._reconst_depth]|![JNet_577_pretrain_beads_roi020_im024._heatmap_depth]|
  
volume: 2.978897216796876, MSE: 0.0013349397340789437, quantized loss: 0.00024298207426909357  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi021_im026._original_depth]|![JNet_577_pretrain_beads_roi021_im026._output_depth]|![JNet_577_pretrain_beads_roi021_im026._reconst_depth]|![JNet_577_pretrain_beads_roi021_im026._heatmap_depth]|
  
volume: 2.8422709960937507, MSE: 0.0012662671506404877, quantized loss: 0.00023673949181102216  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi022_im027._original_depth]|![JNet_577_pretrain_beads_roi022_im027._output_depth]|![JNet_577_pretrain_beads_roi022_im027._reconst_depth]|![JNet_577_pretrain_beads_roi022_im027._heatmap_depth]|
  
volume: 2.7622773437500006, MSE: 0.0013353294925764203, quantized loss: 0.00022634610650129616  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi023_im028._original_depth]|![JNet_577_pretrain_beads_roi023_im028._output_depth]|![JNet_577_pretrain_beads_roi023_im028._reconst_depth]|![JNet_577_pretrain_beads_roi023_im028._heatmap_depth]|
  
volume: 3.0319616699218757, MSE: 0.0011502481065690517, quantized loss: 0.00026807558606378734  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi024_im028._original_depth]|![JNet_577_pretrain_beads_roi024_im028._output_depth]|![JNet_577_pretrain_beads_roi024_im028._reconst_depth]|![JNet_577_pretrain_beads_roi024_im028._heatmap_depth]|
  
volume: 2.948649658203126, MSE: 0.0012167003005743027, quantized loss: 0.00024959901929832995  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi025_im028._original_depth]|![JNet_577_pretrain_beads_roi025_im028._output_depth]|![JNet_577_pretrain_beads_roi025_im028._reconst_depth]|![JNet_577_pretrain_beads_roi025_im028._heatmap_depth]|
  
volume: 2.948649658203126, MSE: 0.0012167003005743027, quantized loss: 0.00024959901929832995  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi026_im029._original_depth]|![JNet_577_pretrain_beads_roi026_im029._output_depth]|![JNet_577_pretrain_beads_roi026_im029._reconst_depth]|![JNet_577_pretrain_beads_roi026_im029._heatmap_depth]|
  
volume: 2.9874494628906256, MSE: 0.0013852580450475216, quantized loss: 0.0002438267256366089  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi027_im029._original_depth]|![JNet_577_pretrain_beads_roi027_im029._output_depth]|![JNet_577_pretrain_beads_roi027_im029._reconst_depth]|![JNet_577_pretrain_beads_roi027_im029._heatmap_depth]|
  
volume: 2.7193913574218755, MSE: 0.0013329843059182167, quantized loss: 0.00022487954993266612  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi028_im030._original_depth]|![JNet_577_pretrain_beads_roi028_im030._output_depth]|![JNet_577_pretrain_beads_roi028_im030._reconst_depth]|![JNet_577_pretrain_beads_roi028_im030._heatmap_depth]|
  
volume: 2.6573942871093754, MSE: 0.0012786192819476128, quantized loss: 0.00022454318241216242  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_577_pretrain_beads_roi029_im030._original_depth]|![JNet_577_pretrain_beads_roi029_im030._output_depth]|![JNet_577_pretrain_beads_roi029_im030._reconst_depth]|![JNet_577_pretrain_beads_roi029_im030._heatmap_depth]|
  
volume: 2.795043457031251, MSE: 0.0013435357250273228, quantized loss: 0.00023112099734134972  

### beads_roi000_im000.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi000_im000._original_depth]|![JNet_578_beads_roi000_im000._output_depth]|![JNet_578_beads_roi000_im000._reconst_depth]|![JNet_578_beads_roi000_im000._heatmap_depth]|
  
volume: 1.4489619140625003, MSE: 0.0013066682731732726, quantized loss: 0.0001459438499296084  

### beads_roi001_im004.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi001_im004._original_depth]|![JNet_578_beads_roi001_im004._output_depth]|![JNet_578_beads_roi001_im004._reconst_depth]|![JNet_578_beads_roi001_im004._heatmap_depth]|
  
volume: 1.6278094482421879, MSE: 0.0015347044682130218, quantized loss: 0.00016247809980995953  

### beads_roi002_im005.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi002_im005._original_depth]|![JNet_578_beads_roi002_im005._output_depth]|![JNet_578_beads_roi002_im005._reconst_depth]|![JNet_578_beads_roi002_im005._heatmap_depth]|
  
volume: 1.4003919677734378, MSE: 0.0013445443473756313, quantized loss: 0.00013364801998250186  

### beads_roi003_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi003_im006._original_depth]|![JNet_578_beads_roi003_im006._output_depth]|![JNet_578_beads_roi003_im006._reconst_depth]|![JNet_578_beads_roi003_im006._heatmap_depth]|
  
volume: 1.4829915771484379, MSE: 0.0013745781034231186, quantized loss: 0.00015540955064352602  

### beads_roi004_im006.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi004_im006._original_depth]|![JNet_578_beads_roi004_im006._output_depth]|![JNet_578_beads_roi004_im006._reconst_depth]|![JNet_578_beads_roi004_im006._heatmap_depth]|
  
volume: 1.5177760009765628, MSE: 0.001443159650079906, quantized loss: 0.00016551239241380244  

### beads_roi005_im007.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi005_im007._original_depth]|![JNet_578_beads_roi005_im007._output_depth]|![JNet_578_beads_roi005_im007._reconst_depth]|![JNet_578_beads_roi005_im007._heatmap_depth]|
  
volume: 1.4759121093750003, MSE: 0.0014289839891716838, quantized loss: 0.00014943996211513877  

### beads_roi006_im008.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi006_im008._original_depth]|![JNet_578_beads_roi006_im008._output_depth]|![JNet_578_beads_roi006_im008._reconst_depth]|![JNet_578_beads_roi006_im008._heatmap_depth]|
  
volume: 1.5675045166015629, MSE: 0.0013655099319294095, quantized loss: 0.00017023914551828057  

### beads_roi007_im009.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi007_im009._original_depth]|![JNet_578_beads_roi007_im009._output_depth]|![JNet_578_beads_roi007_im009._reconst_depth]|![JNet_578_beads_roi007_im009._heatmap_depth]|
  
volume: 1.4191567382812504, MSE: 0.0014900467358529568, quantized loss: 0.0001463240769226104  

### beads_roi008_im010.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi008_im010._original_depth]|![JNet_578_beads_roi008_im010._output_depth]|![JNet_578_beads_roi008_im010._reconst_depth]|![JNet_578_beads_roi008_im010._heatmap_depth]|
  
volume: 1.585134399414063, MSE: 0.001475367578677833, quantized loss: 0.0001548239088151604  

### beads_roi009_im011.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi009_im011._original_depth]|![JNet_578_beads_roi009_im011._output_depth]|![JNet_578_beads_roi009_im011._reconst_depth]|![JNet_578_beads_roi009_im011._heatmap_depth]|
  
volume: 1.5296907958984378, MSE: 0.0013675219379365444, quantized loss: 0.00014975000522099435  

### beads_roi010_im012.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi010_im012._original_depth]|![JNet_578_beads_roi010_im012._output_depth]|![JNet_578_beads_roi010_im012._reconst_depth]|![JNet_578_beads_roi010_im012._heatmap_depth]|
  
volume: 1.5392038574218754, MSE: 0.0016087022377178073, quantized loss: 0.00014360727800522  

### beads_roi011_im013.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi011_im013._original_depth]|![JNet_578_beads_roi011_im013._output_depth]|![JNet_578_beads_roi011_im013._reconst_depth]|![JNet_578_beads_roi011_im013._heatmap_depth]|
  
volume: 1.5831362304687504, MSE: 0.0014439304359257221, quantized loss: 0.00015897229604888707  

### beads_roi012_im014.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi012_im014._original_depth]|![JNet_578_beads_roi012_im014._output_depth]|![JNet_578_beads_roi012_im014._reconst_depth]|![JNet_578_beads_roi012_im014._heatmap_depth]|
  
volume: 1.4204418945312502, MSE: 0.001373978448100388, quantized loss: 0.00015753137995488942  

### beads_roi013_im015.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi013_im015._original_depth]|![JNet_578_beads_roi013_im015._output_depth]|![JNet_578_beads_roi013_im015._reconst_depth]|![JNet_578_beads_roi013_im015._heatmap_depth]|
  
volume: 1.477904174804688, MSE: 0.0012577945599332452, quantized loss: 0.00017674360424280167  

### beads_roi014_im016.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi014_im016._original_depth]|![JNet_578_beads_roi014_im016._output_depth]|![JNet_578_beads_roi014_im016._reconst_depth]|![JNet_578_beads_roi014_im016._heatmap_depth]|
  
volume: 1.4141567382812503, MSE: 0.0012810551561415195, quantized loss: 0.0001571421598782763  

### beads_roi015_im017.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi015_im017._original_depth]|![JNet_578_beads_roi015_im017._output_depth]|![JNet_578_beads_roi015_im017._reconst_depth]|![JNet_578_beads_roi015_im017._heatmap_depth]|
  
volume: 1.4046379394531254, MSE: 0.0013284659944474697, quantized loss: 0.00013579432561527938  

### beads_roi016_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi016_im018._original_depth]|![JNet_578_beads_roi016_im018._output_depth]|![JNet_578_beads_roi016_im018._reconst_depth]|![JNet_578_beads_roi016_im018._heatmap_depth]|
  
volume: 1.4995699462890628, MSE: 0.0015739531954750419, quantized loss: 0.00013512301666196436  

### beads_roi017_im018.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi017_im018._original_depth]|![JNet_578_beads_roi017_im018._output_depth]|![JNet_578_beads_roi017_im018._reconst_depth]|![JNet_578_beads_roi017_im018._heatmap_depth]|
  
volume: 1.432094848632813, MSE: 0.0014982492430135608, quantized loss: 0.00013817647413816303  

### beads_roi018_im022.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi018_im022._original_depth]|![JNet_578_beads_roi018_im022._output_depth]|![JNet_578_beads_roi018_im022._reconst_depth]|![JNet_578_beads_roi018_im022._heatmap_depth]|
  
volume: 1.2980964355468754, MSE: 0.0011989527847617865, quantized loss: 0.00013372836110647768  

### beads_roi019_im023.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi019_im023._original_depth]|![JNet_578_beads_roi019_im023._output_depth]|![JNet_578_beads_roi019_im023._reconst_depth]|![JNet_578_beads_roi019_im023._heatmap_depth]|
  
volume: 1.2881480712890627, MSE: 0.001217984943650663, quantized loss: 0.00013342358579393476  

### beads_roi020_im024.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi020_im024._original_depth]|![JNet_578_beads_roi020_im024._output_depth]|![JNet_578_beads_roi020_im024._reconst_depth]|![JNet_578_beads_roi020_im024._heatmap_depth]|
  
volume: 1.5240583496093754, MSE: 0.0014614170650020242, quantized loss: 0.0001466836256440729  

### beads_roi021_im026.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi021_im026._original_depth]|![JNet_578_beads_roi021_im026._output_depth]|![JNet_578_beads_roi021_im026._reconst_depth]|![JNet_578_beads_roi021_im026._heatmap_depth]|
  
volume: 1.5122371826171879, MSE: 0.0013801013119518757, quantized loss: 0.00015424152661580592  

### beads_roi022_im027.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi022_im027._original_depth]|![JNet_578_beads_roi022_im027._output_depth]|![JNet_578_beads_roi022_im027._reconst_depth]|![JNet_578_beads_roi022_im027._heatmap_depth]|
  
volume: 1.5191352539062504, MSE: 0.0014304259093478322, quantized loss: 0.00015204095689114183  

### beads_roi023_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi023_im028._original_depth]|![JNet_578_beads_roi023_im028._output_depth]|![JNet_578_beads_roi023_im028._reconst_depth]|![JNet_578_beads_roi023_im028._heatmap_depth]|
  
volume: 1.6373847656250005, MSE: 0.0014050276950001717, quantized loss: 0.00017972133355215192  

### beads_roi024_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi024_im028._original_depth]|![JNet_578_beads_roi024_im028._output_depth]|![JNet_578_beads_roi024_im028._reconst_depth]|![JNet_578_beads_roi024_im028._heatmap_depth]|
  
volume: 1.5935183105468753, MSE: 0.0013489645207300782, quantized loss: 0.00016879390750546008  

### beads_roi025_im028.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi025_im028._original_depth]|![JNet_578_beads_roi025_im028._output_depth]|![JNet_578_beads_roi025_im028._reconst_depth]|![JNet_578_beads_roi025_im028._heatmap_depth]|
  
volume: 1.5935183105468753, MSE: 0.0013489645207300782, quantized loss: 0.00016879390750546008  

### beads_roi026_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi026_im029._original_depth]|![JNet_578_beads_roi026_im029._output_depth]|![JNet_578_beads_roi026_im029._reconst_depth]|![JNet_578_beads_roi026_im029._heatmap_depth]|
  
volume: 1.5482817382812504, MSE: 0.0015288298018276691, quantized loss: 0.00015450471255462617  

### beads_roi027_im029.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi027_im029._original_depth]|![JNet_578_beads_roi027_im029._output_depth]|![JNet_578_beads_roi027_im029._reconst_depth]|![JNet_578_beads_roi027_im029._heatmap_depth]|
  
volume: 1.517521362304688, MSE: 0.0011421096278354526, quantized loss: 0.00018757091311272234  

### beads_roi028_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi028_im030._original_depth]|![JNet_578_beads_roi028_im030._output_depth]|![JNet_578_beads_roi028_im030._reconst_depth]|![JNet_578_beads_roi028_im030._heatmap_depth]|
  
volume: 1.3981051025390627, MSE: 0.0012132433475926518, quantized loss: 0.00014224140613805503  

### beads_roi029_im030.

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_578_beads_roi029_im030._original_depth]|![JNet_578_beads_roi029_im030._output_depth]|![JNet_578_beads_roi029_im030._reconst_depth]|![JNet_578_beads_roi029_im030._heatmap_depth]|
  
volume: 1.4332354736328128, MSE: 0.0013631299370899796, quantized loss: 0.00014182108861859888  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_578_psf_pre]|![JNet_578_psf_post]|

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
[JNet_578_beads_roi000_im000._heatmap_depth]: /experiments/images/JNet_578_beads_roi000_im000._heatmap_depth.png
[JNet_578_beads_roi000_im000._original_depth]: /experiments/images/JNet_578_beads_roi000_im000._original_depth.png
[JNet_578_beads_roi000_im000._output_depth]: /experiments/images/JNet_578_beads_roi000_im000._output_depth.png
[JNet_578_beads_roi000_im000._reconst_depth]: /experiments/images/JNet_578_beads_roi000_im000._reconst_depth.png
[JNet_578_beads_roi001_im004._heatmap_depth]: /experiments/images/JNet_578_beads_roi001_im004._heatmap_depth.png
[JNet_578_beads_roi001_im004._original_depth]: /experiments/images/JNet_578_beads_roi001_im004._original_depth.png
[JNet_578_beads_roi001_im004._output_depth]: /experiments/images/JNet_578_beads_roi001_im004._output_depth.png
[JNet_578_beads_roi001_im004._reconst_depth]: /experiments/images/JNet_578_beads_roi001_im004._reconst_depth.png
[JNet_578_beads_roi002_im005._heatmap_depth]: /experiments/images/JNet_578_beads_roi002_im005._heatmap_depth.png
[JNet_578_beads_roi002_im005._original_depth]: /experiments/images/JNet_578_beads_roi002_im005._original_depth.png
[JNet_578_beads_roi002_im005._output_depth]: /experiments/images/JNet_578_beads_roi002_im005._output_depth.png
[JNet_578_beads_roi002_im005._reconst_depth]: /experiments/images/JNet_578_beads_roi002_im005._reconst_depth.png
[JNet_578_beads_roi003_im006._heatmap_depth]: /experiments/images/JNet_578_beads_roi003_im006._heatmap_depth.png
[JNet_578_beads_roi003_im006._original_depth]: /experiments/images/JNet_578_beads_roi003_im006._original_depth.png
[JNet_578_beads_roi003_im006._output_depth]: /experiments/images/JNet_578_beads_roi003_im006._output_depth.png
[JNet_578_beads_roi003_im006._reconst_depth]: /experiments/images/JNet_578_beads_roi003_im006._reconst_depth.png
[JNet_578_beads_roi004_im006._heatmap_depth]: /experiments/images/JNet_578_beads_roi004_im006._heatmap_depth.png
[JNet_578_beads_roi004_im006._original_depth]: /experiments/images/JNet_578_beads_roi004_im006._original_depth.png
[JNet_578_beads_roi004_im006._output_depth]: /experiments/images/JNet_578_beads_roi004_im006._output_depth.png
[JNet_578_beads_roi004_im006._reconst_depth]: /experiments/images/JNet_578_beads_roi004_im006._reconst_depth.png
[JNet_578_beads_roi005_im007._heatmap_depth]: /experiments/images/JNet_578_beads_roi005_im007._heatmap_depth.png
[JNet_578_beads_roi005_im007._original_depth]: /experiments/images/JNet_578_beads_roi005_im007._original_depth.png
[JNet_578_beads_roi005_im007._output_depth]: /experiments/images/JNet_578_beads_roi005_im007._output_depth.png
[JNet_578_beads_roi005_im007._reconst_depth]: /experiments/images/JNet_578_beads_roi005_im007._reconst_depth.png
[JNet_578_beads_roi006_im008._heatmap_depth]: /experiments/images/JNet_578_beads_roi006_im008._heatmap_depth.png
[JNet_578_beads_roi006_im008._original_depth]: /experiments/images/JNet_578_beads_roi006_im008._original_depth.png
[JNet_578_beads_roi006_im008._output_depth]: /experiments/images/JNet_578_beads_roi006_im008._output_depth.png
[JNet_578_beads_roi006_im008._reconst_depth]: /experiments/images/JNet_578_beads_roi006_im008._reconst_depth.png
[JNet_578_beads_roi007_im009._heatmap_depth]: /experiments/images/JNet_578_beads_roi007_im009._heatmap_depth.png
[JNet_578_beads_roi007_im009._original_depth]: /experiments/images/JNet_578_beads_roi007_im009._original_depth.png
[JNet_578_beads_roi007_im009._output_depth]: /experiments/images/JNet_578_beads_roi007_im009._output_depth.png
[JNet_578_beads_roi007_im009._reconst_depth]: /experiments/images/JNet_578_beads_roi007_im009._reconst_depth.png
[JNet_578_beads_roi008_im010._heatmap_depth]: /experiments/images/JNet_578_beads_roi008_im010._heatmap_depth.png
[JNet_578_beads_roi008_im010._original_depth]: /experiments/images/JNet_578_beads_roi008_im010._original_depth.png
[JNet_578_beads_roi008_im010._output_depth]: /experiments/images/JNet_578_beads_roi008_im010._output_depth.png
[JNet_578_beads_roi008_im010._reconst_depth]: /experiments/images/JNet_578_beads_roi008_im010._reconst_depth.png
[JNet_578_beads_roi009_im011._heatmap_depth]: /experiments/images/JNet_578_beads_roi009_im011._heatmap_depth.png
[JNet_578_beads_roi009_im011._original_depth]: /experiments/images/JNet_578_beads_roi009_im011._original_depth.png
[JNet_578_beads_roi009_im011._output_depth]: /experiments/images/JNet_578_beads_roi009_im011._output_depth.png
[JNet_578_beads_roi009_im011._reconst_depth]: /experiments/images/JNet_578_beads_roi009_im011._reconst_depth.png
[JNet_578_beads_roi010_im012._heatmap_depth]: /experiments/images/JNet_578_beads_roi010_im012._heatmap_depth.png
[JNet_578_beads_roi010_im012._original_depth]: /experiments/images/JNet_578_beads_roi010_im012._original_depth.png
[JNet_578_beads_roi010_im012._output_depth]: /experiments/images/JNet_578_beads_roi010_im012._output_depth.png
[JNet_578_beads_roi010_im012._reconst_depth]: /experiments/images/JNet_578_beads_roi010_im012._reconst_depth.png
[JNet_578_beads_roi011_im013._heatmap_depth]: /experiments/images/JNet_578_beads_roi011_im013._heatmap_depth.png
[JNet_578_beads_roi011_im013._original_depth]: /experiments/images/JNet_578_beads_roi011_im013._original_depth.png
[JNet_578_beads_roi011_im013._output_depth]: /experiments/images/JNet_578_beads_roi011_im013._output_depth.png
[JNet_578_beads_roi011_im013._reconst_depth]: /experiments/images/JNet_578_beads_roi011_im013._reconst_depth.png
[JNet_578_beads_roi012_im014._heatmap_depth]: /experiments/images/JNet_578_beads_roi012_im014._heatmap_depth.png
[JNet_578_beads_roi012_im014._original_depth]: /experiments/images/JNet_578_beads_roi012_im014._original_depth.png
[JNet_578_beads_roi012_im014._output_depth]: /experiments/images/JNet_578_beads_roi012_im014._output_depth.png
[JNet_578_beads_roi012_im014._reconst_depth]: /experiments/images/JNet_578_beads_roi012_im014._reconst_depth.png
[JNet_578_beads_roi013_im015._heatmap_depth]: /experiments/images/JNet_578_beads_roi013_im015._heatmap_depth.png
[JNet_578_beads_roi013_im015._original_depth]: /experiments/images/JNet_578_beads_roi013_im015._original_depth.png
[JNet_578_beads_roi013_im015._output_depth]: /experiments/images/JNet_578_beads_roi013_im015._output_depth.png
[JNet_578_beads_roi013_im015._reconst_depth]: /experiments/images/JNet_578_beads_roi013_im015._reconst_depth.png
[JNet_578_beads_roi014_im016._heatmap_depth]: /experiments/images/JNet_578_beads_roi014_im016._heatmap_depth.png
[JNet_578_beads_roi014_im016._original_depth]: /experiments/images/JNet_578_beads_roi014_im016._original_depth.png
[JNet_578_beads_roi014_im016._output_depth]: /experiments/images/JNet_578_beads_roi014_im016._output_depth.png
[JNet_578_beads_roi014_im016._reconst_depth]: /experiments/images/JNet_578_beads_roi014_im016._reconst_depth.png
[JNet_578_beads_roi015_im017._heatmap_depth]: /experiments/images/JNet_578_beads_roi015_im017._heatmap_depth.png
[JNet_578_beads_roi015_im017._original_depth]: /experiments/images/JNet_578_beads_roi015_im017._original_depth.png
[JNet_578_beads_roi015_im017._output_depth]: /experiments/images/JNet_578_beads_roi015_im017._output_depth.png
[JNet_578_beads_roi015_im017._reconst_depth]: /experiments/images/JNet_578_beads_roi015_im017._reconst_depth.png
[JNet_578_beads_roi016_im018._heatmap_depth]: /experiments/images/JNet_578_beads_roi016_im018._heatmap_depth.png
[JNet_578_beads_roi016_im018._original_depth]: /experiments/images/JNet_578_beads_roi016_im018._original_depth.png
[JNet_578_beads_roi016_im018._output_depth]: /experiments/images/JNet_578_beads_roi016_im018._output_depth.png
[JNet_578_beads_roi016_im018._reconst_depth]: /experiments/images/JNet_578_beads_roi016_im018._reconst_depth.png
[JNet_578_beads_roi017_im018._heatmap_depth]: /experiments/images/JNet_578_beads_roi017_im018._heatmap_depth.png
[JNet_578_beads_roi017_im018._original_depth]: /experiments/images/JNet_578_beads_roi017_im018._original_depth.png
[JNet_578_beads_roi017_im018._output_depth]: /experiments/images/JNet_578_beads_roi017_im018._output_depth.png
[JNet_578_beads_roi017_im018._reconst_depth]: /experiments/images/JNet_578_beads_roi017_im018._reconst_depth.png
[JNet_578_beads_roi018_im022._heatmap_depth]: /experiments/images/JNet_578_beads_roi018_im022._heatmap_depth.png
[JNet_578_beads_roi018_im022._original_depth]: /experiments/images/JNet_578_beads_roi018_im022._original_depth.png
[JNet_578_beads_roi018_im022._output_depth]: /experiments/images/JNet_578_beads_roi018_im022._output_depth.png
[JNet_578_beads_roi018_im022._reconst_depth]: /experiments/images/JNet_578_beads_roi018_im022._reconst_depth.png
[JNet_578_beads_roi019_im023._heatmap_depth]: /experiments/images/JNet_578_beads_roi019_im023._heatmap_depth.png
[JNet_578_beads_roi019_im023._original_depth]: /experiments/images/JNet_578_beads_roi019_im023._original_depth.png
[JNet_578_beads_roi019_im023._output_depth]: /experiments/images/JNet_578_beads_roi019_im023._output_depth.png
[JNet_578_beads_roi019_im023._reconst_depth]: /experiments/images/JNet_578_beads_roi019_im023._reconst_depth.png
[JNet_578_beads_roi020_im024._heatmap_depth]: /experiments/images/JNet_578_beads_roi020_im024._heatmap_depth.png
[JNet_578_beads_roi020_im024._original_depth]: /experiments/images/JNet_578_beads_roi020_im024._original_depth.png
[JNet_578_beads_roi020_im024._output_depth]: /experiments/images/JNet_578_beads_roi020_im024._output_depth.png
[JNet_578_beads_roi020_im024._reconst_depth]: /experiments/images/JNet_578_beads_roi020_im024._reconst_depth.png
[JNet_578_beads_roi021_im026._heatmap_depth]: /experiments/images/JNet_578_beads_roi021_im026._heatmap_depth.png
[JNet_578_beads_roi021_im026._original_depth]: /experiments/images/JNet_578_beads_roi021_im026._original_depth.png
[JNet_578_beads_roi021_im026._output_depth]: /experiments/images/JNet_578_beads_roi021_im026._output_depth.png
[JNet_578_beads_roi021_im026._reconst_depth]: /experiments/images/JNet_578_beads_roi021_im026._reconst_depth.png
[JNet_578_beads_roi022_im027._heatmap_depth]: /experiments/images/JNet_578_beads_roi022_im027._heatmap_depth.png
[JNet_578_beads_roi022_im027._original_depth]: /experiments/images/JNet_578_beads_roi022_im027._original_depth.png
[JNet_578_beads_roi022_im027._output_depth]: /experiments/images/JNet_578_beads_roi022_im027._output_depth.png
[JNet_578_beads_roi022_im027._reconst_depth]: /experiments/images/JNet_578_beads_roi022_im027._reconst_depth.png
[JNet_578_beads_roi023_im028._heatmap_depth]: /experiments/images/JNet_578_beads_roi023_im028._heatmap_depth.png
[JNet_578_beads_roi023_im028._original_depth]: /experiments/images/JNet_578_beads_roi023_im028._original_depth.png
[JNet_578_beads_roi023_im028._output_depth]: /experiments/images/JNet_578_beads_roi023_im028._output_depth.png
[JNet_578_beads_roi023_im028._reconst_depth]: /experiments/images/JNet_578_beads_roi023_im028._reconst_depth.png
[JNet_578_beads_roi024_im028._heatmap_depth]: /experiments/images/JNet_578_beads_roi024_im028._heatmap_depth.png
[JNet_578_beads_roi024_im028._original_depth]: /experiments/images/JNet_578_beads_roi024_im028._original_depth.png
[JNet_578_beads_roi024_im028._output_depth]: /experiments/images/JNet_578_beads_roi024_im028._output_depth.png
[JNet_578_beads_roi024_im028._reconst_depth]: /experiments/images/JNet_578_beads_roi024_im028._reconst_depth.png
[JNet_578_beads_roi025_im028._heatmap_depth]: /experiments/images/JNet_578_beads_roi025_im028._heatmap_depth.png
[JNet_578_beads_roi025_im028._original_depth]: /experiments/images/JNet_578_beads_roi025_im028._original_depth.png
[JNet_578_beads_roi025_im028._output_depth]: /experiments/images/JNet_578_beads_roi025_im028._output_depth.png
[JNet_578_beads_roi025_im028._reconst_depth]: /experiments/images/JNet_578_beads_roi025_im028._reconst_depth.png
[JNet_578_beads_roi026_im029._heatmap_depth]: /experiments/images/JNet_578_beads_roi026_im029._heatmap_depth.png
[JNet_578_beads_roi026_im029._original_depth]: /experiments/images/JNet_578_beads_roi026_im029._original_depth.png
[JNet_578_beads_roi026_im029._output_depth]: /experiments/images/JNet_578_beads_roi026_im029._output_depth.png
[JNet_578_beads_roi026_im029._reconst_depth]: /experiments/images/JNet_578_beads_roi026_im029._reconst_depth.png
[JNet_578_beads_roi027_im029._heatmap_depth]: /experiments/images/JNet_578_beads_roi027_im029._heatmap_depth.png
[JNet_578_beads_roi027_im029._original_depth]: /experiments/images/JNet_578_beads_roi027_im029._original_depth.png
[JNet_578_beads_roi027_im029._output_depth]: /experiments/images/JNet_578_beads_roi027_im029._output_depth.png
[JNet_578_beads_roi027_im029._reconst_depth]: /experiments/images/JNet_578_beads_roi027_im029._reconst_depth.png
[JNet_578_beads_roi028_im030._heatmap_depth]: /experiments/images/JNet_578_beads_roi028_im030._heatmap_depth.png
[JNet_578_beads_roi028_im030._original_depth]: /experiments/images/JNet_578_beads_roi028_im030._original_depth.png
[JNet_578_beads_roi028_im030._output_depth]: /experiments/images/JNet_578_beads_roi028_im030._output_depth.png
[JNet_578_beads_roi028_im030._reconst_depth]: /experiments/images/JNet_578_beads_roi028_im030._reconst_depth.png
[JNet_578_beads_roi029_im030._heatmap_depth]: /experiments/images/JNet_578_beads_roi029_im030._heatmap_depth.png
[JNet_578_beads_roi029_im030._original_depth]: /experiments/images/JNet_578_beads_roi029_im030._original_depth.png
[JNet_578_beads_roi029_im030._output_depth]: /experiments/images/JNet_578_beads_roi029_im030._output_depth.png
[JNet_578_beads_roi029_im030._reconst_depth]: /experiments/images/JNet_578_beads_roi029_im030._reconst_depth.png
[JNet_578_psf_post]: /experiments/images/JNet_578_psf_post.png
[JNet_578_psf_pre]: /experiments/images/JNet_578_psf_pre.png
[finetuned]: /experiments/tmp/JNet_578_train.png
[pretrained_model]: /experiments/tmp/JNet_577_pretrain_train.png
