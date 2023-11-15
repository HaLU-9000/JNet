



# JNet_468 Report
  
new data generation with more objects. axon deconv  
pretrained model : JNet_467_pretrain
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
|scale|6||
|device|cuda||

## Datasets and other training details

### simulation_data_generation

|Parameter|Value|
| :--- | :--- |
|dataset_name|_var_num_beadsdataset3|
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
|folderpath|_var_num_beadsdata3|
|labelname|_label|
|size|[1200, 500, 500]|
|cropsize|[240, 112, 112]|
|I|200|
|low|0|
|high|16|
|scale|6|
|mask|True|
|mask_size|[1, 10, 10]|
|mask_num|30|
|surround|False|
|surround_size|[32, 4, 4]|

### pretrain_val_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_var_num_beadsdata3|
|labelname|_label|
|size|[1200, 500, 500]|
|cropsize|[240, 112, 112]|
|I|20|
|low|16|
|high|20|
|scale|6|
|mask|False|
|mask_size|[1, 10, 10]|
|mask_num|False|
|surround|False|
|surround_size|[32, 4, 4]|
|seed|907|

### train_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|spinedata0|
|scorefolderpath|_spinescore0|
|imagename|020|
|size|[282, 512, 512]|
|cropsize|[240, 112, 112]|
|I|200|
|low|0|
|high|1|
|scale|6|
|train|True|
|mask|True|
|mask_size|[1, 10, 10]|
|mask_num|10|
|surround|False|
|surround_size|[32, 4, 4]|
|score_path|./_spinerawscore0/020_score.pt|

### val_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|spinedata0|
|scorefolderpath|_spinescore0|
|imagename|020|
|size|[282, 512, 512]|
|cropsize|[240, 112, 112]|
|I|20|
|low|0|
|high|1|
|scale|6|
|train|False|
|mask|False|
|mask_size|[1, 10, 10]|
|mask_num|10|
|surround|False|
|surround_size|[32, 4, 4]|
|seed|1204|
|score_path|./_spinerawscore0/020_score.pt|

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
|ewc|ewc|
|params|params|
|es_patience|10|
|reconstruct|True|
|is_instantblur|False|
|is_vibrate|True|
|loss_weight|1|
|qloss_weight|1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.02947387658059597, mean BCE: 0.1052783876657486
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_467_pretrain_0_original_plane]|![JNet_467_pretrain_0_output_plane]|![JNet_467_pretrain_0_label_plane]|
  
MSE: 0.027595799416303635, BCE: 0.09863743931055069  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_467_pretrain_0_original_depth]|![JNet_467_pretrain_0_output_depth]|![JNet_467_pretrain_0_label_depth]|
  
MSE: 0.027595799416303635, BCE: 0.09863743931055069  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_467_pretrain_1_original_plane]|![JNet_467_pretrain_1_output_plane]|![JNet_467_pretrain_1_label_plane]|
  
MSE: 0.036776136606931686, BCE: 0.13104359805583954  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_467_pretrain_1_original_depth]|![JNet_467_pretrain_1_output_depth]|![JNet_467_pretrain_1_label_depth]|
  
MSE: 0.036776136606931686, BCE: 0.13104359805583954  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_467_pretrain_2_original_plane]|![JNet_467_pretrain_2_output_plane]|![JNet_467_pretrain_2_label_plane]|
  
MSE: 0.02451910451054573, BCE: 0.09062587469816208  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_467_pretrain_2_original_depth]|![JNet_467_pretrain_2_output_depth]|![JNet_467_pretrain_2_label_depth]|
  
MSE: 0.02451910451054573, BCE: 0.09062587469816208  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_467_pretrain_3_original_plane]|![JNet_467_pretrain_3_output_plane]|![JNet_467_pretrain_3_label_plane]|
  
MSE: 0.035685956478118896, BCE: 0.12834928929805756  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_467_pretrain_3_original_depth]|![JNet_467_pretrain_3_output_depth]|![JNet_467_pretrain_3_label_depth]|
  
MSE: 0.035685956478118896, BCE: 0.12834928929805756  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_467_pretrain_4_original_plane]|![JNet_467_pretrain_4_output_plane]|![JNet_467_pretrain_4_label_plane]|
  
MSE: 0.02279239147901535, BCE: 0.07773570716381073  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_467_pretrain_4_original_depth]|![JNet_467_pretrain_4_output_depth]|![JNet_467_pretrain_4_label_depth]|
  
MSE: 0.02279239147901535, BCE: 0.07773570716381073  
  
mean MSE: 0.07102765142917633, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_468_0_original_plane]|![JNet_468_0_output_plane]|![JNet_468_0_label_plane]|
  
MSE: 0.06729058176279068, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_468_0_original_depth]|![JNet_468_0_output_depth]|![JNet_468_0_label_depth]|
  
MSE: 0.06729058176279068, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_468_1_original_plane]|![JNet_468_1_output_plane]|![JNet_468_1_label_plane]|
  
MSE: 0.06843477487564087, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_468_1_original_depth]|![JNet_468_1_output_depth]|![JNet_468_1_label_depth]|
  
MSE: 0.06843477487564087, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_468_2_original_plane]|![JNet_468_2_output_plane]|![JNet_468_2_label_plane]|
  
MSE: 0.09134647250175476, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_468_2_original_depth]|![JNet_468_2_output_depth]|![JNet_468_2_label_depth]|
  
MSE: 0.09134647250175476, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_468_3_original_plane]|![JNet_468_3_output_plane]|![JNet_468_3_label_plane]|
  
MSE: 0.07358852028846741, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_468_3_original_depth]|![JNet_468_3_output_depth]|![JNet_468_3_label_depth]|
  
MSE: 0.07358852028846741, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_468_4_original_plane]|![JNet_468_4_output_plane]|![JNet_468_4_label_plane]|
  
MSE: 0.05447793006896973, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_468_4_original_depth]|![JNet_468_4_output_depth]|![JNet_468_4_label_depth]|
  
MSE: 0.05447793006896973, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_467_pretrain_beads_001_roi000_original_depth]|![JNet_467_pretrain_beads_001_roi000_output_depth]|![JNet_467_pretrain_beads_001_roi000_reconst_depth]|![JNet_467_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 227.49923200000003, MSE: 0.0026879978831857443, quantized loss: 0.0013893635477870703  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_467_pretrain_beads_001_roi001_original_depth]|![JNet_467_pretrain_beads_001_roi001_output_depth]|![JNet_467_pretrain_beads_001_roi001_reconst_depth]|![JNet_467_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 271.0422080000001, MSE: 0.004844078328460455, quantized loss: 0.001547577092424035  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_467_pretrain_beads_001_roi002_original_depth]|![JNet_467_pretrain_beads_001_roi002_output_depth]|![JNet_467_pretrain_beads_001_roi002_reconst_depth]|![JNet_467_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 207.67750400000003, MSE: 0.0026033802423626184, quantized loss: 0.0012138254242017865  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_467_pretrain_beads_001_roi003_original_depth]|![JNet_467_pretrain_beads_001_roi003_output_depth]|![JNet_467_pretrain_beads_001_roi003_reconst_depth]|![JNet_467_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 280.53206400000005, MSE: 0.004265326540917158, quantized loss: 0.0016447576927021146  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_467_pretrain_beads_001_roi004_original_depth]|![JNet_467_pretrain_beads_001_roi004_output_depth]|![JNet_467_pretrain_beads_001_roi004_reconst_depth]|![JNet_467_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 217.12187200000002, MSE: 0.0031400269363075495, quantized loss: 0.0012824740260839462  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_467_pretrain_beads_002_roi000_original_depth]|![JNet_467_pretrain_beads_002_roi000_output_depth]|![JNet_467_pretrain_beads_002_roi000_reconst_depth]|![JNet_467_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 219.96507200000005, MSE: 0.003450103336945176, quantized loss: 0.0013091432629153132  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_467_pretrain_beads_002_roi001_original_depth]|![JNet_467_pretrain_beads_002_roi001_output_depth]|![JNet_467_pretrain_beads_002_roi001_reconst_depth]|![JNet_467_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 211.94260800000004, MSE: 0.0027462244033813477, quantized loss: 0.001274104113690555  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_467_pretrain_beads_002_roi002_original_depth]|![JNet_467_pretrain_beads_002_roi002_output_depth]|![JNet_467_pretrain_beads_002_roi002_reconst_depth]|![JNet_467_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 211.96824000000004, MSE: 0.003151754615828395, quantized loss: 0.001217361306771636  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_468_beads_001_roi000_original_depth]|![JNet_468_beads_001_roi000_output_depth]|![JNet_468_beads_001_roi000_reconst_depth]|![JNet_468_beads_001_roi000_heatmap_depth]|
  
volume: 183.31158400000004, MSE: 0.0016905717784538865, quantized loss: 0.0003210321592632681  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_468_beads_001_roi001_original_depth]|![JNet_468_beads_001_roi001_output_depth]|![JNet_468_beads_001_roi001_reconst_depth]|![JNet_468_beads_001_roi001_heatmap_depth]|
  
volume: 274.49580800000007, MSE: 0.0027551890816539526, quantized loss: 0.0004134219198022038  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_468_beads_001_roi002_original_depth]|![JNet_468_beads_001_roi002_output_depth]|![JNet_468_beads_001_roi002_reconst_depth]|![JNet_468_beads_001_roi002_heatmap_depth]|
  
volume: 177.14254400000004, MSE: 0.0017310217954218388, quantized loss: 0.0003036431153304875  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_468_beads_001_roi003_original_depth]|![JNet_468_beads_001_roi003_output_depth]|![JNet_468_beads_001_roi003_reconst_depth]|![JNet_468_beads_001_roi003_heatmap_depth]|
  
volume: 312.20873600000004, MSE: 0.0026971313636749983, quantized loss: 0.000533759593963623  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_468_beads_001_roi004_original_depth]|![JNet_468_beads_001_roi004_output_depth]|![JNet_468_beads_001_roi004_reconst_depth]|![JNet_468_beads_001_roi004_heatmap_depth]|
  
volume: 202.28256000000005, MSE: 0.0017853775061666965, quantized loss: 0.0003024680772796273  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_468_beads_002_roi000_original_depth]|![JNet_468_beads_002_roi000_output_depth]|![JNet_468_beads_002_roi000_reconst_depth]|![JNet_468_beads_002_roi000_heatmap_depth]|
  
volume: 224.83937600000004, MSE: 0.001879851333796978, quantized loss: 0.00033446322777308524  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_468_beads_002_roi001_original_depth]|![JNet_468_beads_002_roi001_output_depth]|![JNet_468_beads_002_roi001_reconst_depth]|![JNet_468_beads_002_roi001_heatmap_depth]|
  
volume: 196.32537600000003, MSE: 0.0017431018641218543, quantized loss: 0.00033569190418347716  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_468_beads_002_roi002_original_depth]|![JNet_468_beads_002_roi002_output_depth]|![JNet_468_beads_002_roi002_reconst_depth]|![JNet_468_beads_002_roi002_heatmap_depth]|
  
volume: 203.84272000000004, MSE: 0.0019090734422206879, quantized loss: 0.0003209589922334999  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_468_psf_pre]|![JNet_468_psf_post]|

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
    (blur): Blur()  
    (noise): Noise()  
    (preprocess): PreProcess()  
  )  
  (upsample): JNetUpsample(  
    (upsample): Upsample(scale_factor=(6.0, 1.0, 1.0), mode='trilinear')  
  )  
  (vq): VectorQuantizer()  
)  
```  
  



[JNet_467_pretrain_0_label_depth]: /experiments/images/JNet_467_pretrain_0_label_depth.png
[JNet_467_pretrain_0_label_plane]: /experiments/images/JNet_467_pretrain_0_label_plane.png
[JNet_467_pretrain_0_original_depth]: /experiments/images/JNet_467_pretrain_0_original_depth.png
[JNet_467_pretrain_0_original_plane]: /experiments/images/JNet_467_pretrain_0_original_plane.png
[JNet_467_pretrain_0_output_depth]: /experiments/images/JNet_467_pretrain_0_output_depth.png
[JNet_467_pretrain_0_output_plane]: /experiments/images/JNet_467_pretrain_0_output_plane.png
[JNet_467_pretrain_1_label_depth]: /experiments/images/JNet_467_pretrain_1_label_depth.png
[JNet_467_pretrain_1_label_plane]: /experiments/images/JNet_467_pretrain_1_label_plane.png
[JNet_467_pretrain_1_original_depth]: /experiments/images/JNet_467_pretrain_1_original_depth.png
[JNet_467_pretrain_1_original_plane]: /experiments/images/JNet_467_pretrain_1_original_plane.png
[JNet_467_pretrain_1_output_depth]: /experiments/images/JNet_467_pretrain_1_output_depth.png
[JNet_467_pretrain_1_output_plane]: /experiments/images/JNet_467_pretrain_1_output_plane.png
[JNet_467_pretrain_2_label_depth]: /experiments/images/JNet_467_pretrain_2_label_depth.png
[JNet_467_pretrain_2_label_plane]: /experiments/images/JNet_467_pretrain_2_label_plane.png
[JNet_467_pretrain_2_original_depth]: /experiments/images/JNet_467_pretrain_2_original_depth.png
[JNet_467_pretrain_2_original_plane]: /experiments/images/JNet_467_pretrain_2_original_plane.png
[JNet_467_pretrain_2_output_depth]: /experiments/images/JNet_467_pretrain_2_output_depth.png
[JNet_467_pretrain_2_output_plane]: /experiments/images/JNet_467_pretrain_2_output_plane.png
[JNet_467_pretrain_3_label_depth]: /experiments/images/JNet_467_pretrain_3_label_depth.png
[JNet_467_pretrain_3_label_plane]: /experiments/images/JNet_467_pretrain_3_label_plane.png
[JNet_467_pretrain_3_original_depth]: /experiments/images/JNet_467_pretrain_3_original_depth.png
[JNet_467_pretrain_3_original_plane]: /experiments/images/JNet_467_pretrain_3_original_plane.png
[JNet_467_pretrain_3_output_depth]: /experiments/images/JNet_467_pretrain_3_output_depth.png
[JNet_467_pretrain_3_output_plane]: /experiments/images/JNet_467_pretrain_3_output_plane.png
[JNet_467_pretrain_4_label_depth]: /experiments/images/JNet_467_pretrain_4_label_depth.png
[JNet_467_pretrain_4_label_plane]: /experiments/images/JNet_467_pretrain_4_label_plane.png
[JNet_467_pretrain_4_original_depth]: /experiments/images/JNet_467_pretrain_4_original_depth.png
[JNet_467_pretrain_4_original_plane]: /experiments/images/JNet_467_pretrain_4_original_plane.png
[JNet_467_pretrain_4_output_depth]: /experiments/images/JNet_467_pretrain_4_output_depth.png
[JNet_467_pretrain_4_output_plane]: /experiments/images/JNet_467_pretrain_4_output_plane.png
[JNet_467_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_467_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi000_original_depth.png
[JNet_467_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi000_output_depth.png
[JNet_467_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi000_reconst_depth.png
[JNet_467_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_467_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi001_original_depth.png
[JNet_467_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi001_output_depth.png
[JNet_467_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi001_reconst_depth.png
[JNet_467_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_467_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi002_original_depth.png
[JNet_467_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi002_output_depth.png
[JNet_467_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi002_reconst_depth.png
[JNet_467_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_467_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi003_original_depth.png
[JNet_467_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi003_output_depth.png
[JNet_467_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi003_reconst_depth.png
[JNet_467_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_467_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi004_original_depth.png
[JNet_467_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi004_output_depth.png
[JNet_467_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_467_pretrain_beads_001_roi004_reconst_depth.png
[JNet_467_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_467_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi000_original_depth.png
[JNet_467_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi000_output_depth.png
[JNet_467_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi000_reconst_depth.png
[JNet_467_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_467_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi001_original_depth.png
[JNet_467_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi001_output_depth.png
[JNet_467_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi001_reconst_depth.png
[JNet_467_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_467_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi002_original_depth.png
[JNet_467_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi002_output_depth.png
[JNet_467_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_467_pretrain_beads_002_roi002_reconst_depth.png
[JNet_468_0_label_depth]: /experiments/images/JNet_468_0_label_depth.png
[JNet_468_0_label_plane]: /experiments/images/JNet_468_0_label_plane.png
[JNet_468_0_original_depth]: /experiments/images/JNet_468_0_original_depth.png
[JNet_468_0_original_plane]: /experiments/images/JNet_468_0_original_plane.png
[JNet_468_0_output_depth]: /experiments/images/JNet_468_0_output_depth.png
[JNet_468_0_output_plane]: /experiments/images/JNet_468_0_output_plane.png
[JNet_468_1_label_depth]: /experiments/images/JNet_468_1_label_depth.png
[JNet_468_1_label_plane]: /experiments/images/JNet_468_1_label_plane.png
[JNet_468_1_original_depth]: /experiments/images/JNet_468_1_original_depth.png
[JNet_468_1_original_plane]: /experiments/images/JNet_468_1_original_plane.png
[JNet_468_1_output_depth]: /experiments/images/JNet_468_1_output_depth.png
[JNet_468_1_output_plane]: /experiments/images/JNet_468_1_output_plane.png
[JNet_468_2_label_depth]: /experiments/images/JNet_468_2_label_depth.png
[JNet_468_2_label_plane]: /experiments/images/JNet_468_2_label_plane.png
[JNet_468_2_original_depth]: /experiments/images/JNet_468_2_original_depth.png
[JNet_468_2_original_plane]: /experiments/images/JNet_468_2_original_plane.png
[JNet_468_2_output_depth]: /experiments/images/JNet_468_2_output_depth.png
[JNet_468_2_output_plane]: /experiments/images/JNet_468_2_output_plane.png
[JNet_468_3_label_depth]: /experiments/images/JNet_468_3_label_depth.png
[JNet_468_3_label_plane]: /experiments/images/JNet_468_3_label_plane.png
[JNet_468_3_original_depth]: /experiments/images/JNet_468_3_original_depth.png
[JNet_468_3_original_plane]: /experiments/images/JNet_468_3_original_plane.png
[JNet_468_3_output_depth]: /experiments/images/JNet_468_3_output_depth.png
[JNet_468_3_output_plane]: /experiments/images/JNet_468_3_output_plane.png
[JNet_468_4_label_depth]: /experiments/images/JNet_468_4_label_depth.png
[JNet_468_4_label_plane]: /experiments/images/JNet_468_4_label_plane.png
[JNet_468_4_original_depth]: /experiments/images/JNet_468_4_original_depth.png
[JNet_468_4_original_plane]: /experiments/images/JNet_468_4_original_plane.png
[JNet_468_4_output_depth]: /experiments/images/JNet_468_4_output_depth.png
[JNet_468_4_output_plane]: /experiments/images/JNet_468_4_output_plane.png
[JNet_468_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_468_beads_001_roi000_heatmap_depth.png
[JNet_468_beads_001_roi000_original_depth]: /experiments/images/JNet_468_beads_001_roi000_original_depth.png
[JNet_468_beads_001_roi000_output_depth]: /experiments/images/JNet_468_beads_001_roi000_output_depth.png
[JNet_468_beads_001_roi000_reconst_depth]: /experiments/images/JNet_468_beads_001_roi000_reconst_depth.png
[JNet_468_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_468_beads_001_roi001_heatmap_depth.png
[JNet_468_beads_001_roi001_original_depth]: /experiments/images/JNet_468_beads_001_roi001_original_depth.png
[JNet_468_beads_001_roi001_output_depth]: /experiments/images/JNet_468_beads_001_roi001_output_depth.png
[JNet_468_beads_001_roi001_reconst_depth]: /experiments/images/JNet_468_beads_001_roi001_reconst_depth.png
[JNet_468_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_468_beads_001_roi002_heatmap_depth.png
[JNet_468_beads_001_roi002_original_depth]: /experiments/images/JNet_468_beads_001_roi002_original_depth.png
[JNet_468_beads_001_roi002_output_depth]: /experiments/images/JNet_468_beads_001_roi002_output_depth.png
[JNet_468_beads_001_roi002_reconst_depth]: /experiments/images/JNet_468_beads_001_roi002_reconst_depth.png
[JNet_468_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_468_beads_001_roi003_heatmap_depth.png
[JNet_468_beads_001_roi003_original_depth]: /experiments/images/JNet_468_beads_001_roi003_original_depth.png
[JNet_468_beads_001_roi003_output_depth]: /experiments/images/JNet_468_beads_001_roi003_output_depth.png
[JNet_468_beads_001_roi003_reconst_depth]: /experiments/images/JNet_468_beads_001_roi003_reconst_depth.png
[JNet_468_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_468_beads_001_roi004_heatmap_depth.png
[JNet_468_beads_001_roi004_original_depth]: /experiments/images/JNet_468_beads_001_roi004_original_depth.png
[JNet_468_beads_001_roi004_output_depth]: /experiments/images/JNet_468_beads_001_roi004_output_depth.png
[JNet_468_beads_001_roi004_reconst_depth]: /experiments/images/JNet_468_beads_001_roi004_reconst_depth.png
[JNet_468_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_468_beads_002_roi000_heatmap_depth.png
[JNet_468_beads_002_roi000_original_depth]: /experiments/images/JNet_468_beads_002_roi000_original_depth.png
[JNet_468_beads_002_roi000_output_depth]: /experiments/images/JNet_468_beads_002_roi000_output_depth.png
[JNet_468_beads_002_roi000_reconst_depth]: /experiments/images/JNet_468_beads_002_roi000_reconst_depth.png
[JNet_468_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_468_beads_002_roi001_heatmap_depth.png
[JNet_468_beads_002_roi001_original_depth]: /experiments/images/JNet_468_beads_002_roi001_original_depth.png
[JNet_468_beads_002_roi001_output_depth]: /experiments/images/JNet_468_beads_002_roi001_output_depth.png
[JNet_468_beads_002_roi001_reconst_depth]: /experiments/images/JNet_468_beads_002_roi001_reconst_depth.png
[JNet_468_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_468_beads_002_roi002_heatmap_depth.png
[JNet_468_beads_002_roi002_original_depth]: /experiments/images/JNet_468_beads_002_roi002_original_depth.png
[JNet_468_beads_002_roi002_output_depth]: /experiments/images/JNet_468_beads_002_roi002_output_depth.png
[JNet_468_beads_002_roi002_reconst_depth]: /experiments/images/JNet_468_beads_002_roi002_reconst_depth.png
[JNet_468_psf_post]: /experiments/images/JNet_468_psf_post.png
[JNet_468_psf_pre]: /experiments/images/JNet_468_psf_pre.png
[finetuned]: /experiments/tmp/JNet_468_train.png
[pretrained_model]: /experiments/tmp/JNet_467_pretrain_train.png
