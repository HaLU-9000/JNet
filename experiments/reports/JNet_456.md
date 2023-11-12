



# JNet_456 Report
  
the parameters to replicate the results of JNet_455. no vibrate in fine tuning, bright NA=0.7, mu_z = 1.2, sig_z = 1.27  
pretrained model : JNet_454_pretrain
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
|mu_z|1.5||
|sig_z|0.3||
|blur_mode|gibsonlanni|`gaussian` or `gibsonlanni`|
|size_x|51||
|size_y|51||
|size_z|201||
|NA|0.75||
|wavelength|1.2|microns|
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
|sig_eps|0.0||
|background|0.01||
|scale|10||
|device|cuda||

## Datasets and other training details

### simulation_data_generation

|Parameter|Value|
| :--- | :--- |
|dataset_name|_var_num_beadsdata2|
|train_num|16|
|valid_num|4|
|image_size|[1200, 500, 500]|
|train_object_num_min|2400|
|train_object_num_max|7200|
|valid_object_num_min|4200|
|valid_object_num_max|5400|

### pretrain_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_var_num_beadsdata2_30_fft_blur|
|imagename|_x6|
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
|folderpath|_var_num_beadsdata2_30_fft_blur|
|imagename|_x6|
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
|folderpath|_stackbeadsdata|
|scorefolderpath|_stackbeadsscore|
|imagename|002|
|size|[650, 512, 512]|
|cropsize|[240, 112, 112]|
|I|200|
|low|0|
|high|1|
|scale|10|
|train|True|
|mask|True|
|mask_size|[1, 10, 10]|
|mask_num|10|
|surround|False|
|surround_size|[32, 4, 4]|
|score_path|./_stackbeadsscore/002_score.pt|

### val_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_stackbeadsdata|
|scorefolderpath|_stackbeadsscore|
|imagename|002|
|size|[650, 512, 512]|
|cropsize|[240, 112, 112]|
|I|20|
|low|0|
|high|1|
|scale|10|
|train|False|
|mask|False|
|mask_size|[1, 10, 10]|
|mask_num|10|
|surround|False|
|surround_size|[32, 4, 4]|
|seed|1204|
|score_path|./_stackbeadsscore/002_score.pt|

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
|partial|params['partial']|
|ewc|None|
|params|params|
|es_patience|10|
|reconstruct|True|
|is_instantblur|False|
|is_vibrate|False|
|loss_weight|1|
|qloss_weight|1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.02084825001657009, mean BCE: 0.07918210327625275
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_454_pretrain_0_original_plane]|![JNet_454_pretrain_0_output_plane]|![JNet_454_pretrain_0_label_plane]|
  
MSE: 0.02076825127005577, BCE: 0.08049479871988297  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_454_pretrain_0_original_depth]|![JNet_454_pretrain_0_output_depth]|![JNet_454_pretrain_0_label_depth]|
  
MSE: 0.02076825127005577, BCE: 0.08049479871988297  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_454_pretrain_1_original_plane]|![JNet_454_pretrain_1_output_plane]|![JNet_454_pretrain_1_label_plane]|
  
MSE: 0.028113340958952904, BCE: 0.0987430289387703  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_454_pretrain_1_original_depth]|![JNet_454_pretrain_1_output_depth]|![JNet_454_pretrain_1_label_depth]|
  
MSE: 0.028113340958952904, BCE: 0.0987430289387703  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_454_pretrain_2_original_plane]|![JNet_454_pretrain_2_output_plane]|![JNet_454_pretrain_2_label_plane]|
  
MSE: 0.022855421528220177, BCE: 0.08602037280797958  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_454_pretrain_2_original_depth]|![JNet_454_pretrain_2_output_depth]|![JNet_454_pretrain_2_label_depth]|
  
MSE: 0.022855421528220177, BCE: 0.08602037280797958  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_454_pretrain_3_original_plane]|![JNet_454_pretrain_3_output_plane]|![JNet_454_pretrain_3_label_plane]|
  
MSE: 0.018228977918624878, BCE: 0.06912869960069656  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_454_pretrain_3_original_depth]|![JNet_454_pretrain_3_output_depth]|![JNet_454_pretrain_3_label_depth]|
  
MSE: 0.018228977918624878, BCE: 0.06912869960069656  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_454_pretrain_4_original_plane]|![JNet_454_pretrain_4_output_plane]|![JNet_454_pretrain_4_label_plane]|
  
MSE: 0.014275263994932175, BCE: 0.06152363121509552  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_454_pretrain_4_original_depth]|![JNet_454_pretrain_4_output_depth]|![JNet_454_pretrain_4_label_depth]|
  
MSE: 0.014275263994932175, BCE: 0.06152363121509552  
  
mean MSE: 0.037287306040525436, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_456_0_original_plane]|![JNet_456_0_output_plane]|![JNet_456_0_label_plane]|
  
MSE: 0.03657855466008186, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_456_0_original_depth]|![JNet_456_0_output_depth]|![JNet_456_0_label_depth]|
  
MSE: 0.03657855466008186, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_456_1_original_plane]|![JNet_456_1_output_plane]|![JNet_456_1_label_plane]|
  
MSE: 0.0351884551346302, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_456_1_original_depth]|![JNet_456_1_output_depth]|![JNet_456_1_label_depth]|
  
MSE: 0.0351884551346302, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_456_2_original_plane]|![JNet_456_2_output_plane]|![JNet_456_2_label_plane]|
  
MSE: 0.05312475562095642, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_456_2_original_depth]|![JNet_456_2_output_depth]|![JNet_456_2_label_depth]|
  
MSE: 0.05312475562095642, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_456_3_original_plane]|![JNet_456_3_output_plane]|![JNet_456_3_label_plane]|
  
MSE: 0.025908468291163445, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_456_3_original_depth]|![JNet_456_3_output_depth]|![JNet_456_3_label_depth]|
  
MSE: 0.025908468291163445, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_456_4_original_plane]|![JNet_456_4_output_plane]|![JNet_456_4_label_plane]|
  
MSE: 0.035636305809020996, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_456_4_original_depth]|![JNet_456_4_output_depth]|![JNet_456_4_label_depth]|
  
MSE: 0.035636305809020996, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_454_pretrain_beads_001_roi000_original_depth]|![JNet_454_pretrain_beads_001_roi000_output_depth]|![JNet_454_pretrain_beads_001_roi000_reconst_depth]|![JNet_454_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 15.650760742187504, MSE: 0.001794786541722715, quantized loss: 0.0021483427844941616  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_454_pretrain_beads_001_roi001_original_depth]|![JNet_454_pretrain_beads_001_roi001_output_depth]|![JNet_454_pretrain_beads_001_roi001_reconst_depth]|![JNet_454_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 21.479406250000004, MSE: 0.002747478662058711, quantized loss: 0.0027272445149719715  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_454_pretrain_beads_001_roi002_original_depth]|![JNet_454_pretrain_beads_001_roi002_output_depth]|![JNet_454_pretrain_beads_001_roi002_reconst_depth]|![JNet_454_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 15.572675781250004, MSE: 0.0020759226754307747, quantized loss: 0.002217040164396167  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_454_pretrain_beads_001_roi003_original_depth]|![JNet_454_pretrain_beads_001_roi003_output_depth]|![JNet_454_pretrain_beads_001_roi003_reconst_depth]|![JNet_454_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 24.427794921875005, MSE: 0.0030758993234485388, quantized loss: 0.0032957021612674  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_454_pretrain_beads_001_roi004_original_depth]|![JNet_454_pretrain_beads_001_roi004_output_depth]|![JNet_454_pretrain_beads_001_roi004_reconst_depth]|![JNet_454_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 16.116684570312504, MSE: 0.0023966773878782988, quantized loss: 0.0021837956737726927  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_454_pretrain_beads_002_roi000_original_depth]|![JNet_454_pretrain_beads_002_roi000_output_depth]|![JNet_454_pretrain_beads_002_roi000_reconst_depth]|![JNet_454_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 17.379560546875005, MSE: 0.0026456397026777267, quantized loss: 0.002369888825342059  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_454_pretrain_beads_002_roi001_original_depth]|![JNet_454_pretrain_beads_002_roi001_output_depth]|![JNet_454_pretrain_beads_002_roi001_reconst_depth]|![JNet_454_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 16.477828125000006, MSE: 0.0020741720218211412, quantized loss: 0.0022815887350589037  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_454_pretrain_beads_002_roi002_original_depth]|![JNet_454_pretrain_beads_002_roi002_output_depth]|![JNet_454_pretrain_beads_002_roi002_reconst_depth]|![JNet_454_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 16.443001953125005, MSE: 0.0023428683634847403, quantized loss: 0.002200286602601409  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_456_beads_001_roi000_original_depth]|![JNet_456_beads_001_roi000_output_depth]|![JNet_456_beads_001_roi000_reconst_depth]|![JNet_456_beads_001_roi000_heatmap_depth]|
  
volume: 4.200451660156251, MSE: 0.0003889444924425334, quantized loss: 8.190138032659888e-05  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_456_beads_001_roi001_original_depth]|![JNet_456_beads_001_roi001_output_depth]|![JNet_456_beads_001_roi001_reconst_depth]|![JNet_456_beads_001_roi001_heatmap_depth]|
  
volume: 6.0197172851562515, MSE: 0.0005434606573544443, quantized loss: 7.193497731350362e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_456_beads_001_roi002_original_depth]|![JNet_456_beads_001_roi002_output_depth]|![JNet_456_beads_001_roi002_reconst_depth]|![JNet_456_beads_001_roi002_heatmap_depth]|
  
volume: 3.637862792968751, MSE: 0.00013667164603248239, quantized loss: 1.0319647117285058e-05  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_456_beads_001_roi003_original_depth]|![JNet_456_beads_001_roi003_output_depth]|![JNet_456_beads_001_roi003_reconst_depth]|![JNet_456_beads_001_roi003_heatmap_depth]|
  
volume: 5.876161621093751, MSE: 0.00041788819362409413, quantized loss: 2.430659696983639e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_456_beads_001_roi004_original_depth]|![JNet_456_beads_001_roi004_output_depth]|![JNet_456_beads_001_roi004_reconst_depth]|![JNet_456_beads_001_roi004_heatmap_depth]|
  
volume: 4.008258300781251, MSE: 0.00014391087461262941, quantized loss: 2.223372575826943e-05  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_456_beads_002_roi000_original_depth]|![JNet_456_beads_002_roi000_output_depth]|![JNet_456_beads_002_roi000_reconst_depth]|![JNet_456_beads_002_roi000_heatmap_depth]|
  
volume: 4.261432617187501, MSE: 0.00014892710896674544, quantized loss: 1.9831190002150834e-05  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_456_beads_002_roi001_original_depth]|![JNet_456_beads_002_roi001_output_depth]|![JNet_456_beads_002_roi001_reconst_depth]|![JNet_456_beads_002_roi001_heatmap_depth]|
  
volume: 3.994168945312501, MSE: 0.00013877687160857022, quantized loss: 5.089609112474136e-05  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_456_beads_002_roi002_original_depth]|![JNet_456_beads_002_roi002_output_depth]|![JNet_456_beads_002_roi002_reconst_depth]|![JNet_456_beads_002_roi002_heatmap_depth]|
  
volume: 3.961674804687501, MSE: 0.00014088426542002708, quantized loss: 2.5540673959767446e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_456_psf_pre]|![JNet_456_psf_post]|

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
    (upsample): Upsample(scale_factor=(10.0, 1.0, 1.0), mode='trilinear')  
  )  
  (vq): VectorQuantizer()  
)  
```  
  



[JNet_454_pretrain_0_label_depth]: /experiments/images/JNet_454_pretrain_0_label_depth.png
[JNet_454_pretrain_0_label_plane]: /experiments/images/JNet_454_pretrain_0_label_plane.png
[JNet_454_pretrain_0_original_depth]: /experiments/images/JNet_454_pretrain_0_original_depth.png
[JNet_454_pretrain_0_original_plane]: /experiments/images/JNet_454_pretrain_0_original_plane.png
[JNet_454_pretrain_0_output_depth]: /experiments/images/JNet_454_pretrain_0_output_depth.png
[JNet_454_pretrain_0_output_plane]: /experiments/images/JNet_454_pretrain_0_output_plane.png
[JNet_454_pretrain_1_label_depth]: /experiments/images/JNet_454_pretrain_1_label_depth.png
[JNet_454_pretrain_1_label_plane]: /experiments/images/JNet_454_pretrain_1_label_plane.png
[JNet_454_pretrain_1_original_depth]: /experiments/images/JNet_454_pretrain_1_original_depth.png
[JNet_454_pretrain_1_original_plane]: /experiments/images/JNet_454_pretrain_1_original_plane.png
[JNet_454_pretrain_1_output_depth]: /experiments/images/JNet_454_pretrain_1_output_depth.png
[JNet_454_pretrain_1_output_plane]: /experiments/images/JNet_454_pretrain_1_output_plane.png
[JNet_454_pretrain_2_label_depth]: /experiments/images/JNet_454_pretrain_2_label_depth.png
[JNet_454_pretrain_2_label_plane]: /experiments/images/JNet_454_pretrain_2_label_plane.png
[JNet_454_pretrain_2_original_depth]: /experiments/images/JNet_454_pretrain_2_original_depth.png
[JNet_454_pretrain_2_original_plane]: /experiments/images/JNet_454_pretrain_2_original_plane.png
[JNet_454_pretrain_2_output_depth]: /experiments/images/JNet_454_pretrain_2_output_depth.png
[JNet_454_pretrain_2_output_plane]: /experiments/images/JNet_454_pretrain_2_output_plane.png
[JNet_454_pretrain_3_label_depth]: /experiments/images/JNet_454_pretrain_3_label_depth.png
[JNet_454_pretrain_3_label_plane]: /experiments/images/JNet_454_pretrain_3_label_plane.png
[JNet_454_pretrain_3_original_depth]: /experiments/images/JNet_454_pretrain_3_original_depth.png
[JNet_454_pretrain_3_original_plane]: /experiments/images/JNet_454_pretrain_3_original_plane.png
[JNet_454_pretrain_3_output_depth]: /experiments/images/JNet_454_pretrain_3_output_depth.png
[JNet_454_pretrain_3_output_plane]: /experiments/images/JNet_454_pretrain_3_output_plane.png
[JNet_454_pretrain_4_label_depth]: /experiments/images/JNet_454_pretrain_4_label_depth.png
[JNet_454_pretrain_4_label_plane]: /experiments/images/JNet_454_pretrain_4_label_plane.png
[JNet_454_pretrain_4_original_depth]: /experiments/images/JNet_454_pretrain_4_original_depth.png
[JNet_454_pretrain_4_original_plane]: /experiments/images/JNet_454_pretrain_4_original_plane.png
[JNet_454_pretrain_4_output_depth]: /experiments/images/JNet_454_pretrain_4_output_depth.png
[JNet_454_pretrain_4_output_plane]: /experiments/images/JNet_454_pretrain_4_output_plane.png
[JNet_454_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_454_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi000_original_depth.png
[JNet_454_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi000_output_depth.png
[JNet_454_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi000_reconst_depth.png
[JNet_454_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_454_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi001_original_depth.png
[JNet_454_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi001_output_depth.png
[JNet_454_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi001_reconst_depth.png
[JNet_454_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_454_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi002_original_depth.png
[JNet_454_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi002_output_depth.png
[JNet_454_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi002_reconst_depth.png
[JNet_454_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_454_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi003_original_depth.png
[JNet_454_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi003_output_depth.png
[JNet_454_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi003_reconst_depth.png
[JNet_454_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_454_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi004_original_depth.png
[JNet_454_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi004_output_depth.png
[JNet_454_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_454_pretrain_beads_001_roi004_reconst_depth.png
[JNet_454_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_454_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi000_original_depth.png
[JNet_454_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi000_output_depth.png
[JNet_454_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi000_reconst_depth.png
[JNet_454_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_454_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi001_original_depth.png
[JNet_454_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi001_output_depth.png
[JNet_454_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi001_reconst_depth.png
[JNet_454_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_454_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi002_original_depth.png
[JNet_454_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi002_output_depth.png
[JNet_454_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_454_pretrain_beads_002_roi002_reconst_depth.png
[JNet_456_0_label_depth]: /experiments/images/JNet_456_0_label_depth.png
[JNet_456_0_label_plane]: /experiments/images/JNet_456_0_label_plane.png
[JNet_456_0_original_depth]: /experiments/images/JNet_456_0_original_depth.png
[JNet_456_0_original_plane]: /experiments/images/JNet_456_0_original_plane.png
[JNet_456_0_output_depth]: /experiments/images/JNet_456_0_output_depth.png
[JNet_456_0_output_plane]: /experiments/images/JNet_456_0_output_plane.png
[JNet_456_1_label_depth]: /experiments/images/JNet_456_1_label_depth.png
[JNet_456_1_label_plane]: /experiments/images/JNet_456_1_label_plane.png
[JNet_456_1_original_depth]: /experiments/images/JNet_456_1_original_depth.png
[JNet_456_1_original_plane]: /experiments/images/JNet_456_1_original_plane.png
[JNet_456_1_output_depth]: /experiments/images/JNet_456_1_output_depth.png
[JNet_456_1_output_plane]: /experiments/images/JNet_456_1_output_plane.png
[JNet_456_2_label_depth]: /experiments/images/JNet_456_2_label_depth.png
[JNet_456_2_label_plane]: /experiments/images/JNet_456_2_label_plane.png
[JNet_456_2_original_depth]: /experiments/images/JNet_456_2_original_depth.png
[JNet_456_2_original_plane]: /experiments/images/JNet_456_2_original_plane.png
[JNet_456_2_output_depth]: /experiments/images/JNet_456_2_output_depth.png
[JNet_456_2_output_plane]: /experiments/images/JNet_456_2_output_plane.png
[JNet_456_3_label_depth]: /experiments/images/JNet_456_3_label_depth.png
[JNet_456_3_label_plane]: /experiments/images/JNet_456_3_label_plane.png
[JNet_456_3_original_depth]: /experiments/images/JNet_456_3_original_depth.png
[JNet_456_3_original_plane]: /experiments/images/JNet_456_3_original_plane.png
[JNet_456_3_output_depth]: /experiments/images/JNet_456_3_output_depth.png
[JNet_456_3_output_plane]: /experiments/images/JNet_456_3_output_plane.png
[JNet_456_4_label_depth]: /experiments/images/JNet_456_4_label_depth.png
[JNet_456_4_label_plane]: /experiments/images/JNet_456_4_label_plane.png
[JNet_456_4_original_depth]: /experiments/images/JNet_456_4_original_depth.png
[JNet_456_4_original_plane]: /experiments/images/JNet_456_4_original_plane.png
[JNet_456_4_output_depth]: /experiments/images/JNet_456_4_output_depth.png
[JNet_456_4_output_plane]: /experiments/images/JNet_456_4_output_plane.png
[JNet_456_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_456_beads_001_roi000_heatmap_depth.png
[JNet_456_beads_001_roi000_original_depth]: /experiments/images/JNet_456_beads_001_roi000_original_depth.png
[JNet_456_beads_001_roi000_output_depth]: /experiments/images/JNet_456_beads_001_roi000_output_depth.png
[JNet_456_beads_001_roi000_reconst_depth]: /experiments/images/JNet_456_beads_001_roi000_reconst_depth.png
[JNet_456_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_456_beads_001_roi001_heatmap_depth.png
[JNet_456_beads_001_roi001_original_depth]: /experiments/images/JNet_456_beads_001_roi001_original_depth.png
[JNet_456_beads_001_roi001_output_depth]: /experiments/images/JNet_456_beads_001_roi001_output_depth.png
[JNet_456_beads_001_roi001_reconst_depth]: /experiments/images/JNet_456_beads_001_roi001_reconst_depth.png
[JNet_456_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_456_beads_001_roi002_heatmap_depth.png
[JNet_456_beads_001_roi002_original_depth]: /experiments/images/JNet_456_beads_001_roi002_original_depth.png
[JNet_456_beads_001_roi002_output_depth]: /experiments/images/JNet_456_beads_001_roi002_output_depth.png
[JNet_456_beads_001_roi002_reconst_depth]: /experiments/images/JNet_456_beads_001_roi002_reconst_depth.png
[JNet_456_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_456_beads_001_roi003_heatmap_depth.png
[JNet_456_beads_001_roi003_original_depth]: /experiments/images/JNet_456_beads_001_roi003_original_depth.png
[JNet_456_beads_001_roi003_output_depth]: /experiments/images/JNet_456_beads_001_roi003_output_depth.png
[JNet_456_beads_001_roi003_reconst_depth]: /experiments/images/JNet_456_beads_001_roi003_reconst_depth.png
[JNet_456_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_456_beads_001_roi004_heatmap_depth.png
[JNet_456_beads_001_roi004_original_depth]: /experiments/images/JNet_456_beads_001_roi004_original_depth.png
[JNet_456_beads_001_roi004_output_depth]: /experiments/images/JNet_456_beads_001_roi004_output_depth.png
[JNet_456_beads_001_roi004_reconst_depth]: /experiments/images/JNet_456_beads_001_roi004_reconst_depth.png
[JNet_456_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_456_beads_002_roi000_heatmap_depth.png
[JNet_456_beads_002_roi000_original_depth]: /experiments/images/JNet_456_beads_002_roi000_original_depth.png
[JNet_456_beads_002_roi000_output_depth]: /experiments/images/JNet_456_beads_002_roi000_output_depth.png
[JNet_456_beads_002_roi000_reconst_depth]: /experiments/images/JNet_456_beads_002_roi000_reconst_depth.png
[JNet_456_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_456_beads_002_roi001_heatmap_depth.png
[JNet_456_beads_002_roi001_original_depth]: /experiments/images/JNet_456_beads_002_roi001_original_depth.png
[JNet_456_beads_002_roi001_output_depth]: /experiments/images/JNet_456_beads_002_roi001_output_depth.png
[JNet_456_beads_002_roi001_reconst_depth]: /experiments/images/JNet_456_beads_002_roi001_reconst_depth.png
[JNet_456_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_456_beads_002_roi002_heatmap_depth.png
[JNet_456_beads_002_roi002_original_depth]: /experiments/images/JNet_456_beads_002_roi002_original_depth.png
[JNet_456_beads_002_roi002_output_depth]: /experiments/images/JNet_456_beads_002_roi002_output_depth.png
[JNet_456_beads_002_roi002_reconst_depth]: /experiments/images/JNet_456_beads_002_roi002_reconst_depth.png
[JNet_456_psf_post]: /experiments/images/JNet_456_psf_post.png
[JNet_456_psf_pre]: /experiments/images/JNet_456_psf_pre.png
[finetuned]: /experiments/tmp/JNet_456_train.png
[pretrained_model]: /experiments/tmp/JNet_454_pretrain_train.png
