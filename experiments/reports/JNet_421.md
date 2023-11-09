



# JNet_421 Report
  
the parameters to replicate the results of JNet_421. nearest interp of PSF, logit loss = 1.0, NA = 1.0 vq loss 1 psf upsampling by 5  
pretrained model : JNet_420_pretrain
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
|mu_z|0.1||
|sig_z|0.1||
|blur_mode|gibsonlanni|`gaussian` or `gibsonlanni`|
|size_x|51||
|size_y|51||
|size_z|201||
|NA|1.0||
|wavelength|0.91|microns|
|M|25|magnification|
|ns|1.4|specimen refractive index (RI)|
|ng0|1.5|coverslip RI design value|
|ng|1.5|coverslip RI experimental value|
|ni0|1.5|immersion medium RI design value|
|ni|1.5|immersion medium RI experimental value|
|ti0|150|microns, working distance (immersion medium thickness) design value|
|tg0|170|microns, coverslip thickness design value|
|tg|170|microns, coverslip thickness experimental value|
|res_lateral|0.05|microns|
|res_axial|0.5|microns|
|pZ|0|microns, particle distance from coverslip|
|bet_z|30.0||
|bet_xy|3.0||
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
  
mean MSE: 0.018161943182349205, mean BCE: 0.061405062675476074
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_420_pretrain_0_original_plane]|![JNet_420_pretrain_0_output_plane]|![JNet_420_pretrain_0_label_plane]|
  
MSE: 0.01940298080444336, BCE: 0.06622903794050217  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_420_pretrain_0_original_depth]|![JNet_420_pretrain_0_output_depth]|![JNet_420_pretrain_0_label_depth]|
  
MSE: 0.01940298080444336, BCE: 0.06622903794050217  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_420_pretrain_1_original_plane]|![JNet_420_pretrain_1_output_plane]|![JNet_420_pretrain_1_label_plane]|
  
MSE: 0.015357987023890018, BCE: 0.05005922541022301  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_420_pretrain_1_original_depth]|![JNet_420_pretrain_1_output_depth]|![JNet_420_pretrain_1_label_depth]|
  
MSE: 0.015357987023890018, BCE: 0.05005922541022301  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_420_pretrain_2_original_plane]|![JNet_420_pretrain_2_output_plane]|![JNet_420_pretrain_2_label_plane]|
  
MSE: 0.014921176247298717, BCE: 0.050272852182388306  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_420_pretrain_2_original_depth]|![JNet_420_pretrain_2_output_depth]|![JNet_420_pretrain_2_label_depth]|
  
MSE: 0.014921176247298717, BCE: 0.050272852182388306  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_420_pretrain_3_original_plane]|![JNet_420_pretrain_3_output_plane]|![JNet_420_pretrain_3_label_plane]|
  
MSE: 0.01570715755224228, BCE: 0.05163731426000595  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_420_pretrain_3_original_depth]|![JNet_420_pretrain_3_output_depth]|![JNet_420_pretrain_3_label_depth]|
  
MSE: 0.01570715755224228, BCE: 0.05163731426000595  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_420_pretrain_4_original_plane]|![JNet_420_pretrain_4_output_plane]|![JNet_420_pretrain_4_label_plane]|
  
MSE: 0.025420410558581352, BCE: 0.08882690221071243  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_420_pretrain_4_original_depth]|![JNet_420_pretrain_4_output_depth]|![JNet_420_pretrain_4_label_depth]|
  
MSE: 0.025420410558581352, BCE: 0.08882690221071243  
  
mean MSE: 0.028830990195274353, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_421_0_original_plane]|![JNet_421_0_output_plane]|![JNet_421_0_label_plane]|
  
MSE: 0.03599978983402252, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_421_0_original_depth]|![JNet_421_0_output_depth]|![JNet_421_0_label_depth]|
  
MSE: 0.03599978983402252, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_421_1_original_plane]|![JNet_421_1_output_plane]|![JNet_421_1_label_plane]|
  
MSE: 0.03254789113998413, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_421_1_original_depth]|![JNet_421_1_output_depth]|![JNet_421_1_label_depth]|
  
MSE: 0.03254789113998413, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_421_2_original_plane]|![JNet_421_2_output_plane]|![JNet_421_2_label_plane]|
  
MSE: 0.01945018209517002, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_421_2_original_depth]|![JNet_421_2_output_depth]|![JNet_421_2_label_depth]|
  
MSE: 0.01945018209517002, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_421_3_original_plane]|![JNet_421_3_output_plane]|![JNet_421_3_label_plane]|
  
MSE: 0.027806490659713745, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_421_3_original_depth]|![JNet_421_3_output_depth]|![JNet_421_3_label_depth]|
  
MSE: 0.027806490659713745, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_421_4_original_plane]|![JNet_421_4_output_plane]|![JNet_421_4_label_plane]|
  
MSE: 0.028350593522191048, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_421_4_original_depth]|![JNet_421_4_output_depth]|![JNet_421_4_label_depth]|
  
MSE: 0.028350593522191048, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_420_pretrain_beads_001_roi000_original_depth]|![JNet_420_pretrain_beads_001_roi000_output_depth]|![JNet_420_pretrain_beads_001_roi000_reconst_depth]|![JNet_420_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 7.295744140625001, MSE: 0.015395931899547577, quantized loss: 0.003793770680204034  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_420_pretrain_beads_001_roi001_original_depth]|![JNet_420_pretrain_beads_001_roi001_output_depth]|![JNet_420_pretrain_beads_001_roi001_reconst_depth]|![JNet_420_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 8.235795898437502, MSE: 0.014568012207746506, quantized loss: 0.003992475103586912  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_420_pretrain_beads_001_roi002_original_depth]|![JNet_420_pretrain_beads_001_roi002_output_depth]|![JNet_420_pretrain_beads_001_roi002_reconst_depth]|![JNet_420_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 7.577801757812502, MSE: 0.015992525964975357, quantized loss: 0.003828598652034998  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_420_pretrain_beads_001_roi003_original_depth]|![JNet_420_pretrain_beads_001_roi003_output_depth]|![JNet_420_pretrain_beads_001_roi003_reconst_depth]|![JNet_420_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 8.396110351562502, MSE: 0.017816022038459778, quantized loss: 0.003974582068622112  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_420_pretrain_beads_001_roi004_original_depth]|![JNet_420_pretrain_beads_001_roi004_output_depth]|![JNet_420_pretrain_beads_001_roi004_reconst_depth]|![JNet_420_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 8.533287109375003, MSE: 0.020156702026724815, quantized loss: 0.004319831728935242  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_420_pretrain_beads_002_roi000_original_depth]|![JNet_420_pretrain_beads_002_roi000_output_depth]|![JNet_420_pretrain_beads_002_roi000_reconst_depth]|![JNet_420_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 9.100151367187502, MSE: 0.022097395732998848, quantized loss: 0.0045501189306378365  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_420_pretrain_beads_002_roi001_original_depth]|![JNet_420_pretrain_beads_002_roi001_output_depth]|![JNet_420_pretrain_beads_002_roi001_reconst_depth]|![JNet_420_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 8.472395507812502, MSE: 0.02038676105439663, quantized loss: 0.00429356237873435  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_420_pretrain_beads_002_roi002_original_depth]|![JNet_420_pretrain_beads_002_roi002_output_depth]|![JNet_420_pretrain_beads_002_roi002_reconst_depth]|![JNet_420_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 8.637487304687502, MSE: 0.020389240235090256, quantized loss: 0.004343051929026842  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_421_beads_001_roi000_original_depth]|![JNet_421_beads_001_roi000_output_depth]|![JNet_421_beads_001_roi000_reconst_depth]|![JNet_421_beads_001_roi000_heatmap_depth]|
  
volume: 2.2688022460937507, MSE: 0.00016256245726253837, quantized loss: 1.2770208741130773e-05  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_421_beads_001_roi001_original_depth]|![JNet_421_beads_001_roi001_output_depth]|![JNet_421_beads_001_roi001_reconst_depth]|![JNet_421_beads_001_roi001_heatmap_depth]|
  
volume: 3.480225341796876, MSE: 0.0007498218328692019, quantized loss: 1.696621438895818e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_421_beads_001_roi002_original_depth]|![JNet_421_beads_001_roi002_output_depth]|![JNet_421_beads_001_roi002_reconst_depth]|![JNet_421_beads_001_roi002_heatmap_depth]|
  
volume: 2.3117998046875003, MSE: 0.0001039532435243018, quantized loss: 1.1205198461539112e-05  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_421_beads_001_roi003_original_depth]|![JNet_421_beads_001_roi003_output_depth]|![JNet_421_beads_001_roi003_reconst_depth]|![JNet_421_beads_001_roi003_heatmap_depth]|
  
volume: 3.4761865234375007, MSE: 0.0005546518368646502, quantized loss: 1.5791485566296615e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_421_beads_001_roi004_original_depth]|![JNet_421_beads_001_roi004_output_depth]|![JNet_421_beads_001_roi004_reconst_depth]|![JNet_421_beads_001_roi004_heatmap_depth]|
  
volume: 2.3619519042968755, MSE: 0.00017383090744260699, quantized loss: 1.1919149983441457e-05  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_421_beads_002_roi000_original_depth]|![JNet_421_beads_002_roi000_output_depth]|![JNet_421_beads_002_roi000_reconst_depth]|![JNet_421_beads_002_roi000_heatmap_depth]|
  
volume: 2.4549433593750005, MSE: 0.00023545033764094114, quantized loss: 1.1367190381861292e-05  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_421_beads_002_roi001_original_depth]|![JNet_421_beads_002_roi001_output_depth]|![JNet_421_beads_002_roi001_reconst_depth]|![JNet_421_beads_002_roi001_heatmap_depth]|
  
volume: 2.2946113281250007, MSE: 0.00013123205280862749, quantized loss: 1.2191106179670896e-05  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_421_beads_002_roi002_original_depth]|![JNet_421_beads_002_roi002_output_depth]|![JNet_421_beads_002_roi002_reconst_depth]|![JNet_421_beads_002_roi002_heatmap_depth]|
  
volume: 2.3806816406250007, MSE: 0.00016825621423777193, quantized loss: 1.1590139365580399e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_421_psf_pre]|![JNet_421_psf_post]|

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
  



[JNet_420_pretrain_0_label_depth]: /experiments/images/JNet_420_pretrain_0_label_depth.png
[JNet_420_pretrain_0_label_plane]: /experiments/images/JNet_420_pretrain_0_label_plane.png
[JNet_420_pretrain_0_original_depth]: /experiments/images/JNet_420_pretrain_0_original_depth.png
[JNet_420_pretrain_0_original_plane]: /experiments/images/JNet_420_pretrain_0_original_plane.png
[JNet_420_pretrain_0_output_depth]: /experiments/images/JNet_420_pretrain_0_output_depth.png
[JNet_420_pretrain_0_output_plane]: /experiments/images/JNet_420_pretrain_0_output_plane.png
[JNet_420_pretrain_1_label_depth]: /experiments/images/JNet_420_pretrain_1_label_depth.png
[JNet_420_pretrain_1_label_plane]: /experiments/images/JNet_420_pretrain_1_label_plane.png
[JNet_420_pretrain_1_original_depth]: /experiments/images/JNet_420_pretrain_1_original_depth.png
[JNet_420_pretrain_1_original_plane]: /experiments/images/JNet_420_pretrain_1_original_plane.png
[JNet_420_pretrain_1_output_depth]: /experiments/images/JNet_420_pretrain_1_output_depth.png
[JNet_420_pretrain_1_output_plane]: /experiments/images/JNet_420_pretrain_1_output_plane.png
[JNet_420_pretrain_2_label_depth]: /experiments/images/JNet_420_pretrain_2_label_depth.png
[JNet_420_pretrain_2_label_plane]: /experiments/images/JNet_420_pretrain_2_label_plane.png
[JNet_420_pretrain_2_original_depth]: /experiments/images/JNet_420_pretrain_2_original_depth.png
[JNet_420_pretrain_2_original_plane]: /experiments/images/JNet_420_pretrain_2_original_plane.png
[JNet_420_pretrain_2_output_depth]: /experiments/images/JNet_420_pretrain_2_output_depth.png
[JNet_420_pretrain_2_output_plane]: /experiments/images/JNet_420_pretrain_2_output_plane.png
[JNet_420_pretrain_3_label_depth]: /experiments/images/JNet_420_pretrain_3_label_depth.png
[JNet_420_pretrain_3_label_plane]: /experiments/images/JNet_420_pretrain_3_label_plane.png
[JNet_420_pretrain_3_original_depth]: /experiments/images/JNet_420_pretrain_3_original_depth.png
[JNet_420_pretrain_3_original_plane]: /experiments/images/JNet_420_pretrain_3_original_plane.png
[JNet_420_pretrain_3_output_depth]: /experiments/images/JNet_420_pretrain_3_output_depth.png
[JNet_420_pretrain_3_output_plane]: /experiments/images/JNet_420_pretrain_3_output_plane.png
[JNet_420_pretrain_4_label_depth]: /experiments/images/JNet_420_pretrain_4_label_depth.png
[JNet_420_pretrain_4_label_plane]: /experiments/images/JNet_420_pretrain_4_label_plane.png
[JNet_420_pretrain_4_original_depth]: /experiments/images/JNet_420_pretrain_4_original_depth.png
[JNet_420_pretrain_4_original_plane]: /experiments/images/JNet_420_pretrain_4_original_plane.png
[JNet_420_pretrain_4_output_depth]: /experiments/images/JNet_420_pretrain_4_output_depth.png
[JNet_420_pretrain_4_output_plane]: /experiments/images/JNet_420_pretrain_4_output_plane.png
[JNet_420_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_420_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi000_original_depth.png
[JNet_420_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi000_output_depth.png
[JNet_420_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi000_reconst_depth.png
[JNet_420_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_420_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi001_original_depth.png
[JNet_420_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi001_output_depth.png
[JNet_420_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi001_reconst_depth.png
[JNet_420_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_420_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi002_original_depth.png
[JNet_420_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi002_output_depth.png
[JNet_420_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi002_reconst_depth.png
[JNet_420_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_420_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi003_original_depth.png
[JNet_420_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi003_output_depth.png
[JNet_420_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi003_reconst_depth.png
[JNet_420_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_420_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi004_original_depth.png
[JNet_420_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi004_output_depth.png
[JNet_420_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_420_pretrain_beads_001_roi004_reconst_depth.png
[JNet_420_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_420_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi000_original_depth.png
[JNet_420_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi000_output_depth.png
[JNet_420_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi000_reconst_depth.png
[JNet_420_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_420_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi001_original_depth.png
[JNet_420_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi001_output_depth.png
[JNet_420_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi001_reconst_depth.png
[JNet_420_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_420_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi002_original_depth.png
[JNet_420_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi002_output_depth.png
[JNet_420_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_420_pretrain_beads_002_roi002_reconst_depth.png
[JNet_421_0_label_depth]: /experiments/images/JNet_421_0_label_depth.png
[JNet_421_0_label_plane]: /experiments/images/JNet_421_0_label_plane.png
[JNet_421_0_original_depth]: /experiments/images/JNet_421_0_original_depth.png
[JNet_421_0_original_plane]: /experiments/images/JNet_421_0_original_plane.png
[JNet_421_0_output_depth]: /experiments/images/JNet_421_0_output_depth.png
[JNet_421_0_output_plane]: /experiments/images/JNet_421_0_output_plane.png
[JNet_421_1_label_depth]: /experiments/images/JNet_421_1_label_depth.png
[JNet_421_1_label_plane]: /experiments/images/JNet_421_1_label_plane.png
[JNet_421_1_original_depth]: /experiments/images/JNet_421_1_original_depth.png
[JNet_421_1_original_plane]: /experiments/images/JNet_421_1_original_plane.png
[JNet_421_1_output_depth]: /experiments/images/JNet_421_1_output_depth.png
[JNet_421_1_output_plane]: /experiments/images/JNet_421_1_output_plane.png
[JNet_421_2_label_depth]: /experiments/images/JNet_421_2_label_depth.png
[JNet_421_2_label_plane]: /experiments/images/JNet_421_2_label_plane.png
[JNet_421_2_original_depth]: /experiments/images/JNet_421_2_original_depth.png
[JNet_421_2_original_plane]: /experiments/images/JNet_421_2_original_plane.png
[JNet_421_2_output_depth]: /experiments/images/JNet_421_2_output_depth.png
[JNet_421_2_output_plane]: /experiments/images/JNet_421_2_output_plane.png
[JNet_421_3_label_depth]: /experiments/images/JNet_421_3_label_depth.png
[JNet_421_3_label_plane]: /experiments/images/JNet_421_3_label_plane.png
[JNet_421_3_original_depth]: /experiments/images/JNet_421_3_original_depth.png
[JNet_421_3_original_plane]: /experiments/images/JNet_421_3_original_plane.png
[JNet_421_3_output_depth]: /experiments/images/JNet_421_3_output_depth.png
[JNet_421_3_output_plane]: /experiments/images/JNet_421_3_output_plane.png
[JNet_421_4_label_depth]: /experiments/images/JNet_421_4_label_depth.png
[JNet_421_4_label_plane]: /experiments/images/JNet_421_4_label_plane.png
[JNet_421_4_original_depth]: /experiments/images/JNet_421_4_original_depth.png
[JNet_421_4_original_plane]: /experiments/images/JNet_421_4_original_plane.png
[JNet_421_4_output_depth]: /experiments/images/JNet_421_4_output_depth.png
[JNet_421_4_output_plane]: /experiments/images/JNet_421_4_output_plane.png
[JNet_421_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_421_beads_001_roi000_heatmap_depth.png
[JNet_421_beads_001_roi000_original_depth]: /experiments/images/JNet_421_beads_001_roi000_original_depth.png
[JNet_421_beads_001_roi000_output_depth]: /experiments/images/JNet_421_beads_001_roi000_output_depth.png
[JNet_421_beads_001_roi000_reconst_depth]: /experiments/images/JNet_421_beads_001_roi000_reconst_depth.png
[JNet_421_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_421_beads_001_roi001_heatmap_depth.png
[JNet_421_beads_001_roi001_original_depth]: /experiments/images/JNet_421_beads_001_roi001_original_depth.png
[JNet_421_beads_001_roi001_output_depth]: /experiments/images/JNet_421_beads_001_roi001_output_depth.png
[JNet_421_beads_001_roi001_reconst_depth]: /experiments/images/JNet_421_beads_001_roi001_reconst_depth.png
[JNet_421_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_421_beads_001_roi002_heatmap_depth.png
[JNet_421_beads_001_roi002_original_depth]: /experiments/images/JNet_421_beads_001_roi002_original_depth.png
[JNet_421_beads_001_roi002_output_depth]: /experiments/images/JNet_421_beads_001_roi002_output_depth.png
[JNet_421_beads_001_roi002_reconst_depth]: /experiments/images/JNet_421_beads_001_roi002_reconst_depth.png
[JNet_421_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_421_beads_001_roi003_heatmap_depth.png
[JNet_421_beads_001_roi003_original_depth]: /experiments/images/JNet_421_beads_001_roi003_original_depth.png
[JNet_421_beads_001_roi003_output_depth]: /experiments/images/JNet_421_beads_001_roi003_output_depth.png
[JNet_421_beads_001_roi003_reconst_depth]: /experiments/images/JNet_421_beads_001_roi003_reconst_depth.png
[JNet_421_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_421_beads_001_roi004_heatmap_depth.png
[JNet_421_beads_001_roi004_original_depth]: /experiments/images/JNet_421_beads_001_roi004_original_depth.png
[JNet_421_beads_001_roi004_output_depth]: /experiments/images/JNet_421_beads_001_roi004_output_depth.png
[JNet_421_beads_001_roi004_reconst_depth]: /experiments/images/JNet_421_beads_001_roi004_reconst_depth.png
[JNet_421_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_421_beads_002_roi000_heatmap_depth.png
[JNet_421_beads_002_roi000_original_depth]: /experiments/images/JNet_421_beads_002_roi000_original_depth.png
[JNet_421_beads_002_roi000_output_depth]: /experiments/images/JNet_421_beads_002_roi000_output_depth.png
[JNet_421_beads_002_roi000_reconst_depth]: /experiments/images/JNet_421_beads_002_roi000_reconst_depth.png
[JNet_421_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_421_beads_002_roi001_heatmap_depth.png
[JNet_421_beads_002_roi001_original_depth]: /experiments/images/JNet_421_beads_002_roi001_original_depth.png
[JNet_421_beads_002_roi001_output_depth]: /experiments/images/JNet_421_beads_002_roi001_output_depth.png
[JNet_421_beads_002_roi001_reconst_depth]: /experiments/images/JNet_421_beads_002_roi001_reconst_depth.png
[JNet_421_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_421_beads_002_roi002_heatmap_depth.png
[JNet_421_beads_002_roi002_original_depth]: /experiments/images/JNet_421_beads_002_roi002_original_depth.png
[JNet_421_beads_002_roi002_output_depth]: /experiments/images/JNet_421_beads_002_roi002_output_depth.png
[JNet_421_beads_002_roi002_reconst_depth]: /experiments/images/JNet_421_beads_002_roi002_reconst_depth.png
[JNet_421_psf_post]: /experiments/images/JNet_421_psf_post.png
[JNet_421_psf_pre]: /experiments/images/JNet_421_psf_pre.png
[finetuned]: /experiments/tmp/JNet_421_train.png
[pretrained_model]: /experiments/tmp/JNet_420_pretrain_train.png
