



# JNet_414 Report
  
the parameters to replicate the results of JNet_414. nearest interp of PSF, logit loss = 0.6, NA = 0.8  
pretrained model : JNet_413_pretrain
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
|NA|0.8||
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
|loss_fn|nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.6], device=params['device']))|
|path|model|
|savefig_path|train|
|partial|params['partial']|
|ewc|None|
|params|params|
|es_patience|20|
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
|es_patience|20|
|reconstruct|True|
|is_instantblur|False|
|is_vibrate|True|
|loss_weight|1|
|qloss_weight|0.1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.022719386965036392, mean BCE: 0.08204274624586105
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_413_pretrain_0_original_plane]|![JNet_413_pretrain_0_output_plane]|![JNet_413_pretrain_0_label_plane]|
  
MSE: 0.017851537093520164, BCE: 0.06215093657374382  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_413_pretrain_0_original_depth]|![JNet_413_pretrain_0_output_depth]|![JNet_413_pretrain_0_label_depth]|
  
MSE: 0.017851537093520164, BCE: 0.06215093657374382  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_413_pretrain_1_original_plane]|![JNet_413_pretrain_1_output_plane]|![JNet_413_pretrain_1_label_plane]|
  
MSE: 0.0192185677587986, BCE: 0.06874255836009979  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_413_pretrain_1_original_depth]|![JNet_413_pretrain_1_output_depth]|![JNet_413_pretrain_1_label_depth]|
  
MSE: 0.0192185677587986, BCE: 0.06874255836009979  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_413_pretrain_2_original_plane]|![JNet_413_pretrain_2_output_plane]|![JNet_413_pretrain_2_label_plane]|
  
MSE: 0.026960650458931923, BCE: 0.10295401513576508  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_413_pretrain_2_original_depth]|![JNet_413_pretrain_2_output_depth]|![JNet_413_pretrain_2_label_depth]|
  
MSE: 0.026960650458931923, BCE: 0.10295401513576508  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_413_pretrain_3_original_plane]|![JNet_413_pretrain_3_output_plane]|![JNet_413_pretrain_3_label_plane]|
  
MSE: 0.022496966645121574, BCE: 0.08075548708438873  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_413_pretrain_3_original_depth]|![JNet_413_pretrain_3_output_depth]|![JNet_413_pretrain_3_label_depth]|
  
MSE: 0.022496966645121574, BCE: 0.08075548708438873  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_413_pretrain_4_original_plane]|![JNet_413_pretrain_4_output_plane]|![JNet_413_pretrain_4_label_plane]|
  
MSE: 0.0270692091435194, BCE: 0.09561073035001755  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_413_pretrain_4_original_depth]|![JNet_413_pretrain_4_output_depth]|![JNet_413_pretrain_4_label_depth]|
  
MSE: 0.0270692091435194, BCE: 0.09561073035001755  
  
mean MSE: 0.032813332974910736, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_414_0_original_plane]|![JNet_414_0_output_plane]|![JNet_414_0_label_plane]|
  
MSE: 0.02580493502318859, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_414_0_original_depth]|![JNet_414_0_output_depth]|![JNet_414_0_label_depth]|
  
MSE: 0.02580493502318859, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_414_1_original_plane]|![JNet_414_1_output_plane]|![JNet_414_1_label_plane]|
  
MSE: 0.0338996946811676, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_414_1_original_depth]|![JNet_414_1_output_depth]|![JNet_414_1_label_depth]|
  
MSE: 0.0338996946811676, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_414_2_original_plane]|![JNet_414_2_output_plane]|![JNet_414_2_label_plane]|
  
MSE: 0.03321453928947449, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_414_2_original_depth]|![JNet_414_2_output_depth]|![JNet_414_2_label_depth]|
  
MSE: 0.03321453928947449, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_414_3_original_plane]|![JNet_414_3_output_plane]|![JNet_414_3_label_plane]|
  
MSE: 0.030010640621185303, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_414_3_original_depth]|![JNet_414_3_output_depth]|![JNet_414_3_label_depth]|
  
MSE: 0.030010640621185303, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_414_4_original_plane]|![JNet_414_4_output_plane]|![JNet_414_4_label_plane]|
  
MSE: 0.04113685339689255, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_414_4_original_depth]|![JNet_414_4_output_depth]|![JNet_414_4_label_depth]|
  
MSE: 0.04113685339689255, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_413_pretrain_beads_001_roi000_original_depth]|![JNet_413_pretrain_beads_001_roi000_output_depth]|![JNet_413_pretrain_beads_001_roi000_reconst_depth]|![JNet_413_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 1.0495837402343753, MSE: 0.003450176678597927, quantized loss: 0.000204894517082721  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_413_pretrain_beads_001_roi001_original_depth]|![JNet_413_pretrain_beads_001_roi001_output_depth]|![JNet_413_pretrain_beads_001_roi001_reconst_depth]|![JNet_413_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 1.3586280517578129, MSE: 0.006157765164971352, quantized loss: 0.0003017801500391215  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_413_pretrain_beads_001_roi002_original_depth]|![JNet_413_pretrain_beads_001_roi002_output_depth]|![JNet_413_pretrain_beads_001_roi002_reconst_depth]|![JNet_413_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 1.1696245117187503, MSE: 0.0029298418667167425, quantized loss: 0.00026418629568070173  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_413_pretrain_beads_001_roi003_original_depth]|![JNet_413_pretrain_beads_001_roi003_output_depth]|![JNet_413_pretrain_beads_001_roi003_reconst_depth]|![JNet_413_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 1.5102644042968754, MSE: 0.005896488670259714, quantized loss: 0.0004139603115618229  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_413_pretrain_beads_001_roi004_original_depth]|![JNet_413_pretrain_beads_001_roi004_output_depth]|![JNet_413_pretrain_beads_001_roi004_reconst_depth]|![JNet_413_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 1.2534108886718753, MSE: 0.0032633983064442873, quantized loss: 0.0003478886792436242  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_413_pretrain_beads_002_roi000_original_depth]|![JNet_413_pretrain_beads_002_roi000_output_depth]|![JNet_413_pretrain_beads_002_roi000_reconst_depth]|![JNet_413_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 1.3295063476562503, MSE: 0.0036301380023360252, quantized loss: 0.0003824200539384037  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_413_pretrain_beads_002_roi001_original_depth]|![JNet_413_pretrain_beads_002_roi001_output_depth]|![JNet_413_pretrain_beads_002_roi001_reconst_depth]|![JNet_413_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 1.293590942382813, MSE: 0.0028572941664606333, quantized loss: 0.0003606198006309569  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_413_pretrain_beads_002_roi002_original_depth]|![JNet_413_pretrain_beads_002_roi002_output_depth]|![JNet_413_pretrain_beads_002_roi002_reconst_depth]|![JNet_413_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 1.2753571777343753, MSE: 0.0034566528629511595, quantized loss: 0.0003223131352569908  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_414_beads_001_roi000_original_depth]|![JNet_414_beads_001_roi000_output_depth]|![JNet_414_beads_001_roi000_reconst_depth]|![JNet_414_beads_001_roi000_heatmap_depth]|
  
volume: 1.1856271972656254, MSE: 0.0002065822045551613, quantized loss: 0.00014797813491895795  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_414_beads_001_roi001_original_depth]|![JNet_414_beads_001_roi001_output_depth]|![JNet_414_beads_001_roi001_reconst_depth]|![JNet_414_beads_001_roi001_heatmap_depth]|
  
volume: 1.790310668945313, MSE: 0.0006429263739846647, quantized loss: 0.00019710557535290718  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_414_beads_001_roi002_original_depth]|![JNet_414_beads_001_roi002_output_depth]|![JNet_414_beads_001_roi002_reconst_depth]|![JNet_414_beads_001_roi002_heatmap_depth]|
  
volume: 1.1526127929687502, MSE: 0.000169888895470649, quantized loss: 0.00013359238801058382  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_414_beads_001_roi003_original_depth]|![JNet_414_beads_001_roi003_output_depth]|![JNet_414_beads_001_roi003_reconst_depth]|![JNet_414_beads_001_roi003_heatmap_depth]|
  
volume: 1.984852905273438, MSE: 0.00034138039336539805, quantized loss: 0.00020179302373435348  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_414_beads_001_roi004_original_depth]|![JNet_414_beads_001_roi004_output_depth]|![JNet_414_beads_001_roi004_reconst_depth]|![JNet_414_beads_001_roi004_heatmap_depth]|
  
volume: 1.2599068603515629, MSE: 0.00015417397662531585, quantized loss: 0.00012565967335831374  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_414_beads_002_roi000_original_depth]|![JNet_414_beads_002_roi000_output_depth]|![JNet_414_beads_002_roi000_reconst_depth]|![JNet_414_beads_002_roi000_heatmap_depth]|
  
volume: 1.364194458007813, MSE: 0.00014818846830166876, quantized loss: 0.00011881910177180544  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_414_beads_002_roi001_original_depth]|![JNet_414_beads_002_roi001_output_depth]|![JNet_414_beads_002_roi001_reconst_depth]|![JNet_414_beads_002_roi001_heatmap_depth]|
  
volume: 1.2704780273437504, MSE: 0.0001538552314741537, quantized loss: 0.00011963415454374626  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_414_beads_002_roi002_original_depth]|![JNet_414_beads_002_roi002_output_depth]|![JNet_414_beads_002_roi002_reconst_depth]|![JNet_414_beads_002_roi002_heatmap_depth]|
  
volume: 1.3112641601562502, MSE: 0.0001444339141016826, quantized loss: 0.00012109369708923623  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_414_psf_pre]|![JNet_414_psf_post]|

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
  



[JNet_413_pretrain_0_label_depth]: /experiments/images/JNet_413_pretrain_0_label_depth.png
[JNet_413_pretrain_0_label_plane]: /experiments/images/JNet_413_pretrain_0_label_plane.png
[JNet_413_pretrain_0_original_depth]: /experiments/images/JNet_413_pretrain_0_original_depth.png
[JNet_413_pretrain_0_original_plane]: /experiments/images/JNet_413_pretrain_0_original_plane.png
[JNet_413_pretrain_0_output_depth]: /experiments/images/JNet_413_pretrain_0_output_depth.png
[JNet_413_pretrain_0_output_plane]: /experiments/images/JNet_413_pretrain_0_output_plane.png
[JNet_413_pretrain_1_label_depth]: /experiments/images/JNet_413_pretrain_1_label_depth.png
[JNet_413_pretrain_1_label_plane]: /experiments/images/JNet_413_pretrain_1_label_plane.png
[JNet_413_pretrain_1_original_depth]: /experiments/images/JNet_413_pretrain_1_original_depth.png
[JNet_413_pretrain_1_original_plane]: /experiments/images/JNet_413_pretrain_1_original_plane.png
[JNet_413_pretrain_1_output_depth]: /experiments/images/JNet_413_pretrain_1_output_depth.png
[JNet_413_pretrain_1_output_plane]: /experiments/images/JNet_413_pretrain_1_output_plane.png
[JNet_413_pretrain_2_label_depth]: /experiments/images/JNet_413_pretrain_2_label_depth.png
[JNet_413_pretrain_2_label_plane]: /experiments/images/JNet_413_pretrain_2_label_plane.png
[JNet_413_pretrain_2_original_depth]: /experiments/images/JNet_413_pretrain_2_original_depth.png
[JNet_413_pretrain_2_original_plane]: /experiments/images/JNet_413_pretrain_2_original_plane.png
[JNet_413_pretrain_2_output_depth]: /experiments/images/JNet_413_pretrain_2_output_depth.png
[JNet_413_pretrain_2_output_plane]: /experiments/images/JNet_413_pretrain_2_output_plane.png
[JNet_413_pretrain_3_label_depth]: /experiments/images/JNet_413_pretrain_3_label_depth.png
[JNet_413_pretrain_3_label_plane]: /experiments/images/JNet_413_pretrain_3_label_plane.png
[JNet_413_pretrain_3_original_depth]: /experiments/images/JNet_413_pretrain_3_original_depth.png
[JNet_413_pretrain_3_original_plane]: /experiments/images/JNet_413_pretrain_3_original_plane.png
[JNet_413_pretrain_3_output_depth]: /experiments/images/JNet_413_pretrain_3_output_depth.png
[JNet_413_pretrain_3_output_plane]: /experiments/images/JNet_413_pretrain_3_output_plane.png
[JNet_413_pretrain_4_label_depth]: /experiments/images/JNet_413_pretrain_4_label_depth.png
[JNet_413_pretrain_4_label_plane]: /experiments/images/JNet_413_pretrain_4_label_plane.png
[JNet_413_pretrain_4_original_depth]: /experiments/images/JNet_413_pretrain_4_original_depth.png
[JNet_413_pretrain_4_original_plane]: /experiments/images/JNet_413_pretrain_4_original_plane.png
[JNet_413_pretrain_4_output_depth]: /experiments/images/JNet_413_pretrain_4_output_depth.png
[JNet_413_pretrain_4_output_plane]: /experiments/images/JNet_413_pretrain_4_output_plane.png
[JNet_413_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_413_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi000_original_depth.png
[JNet_413_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi000_output_depth.png
[JNet_413_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi000_reconst_depth.png
[JNet_413_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_413_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi001_original_depth.png
[JNet_413_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi001_output_depth.png
[JNet_413_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi001_reconst_depth.png
[JNet_413_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_413_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi002_original_depth.png
[JNet_413_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi002_output_depth.png
[JNet_413_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi002_reconst_depth.png
[JNet_413_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_413_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi003_original_depth.png
[JNet_413_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi003_output_depth.png
[JNet_413_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi003_reconst_depth.png
[JNet_413_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_413_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi004_original_depth.png
[JNet_413_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi004_output_depth.png
[JNet_413_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_413_pretrain_beads_001_roi004_reconst_depth.png
[JNet_413_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_413_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi000_original_depth.png
[JNet_413_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi000_output_depth.png
[JNet_413_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi000_reconst_depth.png
[JNet_413_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_413_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi001_original_depth.png
[JNet_413_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi001_output_depth.png
[JNet_413_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi001_reconst_depth.png
[JNet_413_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_413_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi002_original_depth.png
[JNet_413_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi002_output_depth.png
[JNet_413_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_413_pretrain_beads_002_roi002_reconst_depth.png
[JNet_414_0_label_depth]: /experiments/images/JNet_414_0_label_depth.png
[JNet_414_0_label_plane]: /experiments/images/JNet_414_0_label_plane.png
[JNet_414_0_original_depth]: /experiments/images/JNet_414_0_original_depth.png
[JNet_414_0_original_plane]: /experiments/images/JNet_414_0_original_plane.png
[JNet_414_0_output_depth]: /experiments/images/JNet_414_0_output_depth.png
[JNet_414_0_output_plane]: /experiments/images/JNet_414_0_output_plane.png
[JNet_414_1_label_depth]: /experiments/images/JNet_414_1_label_depth.png
[JNet_414_1_label_plane]: /experiments/images/JNet_414_1_label_plane.png
[JNet_414_1_original_depth]: /experiments/images/JNet_414_1_original_depth.png
[JNet_414_1_original_plane]: /experiments/images/JNet_414_1_original_plane.png
[JNet_414_1_output_depth]: /experiments/images/JNet_414_1_output_depth.png
[JNet_414_1_output_plane]: /experiments/images/JNet_414_1_output_plane.png
[JNet_414_2_label_depth]: /experiments/images/JNet_414_2_label_depth.png
[JNet_414_2_label_plane]: /experiments/images/JNet_414_2_label_plane.png
[JNet_414_2_original_depth]: /experiments/images/JNet_414_2_original_depth.png
[JNet_414_2_original_plane]: /experiments/images/JNet_414_2_original_plane.png
[JNet_414_2_output_depth]: /experiments/images/JNet_414_2_output_depth.png
[JNet_414_2_output_plane]: /experiments/images/JNet_414_2_output_plane.png
[JNet_414_3_label_depth]: /experiments/images/JNet_414_3_label_depth.png
[JNet_414_3_label_plane]: /experiments/images/JNet_414_3_label_plane.png
[JNet_414_3_original_depth]: /experiments/images/JNet_414_3_original_depth.png
[JNet_414_3_original_plane]: /experiments/images/JNet_414_3_original_plane.png
[JNet_414_3_output_depth]: /experiments/images/JNet_414_3_output_depth.png
[JNet_414_3_output_plane]: /experiments/images/JNet_414_3_output_plane.png
[JNet_414_4_label_depth]: /experiments/images/JNet_414_4_label_depth.png
[JNet_414_4_label_plane]: /experiments/images/JNet_414_4_label_plane.png
[JNet_414_4_original_depth]: /experiments/images/JNet_414_4_original_depth.png
[JNet_414_4_original_plane]: /experiments/images/JNet_414_4_original_plane.png
[JNet_414_4_output_depth]: /experiments/images/JNet_414_4_output_depth.png
[JNet_414_4_output_plane]: /experiments/images/JNet_414_4_output_plane.png
[JNet_414_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_414_beads_001_roi000_heatmap_depth.png
[JNet_414_beads_001_roi000_original_depth]: /experiments/images/JNet_414_beads_001_roi000_original_depth.png
[JNet_414_beads_001_roi000_output_depth]: /experiments/images/JNet_414_beads_001_roi000_output_depth.png
[JNet_414_beads_001_roi000_reconst_depth]: /experiments/images/JNet_414_beads_001_roi000_reconst_depth.png
[JNet_414_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_414_beads_001_roi001_heatmap_depth.png
[JNet_414_beads_001_roi001_original_depth]: /experiments/images/JNet_414_beads_001_roi001_original_depth.png
[JNet_414_beads_001_roi001_output_depth]: /experiments/images/JNet_414_beads_001_roi001_output_depth.png
[JNet_414_beads_001_roi001_reconst_depth]: /experiments/images/JNet_414_beads_001_roi001_reconst_depth.png
[JNet_414_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_414_beads_001_roi002_heatmap_depth.png
[JNet_414_beads_001_roi002_original_depth]: /experiments/images/JNet_414_beads_001_roi002_original_depth.png
[JNet_414_beads_001_roi002_output_depth]: /experiments/images/JNet_414_beads_001_roi002_output_depth.png
[JNet_414_beads_001_roi002_reconst_depth]: /experiments/images/JNet_414_beads_001_roi002_reconst_depth.png
[JNet_414_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_414_beads_001_roi003_heatmap_depth.png
[JNet_414_beads_001_roi003_original_depth]: /experiments/images/JNet_414_beads_001_roi003_original_depth.png
[JNet_414_beads_001_roi003_output_depth]: /experiments/images/JNet_414_beads_001_roi003_output_depth.png
[JNet_414_beads_001_roi003_reconst_depth]: /experiments/images/JNet_414_beads_001_roi003_reconst_depth.png
[JNet_414_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_414_beads_001_roi004_heatmap_depth.png
[JNet_414_beads_001_roi004_original_depth]: /experiments/images/JNet_414_beads_001_roi004_original_depth.png
[JNet_414_beads_001_roi004_output_depth]: /experiments/images/JNet_414_beads_001_roi004_output_depth.png
[JNet_414_beads_001_roi004_reconst_depth]: /experiments/images/JNet_414_beads_001_roi004_reconst_depth.png
[JNet_414_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_414_beads_002_roi000_heatmap_depth.png
[JNet_414_beads_002_roi000_original_depth]: /experiments/images/JNet_414_beads_002_roi000_original_depth.png
[JNet_414_beads_002_roi000_output_depth]: /experiments/images/JNet_414_beads_002_roi000_output_depth.png
[JNet_414_beads_002_roi000_reconst_depth]: /experiments/images/JNet_414_beads_002_roi000_reconst_depth.png
[JNet_414_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_414_beads_002_roi001_heatmap_depth.png
[JNet_414_beads_002_roi001_original_depth]: /experiments/images/JNet_414_beads_002_roi001_original_depth.png
[JNet_414_beads_002_roi001_output_depth]: /experiments/images/JNet_414_beads_002_roi001_output_depth.png
[JNet_414_beads_002_roi001_reconst_depth]: /experiments/images/JNet_414_beads_002_roi001_reconst_depth.png
[JNet_414_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_414_beads_002_roi002_heatmap_depth.png
[JNet_414_beads_002_roi002_original_depth]: /experiments/images/JNet_414_beads_002_roi002_original_depth.png
[JNet_414_beads_002_roi002_output_depth]: /experiments/images/JNet_414_beads_002_roi002_output_depth.png
[JNet_414_beads_002_roi002_reconst_depth]: /experiments/images/JNet_414_beads_002_roi002_reconst_depth.png
[JNet_414_psf_post]: /experiments/images/JNet_414_psf_post.png
[JNet_414_psf_pre]: /experiments/images/JNet_414_psf_pre.png
[finetuned]: /experiments/tmp/JNet_414_train.png
[pretrained_model]: /experiments/tmp/JNet_413_pretrain_train.png
