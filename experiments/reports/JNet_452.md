



# JNet_452 Report
  
the parameters to replicate the results of JNet_452. no vibrate in fine tuning, bright NA=0.7, mu_z = 1.2, sig_z = 1.27  
pretrained model : JNet_451_pretrain
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
|mu_z|2.0||
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
  
mean MSE: 0.02496308460831642, mean BCE: 0.08918201178312302
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_451_pretrain_0_original_plane]|![JNet_451_pretrain_0_output_plane]|![JNet_451_pretrain_0_label_plane]|
  
MSE: 0.024371495470404625, BCE: 0.08489940315485  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_451_pretrain_0_original_depth]|![JNet_451_pretrain_0_output_depth]|![JNet_451_pretrain_0_label_depth]|
  
MSE: 0.024371495470404625, BCE: 0.08489940315485  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_451_pretrain_1_original_plane]|![JNet_451_pretrain_1_output_plane]|![JNet_451_pretrain_1_label_plane]|
  
MSE: 0.025476638227701187, BCE: 0.09466719627380371  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_451_pretrain_1_original_depth]|![JNet_451_pretrain_1_output_depth]|![JNet_451_pretrain_1_label_depth]|
  
MSE: 0.025476638227701187, BCE: 0.09466719627380371  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_451_pretrain_2_original_plane]|![JNet_451_pretrain_2_output_plane]|![JNet_451_pretrain_2_label_plane]|
  
MSE: 0.02498013712465763, BCE: 0.08944958448410034  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_451_pretrain_2_original_depth]|![JNet_451_pretrain_2_output_depth]|![JNet_451_pretrain_2_label_depth]|
  
MSE: 0.02498013712465763, BCE: 0.08944958448410034  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_451_pretrain_3_original_plane]|![JNet_451_pretrain_3_output_plane]|![JNet_451_pretrain_3_label_plane]|
  
MSE: 0.03338116779923439, BCE: 0.11806876957416534  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_451_pretrain_3_original_depth]|![JNet_451_pretrain_3_output_depth]|![JNet_451_pretrain_3_label_depth]|
  
MSE: 0.03338116779923439, BCE: 0.11806876957416534  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_451_pretrain_4_original_plane]|![JNet_451_pretrain_4_output_plane]|![JNet_451_pretrain_4_label_plane]|
  
MSE: 0.016605986282229424, BCE: 0.058825116604566574  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_451_pretrain_4_original_depth]|![JNet_451_pretrain_4_output_depth]|![JNet_451_pretrain_4_label_depth]|
  
MSE: 0.016605986282229424, BCE: 0.058825116604566574  
  
mean MSE: 0.03526186943054199, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_452_0_original_plane]|![JNet_452_0_output_plane]|![JNet_452_0_label_plane]|
  
MSE: 0.03398126736283302, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_452_0_original_depth]|![JNet_452_0_output_depth]|![JNet_452_0_label_depth]|
  
MSE: 0.03398126736283302, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_452_1_original_plane]|![JNet_452_1_output_plane]|![JNet_452_1_label_plane]|
  
MSE: 0.0302239079028368, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_452_1_original_depth]|![JNet_452_1_output_depth]|![JNet_452_1_label_depth]|
  
MSE: 0.0302239079028368, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_452_2_original_plane]|![JNet_452_2_output_plane]|![JNet_452_2_label_plane]|
  
MSE: 0.024320857599377632, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_452_2_original_depth]|![JNet_452_2_output_depth]|![JNet_452_2_label_depth]|
  
MSE: 0.024320857599377632, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_452_3_original_plane]|![JNet_452_3_output_plane]|![JNet_452_3_label_plane]|
  
MSE: 0.04845177382230759, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_452_3_original_depth]|![JNet_452_3_output_depth]|![JNet_452_3_label_depth]|
  
MSE: 0.04845177382230759, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_452_4_original_plane]|![JNet_452_4_output_plane]|![JNet_452_4_label_plane]|
  
MSE: 0.03933154419064522, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_452_4_original_depth]|![JNet_452_4_output_depth]|![JNet_452_4_label_depth]|
  
MSE: 0.03933154419064522, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_451_pretrain_beads_001_roi000_original_depth]|![JNet_451_pretrain_beads_001_roi000_output_depth]|![JNet_451_pretrain_beads_001_roi000_reconst_depth]|![JNet_451_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 17.484445312500004, MSE: 0.0017267762450501323, quantized loss: 0.002206329954788089  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_451_pretrain_beads_001_roi001_original_depth]|![JNet_451_pretrain_beads_001_roi001_output_depth]|![JNet_451_pretrain_beads_001_roi001_reconst_depth]|![JNet_451_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 26.860521484375006, MSE: 0.0027949924115091562, quantized loss: 0.002957952208817005  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_451_pretrain_beads_001_roi002_original_depth]|![JNet_451_pretrain_beads_001_roi002_output_depth]|![JNet_451_pretrain_beads_001_roi002_reconst_depth]|![JNet_451_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 18.664302734375003, MSE: 0.0019627008587121964, quantized loss: 0.002562417183071375  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_451_pretrain_beads_001_roi003_original_depth]|![JNet_451_pretrain_beads_001_roi003_output_depth]|![JNet_451_pretrain_beads_001_roi003_reconst_depth]|![JNet_451_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 29.532615234375008, MSE: 0.0031799820717424154, quantized loss: 0.003590658074244857  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_451_pretrain_beads_001_roi004_original_depth]|![JNet_451_pretrain_beads_001_roi004_output_depth]|![JNet_451_pretrain_beads_001_roi004_reconst_depth]|![JNet_451_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 20.691775390625004, MSE: 0.0024743082467466593, quantized loss: 0.0028762705624103546  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_451_pretrain_beads_002_roi000_original_depth]|![JNet_451_pretrain_beads_002_roi000_output_depth]|![JNet_451_pretrain_beads_002_roi000_reconst_depth]|![JNet_451_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 21.841496093750006, MSE: 0.0027309453580528498, quantized loss: 0.002882840810343623  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_451_pretrain_beads_002_roi001_original_depth]|![JNet_451_pretrain_beads_002_roi001_output_depth]|![JNet_451_pretrain_beads_002_roi001_reconst_depth]|![JNet_451_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 20.364451171875004, MSE: 0.0021486624609678984, quantized loss: 0.00272406917065382  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_451_pretrain_beads_002_roi002_original_depth]|![JNet_451_pretrain_beads_002_roi002_output_depth]|![JNet_451_pretrain_beads_002_roi002_reconst_depth]|![JNet_451_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 20.802427734375005, MSE: 0.002423937665298581, quantized loss: 0.002884809859097004  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_452_beads_001_roi000_original_depth]|![JNet_452_beads_001_roi000_output_depth]|![JNet_452_beads_001_roi000_reconst_depth]|![JNet_452_beads_001_roi000_heatmap_depth]|
  
volume: 4.841217773437501, MSE: 0.0002738474286161363, quantized loss: 9.510237759968732e-06  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_452_beads_001_roi001_original_depth]|![JNet_452_beads_001_roi001_output_depth]|![JNet_452_beads_001_roi001_reconst_depth]|![JNet_452_beads_001_roi001_heatmap_depth]|
  
volume: 7.743201660156251, MSE: 0.0007360260933637619, quantized loss: 1.2792622328561265e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_452_beads_001_roi002_original_depth]|![JNet_452_beads_001_roi002_output_depth]|![JNet_452_beads_001_roi002_reconst_depth]|![JNet_452_beads_001_roi002_heatmap_depth]|
  
volume: 4.893846679687501, MSE: 0.00022430927492678165, quantized loss: 8.999251804198138e-06  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_452_beads_001_roi003_original_depth]|![JNet_452_beads_001_roi003_output_depth]|![JNet_452_beads_001_roi003_reconst_depth]|![JNet_452_beads_001_roi003_heatmap_depth]|
  
volume: 8.691108398437501, MSE: 0.0005856744828633964, quantized loss: 1.627604251552839e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_452_beads_001_roi004_original_depth]|![JNet_452_beads_001_roi004_output_depth]|![JNet_452_beads_001_roi004_reconst_depth]|![JNet_452_beads_001_roi004_heatmap_depth]|
  
volume: 5.223159667968751, MSE: 0.00030500837601721287, quantized loss: 1.0289052625012118e-05  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_452_beads_002_roi000_original_depth]|![JNet_452_beads_002_roi000_output_depth]|![JNet_452_beads_002_roi000_reconst_depth]|![JNet_452_beads_002_roi000_heatmap_depth]|
  
volume: 5.576291503906251, MSE: 0.00033912647631950676, quantized loss: 9.976903129427228e-06  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_452_beads_002_roi001_original_depth]|![JNet_452_beads_002_roi001_output_depth]|![JNet_452_beads_002_roi001_reconst_depth]|![JNet_452_beads_002_roi001_heatmap_depth]|
  
volume: 5.2624062500000015, MSE: 0.00026920036179944873, quantized loss: 8.046432412811555e-06  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_452_beads_002_roi002_original_depth]|![JNet_452_beads_002_roi002_output_depth]|![JNet_452_beads_002_roi002_reconst_depth]|![JNet_452_beads_002_roi002_heatmap_depth]|
  
volume: 5.368697753906251, MSE: 0.00029248124337755144, quantized loss: 9.188205694954377e-06  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_452_psf_pre]|![JNet_452_psf_post]|

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
  



[JNet_451_pretrain_0_label_depth]: /experiments/images/JNet_451_pretrain_0_label_depth.png
[JNet_451_pretrain_0_label_plane]: /experiments/images/JNet_451_pretrain_0_label_plane.png
[JNet_451_pretrain_0_original_depth]: /experiments/images/JNet_451_pretrain_0_original_depth.png
[JNet_451_pretrain_0_original_plane]: /experiments/images/JNet_451_pretrain_0_original_plane.png
[JNet_451_pretrain_0_output_depth]: /experiments/images/JNet_451_pretrain_0_output_depth.png
[JNet_451_pretrain_0_output_plane]: /experiments/images/JNet_451_pretrain_0_output_plane.png
[JNet_451_pretrain_1_label_depth]: /experiments/images/JNet_451_pretrain_1_label_depth.png
[JNet_451_pretrain_1_label_plane]: /experiments/images/JNet_451_pretrain_1_label_plane.png
[JNet_451_pretrain_1_original_depth]: /experiments/images/JNet_451_pretrain_1_original_depth.png
[JNet_451_pretrain_1_original_plane]: /experiments/images/JNet_451_pretrain_1_original_plane.png
[JNet_451_pretrain_1_output_depth]: /experiments/images/JNet_451_pretrain_1_output_depth.png
[JNet_451_pretrain_1_output_plane]: /experiments/images/JNet_451_pretrain_1_output_plane.png
[JNet_451_pretrain_2_label_depth]: /experiments/images/JNet_451_pretrain_2_label_depth.png
[JNet_451_pretrain_2_label_plane]: /experiments/images/JNet_451_pretrain_2_label_plane.png
[JNet_451_pretrain_2_original_depth]: /experiments/images/JNet_451_pretrain_2_original_depth.png
[JNet_451_pretrain_2_original_plane]: /experiments/images/JNet_451_pretrain_2_original_plane.png
[JNet_451_pretrain_2_output_depth]: /experiments/images/JNet_451_pretrain_2_output_depth.png
[JNet_451_pretrain_2_output_plane]: /experiments/images/JNet_451_pretrain_2_output_plane.png
[JNet_451_pretrain_3_label_depth]: /experiments/images/JNet_451_pretrain_3_label_depth.png
[JNet_451_pretrain_3_label_plane]: /experiments/images/JNet_451_pretrain_3_label_plane.png
[JNet_451_pretrain_3_original_depth]: /experiments/images/JNet_451_pretrain_3_original_depth.png
[JNet_451_pretrain_3_original_plane]: /experiments/images/JNet_451_pretrain_3_original_plane.png
[JNet_451_pretrain_3_output_depth]: /experiments/images/JNet_451_pretrain_3_output_depth.png
[JNet_451_pretrain_3_output_plane]: /experiments/images/JNet_451_pretrain_3_output_plane.png
[JNet_451_pretrain_4_label_depth]: /experiments/images/JNet_451_pretrain_4_label_depth.png
[JNet_451_pretrain_4_label_plane]: /experiments/images/JNet_451_pretrain_4_label_plane.png
[JNet_451_pretrain_4_original_depth]: /experiments/images/JNet_451_pretrain_4_original_depth.png
[JNet_451_pretrain_4_original_plane]: /experiments/images/JNet_451_pretrain_4_original_plane.png
[JNet_451_pretrain_4_output_depth]: /experiments/images/JNet_451_pretrain_4_output_depth.png
[JNet_451_pretrain_4_output_plane]: /experiments/images/JNet_451_pretrain_4_output_plane.png
[JNet_451_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_451_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi000_original_depth.png
[JNet_451_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi000_output_depth.png
[JNet_451_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi000_reconst_depth.png
[JNet_451_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_451_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi001_original_depth.png
[JNet_451_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi001_output_depth.png
[JNet_451_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi001_reconst_depth.png
[JNet_451_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_451_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi002_original_depth.png
[JNet_451_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi002_output_depth.png
[JNet_451_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi002_reconst_depth.png
[JNet_451_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_451_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi003_original_depth.png
[JNet_451_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi003_output_depth.png
[JNet_451_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi003_reconst_depth.png
[JNet_451_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_451_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi004_original_depth.png
[JNet_451_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi004_output_depth.png
[JNet_451_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_451_pretrain_beads_001_roi004_reconst_depth.png
[JNet_451_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_451_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi000_original_depth.png
[JNet_451_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi000_output_depth.png
[JNet_451_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi000_reconst_depth.png
[JNet_451_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_451_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi001_original_depth.png
[JNet_451_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi001_output_depth.png
[JNet_451_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi001_reconst_depth.png
[JNet_451_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_451_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi002_original_depth.png
[JNet_451_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi002_output_depth.png
[JNet_451_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_451_pretrain_beads_002_roi002_reconst_depth.png
[JNet_452_0_label_depth]: /experiments/images/JNet_452_0_label_depth.png
[JNet_452_0_label_plane]: /experiments/images/JNet_452_0_label_plane.png
[JNet_452_0_original_depth]: /experiments/images/JNet_452_0_original_depth.png
[JNet_452_0_original_plane]: /experiments/images/JNet_452_0_original_plane.png
[JNet_452_0_output_depth]: /experiments/images/JNet_452_0_output_depth.png
[JNet_452_0_output_plane]: /experiments/images/JNet_452_0_output_plane.png
[JNet_452_1_label_depth]: /experiments/images/JNet_452_1_label_depth.png
[JNet_452_1_label_plane]: /experiments/images/JNet_452_1_label_plane.png
[JNet_452_1_original_depth]: /experiments/images/JNet_452_1_original_depth.png
[JNet_452_1_original_plane]: /experiments/images/JNet_452_1_original_plane.png
[JNet_452_1_output_depth]: /experiments/images/JNet_452_1_output_depth.png
[JNet_452_1_output_plane]: /experiments/images/JNet_452_1_output_plane.png
[JNet_452_2_label_depth]: /experiments/images/JNet_452_2_label_depth.png
[JNet_452_2_label_plane]: /experiments/images/JNet_452_2_label_plane.png
[JNet_452_2_original_depth]: /experiments/images/JNet_452_2_original_depth.png
[JNet_452_2_original_plane]: /experiments/images/JNet_452_2_original_plane.png
[JNet_452_2_output_depth]: /experiments/images/JNet_452_2_output_depth.png
[JNet_452_2_output_plane]: /experiments/images/JNet_452_2_output_plane.png
[JNet_452_3_label_depth]: /experiments/images/JNet_452_3_label_depth.png
[JNet_452_3_label_plane]: /experiments/images/JNet_452_3_label_plane.png
[JNet_452_3_original_depth]: /experiments/images/JNet_452_3_original_depth.png
[JNet_452_3_original_plane]: /experiments/images/JNet_452_3_original_plane.png
[JNet_452_3_output_depth]: /experiments/images/JNet_452_3_output_depth.png
[JNet_452_3_output_plane]: /experiments/images/JNet_452_3_output_plane.png
[JNet_452_4_label_depth]: /experiments/images/JNet_452_4_label_depth.png
[JNet_452_4_label_plane]: /experiments/images/JNet_452_4_label_plane.png
[JNet_452_4_original_depth]: /experiments/images/JNet_452_4_original_depth.png
[JNet_452_4_original_plane]: /experiments/images/JNet_452_4_original_plane.png
[JNet_452_4_output_depth]: /experiments/images/JNet_452_4_output_depth.png
[JNet_452_4_output_plane]: /experiments/images/JNet_452_4_output_plane.png
[JNet_452_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_452_beads_001_roi000_heatmap_depth.png
[JNet_452_beads_001_roi000_original_depth]: /experiments/images/JNet_452_beads_001_roi000_original_depth.png
[JNet_452_beads_001_roi000_output_depth]: /experiments/images/JNet_452_beads_001_roi000_output_depth.png
[JNet_452_beads_001_roi000_reconst_depth]: /experiments/images/JNet_452_beads_001_roi000_reconst_depth.png
[JNet_452_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_452_beads_001_roi001_heatmap_depth.png
[JNet_452_beads_001_roi001_original_depth]: /experiments/images/JNet_452_beads_001_roi001_original_depth.png
[JNet_452_beads_001_roi001_output_depth]: /experiments/images/JNet_452_beads_001_roi001_output_depth.png
[JNet_452_beads_001_roi001_reconst_depth]: /experiments/images/JNet_452_beads_001_roi001_reconst_depth.png
[JNet_452_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_452_beads_001_roi002_heatmap_depth.png
[JNet_452_beads_001_roi002_original_depth]: /experiments/images/JNet_452_beads_001_roi002_original_depth.png
[JNet_452_beads_001_roi002_output_depth]: /experiments/images/JNet_452_beads_001_roi002_output_depth.png
[JNet_452_beads_001_roi002_reconst_depth]: /experiments/images/JNet_452_beads_001_roi002_reconst_depth.png
[JNet_452_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_452_beads_001_roi003_heatmap_depth.png
[JNet_452_beads_001_roi003_original_depth]: /experiments/images/JNet_452_beads_001_roi003_original_depth.png
[JNet_452_beads_001_roi003_output_depth]: /experiments/images/JNet_452_beads_001_roi003_output_depth.png
[JNet_452_beads_001_roi003_reconst_depth]: /experiments/images/JNet_452_beads_001_roi003_reconst_depth.png
[JNet_452_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_452_beads_001_roi004_heatmap_depth.png
[JNet_452_beads_001_roi004_original_depth]: /experiments/images/JNet_452_beads_001_roi004_original_depth.png
[JNet_452_beads_001_roi004_output_depth]: /experiments/images/JNet_452_beads_001_roi004_output_depth.png
[JNet_452_beads_001_roi004_reconst_depth]: /experiments/images/JNet_452_beads_001_roi004_reconst_depth.png
[JNet_452_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_452_beads_002_roi000_heatmap_depth.png
[JNet_452_beads_002_roi000_original_depth]: /experiments/images/JNet_452_beads_002_roi000_original_depth.png
[JNet_452_beads_002_roi000_output_depth]: /experiments/images/JNet_452_beads_002_roi000_output_depth.png
[JNet_452_beads_002_roi000_reconst_depth]: /experiments/images/JNet_452_beads_002_roi000_reconst_depth.png
[JNet_452_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_452_beads_002_roi001_heatmap_depth.png
[JNet_452_beads_002_roi001_original_depth]: /experiments/images/JNet_452_beads_002_roi001_original_depth.png
[JNet_452_beads_002_roi001_output_depth]: /experiments/images/JNet_452_beads_002_roi001_output_depth.png
[JNet_452_beads_002_roi001_reconst_depth]: /experiments/images/JNet_452_beads_002_roi001_reconst_depth.png
[JNet_452_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_452_beads_002_roi002_heatmap_depth.png
[JNet_452_beads_002_roi002_original_depth]: /experiments/images/JNet_452_beads_002_roi002_original_depth.png
[JNet_452_beads_002_roi002_output_depth]: /experiments/images/JNet_452_beads_002_roi002_output_depth.png
[JNet_452_beads_002_roi002_reconst_depth]: /experiments/images/JNet_452_beads_002_roi002_reconst_depth.png
[JNet_452_psf_post]: /experiments/images/JNet_452_psf_post.png
[JNet_452_psf_pre]: /experiments/images/JNet_452_psf_pre.png
[finetuned]: /experiments/tmp/JNet_452_train.png
[pretrained_model]: /experiments/tmp/JNet_451_pretrain_train.png
