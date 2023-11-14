



# JNet_465 Report
  
the parameters to replicate the results of JNet_464. AXONS deconvolution! vibrate in fine tuning, , mu_z = 1.2, sig_z = 1.27  
pretrained model : JNet_463_pretrain
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
|scale|6|
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
|folderpath|spinerawdata0|
|scorefolderpath|_spinerawscore0|
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
|folderpath|spinerawdata0|
|scorefolderpath|_spinerawscore0|
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
|ploss_weight|0.005|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.02495647594332695, mean BCE: 0.09403939545154572
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_463_pretrain_0_original_plane]|![JNet_463_pretrain_0_output_plane]|![JNet_463_pretrain_0_label_plane]|
  
MSE: 0.025020459666848183, BCE: 0.09526634961366653  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_463_pretrain_0_original_depth]|![JNet_463_pretrain_0_output_depth]|![JNet_463_pretrain_0_label_depth]|
  
MSE: 0.025020459666848183, BCE: 0.09526634961366653  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_463_pretrain_1_original_plane]|![JNet_463_pretrain_1_output_plane]|![JNet_463_pretrain_1_label_plane]|
  
MSE: 0.031545087695121765, BCE: 0.11218760907649994  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_463_pretrain_1_original_depth]|![JNet_463_pretrain_1_output_depth]|![JNet_463_pretrain_1_label_depth]|
  
MSE: 0.031545087695121765, BCE: 0.11218760907649994  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_463_pretrain_2_original_plane]|![JNet_463_pretrain_2_output_plane]|![JNet_463_pretrain_2_label_plane]|
  
MSE: 0.022913271561264992, BCE: 0.0805157795548439  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_463_pretrain_2_original_depth]|![JNet_463_pretrain_2_output_depth]|![JNet_463_pretrain_2_label_depth]|
  
MSE: 0.022913271561264992, BCE: 0.0805157795548439  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_463_pretrain_3_original_plane]|![JNet_463_pretrain_3_output_plane]|![JNet_463_pretrain_3_label_plane]|
  
MSE: 0.018708691000938416, BCE: 0.06894847005605698  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_463_pretrain_3_original_depth]|![JNet_463_pretrain_3_output_depth]|![JNet_463_pretrain_3_label_depth]|
  
MSE: 0.018708691000938416, BCE: 0.06894847005605698  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_463_pretrain_4_original_plane]|![JNet_463_pretrain_4_output_plane]|![JNet_463_pretrain_4_label_plane]|
  
MSE: 0.02659488096833229, BCE: 0.11327873915433884  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_463_pretrain_4_original_depth]|![JNet_463_pretrain_4_output_depth]|![JNet_463_pretrain_4_label_depth]|
  
MSE: 0.02659488096833229, BCE: 0.11327873915433884  
  
mean MSE: 0.04301854223012924, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_465_0_original_plane]|![JNet_465_0_output_plane]|![JNet_465_0_label_plane]|
  
MSE: 0.04934891313314438, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_465_0_original_depth]|![JNet_465_0_output_depth]|![JNet_465_0_label_depth]|
  
MSE: 0.04934891313314438, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_465_1_original_plane]|![JNet_465_1_output_plane]|![JNet_465_1_label_plane]|
  
MSE: 0.03959088400006294, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_465_1_original_depth]|![JNet_465_1_output_depth]|![JNet_465_1_label_depth]|
  
MSE: 0.03959088400006294, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_465_2_original_plane]|![JNet_465_2_output_plane]|![JNet_465_2_label_plane]|
  
MSE: 0.03278280422091484, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_465_2_original_depth]|![JNet_465_2_output_depth]|![JNet_465_2_label_depth]|
  
MSE: 0.03278280422091484, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_465_3_original_plane]|![JNet_465_3_output_plane]|![JNet_465_3_label_plane]|
  
MSE: 0.027832506224513054, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_465_3_original_depth]|![JNet_465_3_output_depth]|![JNet_465_3_label_depth]|
  
MSE: 0.027832506224513054, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_465_4_original_plane]|![JNet_465_4_output_plane]|![JNet_465_4_label_plane]|
  
MSE: 0.06553760915994644, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_465_4_original_depth]|![JNet_465_4_output_depth]|![JNet_465_4_label_depth]|
  
MSE: 0.06553760915994644, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_463_pretrain_beads_001_roi000_original_depth]|![JNet_463_pretrain_beads_001_roi000_output_depth]|![JNet_463_pretrain_beads_001_roi000_reconst_depth]|![JNet_463_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 413.78694400000006, MSE: 0.003774357959628105, quantized loss: 0.003791636088863015  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_463_pretrain_beads_001_roi001_original_depth]|![JNet_463_pretrain_beads_001_roi001_output_depth]|![JNet_463_pretrain_beads_001_roi001_reconst_depth]|![JNet_463_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 507.0311680000001, MSE: 0.005448403302580118, quantized loss: 0.004280856344848871  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_463_pretrain_beads_001_roi002_original_depth]|![JNet_463_pretrain_beads_001_roi002_output_depth]|![JNet_463_pretrain_beads_001_roi002_reconst_depth]|![JNet_463_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 442.52892800000006, MSE: 0.0036742871161550283, quantized loss: 0.0037118212785571814  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_463_pretrain_beads_001_roi003_original_depth]|![JNet_463_pretrain_beads_001_roi003_output_depth]|![JNet_463_pretrain_beads_001_roi003_reconst_depth]|![JNet_463_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 656.8306560000001, MSE: 0.006486681755632162, quantized loss: 0.006160588935017586  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_463_pretrain_beads_001_roi004_original_depth]|![JNet_463_pretrain_beads_001_roi004_output_depth]|![JNet_463_pretrain_beads_001_roi004_reconst_depth]|![JNet_463_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 473.3176000000001, MSE: 0.004413239192217588, quantized loss: 0.004018007777631283  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_463_pretrain_beads_002_roi000_original_depth]|![JNet_463_pretrain_beads_002_roi000_output_depth]|![JNet_463_pretrain_beads_002_roi000_reconst_depth]|![JNet_463_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 480.4753600000001, MSE: 0.004870048724114895, quantized loss: 0.004097913391888142  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_463_pretrain_beads_002_roi001_original_depth]|![JNet_463_pretrain_beads_002_roi001_output_depth]|![JNet_463_pretrain_beads_002_roi001_reconst_depth]|![JNet_463_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 412.3456960000001, MSE: 0.003984570968896151, quantized loss: 0.003394774394109845  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_463_pretrain_beads_002_roi002_original_depth]|![JNet_463_pretrain_beads_002_roi002_output_depth]|![JNet_463_pretrain_beads_002_roi002_reconst_depth]|![JNet_463_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 431.96140800000006, MSE: 0.004263313952833414, quantized loss: 0.0035780619364231825  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_465_beads_001_roi000_original_depth]|![JNet_465_beads_001_roi000_output_depth]|![JNet_465_beads_001_roi000_reconst_depth]|![JNet_465_beads_001_roi000_heatmap_depth]|
  
volume: 59.76016000000001, MSE: 0.002793840831145644, quantized loss: 2.7487061743158847e-05  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_465_beads_001_roi001_original_depth]|![JNet_465_beads_001_roi001_output_depth]|![JNet_465_beads_001_roi001_reconst_depth]|![JNet_465_beads_001_roi001_heatmap_depth]|
  
volume: 102.38483200000002, MSE: 0.0044959657825529575, quantized loss: 4.89393605676014e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_465_beads_001_roi002_original_depth]|![JNet_465_beads_001_roi002_output_depth]|![JNet_465_beads_001_roi002_reconst_depth]|![JNet_465_beads_001_roi002_heatmap_depth]|
  
volume: 55.95383200000001, MSE: 0.0030040096025913954, quantized loss: 3.087372533627786e-05  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_465_beads_001_roi003_original_depth]|![JNet_465_beads_001_roi003_output_depth]|![JNet_465_beads_001_roi003_reconst_depth]|![JNet_465_beads_001_roi003_heatmap_depth]|
  
volume: 99.31276000000001, MSE: 0.0050314925611019135, quantized loss: 4.5950797357363626e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_465_beads_001_roi004_original_depth]|![JNet_465_beads_001_roi004_output_depth]|![JNet_465_beads_001_roi004_reconst_depth]|![JNet_465_beads_001_roi004_heatmap_depth]|
  
volume: 77.72199200000001, MSE: 0.002767295576632023, quantized loss: 2.6170939236180857e-05  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_465_beads_002_roi000_original_depth]|![JNet_465_beads_002_roi000_output_depth]|![JNet_465_beads_002_roi000_reconst_depth]|![JNet_465_beads_002_roi000_heatmap_depth]|
  
volume: 91.60313600000002, MSE: 0.0027809971943497658, quantized loss: 2.509339537937194e-05  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_465_beads_002_roi001_original_depth]|![JNet_465_beads_002_roi001_output_depth]|![JNet_465_beads_002_roi001_reconst_depth]|![JNet_465_beads_002_roi001_heatmap_depth]|
  
volume: 70.33044000000001, MSE: 0.0028697382658720016, quantized loss: 2.6461373636266217e-05  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_465_beads_002_roi002_original_depth]|![JNet_465_beads_002_roi002_output_depth]|![JNet_465_beads_002_roi002_reconst_depth]|![JNet_465_beads_002_roi002_heatmap_depth]|
  
volume: 77.97409600000002, MSE: 0.002907515736296773, quantized loss: 2.7018551918445155e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_465_psf_pre]|![JNet_465_psf_post]|

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
  



[JNet_463_pretrain_0_label_depth]: /experiments/images/JNet_463_pretrain_0_label_depth.png
[JNet_463_pretrain_0_label_plane]: /experiments/images/JNet_463_pretrain_0_label_plane.png
[JNet_463_pretrain_0_original_depth]: /experiments/images/JNet_463_pretrain_0_original_depth.png
[JNet_463_pretrain_0_original_plane]: /experiments/images/JNet_463_pretrain_0_original_plane.png
[JNet_463_pretrain_0_output_depth]: /experiments/images/JNet_463_pretrain_0_output_depth.png
[JNet_463_pretrain_0_output_plane]: /experiments/images/JNet_463_pretrain_0_output_plane.png
[JNet_463_pretrain_1_label_depth]: /experiments/images/JNet_463_pretrain_1_label_depth.png
[JNet_463_pretrain_1_label_plane]: /experiments/images/JNet_463_pretrain_1_label_plane.png
[JNet_463_pretrain_1_original_depth]: /experiments/images/JNet_463_pretrain_1_original_depth.png
[JNet_463_pretrain_1_original_plane]: /experiments/images/JNet_463_pretrain_1_original_plane.png
[JNet_463_pretrain_1_output_depth]: /experiments/images/JNet_463_pretrain_1_output_depth.png
[JNet_463_pretrain_1_output_plane]: /experiments/images/JNet_463_pretrain_1_output_plane.png
[JNet_463_pretrain_2_label_depth]: /experiments/images/JNet_463_pretrain_2_label_depth.png
[JNet_463_pretrain_2_label_plane]: /experiments/images/JNet_463_pretrain_2_label_plane.png
[JNet_463_pretrain_2_original_depth]: /experiments/images/JNet_463_pretrain_2_original_depth.png
[JNet_463_pretrain_2_original_plane]: /experiments/images/JNet_463_pretrain_2_original_plane.png
[JNet_463_pretrain_2_output_depth]: /experiments/images/JNet_463_pretrain_2_output_depth.png
[JNet_463_pretrain_2_output_plane]: /experiments/images/JNet_463_pretrain_2_output_plane.png
[JNet_463_pretrain_3_label_depth]: /experiments/images/JNet_463_pretrain_3_label_depth.png
[JNet_463_pretrain_3_label_plane]: /experiments/images/JNet_463_pretrain_3_label_plane.png
[JNet_463_pretrain_3_original_depth]: /experiments/images/JNet_463_pretrain_3_original_depth.png
[JNet_463_pretrain_3_original_plane]: /experiments/images/JNet_463_pretrain_3_original_plane.png
[JNet_463_pretrain_3_output_depth]: /experiments/images/JNet_463_pretrain_3_output_depth.png
[JNet_463_pretrain_3_output_plane]: /experiments/images/JNet_463_pretrain_3_output_plane.png
[JNet_463_pretrain_4_label_depth]: /experiments/images/JNet_463_pretrain_4_label_depth.png
[JNet_463_pretrain_4_label_plane]: /experiments/images/JNet_463_pretrain_4_label_plane.png
[JNet_463_pretrain_4_original_depth]: /experiments/images/JNet_463_pretrain_4_original_depth.png
[JNet_463_pretrain_4_original_plane]: /experiments/images/JNet_463_pretrain_4_original_plane.png
[JNet_463_pretrain_4_output_depth]: /experiments/images/JNet_463_pretrain_4_output_depth.png
[JNet_463_pretrain_4_output_plane]: /experiments/images/JNet_463_pretrain_4_output_plane.png
[JNet_463_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_463_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi000_original_depth.png
[JNet_463_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi000_output_depth.png
[JNet_463_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi000_reconst_depth.png
[JNet_463_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_463_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi001_original_depth.png
[JNet_463_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi001_output_depth.png
[JNet_463_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi001_reconst_depth.png
[JNet_463_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_463_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi002_original_depth.png
[JNet_463_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi002_output_depth.png
[JNet_463_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi002_reconst_depth.png
[JNet_463_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_463_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi003_original_depth.png
[JNet_463_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi003_output_depth.png
[JNet_463_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi003_reconst_depth.png
[JNet_463_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_463_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi004_original_depth.png
[JNet_463_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi004_output_depth.png
[JNet_463_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_463_pretrain_beads_001_roi004_reconst_depth.png
[JNet_463_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_463_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi000_original_depth.png
[JNet_463_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi000_output_depth.png
[JNet_463_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi000_reconst_depth.png
[JNet_463_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_463_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi001_original_depth.png
[JNet_463_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi001_output_depth.png
[JNet_463_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi001_reconst_depth.png
[JNet_463_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_463_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi002_original_depth.png
[JNet_463_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi002_output_depth.png
[JNet_463_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_463_pretrain_beads_002_roi002_reconst_depth.png
[JNet_465_0_label_depth]: /experiments/images/JNet_465_0_label_depth.png
[JNet_465_0_label_plane]: /experiments/images/JNet_465_0_label_plane.png
[JNet_465_0_original_depth]: /experiments/images/JNet_465_0_original_depth.png
[JNet_465_0_original_plane]: /experiments/images/JNet_465_0_original_plane.png
[JNet_465_0_output_depth]: /experiments/images/JNet_465_0_output_depth.png
[JNet_465_0_output_plane]: /experiments/images/JNet_465_0_output_plane.png
[JNet_465_1_label_depth]: /experiments/images/JNet_465_1_label_depth.png
[JNet_465_1_label_plane]: /experiments/images/JNet_465_1_label_plane.png
[JNet_465_1_original_depth]: /experiments/images/JNet_465_1_original_depth.png
[JNet_465_1_original_plane]: /experiments/images/JNet_465_1_original_plane.png
[JNet_465_1_output_depth]: /experiments/images/JNet_465_1_output_depth.png
[JNet_465_1_output_plane]: /experiments/images/JNet_465_1_output_plane.png
[JNet_465_2_label_depth]: /experiments/images/JNet_465_2_label_depth.png
[JNet_465_2_label_plane]: /experiments/images/JNet_465_2_label_plane.png
[JNet_465_2_original_depth]: /experiments/images/JNet_465_2_original_depth.png
[JNet_465_2_original_plane]: /experiments/images/JNet_465_2_original_plane.png
[JNet_465_2_output_depth]: /experiments/images/JNet_465_2_output_depth.png
[JNet_465_2_output_plane]: /experiments/images/JNet_465_2_output_plane.png
[JNet_465_3_label_depth]: /experiments/images/JNet_465_3_label_depth.png
[JNet_465_3_label_plane]: /experiments/images/JNet_465_3_label_plane.png
[JNet_465_3_original_depth]: /experiments/images/JNet_465_3_original_depth.png
[JNet_465_3_original_plane]: /experiments/images/JNet_465_3_original_plane.png
[JNet_465_3_output_depth]: /experiments/images/JNet_465_3_output_depth.png
[JNet_465_3_output_plane]: /experiments/images/JNet_465_3_output_plane.png
[JNet_465_4_label_depth]: /experiments/images/JNet_465_4_label_depth.png
[JNet_465_4_label_plane]: /experiments/images/JNet_465_4_label_plane.png
[JNet_465_4_original_depth]: /experiments/images/JNet_465_4_original_depth.png
[JNet_465_4_original_plane]: /experiments/images/JNet_465_4_original_plane.png
[JNet_465_4_output_depth]: /experiments/images/JNet_465_4_output_depth.png
[JNet_465_4_output_plane]: /experiments/images/JNet_465_4_output_plane.png
[JNet_465_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_465_beads_001_roi000_heatmap_depth.png
[JNet_465_beads_001_roi000_original_depth]: /experiments/images/JNet_465_beads_001_roi000_original_depth.png
[JNet_465_beads_001_roi000_output_depth]: /experiments/images/JNet_465_beads_001_roi000_output_depth.png
[JNet_465_beads_001_roi000_reconst_depth]: /experiments/images/JNet_465_beads_001_roi000_reconst_depth.png
[JNet_465_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_465_beads_001_roi001_heatmap_depth.png
[JNet_465_beads_001_roi001_original_depth]: /experiments/images/JNet_465_beads_001_roi001_original_depth.png
[JNet_465_beads_001_roi001_output_depth]: /experiments/images/JNet_465_beads_001_roi001_output_depth.png
[JNet_465_beads_001_roi001_reconst_depth]: /experiments/images/JNet_465_beads_001_roi001_reconst_depth.png
[JNet_465_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_465_beads_001_roi002_heatmap_depth.png
[JNet_465_beads_001_roi002_original_depth]: /experiments/images/JNet_465_beads_001_roi002_original_depth.png
[JNet_465_beads_001_roi002_output_depth]: /experiments/images/JNet_465_beads_001_roi002_output_depth.png
[JNet_465_beads_001_roi002_reconst_depth]: /experiments/images/JNet_465_beads_001_roi002_reconst_depth.png
[JNet_465_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_465_beads_001_roi003_heatmap_depth.png
[JNet_465_beads_001_roi003_original_depth]: /experiments/images/JNet_465_beads_001_roi003_original_depth.png
[JNet_465_beads_001_roi003_output_depth]: /experiments/images/JNet_465_beads_001_roi003_output_depth.png
[JNet_465_beads_001_roi003_reconst_depth]: /experiments/images/JNet_465_beads_001_roi003_reconst_depth.png
[JNet_465_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_465_beads_001_roi004_heatmap_depth.png
[JNet_465_beads_001_roi004_original_depth]: /experiments/images/JNet_465_beads_001_roi004_original_depth.png
[JNet_465_beads_001_roi004_output_depth]: /experiments/images/JNet_465_beads_001_roi004_output_depth.png
[JNet_465_beads_001_roi004_reconst_depth]: /experiments/images/JNet_465_beads_001_roi004_reconst_depth.png
[JNet_465_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_465_beads_002_roi000_heatmap_depth.png
[JNet_465_beads_002_roi000_original_depth]: /experiments/images/JNet_465_beads_002_roi000_original_depth.png
[JNet_465_beads_002_roi000_output_depth]: /experiments/images/JNet_465_beads_002_roi000_output_depth.png
[JNet_465_beads_002_roi000_reconst_depth]: /experiments/images/JNet_465_beads_002_roi000_reconst_depth.png
[JNet_465_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_465_beads_002_roi001_heatmap_depth.png
[JNet_465_beads_002_roi001_original_depth]: /experiments/images/JNet_465_beads_002_roi001_original_depth.png
[JNet_465_beads_002_roi001_output_depth]: /experiments/images/JNet_465_beads_002_roi001_output_depth.png
[JNet_465_beads_002_roi001_reconst_depth]: /experiments/images/JNet_465_beads_002_roi001_reconst_depth.png
[JNet_465_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_465_beads_002_roi002_heatmap_depth.png
[JNet_465_beads_002_roi002_original_depth]: /experiments/images/JNet_465_beads_002_roi002_original_depth.png
[JNet_465_beads_002_roi002_output_depth]: /experiments/images/JNet_465_beads_002_roi002_output_depth.png
[JNet_465_beads_002_roi002_reconst_depth]: /experiments/images/JNet_465_beads_002_roi002_reconst_depth.png
[JNet_465_psf_post]: /experiments/images/JNet_465_psf_post.png
[JNet_465_psf_pre]: /experiments/images/JNet_465_psf_pre.png
[finetuned]: /experiments/tmp/JNet_465_train.png
[pretrained_model]: /experiments/tmp/JNet_463_pretrain_train.png
