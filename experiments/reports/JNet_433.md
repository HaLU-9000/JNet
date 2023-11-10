



# JNet_433 Report
  
the parameters to replicate the results of JNet_427. nearest interp of PSF, NA=0.7, mu_z = 1.0  
pretrained model : JNet_430_pretrain
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
|mu_z|1.0||
|sig_z|0.3||
|blur_mode|gibsonlanni|`gaussian` or `gibsonlanni`|
|size_x|51||
|size_y|51||
|size_z|201||
|NA|0.7||
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
  
mean MSE: 0.01987607404589653, mean BCE: 0.07508012652397156
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_0_original_plane]|![JNet_430_pretrain_0_output_plane]|![JNet_430_pretrain_0_label_plane]|
  
MSE: 0.02469646744430065, BCE: 0.09098271280527115  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_0_original_depth]|![JNet_430_pretrain_0_output_depth]|![JNet_430_pretrain_0_label_depth]|
  
MSE: 0.02469646744430065, BCE: 0.09098271280527115  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_1_original_plane]|![JNet_430_pretrain_1_output_plane]|![JNet_430_pretrain_1_label_plane]|
  
MSE: 0.01996089331805706, BCE: 0.07467450946569443  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_1_original_depth]|![JNet_430_pretrain_1_output_depth]|![JNet_430_pretrain_1_label_depth]|
  
MSE: 0.01996089331805706, BCE: 0.07467450946569443  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_2_original_plane]|![JNet_430_pretrain_2_output_plane]|![JNet_430_pretrain_2_label_plane]|
  
MSE: 0.018500899896025658, BCE: 0.06524041295051575  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_2_original_depth]|![JNet_430_pretrain_2_output_depth]|![JNet_430_pretrain_2_label_depth]|
  
MSE: 0.018500899896025658, BCE: 0.06524041295051575  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_3_original_plane]|![JNet_430_pretrain_3_output_plane]|![JNet_430_pretrain_3_label_plane]|
  
MSE: 0.017753351479768753, BCE: 0.0796632394194603  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_3_original_depth]|![JNet_430_pretrain_3_output_depth]|![JNet_430_pretrain_3_label_depth]|
  
MSE: 0.017753351479768753, BCE: 0.0796632394194603  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_4_original_plane]|![JNet_430_pretrain_4_output_plane]|![JNet_430_pretrain_4_label_plane]|
  
MSE: 0.018468759953975677, BCE: 0.06483974307775497  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_4_original_depth]|![JNet_430_pretrain_4_output_depth]|![JNet_430_pretrain_4_label_depth]|
  
MSE: 0.018468759953975677, BCE: 0.06483974307775497  
  
mean MSE: 0.030613726004958153, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_433_0_original_plane]|![JNet_433_0_output_plane]|![JNet_433_0_label_plane]|
  
MSE: 0.044124454259872437, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_433_0_original_depth]|![JNet_433_0_output_depth]|![JNet_433_0_label_depth]|
  
MSE: 0.044124454259872437, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_433_1_original_plane]|![JNet_433_1_output_plane]|![JNet_433_1_label_plane]|
  
MSE: 0.03174525499343872, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_433_1_original_depth]|![JNet_433_1_output_depth]|![JNet_433_1_label_depth]|
  
MSE: 0.03174525499343872, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_433_2_original_plane]|![JNet_433_2_output_plane]|![JNet_433_2_label_plane]|
  
MSE: 0.0289467740803957, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_433_2_original_depth]|![JNet_433_2_output_depth]|![JNet_433_2_label_depth]|
  
MSE: 0.0289467740803957, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_433_3_original_plane]|![JNet_433_3_output_plane]|![JNet_433_3_label_plane]|
  
MSE: 0.024405425414443016, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_433_3_original_depth]|![JNet_433_3_output_depth]|![JNet_433_3_label_depth]|
  
MSE: 0.024405425414443016, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_433_4_original_plane]|![JNet_433_4_output_plane]|![JNet_433_4_label_plane]|
  
MSE: 0.02384672686457634, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_433_4_original_depth]|![JNet_433_4_output_depth]|![JNet_433_4_label_depth]|
  
MSE: 0.02384672686457634, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_001_roi000_original_depth]|![JNet_430_pretrain_beads_001_roi000_output_depth]|![JNet_430_pretrain_beads_001_roi000_reconst_depth]|![JNet_430_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 22.792455078125005, MSE: 0.040532778948545456, quantized loss: 0.0034187473356723785  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_001_roi001_original_depth]|![JNet_430_pretrain_beads_001_roi001_output_depth]|![JNet_430_pretrain_beads_001_roi001_reconst_depth]|![JNet_430_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 28.819158203125006, MSE: 0.0385478138923645, quantized loss: 0.003622268093749881  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_001_roi002_original_depth]|![JNet_430_pretrain_beads_001_roi002_output_depth]|![JNet_430_pretrain_beads_001_roi002_reconst_depth]|![JNet_430_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 24.747093750000005, MSE: 0.04728592932224274, quantized loss: 0.004198620095849037  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_001_roi003_original_depth]|![JNet_430_pretrain_beads_001_roi003_output_depth]|![JNet_430_pretrain_beads_001_roi003_reconst_depth]|![JNet_430_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 33.56187109375001, MSE: 0.05257509648799896, quantized loss: 0.004373582080006599  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_001_roi004_original_depth]|![JNet_430_pretrain_beads_001_roi004_output_depth]|![JNet_430_pretrain_beads_001_roi004_reconst_depth]|![JNet_430_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 26.295203125000008, MSE: 0.05005927011370659, quantized loss: 0.0044042011722922325  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_002_roi000_original_depth]|![JNet_430_pretrain_beads_002_roi000_output_depth]|![JNet_430_pretrain_beads_002_roi000_reconst_depth]|![JNet_430_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 29.348919921875005, MSE: 0.05789671465754509, quantized loss: 0.00512992637231946  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_002_roi001_original_depth]|![JNet_430_pretrain_beads_002_roi001_output_depth]|![JNet_430_pretrain_beads_002_roi001_reconst_depth]|![JNet_430_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 26.593306640625006, MSE: 0.05194781720638275, quantized loss: 0.00427400553599  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_002_roi002_original_depth]|![JNet_430_pretrain_beads_002_roi002_output_depth]|![JNet_430_pretrain_beads_002_roi002_reconst_depth]|![JNet_430_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 27.610775390625008, MSE: 0.05414294824004173, quantized loss: 0.00472274050116539  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_433_beads_001_roi000_original_depth]|![JNet_433_beads_001_roi000_output_depth]|![JNet_433_beads_001_roi000_reconst_depth]|![JNet_433_beads_001_roi000_heatmap_depth]|
  
volume: 4.740982421875001, MSE: 0.0001839540636865422, quantized loss: 8.534895641787443e-06  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_433_beads_001_roi001_original_depth]|![JNet_433_beads_001_roi001_output_depth]|![JNet_433_beads_001_roi001_reconst_depth]|![JNet_433_beads_001_roi001_heatmap_depth]|
  
volume: 7.282989257812502, MSE: 0.0004665656015276909, quantized loss: 1.1599933714023791e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_433_beads_001_roi002_original_depth]|![JNet_433_beads_001_roi002_output_depth]|![JNet_433_beads_001_roi002_reconst_depth]|![JNet_433_beads_001_roi002_heatmap_depth]|
  
volume: 4.686053710937501, MSE: 0.00012896601401735097, quantized loss: 8.645281923236325e-06  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_433_beads_001_roi003_original_depth]|![JNet_433_beads_001_roi003_output_depth]|![JNet_433_beads_001_roi003_reconst_depth]|![JNet_433_beads_001_roi003_heatmap_depth]|
  
volume: 7.488437500000002, MSE: 0.000360796315362677, quantized loss: 1.1720076145138592e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_433_beads_001_roi004_original_depth]|![JNet_433_beads_001_roi004_output_depth]|![JNet_433_beads_001_roi004_reconst_depth]|![JNet_433_beads_001_roi004_heatmap_depth]|
  
volume: 4.974892089843751, MSE: 9.799483086680993e-05, quantized loss: 7.943431228341069e-06  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_433_beads_002_roi000_original_depth]|![JNet_433_beads_002_roi000_output_depth]|![JNet_433_beads_002_roi000_reconst_depth]|![JNet_433_beads_002_roi000_heatmap_depth]|
  
volume: 5.282258300781251, MSE: 9.664992830948904e-05, quantized loss: 8.102055289782584e-06  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_433_beads_002_roi001_original_depth]|![JNet_433_beads_002_roi001_output_depth]|![JNet_433_beads_002_roi001_reconst_depth]|![JNet_433_beads_002_roi001_heatmap_depth]|
  
volume: 4.802892089843751, MSE: 0.00011222852481296286, quantized loss: 7.607745374116348e-06  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_433_beads_002_roi002_original_depth]|![JNet_433_beads_002_roi002_output_depth]|![JNet_433_beads_002_roi002_reconst_depth]|![JNet_433_beads_002_roi002_heatmap_depth]|
  
volume: 5.040486328125001, MSE: 9.528244117973372e-05, quantized loss: 8.383407475776039e-06  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_433_psf_pre]|![JNet_433_psf_post]|

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
  



[JNet_430_pretrain_0_label_depth]: /experiments/images/JNet_430_pretrain_0_label_depth.png
[JNet_430_pretrain_0_label_plane]: /experiments/images/JNet_430_pretrain_0_label_plane.png
[JNet_430_pretrain_0_original_depth]: /experiments/images/JNet_430_pretrain_0_original_depth.png
[JNet_430_pretrain_0_original_plane]: /experiments/images/JNet_430_pretrain_0_original_plane.png
[JNet_430_pretrain_0_output_depth]: /experiments/images/JNet_430_pretrain_0_output_depth.png
[JNet_430_pretrain_0_output_plane]: /experiments/images/JNet_430_pretrain_0_output_plane.png
[JNet_430_pretrain_1_label_depth]: /experiments/images/JNet_430_pretrain_1_label_depth.png
[JNet_430_pretrain_1_label_plane]: /experiments/images/JNet_430_pretrain_1_label_plane.png
[JNet_430_pretrain_1_original_depth]: /experiments/images/JNet_430_pretrain_1_original_depth.png
[JNet_430_pretrain_1_original_plane]: /experiments/images/JNet_430_pretrain_1_original_plane.png
[JNet_430_pretrain_1_output_depth]: /experiments/images/JNet_430_pretrain_1_output_depth.png
[JNet_430_pretrain_1_output_plane]: /experiments/images/JNet_430_pretrain_1_output_plane.png
[JNet_430_pretrain_2_label_depth]: /experiments/images/JNet_430_pretrain_2_label_depth.png
[JNet_430_pretrain_2_label_plane]: /experiments/images/JNet_430_pretrain_2_label_plane.png
[JNet_430_pretrain_2_original_depth]: /experiments/images/JNet_430_pretrain_2_original_depth.png
[JNet_430_pretrain_2_original_plane]: /experiments/images/JNet_430_pretrain_2_original_plane.png
[JNet_430_pretrain_2_output_depth]: /experiments/images/JNet_430_pretrain_2_output_depth.png
[JNet_430_pretrain_2_output_plane]: /experiments/images/JNet_430_pretrain_2_output_plane.png
[JNet_430_pretrain_3_label_depth]: /experiments/images/JNet_430_pretrain_3_label_depth.png
[JNet_430_pretrain_3_label_plane]: /experiments/images/JNet_430_pretrain_3_label_plane.png
[JNet_430_pretrain_3_original_depth]: /experiments/images/JNet_430_pretrain_3_original_depth.png
[JNet_430_pretrain_3_original_plane]: /experiments/images/JNet_430_pretrain_3_original_plane.png
[JNet_430_pretrain_3_output_depth]: /experiments/images/JNet_430_pretrain_3_output_depth.png
[JNet_430_pretrain_3_output_plane]: /experiments/images/JNet_430_pretrain_3_output_plane.png
[JNet_430_pretrain_4_label_depth]: /experiments/images/JNet_430_pretrain_4_label_depth.png
[JNet_430_pretrain_4_label_plane]: /experiments/images/JNet_430_pretrain_4_label_plane.png
[JNet_430_pretrain_4_original_depth]: /experiments/images/JNet_430_pretrain_4_original_depth.png
[JNet_430_pretrain_4_original_plane]: /experiments/images/JNet_430_pretrain_4_original_plane.png
[JNet_430_pretrain_4_output_depth]: /experiments/images/JNet_430_pretrain_4_output_depth.png
[JNet_430_pretrain_4_output_plane]: /experiments/images/JNet_430_pretrain_4_output_plane.png
[JNet_430_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_430_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi000_original_depth.png
[JNet_430_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi000_output_depth.png
[JNet_430_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi000_reconst_depth.png
[JNet_430_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_430_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi001_original_depth.png
[JNet_430_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi001_output_depth.png
[JNet_430_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi001_reconst_depth.png
[JNet_430_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_430_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi002_original_depth.png
[JNet_430_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi002_output_depth.png
[JNet_430_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi002_reconst_depth.png
[JNet_430_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_430_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi003_original_depth.png
[JNet_430_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi003_output_depth.png
[JNet_430_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi003_reconst_depth.png
[JNet_430_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_430_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi004_original_depth.png
[JNet_430_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi004_output_depth.png
[JNet_430_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi004_reconst_depth.png
[JNet_430_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_430_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi000_original_depth.png
[JNet_430_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi000_output_depth.png
[JNet_430_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi000_reconst_depth.png
[JNet_430_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_430_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi001_original_depth.png
[JNet_430_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi001_output_depth.png
[JNet_430_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi001_reconst_depth.png
[JNet_430_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_430_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi002_original_depth.png
[JNet_430_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi002_output_depth.png
[JNet_430_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi002_reconst_depth.png
[JNet_433_0_label_depth]: /experiments/images/JNet_433_0_label_depth.png
[JNet_433_0_label_plane]: /experiments/images/JNet_433_0_label_plane.png
[JNet_433_0_original_depth]: /experiments/images/JNet_433_0_original_depth.png
[JNet_433_0_original_plane]: /experiments/images/JNet_433_0_original_plane.png
[JNet_433_0_output_depth]: /experiments/images/JNet_433_0_output_depth.png
[JNet_433_0_output_plane]: /experiments/images/JNet_433_0_output_plane.png
[JNet_433_1_label_depth]: /experiments/images/JNet_433_1_label_depth.png
[JNet_433_1_label_plane]: /experiments/images/JNet_433_1_label_plane.png
[JNet_433_1_original_depth]: /experiments/images/JNet_433_1_original_depth.png
[JNet_433_1_original_plane]: /experiments/images/JNet_433_1_original_plane.png
[JNet_433_1_output_depth]: /experiments/images/JNet_433_1_output_depth.png
[JNet_433_1_output_plane]: /experiments/images/JNet_433_1_output_plane.png
[JNet_433_2_label_depth]: /experiments/images/JNet_433_2_label_depth.png
[JNet_433_2_label_plane]: /experiments/images/JNet_433_2_label_plane.png
[JNet_433_2_original_depth]: /experiments/images/JNet_433_2_original_depth.png
[JNet_433_2_original_plane]: /experiments/images/JNet_433_2_original_plane.png
[JNet_433_2_output_depth]: /experiments/images/JNet_433_2_output_depth.png
[JNet_433_2_output_plane]: /experiments/images/JNet_433_2_output_plane.png
[JNet_433_3_label_depth]: /experiments/images/JNet_433_3_label_depth.png
[JNet_433_3_label_plane]: /experiments/images/JNet_433_3_label_plane.png
[JNet_433_3_original_depth]: /experiments/images/JNet_433_3_original_depth.png
[JNet_433_3_original_plane]: /experiments/images/JNet_433_3_original_plane.png
[JNet_433_3_output_depth]: /experiments/images/JNet_433_3_output_depth.png
[JNet_433_3_output_plane]: /experiments/images/JNet_433_3_output_plane.png
[JNet_433_4_label_depth]: /experiments/images/JNet_433_4_label_depth.png
[JNet_433_4_label_plane]: /experiments/images/JNet_433_4_label_plane.png
[JNet_433_4_original_depth]: /experiments/images/JNet_433_4_original_depth.png
[JNet_433_4_original_plane]: /experiments/images/JNet_433_4_original_plane.png
[JNet_433_4_output_depth]: /experiments/images/JNet_433_4_output_depth.png
[JNet_433_4_output_plane]: /experiments/images/JNet_433_4_output_plane.png
[JNet_433_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_433_beads_001_roi000_heatmap_depth.png
[JNet_433_beads_001_roi000_original_depth]: /experiments/images/JNet_433_beads_001_roi000_original_depth.png
[JNet_433_beads_001_roi000_output_depth]: /experiments/images/JNet_433_beads_001_roi000_output_depth.png
[JNet_433_beads_001_roi000_reconst_depth]: /experiments/images/JNet_433_beads_001_roi000_reconst_depth.png
[JNet_433_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_433_beads_001_roi001_heatmap_depth.png
[JNet_433_beads_001_roi001_original_depth]: /experiments/images/JNet_433_beads_001_roi001_original_depth.png
[JNet_433_beads_001_roi001_output_depth]: /experiments/images/JNet_433_beads_001_roi001_output_depth.png
[JNet_433_beads_001_roi001_reconst_depth]: /experiments/images/JNet_433_beads_001_roi001_reconst_depth.png
[JNet_433_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_433_beads_001_roi002_heatmap_depth.png
[JNet_433_beads_001_roi002_original_depth]: /experiments/images/JNet_433_beads_001_roi002_original_depth.png
[JNet_433_beads_001_roi002_output_depth]: /experiments/images/JNet_433_beads_001_roi002_output_depth.png
[JNet_433_beads_001_roi002_reconst_depth]: /experiments/images/JNet_433_beads_001_roi002_reconst_depth.png
[JNet_433_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_433_beads_001_roi003_heatmap_depth.png
[JNet_433_beads_001_roi003_original_depth]: /experiments/images/JNet_433_beads_001_roi003_original_depth.png
[JNet_433_beads_001_roi003_output_depth]: /experiments/images/JNet_433_beads_001_roi003_output_depth.png
[JNet_433_beads_001_roi003_reconst_depth]: /experiments/images/JNet_433_beads_001_roi003_reconst_depth.png
[JNet_433_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_433_beads_001_roi004_heatmap_depth.png
[JNet_433_beads_001_roi004_original_depth]: /experiments/images/JNet_433_beads_001_roi004_original_depth.png
[JNet_433_beads_001_roi004_output_depth]: /experiments/images/JNet_433_beads_001_roi004_output_depth.png
[JNet_433_beads_001_roi004_reconst_depth]: /experiments/images/JNet_433_beads_001_roi004_reconst_depth.png
[JNet_433_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_433_beads_002_roi000_heatmap_depth.png
[JNet_433_beads_002_roi000_original_depth]: /experiments/images/JNet_433_beads_002_roi000_original_depth.png
[JNet_433_beads_002_roi000_output_depth]: /experiments/images/JNet_433_beads_002_roi000_output_depth.png
[JNet_433_beads_002_roi000_reconst_depth]: /experiments/images/JNet_433_beads_002_roi000_reconst_depth.png
[JNet_433_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_433_beads_002_roi001_heatmap_depth.png
[JNet_433_beads_002_roi001_original_depth]: /experiments/images/JNet_433_beads_002_roi001_original_depth.png
[JNet_433_beads_002_roi001_output_depth]: /experiments/images/JNet_433_beads_002_roi001_output_depth.png
[JNet_433_beads_002_roi001_reconst_depth]: /experiments/images/JNet_433_beads_002_roi001_reconst_depth.png
[JNet_433_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_433_beads_002_roi002_heatmap_depth.png
[JNet_433_beads_002_roi002_original_depth]: /experiments/images/JNet_433_beads_002_roi002_original_depth.png
[JNet_433_beads_002_roi002_output_depth]: /experiments/images/JNet_433_beads_002_roi002_output_depth.png
[JNet_433_beads_002_roi002_reconst_depth]: /experiments/images/JNet_433_beads_002_roi002_reconst_depth.png
[JNet_433_psf_post]: /experiments/images/JNet_433_psf_post.png
[JNet_433_psf_pre]: /experiments/images/JNet_433_psf_pre.png
[finetuned]: /experiments/tmp/JNet_433_train.png
[pretrained_model]: /experiments/tmp/JNet_430_pretrain_train.png
