



# JNet_436 Report
  
the parameters to replicate the results of JNet_436. nearest interp of PSF, NA=0.7, mu_z = 0.3, sig_z = 1.27  
pretrained model : JNet_428_pretrain
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
|mu_z|0.9||
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
  
mean MSE: 0.02964853122830391, mean BCE: 0.11904492229223251
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_428_pretrain_0_original_plane]|![JNet_428_pretrain_0_output_plane]|![JNet_428_pretrain_0_label_plane]|
  
MSE: 0.03916694596409798, BCE: 0.16273698210716248  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_428_pretrain_0_original_depth]|![JNet_428_pretrain_0_output_depth]|![JNet_428_pretrain_0_label_depth]|
  
MSE: 0.03916694596409798, BCE: 0.16273698210716248  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_428_pretrain_1_original_plane]|![JNet_428_pretrain_1_output_plane]|![JNet_428_pretrain_1_label_plane]|
  
MSE: 0.02410181239247322, BCE: 0.08678459376096725  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_428_pretrain_1_original_depth]|![JNet_428_pretrain_1_output_depth]|![JNet_428_pretrain_1_label_depth]|
  
MSE: 0.02410181239247322, BCE: 0.08678459376096725  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_428_pretrain_2_original_plane]|![JNet_428_pretrain_2_output_plane]|![JNet_428_pretrain_2_label_plane]|
  
MSE: 0.02622942440211773, BCE: 0.1171422079205513  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_428_pretrain_2_original_depth]|![JNet_428_pretrain_2_output_depth]|![JNet_428_pretrain_2_label_depth]|
  
MSE: 0.02622942440211773, BCE: 0.1171422079205513  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_428_pretrain_3_original_plane]|![JNet_428_pretrain_3_output_plane]|![JNet_428_pretrain_3_label_plane]|
  
MSE: 0.038032956421375275, BCE: 0.15398477017879486  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_428_pretrain_3_original_depth]|![JNet_428_pretrain_3_output_depth]|![JNet_428_pretrain_3_label_depth]|
  
MSE: 0.038032956421375275, BCE: 0.15398477017879486  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_428_pretrain_4_original_plane]|![JNet_428_pretrain_4_output_plane]|![JNet_428_pretrain_4_label_plane]|
  
MSE: 0.02071150578558445, BCE: 0.07457609474658966  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_428_pretrain_4_original_depth]|![JNet_428_pretrain_4_output_depth]|![JNet_428_pretrain_4_label_depth]|
  
MSE: 0.02071150578558445, BCE: 0.07457609474658966  
  
mean MSE: 0.025257963687181473, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_436_0_original_plane]|![JNet_436_0_output_plane]|![JNet_436_0_label_plane]|
  
MSE: 0.02043573372066021, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_436_0_original_depth]|![JNet_436_0_output_depth]|![JNet_436_0_label_depth]|
  
MSE: 0.02043573372066021, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_436_1_original_plane]|![JNet_436_1_output_plane]|![JNet_436_1_label_plane]|
  
MSE: 0.031127046793699265, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_436_1_original_depth]|![JNet_436_1_output_depth]|![JNet_436_1_label_depth]|
  
MSE: 0.031127046793699265, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_436_2_original_plane]|![JNet_436_2_output_plane]|![JNet_436_2_label_plane]|
  
MSE: 0.02164348214864731, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_436_2_original_depth]|![JNet_436_2_output_depth]|![JNet_436_2_label_depth]|
  
MSE: 0.02164348214864731, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_436_3_original_plane]|![JNet_436_3_output_plane]|![JNet_436_3_label_plane]|
  
MSE: 0.015293831937015057, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_436_3_original_depth]|![JNet_436_3_output_depth]|![JNet_436_3_label_depth]|
  
MSE: 0.015293831937015057, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_436_4_original_plane]|![JNet_436_4_output_plane]|![JNet_436_4_label_plane]|
  
MSE: 0.0377897284924984, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_436_4_original_depth]|![JNet_436_4_output_depth]|![JNet_436_4_label_depth]|
  
MSE: 0.0377897284924984, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_428_pretrain_beads_001_roi000_original_depth]|![JNet_428_pretrain_beads_001_roi000_output_depth]|![JNet_428_pretrain_beads_001_roi000_reconst_depth]|![JNet_428_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 16.342073242187503, MSE: 0.016779156401753426, quantized loss: 0.0019261672860011458  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_428_pretrain_beads_001_roi001_original_depth]|![JNet_428_pretrain_beads_001_roi001_output_depth]|![JNet_428_pretrain_beads_001_roi001_reconst_depth]|![JNet_428_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 23.892419921875007, MSE: 0.020044805482029915, quantized loss: 0.00254199281334877  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_428_pretrain_beads_001_roi002_original_depth]|![JNet_428_pretrain_beads_001_roi002_output_depth]|![JNet_428_pretrain_beads_001_roi002_reconst_depth]|![JNet_428_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 16.867767578125004, MSE: 0.018113786354660988, quantized loss: 0.0022180196829140186  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_428_pretrain_beads_001_roi003_original_depth]|![JNet_428_pretrain_beads_001_roi003_output_depth]|![JNet_428_pretrain_beads_001_roi003_reconst_depth]|![JNet_428_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 26.067439453125004, MSE: 0.024586603045463562, quantized loss: 0.002987432526424527  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_428_pretrain_beads_001_roi004_original_depth]|![JNet_428_pretrain_beads_001_roi004_output_depth]|![JNet_428_pretrain_beads_001_roi004_reconst_depth]|![JNet_428_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 17.870123046875005, MSE: 0.01933441497385502, quantized loss: 0.0021891759242862463  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_428_pretrain_beads_002_roi000_original_depth]|![JNet_428_pretrain_beads_002_roi000_output_depth]|![JNet_428_pretrain_beads_002_roi000_reconst_depth]|![JNet_428_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 19.159074218750003, MSE: 0.02117752470076084, quantized loss: 0.0023188223131000996  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_428_pretrain_beads_002_roi001_original_depth]|![JNet_428_pretrain_beads_002_roi001_output_depth]|![JNet_428_pretrain_beads_002_roi001_reconst_depth]|![JNet_428_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 18.242193359375005, MSE: 0.020624928176403046, quantized loss: 0.00232287822291255  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_428_pretrain_beads_002_roi002_original_depth]|![JNet_428_pretrain_beads_002_roi002_output_depth]|![JNet_428_pretrain_beads_002_roi002_reconst_depth]|![JNet_428_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 18.105873046875004, MSE: 0.019506633281707764, quantized loss: 0.0022198616061359644  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_436_beads_001_roi000_original_depth]|![JNet_436_beads_001_roi000_output_depth]|![JNet_436_beads_001_roi000_reconst_depth]|![JNet_436_beads_001_roi000_heatmap_depth]|
  
volume: 8.698678710937502, MSE: 0.00016979791689664125, quantized loss: 1.6225656054302817e-06  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_436_beads_001_roi001_original_depth]|![JNet_436_beads_001_roi001_output_depth]|![JNet_436_beads_001_roi001_reconst_depth]|![JNet_436_beads_001_roi001_heatmap_depth]|
  
volume: 13.959681640625003, MSE: 0.0004299450374674052, quantized loss: 2.599250137791387e-06  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_436_beads_001_roi002_original_depth]|![JNet_436_beads_001_roi002_output_depth]|![JNet_436_beads_001_roi002_reconst_depth]|![JNet_436_beads_001_roi002_heatmap_depth]|
  
volume: 8.832939453125002, MSE: 0.00013957977353129536, quantized loss: 1.679174943092221e-06  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_436_beads_001_roi003_original_depth]|![JNet_436_beads_001_roi003_output_depth]|![JNet_436_beads_001_roi003_reconst_depth]|![JNet_436_beads_001_roi003_heatmap_depth]|
  
volume: 14.621397460937503, MSE: 0.0003522173792589456, quantized loss: 2.7181101813766873e-06  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_436_beads_001_roi004_original_depth]|![JNet_436_beads_001_roi004_output_depth]|![JNet_436_beads_001_roi004_reconst_depth]|![JNet_436_beads_001_roi004_heatmap_depth]|
  
volume: 9.843459960937503, MSE: 0.00012973193952348083, quantized loss: 2.370805532336817e-06  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_436_beads_002_roi000_original_depth]|![JNet_436_beads_002_roi000_output_depth]|![JNet_436_beads_002_roi000_reconst_depth]|![JNet_436_beads_002_roi000_heatmap_depth]|
  
volume: 10.642192382812503, MSE: 0.00013970481813885272, quantized loss: 2.154801677534124e-06  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_436_beads_002_roi001_original_depth]|![JNet_436_beads_002_roi001_output_depth]|![JNet_436_beads_002_roi001_reconst_depth]|![JNet_436_beads_002_roi001_heatmap_depth]|
  
volume: 9.469467773437502, MSE: 0.00011979840928688645, quantized loss: 1.8067987639369676e-06  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_436_beads_002_roi002_original_depth]|![JNet_436_beads_002_roi002_output_depth]|![JNet_436_beads_002_roi002_reconst_depth]|![JNet_436_beads_002_roi002_heatmap_depth]|
  
volume: 9.972541015625003, MSE: 0.00012300536036491394, quantized loss: 1.6664083659634343e-06  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_436_psf_pre]|![JNet_436_psf_post]|

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
  



[JNet_428_pretrain_0_label_depth]: /experiments/images/JNet_428_pretrain_0_label_depth.png
[JNet_428_pretrain_0_label_plane]: /experiments/images/JNet_428_pretrain_0_label_plane.png
[JNet_428_pretrain_0_original_depth]: /experiments/images/JNet_428_pretrain_0_original_depth.png
[JNet_428_pretrain_0_original_plane]: /experiments/images/JNet_428_pretrain_0_original_plane.png
[JNet_428_pretrain_0_output_depth]: /experiments/images/JNet_428_pretrain_0_output_depth.png
[JNet_428_pretrain_0_output_plane]: /experiments/images/JNet_428_pretrain_0_output_plane.png
[JNet_428_pretrain_1_label_depth]: /experiments/images/JNet_428_pretrain_1_label_depth.png
[JNet_428_pretrain_1_label_plane]: /experiments/images/JNet_428_pretrain_1_label_plane.png
[JNet_428_pretrain_1_original_depth]: /experiments/images/JNet_428_pretrain_1_original_depth.png
[JNet_428_pretrain_1_original_plane]: /experiments/images/JNet_428_pretrain_1_original_plane.png
[JNet_428_pretrain_1_output_depth]: /experiments/images/JNet_428_pretrain_1_output_depth.png
[JNet_428_pretrain_1_output_plane]: /experiments/images/JNet_428_pretrain_1_output_plane.png
[JNet_428_pretrain_2_label_depth]: /experiments/images/JNet_428_pretrain_2_label_depth.png
[JNet_428_pretrain_2_label_plane]: /experiments/images/JNet_428_pretrain_2_label_plane.png
[JNet_428_pretrain_2_original_depth]: /experiments/images/JNet_428_pretrain_2_original_depth.png
[JNet_428_pretrain_2_original_plane]: /experiments/images/JNet_428_pretrain_2_original_plane.png
[JNet_428_pretrain_2_output_depth]: /experiments/images/JNet_428_pretrain_2_output_depth.png
[JNet_428_pretrain_2_output_plane]: /experiments/images/JNet_428_pretrain_2_output_plane.png
[JNet_428_pretrain_3_label_depth]: /experiments/images/JNet_428_pretrain_3_label_depth.png
[JNet_428_pretrain_3_label_plane]: /experiments/images/JNet_428_pretrain_3_label_plane.png
[JNet_428_pretrain_3_original_depth]: /experiments/images/JNet_428_pretrain_3_original_depth.png
[JNet_428_pretrain_3_original_plane]: /experiments/images/JNet_428_pretrain_3_original_plane.png
[JNet_428_pretrain_3_output_depth]: /experiments/images/JNet_428_pretrain_3_output_depth.png
[JNet_428_pretrain_3_output_plane]: /experiments/images/JNet_428_pretrain_3_output_plane.png
[JNet_428_pretrain_4_label_depth]: /experiments/images/JNet_428_pretrain_4_label_depth.png
[JNet_428_pretrain_4_label_plane]: /experiments/images/JNet_428_pretrain_4_label_plane.png
[JNet_428_pretrain_4_original_depth]: /experiments/images/JNet_428_pretrain_4_original_depth.png
[JNet_428_pretrain_4_original_plane]: /experiments/images/JNet_428_pretrain_4_original_plane.png
[JNet_428_pretrain_4_output_depth]: /experiments/images/JNet_428_pretrain_4_output_depth.png
[JNet_428_pretrain_4_output_plane]: /experiments/images/JNet_428_pretrain_4_output_plane.png
[JNet_428_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_428_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi000_original_depth.png
[JNet_428_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi000_output_depth.png
[JNet_428_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi000_reconst_depth.png
[JNet_428_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_428_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi001_original_depth.png
[JNet_428_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi001_output_depth.png
[JNet_428_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi001_reconst_depth.png
[JNet_428_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_428_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi002_original_depth.png
[JNet_428_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi002_output_depth.png
[JNet_428_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi002_reconst_depth.png
[JNet_428_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_428_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi003_original_depth.png
[JNet_428_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi003_output_depth.png
[JNet_428_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi003_reconst_depth.png
[JNet_428_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_428_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi004_original_depth.png
[JNet_428_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi004_output_depth.png
[JNet_428_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_428_pretrain_beads_001_roi004_reconst_depth.png
[JNet_428_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_428_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi000_original_depth.png
[JNet_428_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi000_output_depth.png
[JNet_428_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi000_reconst_depth.png
[JNet_428_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_428_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi001_original_depth.png
[JNet_428_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi001_output_depth.png
[JNet_428_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi001_reconst_depth.png
[JNet_428_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_428_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi002_original_depth.png
[JNet_428_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi002_output_depth.png
[JNet_428_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_428_pretrain_beads_002_roi002_reconst_depth.png
[JNet_436_0_label_depth]: /experiments/images/JNet_436_0_label_depth.png
[JNet_436_0_label_plane]: /experiments/images/JNet_436_0_label_plane.png
[JNet_436_0_original_depth]: /experiments/images/JNet_436_0_original_depth.png
[JNet_436_0_original_plane]: /experiments/images/JNet_436_0_original_plane.png
[JNet_436_0_output_depth]: /experiments/images/JNet_436_0_output_depth.png
[JNet_436_0_output_plane]: /experiments/images/JNet_436_0_output_plane.png
[JNet_436_1_label_depth]: /experiments/images/JNet_436_1_label_depth.png
[JNet_436_1_label_plane]: /experiments/images/JNet_436_1_label_plane.png
[JNet_436_1_original_depth]: /experiments/images/JNet_436_1_original_depth.png
[JNet_436_1_original_plane]: /experiments/images/JNet_436_1_original_plane.png
[JNet_436_1_output_depth]: /experiments/images/JNet_436_1_output_depth.png
[JNet_436_1_output_plane]: /experiments/images/JNet_436_1_output_plane.png
[JNet_436_2_label_depth]: /experiments/images/JNet_436_2_label_depth.png
[JNet_436_2_label_plane]: /experiments/images/JNet_436_2_label_plane.png
[JNet_436_2_original_depth]: /experiments/images/JNet_436_2_original_depth.png
[JNet_436_2_original_plane]: /experiments/images/JNet_436_2_original_plane.png
[JNet_436_2_output_depth]: /experiments/images/JNet_436_2_output_depth.png
[JNet_436_2_output_plane]: /experiments/images/JNet_436_2_output_plane.png
[JNet_436_3_label_depth]: /experiments/images/JNet_436_3_label_depth.png
[JNet_436_3_label_plane]: /experiments/images/JNet_436_3_label_plane.png
[JNet_436_3_original_depth]: /experiments/images/JNet_436_3_original_depth.png
[JNet_436_3_original_plane]: /experiments/images/JNet_436_3_original_plane.png
[JNet_436_3_output_depth]: /experiments/images/JNet_436_3_output_depth.png
[JNet_436_3_output_plane]: /experiments/images/JNet_436_3_output_plane.png
[JNet_436_4_label_depth]: /experiments/images/JNet_436_4_label_depth.png
[JNet_436_4_label_plane]: /experiments/images/JNet_436_4_label_plane.png
[JNet_436_4_original_depth]: /experiments/images/JNet_436_4_original_depth.png
[JNet_436_4_original_plane]: /experiments/images/JNet_436_4_original_plane.png
[JNet_436_4_output_depth]: /experiments/images/JNet_436_4_output_depth.png
[JNet_436_4_output_plane]: /experiments/images/JNet_436_4_output_plane.png
[JNet_436_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_436_beads_001_roi000_heatmap_depth.png
[JNet_436_beads_001_roi000_original_depth]: /experiments/images/JNet_436_beads_001_roi000_original_depth.png
[JNet_436_beads_001_roi000_output_depth]: /experiments/images/JNet_436_beads_001_roi000_output_depth.png
[JNet_436_beads_001_roi000_reconst_depth]: /experiments/images/JNet_436_beads_001_roi000_reconst_depth.png
[JNet_436_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_436_beads_001_roi001_heatmap_depth.png
[JNet_436_beads_001_roi001_original_depth]: /experiments/images/JNet_436_beads_001_roi001_original_depth.png
[JNet_436_beads_001_roi001_output_depth]: /experiments/images/JNet_436_beads_001_roi001_output_depth.png
[JNet_436_beads_001_roi001_reconst_depth]: /experiments/images/JNet_436_beads_001_roi001_reconst_depth.png
[JNet_436_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_436_beads_001_roi002_heatmap_depth.png
[JNet_436_beads_001_roi002_original_depth]: /experiments/images/JNet_436_beads_001_roi002_original_depth.png
[JNet_436_beads_001_roi002_output_depth]: /experiments/images/JNet_436_beads_001_roi002_output_depth.png
[JNet_436_beads_001_roi002_reconst_depth]: /experiments/images/JNet_436_beads_001_roi002_reconst_depth.png
[JNet_436_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_436_beads_001_roi003_heatmap_depth.png
[JNet_436_beads_001_roi003_original_depth]: /experiments/images/JNet_436_beads_001_roi003_original_depth.png
[JNet_436_beads_001_roi003_output_depth]: /experiments/images/JNet_436_beads_001_roi003_output_depth.png
[JNet_436_beads_001_roi003_reconst_depth]: /experiments/images/JNet_436_beads_001_roi003_reconst_depth.png
[JNet_436_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_436_beads_001_roi004_heatmap_depth.png
[JNet_436_beads_001_roi004_original_depth]: /experiments/images/JNet_436_beads_001_roi004_original_depth.png
[JNet_436_beads_001_roi004_output_depth]: /experiments/images/JNet_436_beads_001_roi004_output_depth.png
[JNet_436_beads_001_roi004_reconst_depth]: /experiments/images/JNet_436_beads_001_roi004_reconst_depth.png
[JNet_436_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_436_beads_002_roi000_heatmap_depth.png
[JNet_436_beads_002_roi000_original_depth]: /experiments/images/JNet_436_beads_002_roi000_original_depth.png
[JNet_436_beads_002_roi000_output_depth]: /experiments/images/JNet_436_beads_002_roi000_output_depth.png
[JNet_436_beads_002_roi000_reconst_depth]: /experiments/images/JNet_436_beads_002_roi000_reconst_depth.png
[JNet_436_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_436_beads_002_roi001_heatmap_depth.png
[JNet_436_beads_002_roi001_original_depth]: /experiments/images/JNet_436_beads_002_roi001_original_depth.png
[JNet_436_beads_002_roi001_output_depth]: /experiments/images/JNet_436_beads_002_roi001_output_depth.png
[JNet_436_beads_002_roi001_reconst_depth]: /experiments/images/JNet_436_beads_002_roi001_reconst_depth.png
[JNet_436_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_436_beads_002_roi002_heatmap_depth.png
[JNet_436_beads_002_roi002_original_depth]: /experiments/images/JNet_436_beads_002_roi002_original_depth.png
[JNet_436_beads_002_roi002_output_depth]: /experiments/images/JNet_436_beads_002_roi002_output_depth.png
[JNet_436_beads_002_roi002_reconst_depth]: /experiments/images/JNet_436_beads_002_roi002_reconst_depth.png
[JNet_436_psf_post]: /experiments/images/JNet_436_psf_post.png
[JNet_436_psf_pre]: /experiments/images/JNet_436_psf_pre.png
[finetuned]: /experiments/tmp/JNet_436_train.png
[pretrained_model]: /experiments/tmp/JNet_428_pretrain_train.png
