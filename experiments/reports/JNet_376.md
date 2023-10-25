



# JNet_376 Report
  
the parameters to replicate the results of JNet_376. deterministic background simulation training. no  noise.  
pretrained model : JNet_375_pretrain
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
|use_fftconv|True||
|mu_z|0.1||
|sig_z|0.1||
|blur_mode|gibsonlanni|`gaussian` or `gibsonlanni`|
|size_x|51||
|size_y|51||
|size_z|161||
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
|res_axial|0.05|microns|
|pZ|0|microns, particle distance from coverslip|
|bet_z|30.0||
|bet_xy|3.0||
|sig_eps|0.0||
|background|0.01||
|scale|10||
|device|cuda||

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
    (conv): Conv3d(16, 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
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
|loss_fn|nn.BCELoss()|
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
|qloss_weight|1|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.01915002055466175, mean BCE: 0.07413958013057709
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_375_pretrain_0_original_plane]|![JNet_375_pretrain_0_output_plane]|![JNet_375_pretrain_0_label_plane]|
  
MSE: 0.01439431868493557, BCE: 0.0540987029671669  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_375_pretrain_0_original_depth]|![JNet_375_pretrain_0_output_depth]|![JNet_375_pretrain_0_label_depth]|
  
MSE: 0.01439431868493557, BCE: 0.0540987029671669  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_375_pretrain_1_original_plane]|![JNet_375_pretrain_1_output_plane]|![JNet_375_pretrain_1_label_plane]|
  
MSE: 0.015310611575841904, BCE: 0.05388496443629265  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_375_pretrain_1_original_depth]|![JNet_375_pretrain_1_output_depth]|![JNet_375_pretrain_1_label_depth]|
  
MSE: 0.015310611575841904, BCE: 0.05388496443629265  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_375_pretrain_2_original_plane]|![JNet_375_pretrain_2_output_plane]|![JNet_375_pretrain_2_label_plane]|
  
MSE: 0.021842166781425476, BCE: 0.08953425288200378  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_375_pretrain_2_original_depth]|![JNet_375_pretrain_2_output_depth]|![JNet_375_pretrain_2_label_depth]|
  
MSE: 0.021842166781425476, BCE: 0.08953425288200378  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_375_pretrain_3_original_plane]|![JNet_375_pretrain_3_output_plane]|![JNet_375_pretrain_3_label_plane]|
  
MSE: 0.017968829721212387, BCE: 0.07336310297250748  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_375_pretrain_3_original_depth]|![JNet_375_pretrain_3_output_depth]|![JNet_375_pretrain_3_label_depth]|
  
MSE: 0.017968829721212387, BCE: 0.07336310297250748  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_375_pretrain_4_original_plane]|![JNet_375_pretrain_4_output_plane]|![JNet_375_pretrain_4_label_plane]|
  
MSE: 0.02623417042195797, BCE: 0.09981686621904373  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_375_pretrain_4_original_depth]|![JNet_375_pretrain_4_output_depth]|![JNet_375_pretrain_4_label_depth]|
  
MSE: 0.02623417042195797, BCE: 0.09981686621904373  
  
mean MSE: 0.03257668763399124, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_376_0_original_plane]|![JNet_376_0_output_plane]|![JNet_376_0_label_plane]|
  
MSE: 0.02890351414680481, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_376_0_original_depth]|![JNet_376_0_output_depth]|![JNet_376_0_label_depth]|
  
MSE: 0.02890351414680481, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_376_1_original_plane]|![JNet_376_1_output_plane]|![JNet_376_1_label_plane]|
  
MSE: 0.028008997440338135, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_376_1_original_depth]|![JNet_376_1_output_depth]|![JNet_376_1_label_depth]|
  
MSE: 0.028008997440338135, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_376_2_original_plane]|![JNet_376_2_output_plane]|![JNet_376_2_label_plane]|
  
MSE: 0.04012799635529518, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_376_2_original_depth]|![JNet_376_2_output_depth]|![JNet_376_2_label_depth]|
  
MSE: 0.04012799635529518, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_376_3_original_plane]|![JNet_376_3_output_plane]|![JNet_376_3_label_plane]|
  
MSE: 0.03085995838046074, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_376_3_original_depth]|![JNet_376_3_output_depth]|![JNet_376_3_label_depth]|
  
MSE: 0.03085995838046074, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_376_4_original_plane]|![JNet_376_4_output_plane]|![JNet_376_4_label_plane]|
  
MSE: 0.03498298302292824, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_376_4_original_depth]|![JNet_376_4_output_depth]|![JNet_376_4_label_depth]|
  
MSE: 0.03498298302292824, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_375_pretrain_beads_001_roi000_original_depth]|![JNet_375_pretrain_beads_001_roi000_output_depth]|![JNet_375_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 11.543875000000003, MSE: 0.0011076589580625296, quantized loss: 0.001554663060232997  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_375_pretrain_beads_001_roi001_original_depth]|![JNet_375_pretrain_beads_001_roi001_output_depth]|![JNet_375_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 18.419875000000005, MSE: 0.0019004152854904532, quantized loss: 0.001972902100533247  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_375_pretrain_beads_001_roi002_original_depth]|![JNet_375_pretrain_beads_001_roi002_output_depth]|![JNet_375_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 11.274125000000003, MSE: 0.00111181300599128, quantized loss: 0.001172770163975656  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_375_pretrain_beads_001_roi003_original_depth]|![JNet_375_pretrain_beads_001_roi003_output_depth]|![JNet_375_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 18.534250000000004, MSE: 0.001910274033434689, quantized loss: 0.0019129309803247452  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_375_pretrain_beads_001_roi004_original_depth]|![JNet_375_pretrain_beads_001_roi004_output_depth]|![JNet_375_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 11.995250000000002, MSE: 0.0014374415623024106, quantized loss: 0.001233773655258119  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_375_pretrain_beads_002_roi000_original_depth]|![JNet_375_pretrain_beads_002_roi000_output_depth]|![JNet_375_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 12.715000000000003, MSE: 0.0016312951920554042, quantized loss: 0.001293045119382441  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_375_pretrain_beads_002_roi001_original_depth]|![JNet_375_pretrain_beads_002_roi001_output_depth]|![JNet_375_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 12.055000000000003, MSE: 0.0011400670045986772, quantized loss: 0.0012389479670673609  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_375_pretrain_beads_002_roi002_original_depth]|![JNet_375_pretrain_beads_002_roi002_output_depth]|![JNet_375_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 12.130250000000004, MSE: 0.0014152918010950089, quantized loss: 0.001222185674123466  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_376_beads_001_roi000_original_depth]|![JNet_376_beads_001_roi000_output_depth]|![JNet_376_beads_001_roi000_reconst_depth]|
  
volume: 9.665875000000002, MSE: 0.00019466238154564053, quantized loss: 7.591216672153678e-06  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_376_beads_001_roi001_original_depth]|![JNet_376_beads_001_roi001_output_depth]|![JNet_376_beads_001_roi001_reconst_depth]|
  
volume: 15.045875000000004, MSE: 0.0006031741504557431, quantized loss: 9.945673809852451e-06  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_376_beads_001_roi002_original_depth]|![JNet_376_beads_001_roi002_output_depth]|![JNet_376_beads_001_roi002_reconst_depth]|
  
volume: 9.782500000000002, MSE: 0.0001628173777135089, quantized loss: 6.512988875329029e-06  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_376_beads_001_roi003_original_depth]|![JNet_376_beads_001_roi003_output_depth]|![JNet_376_beads_001_roi003_reconst_depth]|
  
volume: 16.044750000000004, MSE: 0.0003558749158401042, quantized loss: 1.0702542567742057e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_376_beads_001_roi004_original_depth]|![JNet_376_beads_001_roi004_output_depth]|![JNet_376_beads_001_roi004_reconst_depth]|
  
volume: 10.674250000000002, MSE: 0.0001246953324880451, quantized loss: 6.86097655488993e-06  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_376_beads_002_roi000_original_depth]|![JNet_376_beads_002_roi000_output_depth]|![JNet_376_beads_002_roi000_reconst_depth]|
  
volume: 11.383625000000002, MSE: 0.00011285213258815929, quantized loss: 7.171469405875541e-06  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_376_beads_002_roi001_original_depth]|![JNet_376_beads_002_roi001_output_depth]|![JNet_376_beads_002_roi001_reconst_depth]|
  
volume: 10.362750000000002, MSE: 0.00013670188491232693, quantized loss: 7.650167390238494e-06  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_376_beads_002_roi002_original_depth]|![JNet_376_beads_002_roi002_output_depth]|![JNet_376_beads_002_roi002_reconst_depth]|
  
volume: 10.812125000000002, MSE: 0.00012291713210288435, quantized loss: 7.61004457672243e-06  

|pre|post|
| :---: | :---: |
|![JNet_376_psf_pre]|![JNet_376_psf_post]|
  



[JNet_375_pretrain_0_label_depth]: /experiments/images/JNet_375_pretrain_0_label_depth.png
[JNet_375_pretrain_0_label_plane]: /experiments/images/JNet_375_pretrain_0_label_plane.png
[JNet_375_pretrain_0_original_depth]: /experiments/images/JNet_375_pretrain_0_original_depth.png
[JNet_375_pretrain_0_original_plane]: /experiments/images/JNet_375_pretrain_0_original_plane.png
[JNet_375_pretrain_0_output_depth]: /experiments/images/JNet_375_pretrain_0_output_depth.png
[JNet_375_pretrain_0_output_plane]: /experiments/images/JNet_375_pretrain_0_output_plane.png
[JNet_375_pretrain_1_label_depth]: /experiments/images/JNet_375_pretrain_1_label_depth.png
[JNet_375_pretrain_1_label_plane]: /experiments/images/JNet_375_pretrain_1_label_plane.png
[JNet_375_pretrain_1_original_depth]: /experiments/images/JNet_375_pretrain_1_original_depth.png
[JNet_375_pretrain_1_original_plane]: /experiments/images/JNet_375_pretrain_1_original_plane.png
[JNet_375_pretrain_1_output_depth]: /experiments/images/JNet_375_pretrain_1_output_depth.png
[JNet_375_pretrain_1_output_plane]: /experiments/images/JNet_375_pretrain_1_output_plane.png
[JNet_375_pretrain_2_label_depth]: /experiments/images/JNet_375_pretrain_2_label_depth.png
[JNet_375_pretrain_2_label_plane]: /experiments/images/JNet_375_pretrain_2_label_plane.png
[JNet_375_pretrain_2_original_depth]: /experiments/images/JNet_375_pretrain_2_original_depth.png
[JNet_375_pretrain_2_original_plane]: /experiments/images/JNet_375_pretrain_2_original_plane.png
[JNet_375_pretrain_2_output_depth]: /experiments/images/JNet_375_pretrain_2_output_depth.png
[JNet_375_pretrain_2_output_plane]: /experiments/images/JNet_375_pretrain_2_output_plane.png
[JNet_375_pretrain_3_label_depth]: /experiments/images/JNet_375_pretrain_3_label_depth.png
[JNet_375_pretrain_3_label_plane]: /experiments/images/JNet_375_pretrain_3_label_plane.png
[JNet_375_pretrain_3_original_depth]: /experiments/images/JNet_375_pretrain_3_original_depth.png
[JNet_375_pretrain_3_original_plane]: /experiments/images/JNet_375_pretrain_3_original_plane.png
[JNet_375_pretrain_3_output_depth]: /experiments/images/JNet_375_pretrain_3_output_depth.png
[JNet_375_pretrain_3_output_plane]: /experiments/images/JNet_375_pretrain_3_output_plane.png
[JNet_375_pretrain_4_label_depth]: /experiments/images/JNet_375_pretrain_4_label_depth.png
[JNet_375_pretrain_4_label_plane]: /experiments/images/JNet_375_pretrain_4_label_plane.png
[JNet_375_pretrain_4_original_depth]: /experiments/images/JNet_375_pretrain_4_original_depth.png
[JNet_375_pretrain_4_original_plane]: /experiments/images/JNet_375_pretrain_4_original_plane.png
[JNet_375_pretrain_4_output_depth]: /experiments/images/JNet_375_pretrain_4_output_depth.png
[JNet_375_pretrain_4_output_plane]: /experiments/images/JNet_375_pretrain_4_output_plane.png
[JNet_375_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi000_original_depth.png
[JNet_375_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi000_output_depth.png
[JNet_375_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi000_reconst_depth.png
[JNet_375_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi001_original_depth.png
[JNet_375_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi001_output_depth.png
[JNet_375_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi001_reconst_depth.png
[JNet_375_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi002_original_depth.png
[JNet_375_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi002_output_depth.png
[JNet_375_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi002_reconst_depth.png
[JNet_375_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi003_original_depth.png
[JNet_375_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi003_output_depth.png
[JNet_375_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi003_reconst_depth.png
[JNet_375_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi004_original_depth.png
[JNet_375_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi004_output_depth.png
[JNet_375_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_375_pretrain_beads_001_roi004_reconst_depth.png
[JNet_375_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_375_pretrain_beads_002_roi000_original_depth.png
[JNet_375_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_375_pretrain_beads_002_roi000_output_depth.png
[JNet_375_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_375_pretrain_beads_002_roi000_reconst_depth.png
[JNet_375_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_375_pretrain_beads_002_roi001_original_depth.png
[JNet_375_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_375_pretrain_beads_002_roi001_output_depth.png
[JNet_375_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_375_pretrain_beads_002_roi001_reconst_depth.png
[JNet_375_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_375_pretrain_beads_002_roi002_original_depth.png
[JNet_375_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_375_pretrain_beads_002_roi002_output_depth.png
[JNet_375_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_375_pretrain_beads_002_roi002_reconst_depth.png
[JNet_376_0_label_depth]: /experiments/images/JNet_376_0_label_depth.png
[JNet_376_0_label_plane]: /experiments/images/JNet_376_0_label_plane.png
[JNet_376_0_original_depth]: /experiments/images/JNet_376_0_original_depth.png
[JNet_376_0_original_plane]: /experiments/images/JNet_376_0_original_plane.png
[JNet_376_0_output_depth]: /experiments/images/JNet_376_0_output_depth.png
[JNet_376_0_output_plane]: /experiments/images/JNet_376_0_output_plane.png
[JNet_376_1_label_depth]: /experiments/images/JNet_376_1_label_depth.png
[JNet_376_1_label_plane]: /experiments/images/JNet_376_1_label_plane.png
[JNet_376_1_original_depth]: /experiments/images/JNet_376_1_original_depth.png
[JNet_376_1_original_plane]: /experiments/images/JNet_376_1_original_plane.png
[JNet_376_1_output_depth]: /experiments/images/JNet_376_1_output_depth.png
[JNet_376_1_output_plane]: /experiments/images/JNet_376_1_output_plane.png
[JNet_376_2_label_depth]: /experiments/images/JNet_376_2_label_depth.png
[JNet_376_2_label_plane]: /experiments/images/JNet_376_2_label_plane.png
[JNet_376_2_original_depth]: /experiments/images/JNet_376_2_original_depth.png
[JNet_376_2_original_plane]: /experiments/images/JNet_376_2_original_plane.png
[JNet_376_2_output_depth]: /experiments/images/JNet_376_2_output_depth.png
[JNet_376_2_output_plane]: /experiments/images/JNet_376_2_output_plane.png
[JNet_376_3_label_depth]: /experiments/images/JNet_376_3_label_depth.png
[JNet_376_3_label_plane]: /experiments/images/JNet_376_3_label_plane.png
[JNet_376_3_original_depth]: /experiments/images/JNet_376_3_original_depth.png
[JNet_376_3_original_plane]: /experiments/images/JNet_376_3_original_plane.png
[JNet_376_3_output_depth]: /experiments/images/JNet_376_3_output_depth.png
[JNet_376_3_output_plane]: /experiments/images/JNet_376_3_output_plane.png
[JNet_376_4_label_depth]: /experiments/images/JNet_376_4_label_depth.png
[JNet_376_4_label_plane]: /experiments/images/JNet_376_4_label_plane.png
[JNet_376_4_original_depth]: /experiments/images/JNet_376_4_original_depth.png
[JNet_376_4_original_plane]: /experiments/images/JNet_376_4_original_plane.png
[JNet_376_4_output_depth]: /experiments/images/JNet_376_4_output_depth.png
[JNet_376_4_output_plane]: /experiments/images/JNet_376_4_output_plane.png
[JNet_376_beads_001_roi000_original_depth]: /experiments/images/JNet_376_beads_001_roi000_original_depth.png
[JNet_376_beads_001_roi000_output_depth]: /experiments/images/JNet_376_beads_001_roi000_output_depth.png
[JNet_376_beads_001_roi000_reconst_depth]: /experiments/images/JNet_376_beads_001_roi000_reconst_depth.png
[JNet_376_beads_001_roi001_original_depth]: /experiments/images/JNet_376_beads_001_roi001_original_depth.png
[JNet_376_beads_001_roi001_output_depth]: /experiments/images/JNet_376_beads_001_roi001_output_depth.png
[JNet_376_beads_001_roi001_reconst_depth]: /experiments/images/JNet_376_beads_001_roi001_reconst_depth.png
[JNet_376_beads_001_roi002_original_depth]: /experiments/images/JNet_376_beads_001_roi002_original_depth.png
[JNet_376_beads_001_roi002_output_depth]: /experiments/images/JNet_376_beads_001_roi002_output_depth.png
[JNet_376_beads_001_roi002_reconst_depth]: /experiments/images/JNet_376_beads_001_roi002_reconst_depth.png
[JNet_376_beads_001_roi003_original_depth]: /experiments/images/JNet_376_beads_001_roi003_original_depth.png
[JNet_376_beads_001_roi003_output_depth]: /experiments/images/JNet_376_beads_001_roi003_output_depth.png
[JNet_376_beads_001_roi003_reconst_depth]: /experiments/images/JNet_376_beads_001_roi003_reconst_depth.png
[JNet_376_beads_001_roi004_original_depth]: /experiments/images/JNet_376_beads_001_roi004_original_depth.png
[JNet_376_beads_001_roi004_output_depth]: /experiments/images/JNet_376_beads_001_roi004_output_depth.png
[JNet_376_beads_001_roi004_reconst_depth]: /experiments/images/JNet_376_beads_001_roi004_reconst_depth.png
[JNet_376_beads_002_roi000_original_depth]: /experiments/images/JNet_376_beads_002_roi000_original_depth.png
[JNet_376_beads_002_roi000_output_depth]: /experiments/images/JNet_376_beads_002_roi000_output_depth.png
[JNet_376_beads_002_roi000_reconst_depth]: /experiments/images/JNet_376_beads_002_roi000_reconst_depth.png
[JNet_376_beads_002_roi001_original_depth]: /experiments/images/JNet_376_beads_002_roi001_original_depth.png
[JNet_376_beads_002_roi001_output_depth]: /experiments/images/JNet_376_beads_002_roi001_output_depth.png
[JNet_376_beads_002_roi001_reconst_depth]: /experiments/images/JNet_376_beads_002_roi001_reconst_depth.png
[JNet_376_beads_002_roi002_original_depth]: /experiments/images/JNet_376_beads_002_roi002_original_depth.png
[JNet_376_beads_002_roi002_output_depth]: /experiments/images/JNet_376_beads_002_roi002_output_depth.png
[JNet_376_beads_002_roi002_reconst_depth]: /experiments/images/JNet_376_beads_002_roi002_reconst_depth.png
[JNet_376_psf_post]: /experiments/images/JNet_376_psf_post.png
[JNet_376_psf_pre]: /experiments/images/JNet_376_psf_pre.png
[finetuned]: /experiments/tmp/JNet_376_train.png
[pretrained_model]: /experiments/tmp/JNet_375_pretrain_train.png
