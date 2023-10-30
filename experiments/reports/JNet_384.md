



# JNet_384 Report
  
the parameters to replicate the results of JNet_384. argmax f1 threshold applied  
pretrained model : JNet_383_pretrain
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
|threshold|0.165||
|use_fftconv|True||
|mu_z|0.1||
|sig_z|0.1||
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
|qloss_weight|1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.025121605023741722, mean BCE: 0.09168317168951035
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_383_pretrain_0_original_plane]|![JNet_383_pretrain_0_output_plane]|![JNet_383_pretrain_0_label_plane]|
  
MSE: 0.02541649155318737, BCE: 0.09585686773061752  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_383_pretrain_0_original_depth]|![JNet_383_pretrain_0_output_depth]|![JNet_383_pretrain_0_label_depth]|
  
MSE: 0.02541649155318737, BCE: 0.09585686773061752  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_383_pretrain_1_original_plane]|![JNet_383_pretrain_1_output_plane]|![JNet_383_pretrain_1_label_plane]|
  
MSE: 0.025380469858646393, BCE: 0.09192771464586258  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_383_pretrain_1_original_depth]|![JNet_383_pretrain_1_output_depth]|![JNet_383_pretrain_1_label_depth]|
  
MSE: 0.025380469858646393, BCE: 0.09192771464586258  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_383_pretrain_2_original_plane]|![JNet_383_pretrain_2_output_plane]|![JNet_383_pretrain_2_label_plane]|
  
MSE: 0.028016353026032448, BCE: 0.10102340579032898  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_383_pretrain_2_original_depth]|![JNet_383_pretrain_2_output_depth]|![JNet_383_pretrain_2_label_depth]|
  
MSE: 0.028016353026032448, BCE: 0.10102340579032898  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_383_pretrain_3_original_plane]|![JNet_383_pretrain_3_output_plane]|![JNet_383_pretrain_3_label_plane]|
  
MSE: 0.021566500887274742, BCE: 0.07720356434583664  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_383_pretrain_3_original_depth]|![JNet_383_pretrain_3_output_depth]|![JNet_383_pretrain_3_label_depth]|
  
MSE: 0.021566500887274742, BCE: 0.07720356434583664  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_383_pretrain_4_original_plane]|![JNet_383_pretrain_4_output_plane]|![JNet_383_pretrain_4_label_plane]|
  
MSE: 0.02522820420563221, BCE: 0.0924043208360672  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_383_pretrain_4_original_depth]|![JNet_383_pretrain_4_output_depth]|![JNet_383_pretrain_4_label_depth]|
  
MSE: 0.02522820420563221, BCE: 0.0924043208360672  
  
mean MSE: 0.9635376930236816, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_384_0_original_plane]|![JNet_384_0_output_plane]|![JNet_384_0_label_plane]|
  
MSE: 0.9643666744232178, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_384_0_original_depth]|![JNet_384_0_output_depth]|![JNet_384_0_label_depth]|
  
MSE: 0.9643666744232178, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_384_1_original_plane]|![JNet_384_1_output_plane]|![JNet_384_1_label_plane]|
  
MSE: 0.9647128582000732, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_384_1_original_depth]|![JNet_384_1_output_depth]|![JNet_384_1_label_depth]|
  
MSE: 0.9647128582000732, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_384_2_original_plane]|![JNet_384_2_output_plane]|![JNet_384_2_label_plane]|
  
MSE: 0.9794639348983765, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_384_2_original_depth]|![JNet_384_2_output_depth]|![JNet_384_2_label_depth]|
  
MSE: 0.9794639348983765, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_384_3_original_plane]|![JNet_384_3_output_plane]|![JNet_384_3_label_plane]|
  
MSE: 0.9552558064460754, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_384_3_original_depth]|![JNet_384_3_output_depth]|![JNet_384_3_label_depth]|
  
MSE: 0.9552558064460754, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_384_4_original_plane]|![JNet_384_4_output_plane]|![JNet_384_4_label_plane]|
  
MSE: 0.9538891315460205, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_384_4_original_depth]|![JNet_384_4_output_depth]|![JNet_384_4_label_depth]|
  
MSE: 0.9538891315460205, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_383_pretrain_beads_001_roi000_original_depth]|![JNet_383_pretrain_beads_001_roi000_output_depth]|![JNet_383_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 18.037017578125003, MSE: 0.002216769615188241, quantized loss: 0.007489526644349098  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_383_pretrain_beads_001_roi001_original_depth]|![JNet_383_pretrain_beads_001_roi001_output_depth]|![JNet_383_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 27.705529296875007, MSE: 0.003201365238055587, quantized loss: 0.011529110372066498  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_383_pretrain_beads_001_roi002_original_depth]|![JNet_383_pretrain_beads_001_roi002_output_depth]|![JNet_383_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 19.638962890625006, MSE: 0.0027284056413918734, quantized loss: 0.011026730760931969  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_383_pretrain_beads_001_roi003_original_depth]|![JNet_383_pretrain_beads_001_roi003_output_depth]|![JNet_383_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 28.255080078125008, MSE: 0.0034779980778694153, quantized loss: 0.01092518214136362  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_383_pretrain_beads_001_roi004_original_depth]|![JNet_383_pretrain_beads_001_roi004_output_depth]|![JNet_383_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 21.207960937500005, MSE: 0.0031853674445301294, quantized loss: 0.010997975245118141  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_383_pretrain_beads_002_roi000_original_depth]|![JNet_383_pretrain_beads_002_roi000_output_depth]|![JNet_383_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 23.839425781250007, MSE: 0.0038583020213991404, quantized loss: 0.01323652733117342  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_383_pretrain_beads_002_roi001_original_depth]|![JNet_383_pretrain_beads_002_roi001_output_depth]|![JNet_383_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 20.707503906250004, MSE: 0.0029493432957679033, quantized loss: 0.01088588498532772  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_383_pretrain_beads_002_roi002_original_depth]|![JNet_383_pretrain_beads_002_roi002_output_depth]|![JNet_383_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 22.331029296875005, MSE: 0.003416295861825347, quantized loss: 0.01217796839773655  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_384_beads_001_roi000_original_depth]|![JNet_384_beads_001_roi000_output_depth]|![JNet_384_beads_001_roi000_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.0432654470205307, quantized loss: 2.42009222313792e-14  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_384_beads_001_roi001_original_depth]|![JNet_384_beads_001_roi001_output_depth]|![JNet_384_beads_001_roi001_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.04435000196099281, quantized loss: 2.5607244257441937e-14  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_384_beads_001_roi002_original_depth]|![JNet_384_beads_001_roi002_output_depth]|![JNet_384_beads_001_roi002_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.044298410415649414, quantized loss: 1.353826735640178e-14  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_384_beads_001_roi003_original_depth]|![JNet_384_beads_001_roi003_output_depth]|![JNet_384_beads_001_roi003_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.04456178843975067, quantized loss: 1.647242251997562e-14  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_384_beads_001_roi004_original_depth]|![JNet_384_beads_001_roi004_output_depth]|![JNet_384_beads_001_roi004_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.04462502896785736, quantized loss: 1.2959093332123601e-14  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_384_beads_002_roi000_original_depth]|![JNet_384_beads_002_roi000_output_depth]|![JNet_384_beads_002_roi000_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.04493875429034233, quantized loss: 1.1930094837909501e-14  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_384_beads_002_roi001_original_depth]|![JNet_384_beads_002_roi001_output_depth]|![JNet_384_beads_002_roi001_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.04453446343541145, quantized loss: 1.2197654758255242e-14  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_384_beads_002_roi002_original_depth]|![JNet_384_beads_002_roi002_output_depth]|![JNet_384_beads_002_roi002_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.04467135667800903, quantized loss: 1.0903796684731248e-14  

|pre|post|
| :---: | :---: |
|![JNet_384_psf_pre]|![JNet_384_psf_post]|
  



[JNet_383_pretrain_0_label_depth]: /experiments/images/JNet_383_pretrain_0_label_depth.png
[JNet_383_pretrain_0_label_plane]: /experiments/images/JNet_383_pretrain_0_label_plane.png
[JNet_383_pretrain_0_original_depth]: /experiments/images/JNet_383_pretrain_0_original_depth.png
[JNet_383_pretrain_0_original_plane]: /experiments/images/JNet_383_pretrain_0_original_plane.png
[JNet_383_pretrain_0_output_depth]: /experiments/images/JNet_383_pretrain_0_output_depth.png
[JNet_383_pretrain_0_output_plane]: /experiments/images/JNet_383_pretrain_0_output_plane.png
[JNet_383_pretrain_1_label_depth]: /experiments/images/JNet_383_pretrain_1_label_depth.png
[JNet_383_pretrain_1_label_plane]: /experiments/images/JNet_383_pretrain_1_label_plane.png
[JNet_383_pretrain_1_original_depth]: /experiments/images/JNet_383_pretrain_1_original_depth.png
[JNet_383_pretrain_1_original_plane]: /experiments/images/JNet_383_pretrain_1_original_plane.png
[JNet_383_pretrain_1_output_depth]: /experiments/images/JNet_383_pretrain_1_output_depth.png
[JNet_383_pretrain_1_output_plane]: /experiments/images/JNet_383_pretrain_1_output_plane.png
[JNet_383_pretrain_2_label_depth]: /experiments/images/JNet_383_pretrain_2_label_depth.png
[JNet_383_pretrain_2_label_plane]: /experiments/images/JNet_383_pretrain_2_label_plane.png
[JNet_383_pretrain_2_original_depth]: /experiments/images/JNet_383_pretrain_2_original_depth.png
[JNet_383_pretrain_2_original_plane]: /experiments/images/JNet_383_pretrain_2_original_plane.png
[JNet_383_pretrain_2_output_depth]: /experiments/images/JNet_383_pretrain_2_output_depth.png
[JNet_383_pretrain_2_output_plane]: /experiments/images/JNet_383_pretrain_2_output_plane.png
[JNet_383_pretrain_3_label_depth]: /experiments/images/JNet_383_pretrain_3_label_depth.png
[JNet_383_pretrain_3_label_plane]: /experiments/images/JNet_383_pretrain_3_label_plane.png
[JNet_383_pretrain_3_original_depth]: /experiments/images/JNet_383_pretrain_3_original_depth.png
[JNet_383_pretrain_3_original_plane]: /experiments/images/JNet_383_pretrain_3_original_plane.png
[JNet_383_pretrain_3_output_depth]: /experiments/images/JNet_383_pretrain_3_output_depth.png
[JNet_383_pretrain_3_output_plane]: /experiments/images/JNet_383_pretrain_3_output_plane.png
[JNet_383_pretrain_4_label_depth]: /experiments/images/JNet_383_pretrain_4_label_depth.png
[JNet_383_pretrain_4_label_plane]: /experiments/images/JNet_383_pretrain_4_label_plane.png
[JNet_383_pretrain_4_original_depth]: /experiments/images/JNet_383_pretrain_4_original_depth.png
[JNet_383_pretrain_4_original_plane]: /experiments/images/JNet_383_pretrain_4_original_plane.png
[JNet_383_pretrain_4_output_depth]: /experiments/images/JNet_383_pretrain_4_output_depth.png
[JNet_383_pretrain_4_output_plane]: /experiments/images/JNet_383_pretrain_4_output_plane.png
[JNet_383_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi000_original_depth.png
[JNet_383_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi000_output_depth.png
[JNet_383_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi000_reconst_depth.png
[JNet_383_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi001_original_depth.png
[JNet_383_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi001_output_depth.png
[JNet_383_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi001_reconst_depth.png
[JNet_383_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi002_original_depth.png
[JNet_383_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi002_output_depth.png
[JNet_383_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi002_reconst_depth.png
[JNet_383_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi003_original_depth.png
[JNet_383_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi003_output_depth.png
[JNet_383_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi003_reconst_depth.png
[JNet_383_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi004_original_depth.png
[JNet_383_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi004_output_depth.png
[JNet_383_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_383_pretrain_beads_001_roi004_reconst_depth.png
[JNet_383_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_383_pretrain_beads_002_roi000_original_depth.png
[JNet_383_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_383_pretrain_beads_002_roi000_output_depth.png
[JNet_383_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_383_pretrain_beads_002_roi000_reconst_depth.png
[JNet_383_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_383_pretrain_beads_002_roi001_original_depth.png
[JNet_383_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_383_pretrain_beads_002_roi001_output_depth.png
[JNet_383_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_383_pretrain_beads_002_roi001_reconst_depth.png
[JNet_383_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_383_pretrain_beads_002_roi002_original_depth.png
[JNet_383_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_383_pretrain_beads_002_roi002_output_depth.png
[JNet_383_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_383_pretrain_beads_002_roi002_reconst_depth.png
[JNet_384_0_label_depth]: /experiments/images/JNet_384_0_label_depth.png
[JNet_384_0_label_plane]: /experiments/images/JNet_384_0_label_plane.png
[JNet_384_0_original_depth]: /experiments/images/JNet_384_0_original_depth.png
[JNet_384_0_original_plane]: /experiments/images/JNet_384_0_original_plane.png
[JNet_384_0_output_depth]: /experiments/images/JNet_384_0_output_depth.png
[JNet_384_0_output_plane]: /experiments/images/JNet_384_0_output_plane.png
[JNet_384_1_label_depth]: /experiments/images/JNet_384_1_label_depth.png
[JNet_384_1_label_plane]: /experiments/images/JNet_384_1_label_plane.png
[JNet_384_1_original_depth]: /experiments/images/JNet_384_1_original_depth.png
[JNet_384_1_original_plane]: /experiments/images/JNet_384_1_original_plane.png
[JNet_384_1_output_depth]: /experiments/images/JNet_384_1_output_depth.png
[JNet_384_1_output_plane]: /experiments/images/JNet_384_1_output_plane.png
[JNet_384_2_label_depth]: /experiments/images/JNet_384_2_label_depth.png
[JNet_384_2_label_plane]: /experiments/images/JNet_384_2_label_plane.png
[JNet_384_2_original_depth]: /experiments/images/JNet_384_2_original_depth.png
[JNet_384_2_original_plane]: /experiments/images/JNet_384_2_original_plane.png
[JNet_384_2_output_depth]: /experiments/images/JNet_384_2_output_depth.png
[JNet_384_2_output_plane]: /experiments/images/JNet_384_2_output_plane.png
[JNet_384_3_label_depth]: /experiments/images/JNet_384_3_label_depth.png
[JNet_384_3_label_plane]: /experiments/images/JNet_384_3_label_plane.png
[JNet_384_3_original_depth]: /experiments/images/JNet_384_3_original_depth.png
[JNet_384_3_original_plane]: /experiments/images/JNet_384_3_original_plane.png
[JNet_384_3_output_depth]: /experiments/images/JNet_384_3_output_depth.png
[JNet_384_3_output_plane]: /experiments/images/JNet_384_3_output_plane.png
[JNet_384_4_label_depth]: /experiments/images/JNet_384_4_label_depth.png
[JNet_384_4_label_plane]: /experiments/images/JNet_384_4_label_plane.png
[JNet_384_4_original_depth]: /experiments/images/JNet_384_4_original_depth.png
[JNet_384_4_original_plane]: /experiments/images/JNet_384_4_original_plane.png
[JNet_384_4_output_depth]: /experiments/images/JNet_384_4_output_depth.png
[JNet_384_4_output_plane]: /experiments/images/JNet_384_4_output_plane.png
[JNet_384_beads_001_roi000_original_depth]: /experiments/images/JNet_384_beads_001_roi000_original_depth.png
[JNet_384_beads_001_roi000_output_depth]: /experiments/images/JNet_384_beads_001_roi000_output_depth.png
[JNet_384_beads_001_roi000_reconst_depth]: /experiments/images/JNet_384_beads_001_roi000_reconst_depth.png
[JNet_384_beads_001_roi001_original_depth]: /experiments/images/JNet_384_beads_001_roi001_original_depth.png
[JNet_384_beads_001_roi001_output_depth]: /experiments/images/JNet_384_beads_001_roi001_output_depth.png
[JNet_384_beads_001_roi001_reconst_depth]: /experiments/images/JNet_384_beads_001_roi001_reconst_depth.png
[JNet_384_beads_001_roi002_original_depth]: /experiments/images/JNet_384_beads_001_roi002_original_depth.png
[JNet_384_beads_001_roi002_output_depth]: /experiments/images/JNet_384_beads_001_roi002_output_depth.png
[JNet_384_beads_001_roi002_reconst_depth]: /experiments/images/JNet_384_beads_001_roi002_reconst_depth.png
[JNet_384_beads_001_roi003_original_depth]: /experiments/images/JNet_384_beads_001_roi003_original_depth.png
[JNet_384_beads_001_roi003_output_depth]: /experiments/images/JNet_384_beads_001_roi003_output_depth.png
[JNet_384_beads_001_roi003_reconst_depth]: /experiments/images/JNet_384_beads_001_roi003_reconst_depth.png
[JNet_384_beads_001_roi004_original_depth]: /experiments/images/JNet_384_beads_001_roi004_original_depth.png
[JNet_384_beads_001_roi004_output_depth]: /experiments/images/JNet_384_beads_001_roi004_output_depth.png
[JNet_384_beads_001_roi004_reconst_depth]: /experiments/images/JNet_384_beads_001_roi004_reconst_depth.png
[JNet_384_beads_002_roi000_original_depth]: /experiments/images/JNet_384_beads_002_roi000_original_depth.png
[JNet_384_beads_002_roi000_output_depth]: /experiments/images/JNet_384_beads_002_roi000_output_depth.png
[JNet_384_beads_002_roi000_reconst_depth]: /experiments/images/JNet_384_beads_002_roi000_reconst_depth.png
[JNet_384_beads_002_roi001_original_depth]: /experiments/images/JNet_384_beads_002_roi001_original_depth.png
[JNet_384_beads_002_roi001_output_depth]: /experiments/images/JNet_384_beads_002_roi001_output_depth.png
[JNet_384_beads_002_roi001_reconst_depth]: /experiments/images/JNet_384_beads_002_roi001_reconst_depth.png
[JNet_384_beads_002_roi002_original_depth]: /experiments/images/JNet_384_beads_002_roi002_original_depth.png
[JNet_384_beads_002_roi002_output_depth]: /experiments/images/JNet_384_beads_002_roi002_output_depth.png
[JNet_384_beads_002_roi002_reconst_depth]: /experiments/images/JNet_384_beads_002_roi002_reconst_depth.png
[JNet_384_psf_post]: /experiments/images/JNet_384_psf_post.png
[JNet_384_psf_pre]: /experiments/images/JNet_384_psf_pre.png
[finetuned]: /experiments/tmp/JNet_384_train.png
[pretrained_model]: /experiments/tmp/JNet_383_pretrain_train.png
