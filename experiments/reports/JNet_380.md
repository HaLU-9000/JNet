



# JNet_380 Report
  
the parameters to replicate the results of JNet_380. psf loss test. deterministic background simulation training. no  noise.  
pretrained model : JNet_377_pretrain
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
|ploss_weight|0.01|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.024821488186717033, mean BCE: 0.08886165916919708
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_0_original_plane]|![JNet_377_pretrain_0_output_plane]|![JNet_377_pretrain_0_label_plane]|
  
MSE: 0.025643985718488693, BCE: 0.09402754157781601  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_0_original_depth]|![JNet_377_pretrain_0_output_depth]|![JNet_377_pretrain_0_label_depth]|
  
MSE: 0.025643985718488693, BCE: 0.09402754157781601  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_1_original_plane]|![JNet_377_pretrain_1_output_plane]|![JNet_377_pretrain_1_label_plane]|
  
MSE: 0.02961086854338646, BCE: 0.10505993664264679  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_1_original_depth]|![JNet_377_pretrain_1_output_depth]|![JNet_377_pretrain_1_label_depth]|
  
MSE: 0.02961086854338646, BCE: 0.10505993664264679  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_2_original_plane]|![JNet_377_pretrain_2_output_plane]|![JNet_377_pretrain_2_label_plane]|
  
MSE: 0.01876448467373848, BCE: 0.06905651837587357  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_2_original_depth]|![JNet_377_pretrain_2_output_depth]|![JNet_377_pretrain_2_label_depth]|
  
MSE: 0.01876448467373848, BCE: 0.06905651837587357  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_3_original_plane]|![JNet_377_pretrain_3_output_plane]|![JNet_377_pretrain_3_label_plane]|
  
MSE: 0.022062942385673523, BCE: 0.0783681869506836  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_3_original_depth]|![JNet_377_pretrain_3_output_depth]|![JNet_377_pretrain_3_label_depth]|
  
MSE: 0.022062942385673523, BCE: 0.0783681869506836  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_4_original_plane]|![JNet_377_pretrain_4_output_plane]|![JNet_377_pretrain_4_label_plane]|
  
MSE: 0.028025159612298012, BCE: 0.09779614210128784  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_4_original_depth]|![JNet_377_pretrain_4_output_depth]|![JNet_377_pretrain_4_label_depth]|
  
MSE: 0.028025159612298012, BCE: 0.09779614210128784  
  
mean MSE: 0.03198755532503128, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_380_0_original_plane]|![JNet_380_0_output_plane]|![JNet_380_0_label_plane]|
  
MSE: 0.04010298475623131, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_380_0_original_depth]|![JNet_380_0_output_depth]|![JNet_380_0_label_depth]|
  
MSE: 0.04010298475623131, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_380_1_original_plane]|![JNet_380_1_output_plane]|![JNet_380_1_label_plane]|
  
MSE: 0.042269133031368256, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_380_1_original_depth]|![JNet_380_1_output_depth]|![JNet_380_1_label_depth]|
  
MSE: 0.042269133031368256, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_380_2_original_plane]|![JNet_380_2_output_plane]|![JNet_380_2_label_plane]|
  
MSE: 0.018588611856102943, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_380_2_original_depth]|![JNet_380_2_output_depth]|![JNet_380_2_label_depth]|
  
MSE: 0.018588611856102943, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_380_3_original_plane]|![JNet_380_3_output_plane]|![JNet_380_3_label_plane]|
  
MSE: 0.02791888639330864, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_380_3_original_depth]|![JNet_380_3_output_depth]|![JNet_380_3_label_depth]|
  
MSE: 0.02791888639330864, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_380_4_original_plane]|![JNet_380_4_output_plane]|![JNet_380_4_label_plane]|
  
MSE: 0.03105817176401615, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_380_4_original_depth]|![JNet_380_4_output_depth]|![JNet_380_4_label_depth]|
  
MSE: 0.03105817176401615, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_001_roi000_original_depth]|![JNet_377_pretrain_beads_001_roi000_output_depth]|![JNet_377_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 17.495292968750004, MSE: 0.0027611469849944115, quantized loss: 0.0027920580469071865  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_001_roi001_original_depth]|![JNet_377_pretrain_beads_001_roi001_output_depth]|![JNet_377_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 23.712679687500007, MSE: 0.0042731487192213535, quantized loss: 0.0032781800255179405  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_001_roi002_original_depth]|![JNet_377_pretrain_beads_001_roi002_output_depth]|![JNet_377_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 17.780433593750004, MSE: 0.003020011819899082, quantized loss: 0.00265832943841815  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_001_roi003_original_depth]|![JNet_377_pretrain_beads_001_roi003_output_depth]|![JNet_377_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 24.510615234375006, MSE: 0.0048048775643110275, quantized loss: 0.003507738932967186  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_001_roi004_original_depth]|![JNet_377_pretrain_beads_001_roi004_output_depth]|![JNet_377_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 18.611765625000004, MSE: 0.003609846578910947, quantized loss: 0.0028067613020539284  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_002_roi000_original_depth]|![JNet_377_pretrain_beads_002_roi000_output_depth]|![JNet_377_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 21.192730468750003, MSE: 0.00417099054902792, quantized loss: 0.003338450798764825  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_002_roi001_original_depth]|![JNet_377_pretrain_beads_002_roi001_output_depth]|![JNet_377_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 19.580166015625004, MSE: 0.003441053442656994, quantized loss: 0.003032325068488717  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_002_roi002_original_depth]|![JNet_377_pretrain_beads_002_roi002_output_depth]|![JNet_377_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 20.082429687500003, MSE: 0.0037366372998803854, quantized loss: 0.003152257762849331  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_380_beads_001_roi000_original_depth]|![JNet_380_beads_001_roi000_output_depth]|![JNet_380_beads_001_roi000_reconst_depth]|
  
volume: 10.254125000000002, MSE: 0.00023040920495986938, quantized loss: 1.4219268450688105e-05  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_380_beads_001_roi001_original_depth]|![JNet_380_beads_001_roi001_output_depth]|![JNet_380_beads_001_roi001_reconst_depth]|
  
volume: 15.901625000000005, MSE: 0.0006929025403223932, quantized loss: 2.2391499442164786e-05  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_380_beads_001_roi002_original_depth]|![JNet_380_beads_001_roi002_output_depth]|![JNet_380_beads_001_roi002_reconst_depth]|
  
volume: 10.261625000000002, MSE: 0.00018775802163872868, quantized loss: 1.518726276117377e-05  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_380_beads_001_roi003_original_depth]|![JNet_380_beads_001_roi003_output_depth]|![JNet_380_beads_001_roi003_reconst_depth]|
  
volume: 16.770250000000004, MSE: 0.0004155731585342437, quantized loss: 2.2065072698751464e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_380_beads_001_roi004_original_depth]|![JNet_380_beads_001_roi004_output_depth]|![JNet_380_beads_001_roi004_reconst_depth]|
  
volume: 10.848125000000003, MSE: 0.0001482333755120635, quantized loss: 1.3916218449594453e-05  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_380_beads_002_roi000_original_depth]|![JNet_380_beads_002_roi000_output_depth]|![JNet_380_beads_002_roi000_reconst_depth]|
  
volume: 11.454125000000003, MSE: 0.00014404299145098776, quantized loss: 1.4004032891534735e-05  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_380_beads_002_roi001_original_depth]|![JNet_380_beads_002_roi001_output_depth]|![JNet_380_beads_002_roi001_reconst_depth]|
  
volume: 10.809000000000003, MSE: 0.00015964331396389753, quantized loss: 1.492295450589154e-05  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_380_beads_002_roi002_original_depth]|![JNet_380_beads_002_roi002_output_depth]|![JNet_380_beads_002_roi002_reconst_depth]|
  
volume: 11.089625000000003, MSE: 0.00014672998804599047, quantized loss: 1.5258433450071607e-05  

|pre|post|
| :---: | :---: |
|![JNet_380_psf_pre]|![JNet_380_psf_post]|
  



[JNet_377_pretrain_0_label_depth]: /experiments/images/JNet_377_pretrain_0_label_depth.png
[JNet_377_pretrain_0_label_plane]: /experiments/images/JNet_377_pretrain_0_label_plane.png
[JNet_377_pretrain_0_original_depth]: /experiments/images/JNet_377_pretrain_0_original_depth.png
[JNet_377_pretrain_0_original_plane]: /experiments/images/JNet_377_pretrain_0_original_plane.png
[JNet_377_pretrain_0_output_depth]: /experiments/images/JNet_377_pretrain_0_output_depth.png
[JNet_377_pretrain_0_output_plane]: /experiments/images/JNet_377_pretrain_0_output_plane.png
[JNet_377_pretrain_1_label_depth]: /experiments/images/JNet_377_pretrain_1_label_depth.png
[JNet_377_pretrain_1_label_plane]: /experiments/images/JNet_377_pretrain_1_label_plane.png
[JNet_377_pretrain_1_original_depth]: /experiments/images/JNet_377_pretrain_1_original_depth.png
[JNet_377_pretrain_1_original_plane]: /experiments/images/JNet_377_pretrain_1_original_plane.png
[JNet_377_pretrain_1_output_depth]: /experiments/images/JNet_377_pretrain_1_output_depth.png
[JNet_377_pretrain_1_output_plane]: /experiments/images/JNet_377_pretrain_1_output_plane.png
[JNet_377_pretrain_2_label_depth]: /experiments/images/JNet_377_pretrain_2_label_depth.png
[JNet_377_pretrain_2_label_plane]: /experiments/images/JNet_377_pretrain_2_label_plane.png
[JNet_377_pretrain_2_original_depth]: /experiments/images/JNet_377_pretrain_2_original_depth.png
[JNet_377_pretrain_2_original_plane]: /experiments/images/JNet_377_pretrain_2_original_plane.png
[JNet_377_pretrain_2_output_depth]: /experiments/images/JNet_377_pretrain_2_output_depth.png
[JNet_377_pretrain_2_output_plane]: /experiments/images/JNet_377_pretrain_2_output_plane.png
[JNet_377_pretrain_3_label_depth]: /experiments/images/JNet_377_pretrain_3_label_depth.png
[JNet_377_pretrain_3_label_plane]: /experiments/images/JNet_377_pretrain_3_label_plane.png
[JNet_377_pretrain_3_original_depth]: /experiments/images/JNet_377_pretrain_3_original_depth.png
[JNet_377_pretrain_3_original_plane]: /experiments/images/JNet_377_pretrain_3_original_plane.png
[JNet_377_pretrain_3_output_depth]: /experiments/images/JNet_377_pretrain_3_output_depth.png
[JNet_377_pretrain_3_output_plane]: /experiments/images/JNet_377_pretrain_3_output_plane.png
[JNet_377_pretrain_4_label_depth]: /experiments/images/JNet_377_pretrain_4_label_depth.png
[JNet_377_pretrain_4_label_plane]: /experiments/images/JNet_377_pretrain_4_label_plane.png
[JNet_377_pretrain_4_original_depth]: /experiments/images/JNet_377_pretrain_4_original_depth.png
[JNet_377_pretrain_4_original_plane]: /experiments/images/JNet_377_pretrain_4_original_plane.png
[JNet_377_pretrain_4_output_depth]: /experiments/images/JNet_377_pretrain_4_output_depth.png
[JNet_377_pretrain_4_output_plane]: /experiments/images/JNet_377_pretrain_4_output_plane.png
[JNet_377_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi000_original_depth.png
[JNet_377_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi000_output_depth.png
[JNet_377_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi000_reconst_depth.png
[JNet_377_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi001_original_depth.png
[JNet_377_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi001_output_depth.png
[JNet_377_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi001_reconst_depth.png
[JNet_377_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi002_original_depth.png
[JNet_377_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi002_output_depth.png
[JNet_377_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi002_reconst_depth.png
[JNet_377_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi003_original_depth.png
[JNet_377_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi003_output_depth.png
[JNet_377_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi003_reconst_depth.png
[JNet_377_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi004_original_depth.png
[JNet_377_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi004_output_depth.png
[JNet_377_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_377_pretrain_beads_001_roi004_reconst_depth.png
[JNet_377_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_377_pretrain_beads_002_roi000_original_depth.png
[JNet_377_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_377_pretrain_beads_002_roi000_output_depth.png
[JNet_377_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_377_pretrain_beads_002_roi000_reconst_depth.png
[JNet_377_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_377_pretrain_beads_002_roi001_original_depth.png
[JNet_377_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_377_pretrain_beads_002_roi001_output_depth.png
[JNet_377_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_377_pretrain_beads_002_roi001_reconst_depth.png
[JNet_377_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_377_pretrain_beads_002_roi002_original_depth.png
[JNet_377_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_377_pretrain_beads_002_roi002_output_depth.png
[JNet_377_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_377_pretrain_beads_002_roi002_reconst_depth.png
[JNet_380_0_label_depth]: /experiments/images/JNet_380_0_label_depth.png
[JNet_380_0_label_plane]: /experiments/images/JNet_380_0_label_plane.png
[JNet_380_0_original_depth]: /experiments/images/JNet_380_0_original_depth.png
[JNet_380_0_original_plane]: /experiments/images/JNet_380_0_original_plane.png
[JNet_380_0_output_depth]: /experiments/images/JNet_380_0_output_depth.png
[JNet_380_0_output_plane]: /experiments/images/JNet_380_0_output_plane.png
[JNet_380_1_label_depth]: /experiments/images/JNet_380_1_label_depth.png
[JNet_380_1_label_plane]: /experiments/images/JNet_380_1_label_plane.png
[JNet_380_1_original_depth]: /experiments/images/JNet_380_1_original_depth.png
[JNet_380_1_original_plane]: /experiments/images/JNet_380_1_original_plane.png
[JNet_380_1_output_depth]: /experiments/images/JNet_380_1_output_depth.png
[JNet_380_1_output_plane]: /experiments/images/JNet_380_1_output_plane.png
[JNet_380_2_label_depth]: /experiments/images/JNet_380_2_label_depth.png
[JNet_380_2_label_plane]: /experiments/images/JNet_380_2_label_plane.png
[JNet_380_2_original_depth]: /experiments/images/JNet_380_2_original_depth.png
[JNet_380_2_original_plane]: /experiments/images/JNet_380_2_original_plane.png
[JNet_380_2_output_depth]: /experiments/images/JNet_380_2_output_depth.png
[JNet_380_2_output_plane]: /experiments/images/JNet_380_2_output_plane.png
[JNet_380_3_label_depth]: /experiments/images/JNet_380_3_label_depth.png
[JNet_380_3_label_plane]: /experiments/images/JNet_380_3_label_plane.png
[JNet_380_3_original_depth]: /experiments/images/JNet_380_3_original_depth.png
[JNet_380_3_original_plane]: /experiments/images/JNet_380_3_original_plane.png
[JNet_380_3_output_depth]: /experiments/images/JNet_380_3_output_depth.png
[JNet_380_3_output_plane]: /experiments/images/JNet_380_3_output_plane.png
[JNet_380_4_label_depth]: /experiments/images/JNet_380_4_label_depth.png
[JNet_380_4_label_plane]: /experiments/images/JNet_380_4_label_plane.png
[JNet_380_4_original_depth]: /experiments/images/JNet_380_4_original_depth.png
[JNet_380_4_original_plane]: /experiments/images/JNet_380_4_original_plane.png
[JNet_380_4_output_depth]: /experiments/images/JNet_380_4_output_depth.png
[JNet_380_4_output_plane]: /experiments/images/JNet_380_4_output_plane.png
[JNet_380_beads_001_roi000_original_depth]: /experiments/images/JNet_380_beads_001_roi000_original_depth.png
[JNet_380_beads_001_roi000_output_depth]: /experiments/images/JNet_380_beads_001_roi000_output_depth.png
[JNet_380_beads_001_roi000_reconst_depth]: /experiments/images/JNet_380_beads_001_roi000_reconst_depth.png
[JNet_380_beads_001_roi001_original_depth]: /experiments/images/JNet_380_beads_001_roi001_original_depth.png
[JNet_380_beads_001_roi001_output_depth]: /experiments/images/JNet_380_beads_001_roi001_output_depth.png
[JNet_380_beads_001_roi001_reconst_depth]: /experiments/images/JNet_380_beads_001_roi001_reconst_depth.png
[JNet_380_beads_001_roi002_original_depth]: /experiments/images/JNet_380_beads_001_roi002_original_depth.png
[JNet_380_beads_001_roi002_output_depth]: /experiments/images/JNet_380_beads_001_roi002_output_depth.png
[JNet_380_beads_001_roi002_reconst_depth]: /experiments/images/JNet_380_beads_001_roi002_reconst_depth.png
[JNet_380_beads_001_roi003_original_depth]: /experiments/images/JNet_380_beads_001_roi003_original_depth.png
[JNet_380_beads_001_roi003_output_depth]: /experiments/images/JNet_380_beads_001_roi003_output_depth.png
[JNet_380_beads_001_roi003_reconst_depth]: /experiments/images/JNet_380_beads_001_roi003_reconst_depth.png
[JNet_380_beads_001_roi004_original_depth]: /experiments/images/JNet_380_beads_001_roi004_original_depth.png
[JNet_380_beads_001_roi004_output_depth]: /experiments/images/JNet_380_beads_001_roi004_output_depth.png
[JNet_380_beads_001_roi004_reconst_depth]: /experiments/images/JNet_380_beads_001_roi004_reconst_depth.png
[JNet_380_beads_002_roi000_original_depth]: /experiments/images/JNet_380_beads_002_roi000_original_depth.png
[JNet_380_beads_002_roi000_output_depth]: /experiments/images/JNet_380_beads_002_roi000_output_depth.png
[JNet_380_beads_002_roi000_reconst_depth]: /experiments/images/JNet_380_beads_002_roi000_reconst_depth.png
[JNet_380_beads_002_roi001_original_depth]: /experiments/images/JNet_380_beads_002_roi001_original_depth.png
[JNet_380_beads_002_roi001_output_depth]: /experiments/images/JNet_380_beads_002_roi001_output_depth.png
[JNet_380_beads_002_roi001_reconst_depth]: /experiments/images/JNet_380_beads_002_roi001_reconst_depth.png
[JNet_380_beads_002_roi002_original_depth]: /experiments/images/JNet_380_beads_002_roi002_original_depth.png
[JNet_380_beads_002_roi002_output_depth]: /experiments/images/JNet_380_beads_002_roi002_output_depth.png
[JNet_380_beads_002_roi002_reconst_depth]: /experiments/images/JNet_380_beads_002_roi002_reconst_depth.png
[JNet_380_psf_post]: /experiments/images/JNet_380_psf_post.png
[JNet_380_psf_pre]: /experiments/images/JNet_380_psf_pre.png
[finetuned]: /experiments/tmp/JNet_380_train.png
[pretrained_model]: /experiments/tmp/JNet_377_pretrain_train.png
