



# JNet_358 Report
  
the parameters to replicate the results of JNet_358. mask added.  
pretrained model : JNet_357_pretrain
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
|size_x|101||
|size_y|101||
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
|sig_eps|0.02||
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

## Results
  
mean MSE: 0.02258053794503212, mean BCE: 0.08191520720720291
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_357_pretrain_0_original_plane]|![JNet_357_pretrain_0_output_plane]|![JNet_357_pretrain_0_label_plane]|
  
MSE: 0.025935828685760498, BCE: 0.0942736566066742  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_357_pretrain_0_original_depth]|![JNet_357_pretrain_0_output_depth]|![JNet_357_pretrain_0_label_depth]|
  
MSE: 0.025935828685760498, BCE: 0.0942736566066742  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_357_pretrain_1_original_plane]|![JNet_357_pretrain_1_output_plane]|![JNet_357_pretrain_1_label_plane]|
  
MSE: 0.025120336562395096, BCE: 0.08905192464590073  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_357_pretrain_1_original_depth]|![JNet_357_pretrain_1_output_depth]|![JNet_357_pretrain_1_label_depth]|
  
MSE: 0.025120336562395096, BCE: 0.08905192464590073  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_357_pretrain_2_original_plane]|![JNet_357_pretrain_2_output_plane]|![JNet_357_pretrain_2_label_plane]|
  
MSE: 0.023288745433092117, BCE: 0.08606850355863571  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_357_pretrain_2_original_depth]|![JNet_357_pretrain_2_output_depth]|![JNet_357_pretrain_2_label_depth]|
  
MSE: 0.023288745433092117, BCE: 0.08606850355863571  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_357_pretrain_3_original_plane]|![JNet_357_pretrain_3_output_plane]|![JNet_357_pretrain_3_label_plane]|
  
MSE: 0.01606200821697712, BCE: 0.05830899253487587  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_357_pretrain_3_original_depth]|![JNet_357_pretrain_3_output_depth]|![JNet_357_pretrain_3_label_depth]|
  
MSE: 0.01606200821697712, BCE: 0.05830899253487587  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_357_pretrain_4_original_plane]|![JNet_357_pretrain_4_output_plane]|![JNet_357_pretrain_4_label_plane]|
  
MSE: 0.022495770826935768, BCE: 0.08187294751405716  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_357_pretrain_4_original_depth]|![JNet_357_pretrain_4_output_depth]|![JNet_357_pretrain_4_label_depth]|
  
MSE: 0.022495770826935768, BCE: 0.08187294751405716  
  
mean MSE: 0.20483574271202087, mean BCE: 0.6025126576423645
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_358_0_original_plane]|![JNet_358_0_output_plane]|![JNet_358_0_label_plane]|
  
MSE: 0.20508632063865662, BCE: 0.60301673412323  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_358_0_original_depth]|![JNet_358_0_output_depth]|![JNet_358_0_label_depth]|
  
MSE: 0.20508632063865662, BCE: 0.60301673412323  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_358_1_original_plane]|![JNet_358_1_output_plane]|![JNet_358_1_label_plane]|
  
MSE: 0.2056417316198349, BCE: 0.6041311621665955  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_358_1_original_depth]|![JNet_358_1_output_depth]|![JNet_358_1_label_depth]|
  
MSE: 0.2056417316198349, BCE: 0.6041311621665955  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_358_2_original_plane]|![JNet_358_2_output_plane]|![JNet_358_2_label_plane]|
  
MSE: 0.20521044731140137, BCE: 0.603265643119812  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_358_2_original_depth]|![JNet_358_2_output_depth]|![JNet_358_2_label_depth]|
  
MSE: 0.20521044731140137, BCE: 0.603265643119812  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_358_3_original_plane]|![JNet_358_3_output_plane]|![JNet_358_3_label_plane]|
  
MSE: 0.20434215664863586, BCE: 0.6015207767486572  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_358_3_original_depth]|![JNet_358_3_output_depth]|![JNet_358_3_label_depth]|
  
MSE: 0.20434215664863586, BCE: 0.6015207767486572  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_358_4_original_plane]|![JNet_358_4_output_plane]|![JNet_358_4_label_plane]|
  
MSE: 0.20389802753925323, BCE: 0.6006289720535278  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_358_4_original_depth]|![JNet_358_4_output_depth]|![JNet_358_4_label_depth]|
  
MSE: 0.20389802753925323, BCE: 0.6006289720535278  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_357_pretrain_beads_001_roi000_original_depth]|![JNet_357_pretrain_beads_001_roi000_output_depth]|![JNet_357_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 17.778000000000006, MSE: 0.002108827931806445, quantized loss: 0.00213285512290895  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_357_pretrain_beads_001_roi001_original_depth]|![JNet_357_pretrain_beads_001_roi001_output_depth]|![JNet_357_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 24.832875000000005, MSE: 0.003597170114517212, quantized loss: 0.0025859272573143244  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_357_pretrain_beads_001_roi002_original_depth]|![JNet_357_pretrain_beads_001_roi002_output_depth]|![JNet_357_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 16.268250000000005, MSE: 0.0021668155677616596, quantized loss: 0.0017707888036966324  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_357_pretrain_beads_001_roi003_original_depth]|![JNet_357_pretrain_beads_001_roi003_output_depth]|![JNet_357_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 26.614125000000005, MSE: 0.0037870658561587334, quantized loss: 0.0025602709501981735  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_357_pretrain_beads_001_roi004_original_depth]|![JNet_357_pretrain_beads_001_roi004_output_depth]|![JNet_357_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 17.780500000000004, MSE: 0.002677113516256213, quantized loss: 0.0018609706312417984  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_357_pretrain_beads_002_roi000_original_depth]|![JNet_357_pretrain_beads_002_roi000_output_depth]|![JNet_357_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 18.921250000000004, MSE: 0.0030060652643442154, quantized loss: 0.0019118397030979395  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_357_pretrain_beads_002_roi001_original_depth]|![JNet_357_pretrain_beads_002_roi001_output_depth]|![JNet_357_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 17.314250000000005, MSE: 0.002354885684326291, quantized loss: 0.0017178760608658195  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_357_pretrain_beads_002_roi002_original_depth]|![JNet_357_pretrain_beads_002_roi002_output_depth]|![JNet_357_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 18.096125000000004, MSE: 0.0026699297595769167, quantized loss: 0.001968628726899624  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_358_beads_001_roi000_original_depth]|![JNet_358_beads_001_roi000_output_depth]|![JNet_358_beads_001_roi000_reconst_depth]|
  
volume: 0.0, MSE: 0.009335086680948734, quantized loss: 0.2402193695306778  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_358_beads_001_roi001_original_depth]|![JNet_358_beads_001_roi001_output_depth]|![JNet_358_beads_001_roi001_reconst_depth]|
  
volume: 0.0, MSE: 0.014853984117507935, quantized loss: 0.23781655728816986  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_358_beads_001_roi002_original_depth]|![JNet_358_beads_001_roi002_output_depth]|![JNet_358_beads_001_roi002_reconst_depth]|
  
volume: 0.0, MSE: 0.009056191891431808, quantized loss: 0.24071945250034332  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_358_beads_001_roi003_original_depth]|![JNet_358_beads_001_roi003_output_depth]|![JNet_358_beads_001_roi003_reconst_depth]|
  
volume: 0.0, MSE: 0.015940722078084946, quantized loss: 0.2378564029932022  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_358_beads_001_roi004_original_depth]|![JNet_358_beads_001_roi004_output_depth]|![JNet_358_beads_001_roi004_reconst_depth]|
  
volume: 0.0, MSE: 0.010938753373920918, quantized loss: 0.24024613201618195  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_358_beads_002_roi000_original_depth]|![JNet_358_beads_002_roi000_output_depth]|![JNet_358_beads_002_roi000_reconst_depth]|
  
volume: 0.0, MSE: 0.012265910394489765, quantized loss: 0.23995722830295563  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_358_beads_002_roi001_original_depth]|![JNet_358_beads_002_roi001_output_depth]|![JNet_358_beads_002_roi001_reconst_depth]|
  
volume: 0.0, MSE: 0.010307948105037212, quantized loss: 0.24040140211582184  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_358_beads_002_roi002_original_depth]|![JNet_358_beads_002_roi002_output_depth]|![JNet_358_beads_002_roi002_reconst_depth]|
  
volume: 0.0, MSE: 0.011078033596277237, quantized loss: 0.24025292694568634  

|pre|post|
| :---: | :---: |
|![JNet_358_psf_pre]|![JNet_358_psf_post]|
  



[JNet_357_pretrain_0_label_depth]: /experiments/images/JNet_357_pretrain_0_label_depth.png
[JNet_357_pretrain_0_label_plane]: /experiments/images/JNet_357_pretrain_0_label_plane.png
[JNet_357_pretrain_0_original_depth]: /experiments/images/JNet_357_pretrain_0_original_depth.png
[JNet_357_pretrain_0_original_plane]: /experiments/images/JNet_357_pretrain_0_original_plane.png
[JNet_357_pretrain_0_output_depth]: /experiments/images/JNet_357_pretrain_0_output_depth.png
[JNet_357_pretrain_0_output_plane]: /experiments/images/JNet_357_pretrain_0_output_plane.png
[JNet_357_pretrain_1_label_depth]: /experiments/images/JNet_357_pretrain_1_label_depth.png
[JNet_357_pretrain_1_label_plane]: /experiments/images/JNet_357_pretrain_1_label_plane.png
[JNet_357_pretrain_1_original_depth]: /experiments/images/JNet_357_pretrain_1_original_depth.png
[JNet_357_pretrain_1_original_plane]: /experiments/images/JNet_357_pretrain_1_original_plane.png
[JNet_357_pretrain_1_output_depth]: /experiments/images/JNet_357_pretrain_1_output_depth.png
[JNet_357_pretrain_1_output_plane]: /experiments/images/JNet_357_pretrain_1_output_plane.png
[JNet_357_pretrain_2_label_depth]: /experiments/images/JNet_357_pretrain_2_label_depth.png
[JNet_357_pretrain_2_label_plane]: /experiments/images/JNet_357_pretrain_2_label_plane.png
[JNet_357_pretrain_2_original_depth]: /experiments/images/JNet_357_pretrain_2_original_depth.png
[JNet_357_pretrain_2_original_plane]: /experiments/images/JNet_357_pretrain_2_original_plane.png
[JNet_357_pretrain_2_output_depth]: /experiments/images/JNet_357_pretrain_2_output_depth.png
[JNet_357_pretrain_2_output_plane]: /experiments/images/JNet_357_pretrain_2_output_plane.png
[JNet_357_pretrain_3_label_depth]: /experiments/images/JNet_357_pretrain_3_label_depth.png
[JNet_357_pretrain_3_label_plane]: /experiments/images/JNet_357_pretrain_3_label_plane.png
[JNet_357_pretrain_3_original_depth]: /experiments/images/JNet_357_pretrain_3_original_depth.png
[JNet_357_pretrain_3_original_plane]: /experiments/images/JNet_357_pretrain_3_original_plane.png
[JNet_357_pretrain_3_output_depth]: /experiments/images/JNet_357_pretrain_3_output_depth.png
[JNet_357_pretrain_3_output_plane]: /experiments/images/JNet_357_pretrain_3_output_plane.png
[JNet_357_pretrain_4_label_depth]: /experiments/images/JNet_357_pretrain_4_label_depth.png
[JNet_357_pretrain_4_label_plane]: /experiments/images/JNet_357_pretrain_4_label_plane.png
[JNet_357_pretrain_4_original_depth]: /experiments/images/JNet_357_pretrain_4_original_depth.png
[JNet_357_pretrain_4_original_plane]: /experiments/images/JNet_357_pretrain_4_original_plane.png
[JNet_357_pretrain_4_output_depth]: /experiments/images/JNet_357_pretrain_4_output_depth.png
[JNet_357_pretrain_4_output_plane]: /experiments/images/JNet_357_pretrain_4_output_plane.png
[JNet_357_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi000_original_depth.png
[JNet_357_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi000_output_depth.png
[JNet_357_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi000_reconst_depth.png
[JNet_357_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi001_original_depth.png
[JNet_357_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi001_output_depth.png
[JNet_357_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi001_reconst_depth.png
[JNet_357_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi002_original_depth.png
[JNet_357_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi002_output_depth.png
[JNet_357_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi002_reconst_depth.png
[JNet_357_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi003_original_depth.png
[JNet_357_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi003_output_depth.png
[JNet_357_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi003_reconst_depth.png
[JNet_357_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi004_original_depth.png
[JNet_357_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi004_output_depth.png
[JNet_357_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_357_pretrain_beads_001_roi004_reconst_depth.png
[JNet_357_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_357_pretrain_beads_002_roi000_original_depth.png
[JNet_357_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_357_pretrain_beads_002_roi000_output_depth.png
[JNet_357_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_357_pretrain_beads_002_roi000_reconst_depth.png
[JNet_357_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_357_pretrain_beads_002_roi001_original_depth.png
[JNet_357_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_357_pretrain_beads_002_roi001_output_depth.png
[JNet_357_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_357_pretrain_beads_002_roi001_reconst_depth.png
[JNet_357_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_357_pretrain_beads_002_roi002_original_depth.png
[JNet_357_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_357_pretrain_beads_002_roi002_output_depth.png
[JNet_357_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_357_pretrain_beads_002_roi002_reconst_depth.png
[JNet_358_0_label_depth]: /experiments/images/JNet_358_0_label_depth.png
[JNet_358_0_label_plane]: /experiments/images/JNet_358_0_label_plane.png
[JNet_358_0_original_depth]: /experiments/images/JNet_358_0_original_depth.png
[JNet_358_0_original_plane]: /experiments/images/JNet_358_0_original_plane.png
[JNet_358_0_output_depth]: /experiments/images/JNet_358_0_output_depth.png
[JNet_358_0_output_plane]: /experiments/images/JNet_358_0_output_plane.png
[JNet_358_1_label_depth]: /experiments/images/JNet_358_1_label_depth.png
[JNet_358_1_label_plane]: /experiments/images/JNet_358_1_label_plane.png
[JNet_358_1_original_depth]: /experiments/images/JNet_358_1_original_depth.png
[JNet_358_1_original_plane]: /experiments/images/JNet_358_1_original_plane.png
[JNet_358_1_output_depth]: /experiments/images/JNet_358_1_output_depth.png
[JNet_358_1_output_plane]: /experiments/images/JNet_358_1_output_plane.png
[JNet_358_2_label_depth]: /experiments/images/JNet_358_2_label_depth.png
[JNet_358_2_label_plane]: /experiments/images/JNet_358_2_label_plane.png
[JNet_358_2_original_depth]: /experiments/images/JNet_358_2_original_depth.png
[JNet_358_2_original_plane]: /experiments/images/JNet_358_2_original_plane.png
[JNet_358_2_output_depth]: /experiments/images/JNet_358_2_output_depth.png
[JNet_358_2_output_plane]: /experiments/images/JNet_358_2_output_plane.png
[JNet_358_3_label_depth]: /experiments/images/JNet_358_3_label_depth.png
[JNet_358_3_label_plane]: /experiments/images/JNet_358_3_label_plane.png
[JNet_358_3_original_depth]: /experiments/images/JNet_358_3_original_depth.png
[JNet_358_3_original_plane]: /experiments/images/JNet_358_3_original_plane.png
[JNet_358_3_output_depth]: /experiments/images/JNet_358_3_output_depth.png
[JNet_358_3_output_plane]: /experiments/images/JNet_358_3_output_plane.png
[JNet_358_4_label_depth]: /experiments/images/JNet_358_4_label_depth.png
[JNet_358_4_label_plane]: /experiments/images/JNet_358_4_label_plane.png
[JNet_358_4_original_depth]: /experiments/images/JNet_358_4_original_depth.png
[JNet_358_4_original_plane]: /experiments/images/JNet_358_4_original_plane.png
[JNet_358_4_output_depth]: /experiments/images/JNet_358_4_output_depth.png
[JNet_358_4_output_plane]: /experiments/images/JNet_358_4_output_plane.png
[JNet_358_beads_001_roi000_original_depth]: /experiments/images/JNet_358_beads_001_roi000_original_depth.png
[JNet_358_beads_001_roi000_output_depth]: /experiments/images/JNet_358_beads_001_roi000_output_depth.png
[JNet_358_beads_001_roi000_reconst_depth]: /experiments/images/JNet_358_beads_001_roi000_reconst_depth.png
[JNet_358_beads_001_roi001_original_depth]: /experiments/images/JNet_358_beads_001_roi001_original_depth.png
[JNet_358_beads_001_roi001_output_depth]: /experiments/images/JNet_358_beads_001_roi001_output_depth.png
[JNet_358_beads_001_roi001_reconst_depth]: /experiments/images/JNet_358_beads_001_roi001_reconst_depth.png
[JNet_358_beads_001_roi002_original_depth]: /experiments/images/JNet_358_beads_001_roi002_original_depth.png
[JNet_358_beads_001_roi002_output_depth]: /experiments/images/JNet_358_beads_001_roi002_output_depth.png
[JNet_358_beads_001_roi002_reconst_depth]: /experiments/images/JNet_358_beads_001_roi002_reconst_depth.png
[JNet_358_beads_001_roi003_original_depth]: /experiments/images/JNet_358_beads_001_roi003_original_depth.png
[JNet_358_beads_001_roi003_output_depth]: /experiments/images/JNet_358_beads_001_roi003_output_depth.png
[JNet_358_beads_001_roi003_reconst_depth]: /experiments/images/JNet_358_beads_001_roi003_reconst_depth.png
[JNet_358_beads_001_roi004_original_depth]: /experiments/images/JNet_358_beads_001_roi004_original_depth.png
[JNet_358_beads_001_roi004_output_depth]: /experiments/images/JNet_358_beads_001_roi004_output_depth.png
[JNet_358_beads_001_roi004_reconst_depth]: /experiments/images/JNet_358_beads_001_roi004_reconst_depth.png
[JNet_358_beads_002_roi000_original_depth]: /experiments/images/JNet_358_beads_002_roi000_original_depth.png
[JNet_358_beads_002_roi000_output_depth]: /experiments/images/JNet_358_beads_002_roi000_output_depth.png
[JNet_358_beads_002_roi000_reconst_depth]: /experiments/images/JNet_358_beads_002_roi000_reconst_depth.png
[JNet_358_beads_002_roi001_original_depth]: /experiments/images/JNet_358_beads_002_roi001_original_depth.png
[JNet_358_beads_002_roi001_output_depth]: /experiments/images/JNet_358_beads_002_roi001_output_depth.png
[JNet_358_beads_002_roi001_reconst_depth]: /experiments/images/JNet_358_beads_002_roi001_reconst_depth.png
[JNet_358_beads_002_roi002_original_depth]: /experiments/images/JNet_358_beads_002_roi002_original_depth.png
[JNet_358_beads_002_roi002_output_depth]: /experiments/images/JNet_358_beads_002_roi002_output_depth.png
[JNet_358_beads_002_roi002_reconst_depth]: /experiments/images/JNet_358_beads_002_roi002_reconst_depth.png
[JNet_358_psf_post]: /experiments/images/JNet_358_psf_post.png
[JNet_358_psf_pre]: /experiments/images/JNet_358_psf_pre.png
