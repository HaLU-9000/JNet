



# JNet_366 Report
  
the parameters to replicate the results of JNet_366. timm added. q loss 0.01  
pretrained model : JNet_365_pretrain
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
|apply_vq|True||
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
|qloss_weight|0.01|

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
  
mean MSE: 0.023049941286444664, mean BCE: 0.0875668078660965
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_365_pretrain_0_original_plane]|![JNet_365_pretrain_0_output_plane]|![JNet_365_pretrain_0_label_plane]|
  
MSE: 0.024600321426987648, BCE: 0.09199244529008865  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_365_pretrain_0_original_depth]|![JNet_365_pretrain_0_output_depth]|![JNet_365_pretrain_0_label_depth]|
  
MSE: 0.024600321426987648, BCE: 0.09199244529008865  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_365_pretrain_1_original_plane]|![JNet_365_pretrain_1_output_plane]|![JNet_365_pretrain_1_label_plane]|
  
MSE: 0.01735709421336651, BCE: 0.06825337558984756  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_365_pretrain_1_original_depth]|![JNet_365_pretrain_1_output_depth]|![JNet_365_pretrain_1_label_depth]|
  
MSE: 0.01735709421336651, BCE: 0.06825337558984756  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_365_pretrain_2_original_plane]|![JNet_365_pretrain_2_output_plane]|![JNet_365_pretrain_2_label_plane]|
  
MSE: 0.03020527958869934, BCE: 0.10947684198617935  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_365_pretrain_2_original_depth]|![JNet_365_pretrain_2_output_depth]|![JNet_365_pretrain_2_label_depth]|
  
MSE: 0.03020527958869934, BCE: 0.10947684198617935  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_365_pretrain_3_original_plane]|![JNet_365_pretrain_3_output_plane]|![JNet_365_pretrain_3_label_plane]|
  
MSE: 0.01608756184577942, BCE: 0.0618768148124218  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_365_pretrain_3_original_depth]|![JNet_365_pretrain_3_output_depth]|![JNet_365_pretrain_3_label_depth]|
  
MSE: 0.01608756184577942, BCE: 0.0618768148124218  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_365_pretrain_4_original_plane]|![JNet_365_pretrain_4_output_plane]|![JNet_365_pretrain_4_label_plane]|
  
MSE: 0.02699945494532585, BCE: 0.10623456537723541  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_365_pretrain_4_original_depth]|![JNet_365_pretrain_4_output_depth]|![JNet_365_pretrain_4_label_depth]|
  
MSE: 0.02699945494532585, BCE: 0.10623456537723541  
  
mean MSE: 0.2677704989910126, mean BCE: 0.7287061810493469
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_366_0_original_plane]|![JNet_366_0_output_plane]|![JNet_366_0_label_plane]|
  
MSE: 0.26767319440841675, BCE: 0.7285107374191284  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_366_0_original_depth]|![JNet_366_0_output_depth]|![JNet_366_0_label_depth]|
  
MSE: 0.26767319440841675, BCE: 0.7285107374191284  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_366_1_original_plane]|![JNet_366_1_output_plane]|![JNet_366_1_label_plane]|
  
MSE: 0.2681763768196106, BCE: 0.7295211553573608  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_366_1_original_depth]|![JNet_366_1_output_depth]|![JNet_366_1_label_depth]|
  
MSE: 0.2681763768196106, BCE: 0.7295211553573608  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_366_2_original_plane]|![JNet_366_2_output_plane]|![JNet_366_2_label_plane]|
  
MSE: 0.2674938440322876, BCE: 0.7281510829925537  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_366_2_original_depth]|![JNet_366_2_output_depth]|![JNet_366_2_label_depth]|
  
MSE: 0.2674938440322876, BCE: 0.7281510829925537  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_366_3_original_plane]|![JNet_366_3_output_plane]|![JNet_366_3_label_plane]|
  
MSE: 0.26763200759887695, BCE: 0.7284282445907593  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_366_3_original_depth]|![JNet_366_3_output_depth]|![JNet_366_3_label_depth]|
  
MSE: 0.26763200759887695, BCE: 0.7284282445907593  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_366_4_original_plane]|![JNet_366_4_output_plane]|![JNet_366_4_label_plane]|
  
MSE: 0.26787710189819336, BCE: 0.7289196252822876  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_366_4_original_depth]|![JNet_366_4_output_depth]|![JNet_366_4_label_depth]|
  
MSE: 0.26787710189819336, BCE: 0.7289196252822876  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_365_pretrain_beads_001_roi000_original_depth]|![JNet_365_pretrain_beads_001_roi000_output_depth]|![JNet_365_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 25.178000000000004, MSE: 0.003318280912935734, quantized loss: 0.004952607676386833  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_365_pretrain_beads_001_roi001_original_depth]|![JNet_365_pretrain_beads_001_roi001_output_depth]|![JNet_365_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 36.341375000000006, MSE: 0.004718462470918894, quantized loss: 0.005465044640004635  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_365_pretrain_beads_001_roi002_original_depth]|![JNet_365_pretrain_beads_001_roi002_output_depth]|![JNet_365_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 23.931500000000007, MSE: 0.0032659240532666445, quantized loss: 0.004455867689102888  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_365_pretrain_beads_001_roi003_original_depth]|![JNet_365_pretrain_beads_001_roi003_output_depth]|![JNet_365_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 36.20925000000001, MSE: 0.004854182247072458, quantized loss: 0.006083534564822912  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_365_pretrain_beads_001_roi004_original_depth]|![JNet_365_pretrain_beads_001_roi004_output_depth]|![JNet_365_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 25.765750000000008, MSE: 0.0038392452988773584, quantized loss: 0.00482320599257946  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_365_pretrain_beads_002_roi000_original_depth]|![JNet_365_pretrain_beads_002_roi000_output_depth]|![JNet_365_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 26.990375000000007, MSE: 0.004168333951383829, quantized loss: 0.005097334273159504  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_365_pretrain_beads_002_roi001_original_depth]|![JNet_365_pretrain_beads_002_roi001_output_depth]|![JNet_365_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 25.080875000000006, MSE: 0.0034735125955194235, quantized loss: 0.004647931549698114  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_365_pretrain_beads_002_roi002_original_depth]|![JNet_365_pretrain_beads_002_roi002_output_depth]|![JNet_365_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 25.727500000000006, MSE: 0.0037739828694611788, quantized loss: 0.004778013098984957  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_366_beads_001_roi000_original_depth]|![JNet_366_beads_001_roi000_output_depth]|![JNet_366_beads_001_roi000_reconst_depth]|
  
volume: 489.42087500000014, MSE: 0.5272037386894226, quantized loss: 0.24645543098449707  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_366_beads_001_roi001_original_depth]|![JNet_366_beads_001_roi001_output_depth]|![JNet_366_beads_001_roi001_reconst_depth]|
  
volume: 488.93762500000014, MSE: 0.514744222164154, quantized loss: 0.24573467671871185  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_366_beads_001_roi002_original_depth]|![JNet_366_beads_001_roi002_output_depth]|![JNet_366_beads_001_roi002_reconst_depth]|
  
volume: 489.1155000000001, MSE: 0.5302333235740662, quantized loss: 0.2466312199831009  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_366_beads_001_roi003_original_depth]|![JNet_366_beads_001_roi003_output_depth]|![JNet_366_beads_001_roi003_reconst_depth]|
  
volume: 488.3363750000001, MSE: 0.5117490291595459, quantized loss: 0.24584323167800903  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_366_beads_001_roi004_original_depth]|![JNet_366_beads_001_roi004_output_depth]|![JNet_366_beads_001_roi004_reconst_depth]|
  
volume: 488.8933750000001, MSE: 0.5264146327972412, quantized loss: 0.24648834764957428  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_366_beads_002_roi000_original_depth]|![JNet_366_beads_002_roi000_output_depth]|![JNet_366_beads_002_roi000_reconst_depth]|
  
volume: 488.6916250000001, MSE: 0.5238863229751587, quantized loss: 0.24641619622707367  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_366_beads_002_roi001_original_depth]|![JNet_366_beads_002_roi001_output_depth]|![JNet_366_beads_002_roi001_reconst_depth]|
  
volume: 488.8288750000001, MSE: 0.5274270176887512, quantized loss: 0.246537983417511  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_366_beads_002_roi002_original_depth]|![JNet_366_beads_002_roi002_output_depth]|![JNet_366_beads_002_roi002_reconst_depth]|
  
volume: 488.8325000000001, MSE: 0.5260421633720398, quantized loss: 0.24650642275810242  

|pre|post|
| :---: | :---: |
|![JNet_366_psf_pre]|![JNet_366_psf_post]|
  



[JNet_365_pretrain_0_label_depth]: /experiments/images/JNet_365_pretrain_0_label_depth.png
[JNet_365_pretrain_0_label_plane]: /experiments/images/JNet_365_pretrain_0_label_plane.png
[JNet_365_pretrain_0_original_depth]: /experiments/images/JNet_365_pretrain_0_original_depth.png
[JNet_365_pretrain_0_original_plane]: /experiments/images/JNet_365_pretrain_0_original_plane.png
[JNet_365_pretrain_0_output_depth]: /experiments/images/JNet_365_pretrain_0_output_depth.png
[JNet_365_pretrain_0_output_plane]: /experiments/images/JNet_365_pretrain_0_output_plane.png
[JNet_365_pretrain_1_label_depth]: /experiments/images/JNet_365_pretrain_1_label_depth.png
[JNet_365_pretrain_1_label_plane]: /experiments/images/JNet_365_pretrain_1_label_plane.png
[JNet_365_pretrain_1_original_depth]: /experiments/images/JNet_365_pretrain_1_original_depth.png
[JNet_365_pretrain_1_original_plane]: /experiments/images/JNet_365_pretrain_1_original_plane.png
[JNet_365_pretrain_1_output_depth]: /experiments/images/JNet_365_pretrain_1_output_depth.png
[JNet_365_pretrain_1_output_plane]: /experiments/images/JNet_365_pretrain_1_output_plane.png
[JNet_365_pretrain_2_label_depth]: /experiments/images/JNet_365_pretrain_2_label_depth.png
[JNet_365_pretrain_2_label_plane]: /experiments/images/JNet_365_pretrain_2_label_plane.png
[JNet_365_pretrain_2_original_depth]: /experiments/images/JNet_365_pretrain_2_original_depth.png
[JNet_365_pretrain_2_original_plane]: /experiments/images/JNet_365_pretrain_2_original_plane.png
[JNet_365_pretrain_2_output_depth]: /experiments/images/JNet_365_pretrain_2_output_depth.png
[JNet_365_pretrain_2_output_plane]: /experiments/images/JNet_365_pretrain_2_output_plane.png
[JNet_365_pretrain_3_label_depth]: /experiments/images/JNet_365_pretrain_3_label_depth.png
[JNet_365_pretrain_3_label_plane]: /experiments/images/JNet_365_pretrain_3_label_plane.png
[JNet_365_pretrain_3_original_depth]: /experiments/images/JNet_365_pretrain_3_original_depth.png
[JNet_365_pretrain_3_original_plane]: /experiments/images/JNet_365_pretrain_3_original_plane.png
[JNet_365_pretrain_3_output_depth]: /experiments/images/JNet_365_pretrain_3_output_depth.png
[JNet_365_pretrain_3_output_plane]: /experiments/images/JNet_365_pretrain_3_output_plane.png
[JNet_365_pretrain_4_label_depth]: /experiments/images/JNet_365_pretrain_4_label_depth.png
[JNet_365_pretrain_4_label_plane]: /experiments/images/JNet_365_pretrain_4_label_plane.png
[JNet_365_pretrain_4_original_depth]: /experiments/images/JNet_365_pretrain_4_original_depth.png
[JNet_365_pretrain_4_original_plane]: /experiments/images/JNet_365_pretrain_4_original_plane.png
[JNet_365_pretrain_4_output_depth]: /experiments/images/JNet_365_pretrain_4_output_depth.png
[JNet_365_pretrain_4_output_plane]: /experiments/images/JNet_365_pretrain_4_output_plane.png
[JNet_365_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi000_original_depth.png
[JNet_365_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi000_output_depth.png
[JNet_365_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi000_reconst_depth.png
[JNet_365_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi001_original_depth.png
[JNet_365_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi001_output_depth.png
[JNet_365_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi001_reconst_depth.png
[JNet_365_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi002_original_depth.png
[JNet_365_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi002_output_depth.png
[JNet_365_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi002_reconst_depth.png
[JNet_365_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi003_original_depth.png
[JNet_365_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi003_output_depth.png
[JNet_365_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi003_reconst_depth.png
[JNet_365_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi004_original_depth.png
[JNet_365_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi004_output_depth.png
[JNet_365_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_365_pretrain_beads_001_roi004_reconst_depth.png
[JNet_365_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_365_pretrain_beads_002_roi000_original_depth.png
[JNet_365_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_365_pretrain_beads_002_roi000_output_depth.png
[JNet_365_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_365_pretrain_beads_002_roi000_reconst_depth.png
[JNet_365_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_365_pretrain_beads_002_roi001_original_depth.png
[JNet_365_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_365_pretrain_beads_002_roi001_output_depth.png
[JNet_365_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_365_pretrain_beads_002_roi001_reconst_depth.png
[JNet_365_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_365_pretrain_beads_002_roi002_original_depth.png
[JNet_365_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_365_pretrain_beads_002_roi002_output_depth.png
[JNet_365_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_365_pretrain_beads_002_roi002_reconst_depth.png
[JNet_366_0_label_depth]: /experiments/images/JNet_366_0_label_depth.png
[JNet_366_0_label_plane]: /experiments/images/JNet_366_0_label_plane.png
[JNet_366_0_original_depth]: /experiments/images/JNet_366_0_original_depth.png
[JNet_366_0_original_plane]: /experiments/images/JNet_366_0_original_plane.png
[JNet_366_0_output_depth]: /experiments/images/JNet_366_0_output_depth.png
[JNet_366_0_output_plane]: /experiments/images/JNet_366_0_output_plane.png
[JNet_366_1_label_depth]: /experiments/images/JNet_366_1_label_depth.png
[JNet_366_1_label_plane]: /experiments/images/JNet_366_1_label_plane.png
[JNet_366_1_original_depth]: /experiments/images/JNet_366_1_original_depth.png
[JNet_366_1_original_plane]: /experiments/images/JNet_366_1_original_plane.png
[JNet_366_1_output_depth]: /experiments/images/JNet_366_1_output_depth.png
[JNet_366_1_output_plane]: /experiments/images/JNet_366_1_output_plane.png
[JNet_366_2_label_depth]: /experiments/images/JNet_366_2_label_depth.png
[JNet_366_2_label_plane]: /experiments/images/JNet_366_2_label_plane.png
[JNet_366_2_original_depth]: /experiments/images/JNet_366_2_original_depth.png
[JNet_366_2_original_plane]: /experiments/images/JNet_366_2_original_plane.png
[JNet_366_2_output_depth]: /experiments/images/JNet_366_2_output_depth.png
[JNet_366_2_output_plane]: /experiments/images/JNet_366_2_output_plane.png
[JNet_366_3_label_depth]: /experiments/images/JNet_366_3_label_depth.png
[JNet_366_3_label_plane]: /experiments/images/JNet_366_3_label_plane.png
[JNet_366_3_original_depth]: /experiments/images/JNet_366_3_original_depth.png
[JNet_366_3_original_plane]: /experiments/images/JNet_366_3_original_plane.png
[JNet_366_3_output_depth]: /experiments/images/JNet_366_3_output_depth.png
[JNet_366_3_output_plane]: /experiments/images/JNet_366_3_output_plane.png
[JNet_366_4_label_depth]: /experiments/images/JNet_366_4_label_depth.png
[JNet_366_4_label_plane]: /experiments/images/JNet_366_4_label_plane.png
[JNet_366_4_original_depth]: /experiments/images/JNet_366_4_original_depth.png
[JNet_366_4_original_plane]: /experiments/images/JNet_366_4_original_plane.png
[JNet_366_4_output_depth]: /experiments/images/JNet_366_4_output_depth.png
[JNet_366_4_output_plane]: /experiments/images/JNet_366_4_output_plane.png
[JNet_366_beads_001_roi000_original_depth]: /experiments/images/JNet_366_beads_001_roi000_original_depth.png
[JNet_366_beads_001_roi000_output_depth]: /experiments/images/JNet_366_beads_001_roi000_output_depth.png
[JNet_366_beads_001_roi000_reconst_depth]: /experiments/images/JNet_366_beads_001_roi000_reconst_depth.png
[JNet_366_beads_001_roi001_original_depth]: /experiments/images/JNet_366_beads_001_roi001_original_depth.png
[JNet_366_beads_001_roi001_output_depth]: /experiments/images/JNet_366_beads_001_roi001_output_depth.png
[JNet_366_beads_001_roi001_reconst_depth]: /experiments/images/JNet_366_beads_001_roi001_reconst_depth.png
[JNet_366_beads_001_roi002_original_depth]: /experiments/images/JNet_366_beads_001_roi002_original_depth.png
[JNet_366_beads_001_roi002_output_depth]: /experiments/images/JNet_366_beads_001_roi002_output_depth.png
[JNet_366_beads_001_roi002_reconst_depth]: /experiments/images/JNet_366_beads_001_roi002_reconst_depth.png
[JNet_366_beads_001_roi003_original_depth]: /experiments/images/JNet_366_beads_001_roi003_original_depth.png
[JNet_366_beads_001_roi003_output_depth]: /experiments/images/JNet_366_beads_001_roi003_output_depth.png
[JNet_366_beads_001_roi003_reconst_depth]: /experiments/images/JNet_366_beads_001_roi003_reconst_depth.png
[JNet_366_beads_001_roi004_original_depth]: /experiments/images/JNet_366_beads_001_roi004_original_depth.png
[JNet_366_beads_001_roi004_output_depth]: /experiments/images/JNet_366_beads_001_roi004_output_depth.png
[JNet_366_beads_001_roi004_reconst_depth]: /experiments/images/JNet_366_beads_001_roi004_reconst_depth.png
[JNet_366_beads_002_roi000_original_depth]: /experiments/images/JNet_366_beads_002_roi000_original_depth.png
[JNet_366_beads_002_roi000_output_depth]: /experiments/images/JNet_366_beads_002_roi000_output_depth.png
[JNet_366_beads_002_roi000_reconst_depth]: /experiments/images/JNet_366_beads_002_roi000_reconst_depth.png
[JNet_366_beads_002_roi001_original_depth]: /experiments/images/JNet_366_beads_002_roi001_original_depth.png
[JNet_366_beads_002_roi001_output_depth]: /experiments/images/JNet_366_beads_002_roi001_output_depth.png
[JNet_366_beads_002_roi001_reconst_depth]: /experiments/images/JNet_366_beads_002_roi001_reconst_depth.png
[JNet_366_beads_002_roi002_original_depth]: /experiments/images/JNet_366_beads_002_roi002_original_depth.png
[JNet_366_beads_002_roi002_output_depth]: /experiments/images/JNet_366_beads_002_roi002_output_depth.png
[JNet_366_beads_002_roi002_reconst_depth]: /experiments/images/JNet_366_beads_002_roi002_reconst_depth.png
[JNet_366_psf_post]: /experiments/images/JNet_366_psf_post.png
[JNet_366_psf_pre]: /experiments/images/JNet_366_psf_pre.png
