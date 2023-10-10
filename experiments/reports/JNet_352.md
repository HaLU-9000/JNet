



# JNet_352 Report
  
the parameters to replicate the results of JNet_352. large psf  
pretrained model : JNet_351_pretrain
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
|sig_eps|0.01||
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
  
mean MSE: 0.024336999282240868, mean BCE: 0.08694253861904144
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_351_pretrain_0_original_plane]|![JNet_351_pretrain_0_output_plane]|![JNet_351_pretrain_0_label_plane]|
  
MSE: 0.021738748997449875, BCE: 0.07834608852863312  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_351_pretrain_0_original_depth]|![JNet_351_pretrain_0_output_depth]|![JNet_351_pretrain_0_label_depth]|
  
MSE: 0.021738748997449875, BCE: 0.07834608852863312  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_351_pretrain_1_original_plane]|![JNet_351_pretrain_1_output_plane]|![JNet_351_pretrain_1_label_plane]|
  
MSE: 0.022509228438138962, BCE: 0.08203141391277313  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_351_pretrain_1_original_depth]|![JNet_351_pretrain_1_output_depth]|![JNet_351_pretrain_1_label_depth]|
  
MSE: 0.022509228438138962, BCE: 0.08203141391277313  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_351_pretrain_2_original_plane]|![JNet_351_pretrain_2_output_plane]|![JNet_351_pretrain_2_label_plane]|
  
MSE: 0.025605257600545883, BCE: 0.09296760708093643  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_351_pretrain_2_original_depth]|![JNet_351_pretrain_2_output_depth]|![JNet_351_pretrain_2_label_depth]|
  
MSE: 0.025605257600545883, BCE: 0.09296760708093643  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_351_pretrain_3_original_plane]|![JNet_351_pretrain_3_output_plane]|![JNet_351_pretrain_3_label_plane]|
  
MSE: 0.022984817624092102, BCE: 0.08219638466835022  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_351_pretrain_3_original_depth]|![JNet_351_pretrain_3_output_depth]|![JNet_351_pretrain_3_label_depth]|
  
MSE: 0.022984817624092102, BCE: 0.08219638466835022  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_351_pretrain_4_original_plane]|![JNet_351_pretrain_4_output_plane]|![JNet_351_pretrain_4_label_plane]|
  
MSE: 0.028846953064203262, BCE: 0.09917120635509491  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_351_pretrain_4_original_depth]|![JNet_351_pretrain_4_output_depth]|![JNet_351_pretrain_4_label_depth]|
  
MSE: 0.028846953064203262, BCE: 0.09917120635509491  
  
mean MSE: 0.22966976463794708, mean BCE: 0.6524601578712463
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_352_0_original_plane]|![JNet_352_0_output_plane]|![JNet_352_0_label_plane]|
  
MSE: 0.2294914573431015, BCE: 0.6521031856536865  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_352_0_original_depth]|![JNet_352_0_output_depth]|![JNet_352_0_label_depth]|
  
MSE: 0.2294914573431015, BCE: 0.6521031856536865  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_352_1_original_plane]|![JNet_352_1_output_plane]|![JNet_352_1_label_plane]|
  
MSE: 0.22950869798660278, BCE: 0.6521374583244324  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_352_1_original_depth]|![JNet_352_1_output_depth]|![JNet_352_1_label_depth]|
  
MSE: 0.22950869798660278, BCE: 0.6521374583244324  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_352_2_original_plane]|![JNet_352_2_output_plane]|![JNet_352_2_label_plane]|
  
MSE: 0.22925443947315216, BCE: 0.6516287326812744  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_352_2_original_depth]|![JNet_352_2_output_depth]|![JNet_352_2_label_depth]|
  
MSE: 0.22925443947315216, BCE: 0.6516287326812744  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_352_3_original_plane]|![JNet_352_3_output_plane]|![JNet_352_3_label_plane]|
  
MSE: 0.2300989031791687, BCE: 0.6533190011978149  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_352_3_original_depth]|![JNet_352_3_output_depth]|![JNet_352_3_label_depth]|
  
MSE: 0.2300989031791687, BCE: 0.6533190011978149  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_352_4_original_plane]|![JNet_352_4_output_plane]|![JNet_352_4_label_plane]|
  
MSE: 0.22999535501003265, BCE: 0.6531121134757996  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_352_4_original_depth]|![JNet_352_4_output_depth]|![JNet_352_4_label_depth]|
  
MSE: 0.22999535501003265, BCE: 0.6531121134757996  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_351_pretrain_beads_001_roi000_original_depth]|![JNet_351_pretrain_beads_001_roi000_output_depth]|![JNet_351_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 16.278875000000003, MSE: 0.002061236882582307, quantized loss: 0.002546042902395129  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_351_pretrain_beads_001_roi001_original_depth]|![JNet_351_pretrain_beads_001_roi001_output_depth]|![JNet_351_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 25.110000000000007, MSE: 0.0034828863572329283, quantized loss: 0.003306824713945389  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_351_pretrain_beads_001_roi002_original_depth]|![JNet_351_pretrain_beads_001_roi002_output_depth]|![JNet_351_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 16.093000000000004, MSE: 0.002106019062921405, quantized loss: 0.002327385125681758  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_351_pretrain_beads_001_roi003_original_depth]|![JNet_351_pretrain_beads_001_roi003_output_depth]|![JNet_351_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 25.939250000000005, MSE: 0.00354761048220098, quantized loss: 0.0035092865582555532  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_351_pretrain_beads_001_roi004_original_depth]|![JNet_351_pretrain_beads_001_roi004_output_depth]|![JNet_351_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 17.118750000000006, MSE: 0.002621002495288849, quantized loss: 0.0026335231959819794  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_351_pretrain_beads_002_roi000_original_depth]|![JNet_351_pretrain_beads_002_roi000_output_depth]|![JNet_351_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 18.127375000000004, MSE: 0.002951523521915078, quantized loss: 0.002592669799923897  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_351_pretrain_beads_002_roi001_original_depth]|![JNet_351_pretrain_beads_002_roi001_output_depth]|![JNet_351_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 16.976000000000003, MSE: 0.002268698997795582, quantized loss: 0.0022826548665761948  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_351_pretrain_beads_002_roi002_original_depth]|![JNet_351_pretrain_beads_002_roi002_output_depth]|![JNet_351_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 17.392500000000005, MSE: 0.0026021257508546114, quantized loss: 0.002564253518357873  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_352_beads_001_roi000_original_depth]|![JNet_352_beads_001_roi000_output_depth]|![JNet_352_beads_001_roi000_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.532200276851654, quantized loss: 0.17054231464862823  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_352_beads_001_roi001_original_depth]|![JNet_352_beads_001_roi001_output_depth]|![JNet_352_beads_001_roi001_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5208145380020142, quantized loss: 0.1688345968723297  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_352_beads_001_roi002_original_depth]|![JNet_352_beads_001_roi002_output_depth]|![JNet_352_beads_001_roi002_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5362014770507812, quantized loss: 0.17091739177703857  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_352_beads_001_roi003_original_depth]|![JNet_352_beads_001_roi003_output_depth]|![JNet_352_beads_001_roi003_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5194132328033447, quantized loss: 0.16895978152751923  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_352_beads_001_roi004_original_depth]|![JNet_352_beads_001_roi004_output_depth]|![JNet_352_beads_001_roi004_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5328969955444336, quantized loss: 0.17057988047599792  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_352_beads_002_roi000_original_depth]|![JNet_352_beads_002_roi000_output_depth]|![JNet_352_beads_002_roi000_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5308762192726135, quantized loss: 0.1703757643699646  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_352_beads_002_roi001_original_depth]|![JNet_352_beads_002_roi001_output_depth]|![JNet_352_beads_002_roi001_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5341352820396423, quantized loss: 0.1706896275281906  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_352_beads_002_roi002_original_depth]|![JNet_352_beads_002_roi002_output_depth]|![JNet_352_beads_002_roi002_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5327744483947754, quantized loss: 0.1705750823020935  

|pre|post|
| :---: | :---: |
|![JNet_352_psf_pre]|![JNet_352_psf_post]|
  



[JNet_351_pretrain_0_label_depth]: /experiments/images/JNet_351_pretrain_0_label_depth.png
[JNet_351_pretrain_0_label_plane]: /experiments/images/JNet_351_pretrain_0_label_plane.png
[JNet_351_pretrain_0_original_depth]: /experiments/images/JNet_351_pretrain_0_original_depth.png
[JNet_351_pretrain_0_original_plane]: /experiments/images/JNet_351_pretrain_0_original_plane.png
[JNet_351_pretrain_0_output_depth]: /experiments/images/JNet_351_pretrain_0_output_depth.png
[JNet_351_pretrain_0_output_plane]: /experiments/images/JNet_351_pretrain_0_output_plane.png
[JNet_351_pretrain_1_label_depth]: /experiments/images/JNet_351_pretrain_1_label_depth.png
[JNet_351_pretrain_1_label_plane]: /experiments/images/JNet_351_pretrain_1_label_plane.png
[JNet_351_pretrain_1_original_depth]: /experiments/images/JNet_351_pretrain_1_original_depth.png
[JNet_351_pretrain_1_original_plane]: /experiments/images/JNet_351_pretrain_1_original_plane.png
[JNet_351_pretrain_1_output_depth]: /experiments/images/JNet_351_pretrain_1_output_depth.png
[JNet_351_pretrain_1_output_plane]: /experiments/images/JNet_351_pretrain_1_output_plane.png
[JNet_351_pretrain_2_label_depth]: /experiments/images/JNet_351_pretrain_2_label_depth.png
[JNet_351_pretrain_2_label_plane]: /experiments/images/JNet_351_pretrain_2_label_plane.png
[JNet_351_pretrain_2_original_depth]: /experiments/images/JNet_351_pretrain_2_original_depth.png
[JNet_351_pretrain_2_original_plane]: /experiments/images/JNet_351_pretrain_2_original_plane.png
[JNet_351_pretrain_2_output_depth]: /experiments/images/JNet_351_pretrain_2_output_depth.png
[JNet_351_pretrain_2_output_plane]: /experiments/images/JNet_351_pretrain_2_output_plane.png
[JNet_351_pretrain_3_label_depth]: /experiments/images/JNet_351_pretrain_3_label_depth.png
[JNet_351_pretrain_3_label_plane]: /experiments/images/JNet_351_pretrain_3_label_plane.png
[JNet_351_pretrain_3_original_depth]: /experiments/images/JNet_351_pretrain_3_original_depth.png
[JNet_351_pretrain_3_original_plane]: /experiments/images/JNet_351_pretrain_3_original_plane.png
[JNet_351_pretrain_3_output_depth]: /experiments/images/JNet_351_pretrain_3_output_depth.png
[JNet_351_pretrain_3_output_plane]: /experiments/images/JNet_351_pretrain_3_output_plane.png
[JNet_351_pretrain_4_label_depth]: /experiments/images/JNet_351_pretrain_4_label_depth.png
[JNet_351_pretrain_4_label_plane]: /experiments/images/JNet_351_pretrain_4_label_plane.png
[JNet_351_pretrain_4_original_depth]: /experiments/images/JNet_351_pretrain_4_original_depth.png
[JNet_351_pretrain_4_original_plane]: /experiments/images/JNet_351_pretrain_4_original_plane.png
[JNet_351_pretrain_4_output_depth]: /experiments/images/JNet_351_pretrain_4_output_depth.png
[JNet_351_pretrain_4_output_plane]: /experiments/images/JNet_351_pretrain_4_output_plane.png
[JNet_351_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi000_original_depth.png
[JNet_351_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi000_output_depth.png
[JNet_351_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi000_reconst_depth.png
[JNet_351_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi001_original_depth.png
[JNet_351_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi001_output_depth.png
[JNet_351_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi001_reconst_depth.png
[JNet_351_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi002_original_depth.png
[JNet_351_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi002_output_depth.png
[JNet_351_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi002_reconst_depth.png
[JNet_351_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi003_original_depth.png
[JNet_351_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi003_output_depth.png
[JNet_351_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi003_reconst_depth.png
[JNet_351_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi004_original_depth.png
[JNet_351_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi004_output_depth.png
[JNet_351_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_351_pretrain_beads_001_roi004_reconst_depth.png
[JNet_351_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_351_pretrain_beads_002_roi000_original_depth.png
[JNet_351_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_351_pretrain_beads_002_roi000_output_depth.png
[JNet_351_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_351_pretrain_beads_002_roi000_reconst_depth.png
[JNet_351_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_351_pretrain_beads_002_roi001_original_depth.png
[JNet_351_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_351_pretrain_beads_002_roi001_output_depth.png
[JNet_351_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_351_pretrain_beads_002_roi001_reconst_depth.png
[JNet_351_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_351_pretrain_beads_002_roi002_original_depth.png
[JNet_351_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_351_pretrain_beads_002_roi002_output_depth.png
[JNet_351_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_351_pretrain_beads_002_roi002_reconst_depth.png
[JNet_352_0_label_depth]: /experiments/images/JNet_352_0_label_depth.png
[JNet_352_0_label_plane]: /experiments/images/JNet_352_0_label_plane.png
[JNet_352_0_original_depth]: /experiments/images/JNet_352_0_original_depth.png
[JNet_352_0_original_plane]: /experiments/images/JNet_352_0_original_plane.png
[JNet_352_0_output_depth]: /experiments/images/JNet_352_0_output_depth.png
[JNet_352_0_output_plane]: /experiments/images/JNet_352_0_output_plane.png
[JNet_352_1_label_depth]: /experiments/images/JNet_352_1_label_depth.png
[JNet_352_1_label_plane]: /experiments/images/JNet_352_1_label_plane.png
[JNet_352_1_original_depth]: /experiments/images/JNet_352_1_original_depth.png
[JNet_352_1_original_plane]: /experiments/images/JNet_352_1_original_plane.png
[JNet_352_1_output_depth]: /experiments/images/JNet_352_1_output_depth.png
[JNet_352_1_output_plane]: /experiments/images/JNet_352_1_output_plane.png
[JNet_352_2_label_depth]: /experiments/images/JNet_352_2_label_depth.png
[JNet_352_2_label_plane]: /experiments/images/JNet_352_2_label_plane.png
[JNet_352_2_original_depth]: /experiments/images/JNet_352_2_original_depth.png
[JNet_352_2_original_plane]: /experiments/images/JNet_352_2_original_plane.png
[JNet_352_2_output_depth]: /experiments/images/JNet_352_2_output_depth.png
[JNet_352_2_output_plane]: /experiments/images/JNet_352_2_output_plane.png
[JNet_352_3_label_depth]: /experiments/images/JNet_352_3_label_depth.png
[JNet_352_3_label_plane]: /experiments/images/JNet_352_3_label_plane.png
[JNet_352_3_original_depth]: /experiments/images/JNet_352_3_original_depth.png
[JNet_352_3_original_plane]: /experiments/images/JNet_352_3_original_plane.png
[JNet_352_3_output_depth]: /experiments/images/JNet_352_3_output_depth.png
[JNet_352_3_output_plane]: /experiments/images/JNet_352_3_output_plane.png
[JNet_352_4_label_depth]: /experiments/images/JNet_352_4_label_depth.png
[JNet_352_4_label_plane]: /experiments/images/JNet_352_4_label_plane.png
[JNet_352_4_original_depth]: /experiments/images/JNet_352_4_original_depth.png
[JNet_352_4_original_plane]: /experiments/images/JNet_352_4_original_plane.png
[JNet_352_4_output_depth]: /experiments/images/JNet_352_4_output_depth.png
[JNet_352_4_output_plane]: /experiments/images/JNet_352_4_output_plane.png
[JNet_352_beads_001_roi000_original_depth]: /experiments/images/JNet_352_beads_001_roi000_original_depth.png
[JNet_352_beads_001_roi000_output_depth]: /experiments/images/JNet_352_beads_001_roi000_output_depth.png
[JNet_352_beads_001_roi000_reconst_depth]: /experiments/images/JNet_352_beads_001_roi000_reconst_depth.png
[JNet_352_beads_001_roi001_original_depth]: /experiments/images/JNet_352_beads_001_roi001_original_depth.png
[JNet_352_beads_001_roi001_output_depth]: /experiments/images/JNet_352_beads_001_roi001_output_depth.png
[JNet_352_beads_001_roi001_reconst_depth]: /experiments/images/JNet_352_beads_001_roi001_reconst_depth.png
[JNet_352_beads_001_roi002_original_depth]: /experiments/images/JNet_352_beads_001_roi002_original_depth.png
[JNet_352_beads_001_roi002_output_depth]: /experiments/images/JNet_352_beads_001_roi002_output_depth.png
[JNet_352_beads_001_roi002_reconst_depth]: /experiments/images/JNet_352_beads_001_roi002_reconst_depth.png
[JNet_352_beads_001_roi003_original_depth]: /experiments/images/JNet_352_beads_001_roi003_original_depth.png
[JNet_352_beads_001_roi003_output_depth]: /experiments/images/JNet_352_beads_001_roi003_output_depth.png
[JNet_352_beads_001_roi003_reconst_depth]: /experiments/images/JNet_352_beads_001_roi003_reconst_depth.png
[JNet_352_beads_001_roi004_original_depth]: /experiments/images/JNet_352_beads_001_roi004_original_depth.png
[JNet_352_beads_001_roi004_output_depth]: /experiments/images/JNet_352_beads_001_roi004_output_depth.png
[JNet_352_beads_001_roi004_reconst_depth]: /experiments/images/JNet_352_beads_001_roi004_reconst_depth.png
[JNet_352_beads_002_roi000_original_depth]: /experiments/images/JNet_352_beads_002_roi000_original_depth.png
[JNet_352_beads_002_roi000_output_depth]: /experiments/images/JNet_352_beads_002_roi000_output_depth.png
[JNet_352_beads_002_roi000_reconst_depth]: /experiments/images/JNet_352_beads_002_roi000_reconst_depth.png
[JNet_352_beads_002_roi001_original_depth]: /experiments/images/JNet_352_beads_002_roi001_original_depth.png
[JNet_352_beads_002_roi001_output_depth]: /experiments/images/JNet_352_beads_002_roi001_output_depth.png
[JNet_352_beads_002_roi001_reconst_depth]: /experiments/images/JNet_352_beads_002_roi001_reconst_depth.png
[JNet_352_beads_002_roi002_original_depth]: /experiments/images/JNet_352_beads_002_roi002_original_depth.png
[JNet_352_beads_002_roi002_output_depth]: /experiments/images/JNet_352_beads_002_roi002_output_depth.png
[JNet_352_beads_002_roi002_reconst_depth]: /experiments/images/JNet_352_beads_002_roi002_reconst_depth.png
[JNet_352_psf_post]: /experiments/images/JNet_352_psf_post.png
[JNet_352_psf_pre]: /experiments/images/JNet_352_psf_pre.png
