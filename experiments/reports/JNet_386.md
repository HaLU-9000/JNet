



# JNet_386 Report
  
the parameters to replicate the results of JNet_386. vibrate, arg f1 deleted  
pretrained model : JNet_385_pretrain
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
|is_vibrate|False|
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
  
mean MSE: 0.016820644959807396, mean BCE: 0.05758843943476677
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_0_original_plane]|![JNet_385_pretrain_0_output_plane]|![JNet_385_pretrain_0_label_plane]|
  
MSE: 0.015941061079502106, BCE: 0.05397917702794075  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_0_original_depth]|![JNet_385_pretrain_0_output_depth]|![JNet_385_pretrain_0_label_depth]|
  
MSE: 0.015941061079502106, BCE: 0.05397917702794075  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_1_original_plane]|![JNet_385_pretrain_1_output_plane]|![JNet_385_pretrain_1_label_plane]|
  
MSE: 0.017107876017689705, BCE: 0.056716471910476685  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_1_original_depth]|![JNet_385_pretrain_1_output_depth]|![JNet_385_pretrain_1_label_depth]|
  
MSE: 0.017107876017689705, BCE: 0.056716471910476685  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_2_original_plane]|![JNet_385_pretrain_2_output_plane]|![JNet_385_pretrain_2_label_plane]|
  
MSE: 0.021343156695365906, BCE: 0.07338765263557434  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_2_original_depth]|![JNet_385_pretrain_2_output_depth]|![JNet_385_pretrain_2_label_depth]|
  
MSE: 0.021343156695365906, BCE: 0.07338765263557434  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_3_original_plane]|![JNet_385_pretrain_3_output_plane]|![JNet_385_pretrain_3_label_plane]|
  
MSE: 0.013762150891125202, BCE: 0.04772234708070755  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_3_original_depth]|![JNet_385_pretrain_3_output_depth]|![JNet_385_pretrain_3_label_depth]|
  
MSE: 0.013762150891125202, BCE: 0.04772234708070755  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_4_original_plane]|![JNet_385_pretrain_4_output_plane]|![JNet_385_pretrain_4_label_plane]|
  
MSE: 0.015948977321386337, BCE: 0.056136541068553925  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_4_original_depth]|![JNet_385_pretrain_4_output_depth]|![JNet_385_pretrain_4_label_depth]|
  
MSE: 0.015948977321386337, BCE: 0.056136541068553925  
  
mean MSE: 0.024926764890551567, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_386_0_original_plane]|![JNet_386_0_output_plane]|![JNet_386_0_label_plane]|
  
MSE: 0.014807876199483871, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_386_0_original_depth]|![JNet_386_0_output_depth]|![JNet_386_0_label_depth]|
  
MSE: 0.014807876199483871, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_386_1_original_plane]|![JNet_386_1_output_plane]|![JNet_386_1_label_plane]|
  
MSE: 0.025520501658320427, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_386_1_original_depth]|![JNet_386_1_output_depth]|![JNet_386_1_label_depth]|
  
MSE: 0.025520501658320427, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_386_2_original_plane]|![JNet_386_2_output_plane]|![JNet_386_2_label_plane]|
  
MSE: 0.027761081233620644, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_386_2_original_depth]|![JNet_386_2_output_depth]|![JNet_386_2_label_depth]|
  
MSE: 0.027761081233620644, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_386_3_original_plane]|![JNet_386_3_output_plane]|![JNet_386_3_label_plane]|
  
MSE: 0.024741243571043015, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_386_3_original_depth]|![JNet_386_3_output_depth]|![JNet_386_3_label_depth]|
  
MSE: 0.024741243571043015, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_386_4_original_plane]|![JNet_386_4_output_plane]|![JNet_386_4_label_plane]|
  
MSE: 0.03180312365293503, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_386_4_original_depth]|![JNet_386_4_output_depth]|![JNet_386_4_label_depth]|
  
MSE: 0.03180312365293503, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_001_roi000_original_depth]|![JNet_385_pretrain_beads_001_roi000_output_depth]|![JNet_385_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 9.207460937500002, MSE: 0.002633366035297513, quantized loss: 0.0006782306008972228  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_001_roi001_original_depth]|![JNet_385_pretrain_beads_001_roi001_output_depth]|![JNet_385_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 14.162287109375004, MSE: 0.004437160212546587, quantized loss: 0.0009158350294455886  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_001_roi002_original_depth]|![JNet_385_pretrain_beads_001_roi002_output_depth]|![JNet_385_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 8.631785156250002, MSE: 0.002711518434807658, quantized loss: 0.000558068510144949  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_001_roi003_original_depth]|![JNet_385_pretrain_beads_001_roi003_output_depth]|![JNet_385_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 14.662413085937503, MSE: 0.0045382375828921795, quantized loss: 0.0009447999182157218  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_001_roi004_original_depth]|![JNet_385_pretrain_beads_001_roi004_output_depth]|![JNet_385_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 9.022173828125002, MSE: 0.0035021109506487846, quantized loss: 0.0005383387324400246  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_002_roi000_original_depth]|![JNet_385_pretrain_beads_002_roi000_output_depth]|![JNet_385_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 9.590084960937503, MSE: 0.003958486951887608, quantized loss: 0.0005660082679241896  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_002_roi001_original_depth]|![JNet_385_pretrain_beads_002_roi001_output_depth]|![JNet_385_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 9.339628906250002, MSE: 0.002902807667851448, quantized loss: 0.0005866418359801173  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_002_roi002_original_depth]|![JNet_385_pretrain_beads_002_roi002_output_depth]|![JNet_385_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 9.460370117187502, MSE: 0.003445657901465893, quantized loss: 0.0005729333497583866  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_386_beads_001_roi000_original_depth]|![JNet_386_beads_001_roi000_output_depth]|![JNet_386_beads_001_roi000_reconst_depth]|
  
volume: 12.352167968750003, MSE: 0.0008896844810806215, quantized loss: 7.947999620228074e-06  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_386_beads_001_roi001_original_depth]|![JNet_386_beads_001_roi001_output_depth]|![JNet_386_beads_001_roi001_reconst_depth]|
  
volume: 19.363873046875003, MSE: 0.0012608106480911374, quantized loss: 1.1541182175278664e-05  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_386_beads_001_roi002_original_depth]|![JNet_386_beads_001_roi002_output_depth]|![JNet_386_beads_001_roi002_reconst_depth]|
  
volume: 12.259083984375003, MSE: 0.0008519541588611901, quantized loss: 7.450218163285172e-06  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_386_beads_001_roi003_original_depth]|![JNet_386_beads_001_roi003_output_depth]|![JNet_386_beads_001_roi003_reconst_depth]|
  
volume: 20.273482421875006, MSE: 0.00124579097609967, quantized loss: 1.2711851013591513e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_386_beads_001_roi004_original_depth]|![JNet_386_beads_001_roi004_output_depth]|![JNet_386_beads_001_roi004_reconst_depth]|
  
volume: 13.488507812500004, MSE: 0.0009046234772540629, quantized loss: 7.5681314228859264e-06  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_386_beads_002_roi000_original_depth]|![JNet_386_beads_002_roi000_output_depth]|![JNet_386_beads_002_roi000_reconst_depth]|
  
volume: 14.491078125000003, MSE: 0.0010029188124462962, quantized loss: 8.144355888362043e-06  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_386_beads_002_roi001_original_depth]|![JNet_386_beads_002_roi001_output_depth]|![JNet_386_beads_002_roi001_reconst_depth]|
  
volume: 13.124392578125002, MSE: 0.0009090359671972692, quantized loss: 7.97045049694134e-06  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_386_beads_002_roi002_original_depth]|![JNet_386_beads_002_roi002_output_depth]|![JNet_386_beads_002_roi002_reconst_depth]|
  
volume: 13.729384765625003, MSE: 0.0009461829904466867, quantized loss: 8.512164640706033e-06  

|pre|post|
| :---: | :---: |
|![JNet_386_psf_pre]|![JNet_386_psf_post]|
  



[JNet_385_pretrain_0_label_depth]: /experiments/images/JNet_385_pretrain_0_label_depth.png
[JNet_385_pretrain_0_label_plane]: /experiments/images/JNet_385_pretrain_0_label_plane.png
[JNet_385_pretrain_0_original_depth]: /experiments/images/JNet_385_pretrain_0_original_depth.png
[JNet_385_pretrain_0_original_plane]: /experiments/images/JNet_385_pretrain_0_original_plane.png
[JNet_385_pretrain_0_output_depth]: /experiments/images/JNet_385_pretrain_0_output_depth.png
[JNet_385_pretrain_0_output_plane]: /experiments/images/JNet_385_pretrain_0_output_plane.png
[JNet_385_pretrain_1_label_depth]: /experiments/images/JNet_385_pretrain_1_label_depth.png
[JNet_385_pretrain_1_label_plane]: /experiments/images/JNet_385_pretrain_1_label_plane.png
[JNet_385_pretrain_1_original_depth]: /experiments/images/JNet_385_pretrain_1_original_depth.png
[JNet_385_pretrain_1_original_plane]: /experiments/images/JNet_385_pretrain_1_original_plane.png
[JNet_385_pretrain_1_output_depth]: /experiments/images/JNet_385_pretrain_1_output_depth.png
[JNet_385_pretrain_1_output_plane]: /experiments/images/JNet_385_pretrain_1_output_plane.png
[JNet_385_pretrain_2_label_depth]: /experiments/images/JNet_385_pretrain_2_label_depth.png
[JNet_385_pretrain_2_label_plane]: /experiments/images/JNet_385_pretrain_2_label_plane.png
[JNet_385_pretrain_2_original_depth]: /experiments/images/JNet_385_pretrain_2_original_depth.png
[JNet_385_pretrain_2_original_plane]: /experiments/images/JNet_385_pretrain_2_original_plane.png
[JNet_385_pretrain_2_output_depth]: /experiments/images/JNet_385_pretrain_2_output_depth.png
[JNet_385_pretrain_2_output_plane]: /experiments/images/JNet_385_pretrain_2_output_plane.png
[JNet_385_pretrain_3_label_depth]: /experiments/images/JNet_385_pretrain_3_label_depth.png
[JNet_385_pretrain_3_label_plane]: /experiments/images/JNet_385_pretrain_3_label_plane.png
[JNet_385_pretrain_3_original_depth]: /experiments/images/JNet_385_pretrain_3_original_depth.png
[JNet_385_pretrain_3_original_plane]: /experiments/images/JNet_385_pretrain_3_original_plane.png
[JNet_385_pretrain_3_output_depth]: /experiments/images/JNet_385_pretrain_3_output_depth.png
[JNet_385_pretrain_3_output_plane]: /experiments/images/JNet_385_pretrain_3_output_plane.png
[JNet_385_pretrain_4_label_depth]: /experiments/images/JNet_385_pretrain_4_label_depth.png
[JNet_385_pretrain_4_label_plane]: /experiments/images/JNet_385_pretrain_4_label_plane.png
[JNet_385_pretrain_4_original_depth]: /experiments/images/JNet_385_pretrain_4_original_depth.png
[JNet_385_pretrain_4_original_plane]: /experiments/images/JNet_385_pretrain_4_original_plane.png
[JNet_385_pretrain_4_output_depth]: /experiments/images/JNet_385_pretrain_4_output_depth.png
[JNet_385_pretrain_4_output_plane]: /experiments/images/JNet_385_pretrain_4_output_plane.png
[JNet_385_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi000_original_depth.png
[JNet_385_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi000_output_depth.png
[JNet_385_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi000_reconst_depth.png
[JNet_385_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi001_original_depth.png
[JNet_385_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi001_output_depth.png
[JNet_385_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi001_reconst_depth.png
[JNet_385_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi002_original_depth.png
[JNet_385_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi002_output_depth.png
[JNet_385_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi002_reconst_depth.png
[JNet_385_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi003_original_depth.png
[JNet_385_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi003_output_depth.png
[JNet_385_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi003_reconst_depth.png
[JNet_385_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi004_original_depth.png
[JNet_385_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi004_output_depth.png
[JNet_385_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_385_pretrain_beads_001_roi004_reconst_depth.png
[JNet_385_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_385_pretrain_beads_002_roi000_original_depth.png
[JNet_385_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_385_pretrain_beads_002_roi000_output_depth.png
[JNet_385_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_385_pretrain_beads_002_roi000_reconst_depth.png
[JNet_385_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_385_pretrain_beads_002_roi001_original_depth.png
[JNet_385_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_385_pretrain_beads_002_roi001_output_depth.png
[JNet_385_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_385_pretrain_beads_002_roi001_reconst_depth.png
[JNet_385_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_385_pretrain_beads_002_roi002_original_depth.png
[JNet_385_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_385_pretrain_beads_002_roi002_output_depth.png
[JNet_385_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_385_pretrain_beads_002_roi002_reconst_depth.png
[JNet_386_0_label_depth]: /experiments/images/JNet_386_0_label_depth.png
[JNet_386_0_label_plane]: /experiments/images/JNet_386_0_label_plane.png
[JNet_386_0_original_depth]: /experiments/images/JNet_386_0_original_depth.png
[JNet_386_0_original_plane]: /experiments/images/JNet_386_0_original_plane.png
[JNet_386_0_output_depth]: /experiments/images/JNet_386_0_output_depth.png
[JNet_386_0_output_plane]: /experiments/images/JNet_386_0_output_plane.png
[JNet_386_1_label_depth]: /experiments/images/JNet_386_1_label_depth.png
[JNet_386_1_label_plane]: /experiments/images/JNet_386_1_label_plane.png
[JNet_386_1_original_depth]: /experiments/images/JNet_386_1_original_depth.png
[JNet_386_1_original_plane]: /experiments/images/JNet_386_1_original_plane.png
[JNet_386_1_output_depth]: /experiments/images/JNet_386_1_output_depth.png
[JNet_386_1_output_plane]: /experiments/images/JNet_386_1_output_plane.png
[JNet_386_2_label_depth]: /experiments/images/JNet_386_2_label_depth.png
[JNet_386_2_label_plane]: /experiments/images/JNet_386_2_label_plane.png
[JNet_386_2_original_depth]: /experiments/images/JNet_386_2_original_depth.png
[JNet_386_2_original_plane]: /experiments/images/JNet_386_2_original_plane.png
[JNet_386_2_output_depth]: /experiments/images/JNet_386_2_output_depth.png
[JNet_386_2_output_plane]: /experiments/images/JNet_386_2_output_plane.png
[JNet_386_3_label_depth]: /experiments/images/JNet_386_3_label_depth.png
[JNet_386_3_label_plane]: /experiments/images/JNet_386_3_label_plane.png
[JNet_386_3_original_depth]: /experiments/images/JNet_386_3_original_depth.png
[JNet_386_3_original_plane]: /experiments/images/JNet_386_3_original_plane.png
[JNet_386_3_output_depth]: /experiments/images/JNet_386_3_output_depth.png
[JNet_386_3_output_plane]: /experiments/images/JNet_386_3_output_plane.png
[JNet_386_4_label_depth]: /experiments/images/JNet_386_4_label_depth.png
[JNet_386_4_label_plane]: /experiments/images/JNet_386_4_label_plane.png
[JNet_386_4_original_depth]: /experiments/images/JNet_386_4_original_depth.png
[JNet_386_4_original_plane]: /experiments/images/JNet_386_4_original_plane.png
[JNet_386_4_output_depth]: /experiments/images/JNet_386_4_output_depth.png
[JNet_386_4_output_plane]: /experiments/images/JNet_386_4_output_plane.png
[JNet_386_beads_001_roi000_original_depth]: /experiments/images/JNet_386_beads_001_roi000_original_depth.png
[JNet_386_beads_001_roi000_output_depth]: /experiments/images/JNet_386_beads_001_roi000_output_depth.png
[JNet_386_beads_001_roi000_reconst_depth]: /experiments/images/JNet_386_beads_001_roi000_reconst_depth.png
[JNet_386_beads_001_roi001_original_depth]: /experiments/images/JNet_386_beads_001_roi001_original_depth.png
[JNet_386_beads_001_roi001_output_depth]: /experiments/images/JNet_386_beads_001_roi001_output_depth.png
[JNet_386_beads_001_roi001_reconst_depth]: /experiments/images/JNet_386_beads_001_roi001_reconst_depth.png
[JNet_386_beads_001_roi002_original_depth]: /experiments/images/JNet_386_beads_001_roi002_original_depth.png
[JNet_386_beads_001_roi002_output_depth]: /experiments/images/JNet_386_beads_001_roi002_output_depth.png
[JNet_386_beads_001_roi002_reconst_depth]: /experiments/images/JNet_386_beads_001_roi002_reconst_depth.png
[JNet_386_beads_001_roi003_original_depth]: /experiments/images/JNet_386_beads_001_roi003_original_depth.png
[JNet_386_beads_001_roi003_output_depth]: /experiments/images/JNet_386_beads_001_roi003_output_depth.png
[JNet_386_beads_001_roi003_reconst_depth]: /experiments/images/JNet_386_beads_001_roi003_reconst_depth.png
[JNet_386_beads_001_roi004_original_depth]: /experiments/images/JNet_386_beads_001_roi004_original_depth.png
[JNet_386_beads_001_roi004_output_depth]: /experiments/images/JNet_386_beads_001_roi004_output_depth.png
[JNet_386_beads_001_roi004_reconst_depth]: /experiments/images/JNet_386_beads_001_roi004_reconst_depth.png
[JNet_386_beads_002_roi000_original_depth]: /experiments/images/JNet_386_beads_002_roi000_original_depth.png
[JNet_386_beads_002_roi000_output_depth]: /experiments/images/JNet_386_beads_002_roi000_output_depth.png
[JNet_386_beads_002_roi000_reconst_depth]: /experiments/images/JNet_386_beads_002_roi000_reconst_depth.png
[JNet_386_beads_002_roi001_original_depth]: /experiments/images/JNet_386_beads_002_roi001_original_depth.png
[JNet_386_beads_002_roi001_output_depth]: /experiments/images/JNet_386_beads_002_roi001_output_depth.png
[JNet_386_beads_002_roi001_reconst_depth]: /experiments/images/JNet_386_beads_002_roi001_reconst_depth.png
[JNet_386_beads_002_roi002_original_depth]: /experiments/images/JNet_386_beads_002_roi002_original_depth.png
[JNet_386_beads_002_roi002_output_depth]: /experiments/images/JNet_386_beads_002_roi002_output_depth.png
[JNet_386_beads_002_roi002_reconst_depth]: /experiments/images/JNet_386_beads_002_roi002_reconst_depth.png
[JNet_386_psf_post]: /experiments/images/JNet_386_psf_post.png
[JNet_386_psf_pre]: /experiments/images/JNet_386_psf_pre.png
[finetuned]: /experiments/tmp/JNet_386_train.png
[pretrained_model]: /experiments/tmp/JNet_385_pretrain_train.png
