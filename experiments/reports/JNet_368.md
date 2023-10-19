



# JNet_368 Report
  
the parameters to replicate the results of JNet_368  
pretrained model : JNet_367_pretrain
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
|background|0.02||
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
|mask|False|
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
  
mean MSE: 0.03301897644996643, mean BCE: 0.1488211452960968
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_367_pretrain_0_original_plane]|![JNet_367_pretrain_0_output_plane]|![JNet_367_pretrain_0_label_plane]|
  
MSE: 0.02551477588713169, BCE: 0.1215234249830246  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_367_pretrain_0_original_depth]|![JNet_367_pretrain_0_output_depth]|![JNet_367_pretrain_0_label_depth]|
  
MSE: 0.02551477588713169, BCE: 0.1215234249830246  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_367_pretrain_1_original_plane]|![JNet_367_pretrain_1_output_plane]|![JNet_367_pretrain_1_label_plane]|
  
MSE: 0.03352440148591995, BCE: 0.15081767737865448  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_367_pretrain_1_original_depth]|![JNet_367_pretrain_1_output_depth]|![JNet_367_pretrain_1_label_depth]|
  
MSE: 0.03352440148591995, BCE: 0.15081767737865448  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_367_pretrain_2_original_plane]|![JNet_367_pretrain_2_output_plane]|![JNet_367_pretrain_2_label_plane]|
  
MSE: 0.04127771779894829, BCE: 0.1786523461341858  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_367_pretrain_2_original_depth]|![JNet_367_pretrain_2_output_depth]|![JNet_367_pretrain_2_label_depth]|
  
MSE: 0.04127771779894829, BCE: 0.1786523461341858  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_367_pretrain_3_original_plane]|![JNet_367_pretrain_3_output_plane]|![JNet_367_pretrain_3_label_plane]|
  
MSE: 0.034090589731931686, BCE: 0.15283344686031342  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_367_pretrain_3_original_depth]|![JNet_367_pretrain_3_output_depth]|![JNet_367_pretrain_3_label_depth]|
  
MSE: 0.034090589731931686, BCE: 0.15283344686031342  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_367_pretrain_4_original_plane]|![JNet_367_pretrain_4_output_plane]|![JNet_367_pretrain_4_label_plane]|
  
MSE: 0.030687393620610237, BCE: 0.14027878642082214  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_367_pretrain_4_original_depth]|![JNet_367_pretrain_4_output_depth]|![JNet_367_pretrain_4_label_depth]|
  
MSE: 0.030687393620610237, BCE: 0.14027878642082214  
  
mean MSE: 0.03507460281252861, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_368_0_original_plane]|![JNet_368_0_output_plane]|![JNet_368_0_label_plane]|
  
MSE: 0.036854274570941925, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_368_0_original_depth]|![JNet_368_0_output_depth]|![JNet_368_0_label_depth]|
  
MSE: 0.036854274570941925, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_368_1_original_plane]|![JNet_368_1_output_plane]|![JNet_368_1_label_plane]|
  
MSE: 0.03417138382792473, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_368_1_original_depth]|![JNet_368_1_output_depth]|![JNet_368_1_label_depth]|
  
MSE: 0.03417138382792473, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_368_2_original_plane]|![JNet_368_2_output_plane]|![JNet_368_2_label_plane]|
  
MSE: 0.026201769709587097, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_368_2_original_depth]|![JNet_368_2_output_depth]|![JNet_368_2_label_depth]|
  
MSE: 0.026201769709587097, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_368_3_original_plane]|![JNet_368_3_output_plane]|![JNet_368_3_label_plane]|
  
MSE: 0.03811018541455269, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_368_3_original_depth]|![JNet_368_3_output_depth]|![JNet_368_3_label_depth]|
  
MSE: 0.03811018541455269, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_368_4_original_plane]|![JNet_368_4_output_plane]|![JNet_368_4_label_plane]|
  
MSE: 0.04003540799021721, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_368_4_original_depth]|![JNet_368_4_output_depth]|![JNet_368_4_label_depth]|
  
MSE: 0.04003540799021721, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_367_pretrain_beads_001_roi000_original_depth]|![JNet_367_pretrain_beads_001_roi000_output_depth]|![JNet_367_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 0.0, MSE: 0.00835121888667345, quantized loss: 0.0009973651031032205  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_367_pretrain_beads_001_roi001_original_depth]|![JNet_367_pretrain_beads_001_roi001_output_depth]|![JNet_367_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 0.0, MSE: 0.01334976777434349, quantized loss: 0.0010046090465039015  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_367_pretrain_beads_001_roi002_original_depth]|![JNet_367_pretrain_beads_001_roi002_output_depth]|![JNet_367_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 0.0, MSE: 0.008190272375941277, quantized loss: 0.0009956116555258632  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_367_pretrain_beads_001_roi003_original_depth]|![JNet_367_pretrain_beads_001_roi003_output_depth]|![JNet_367_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 0.0, MSE: 0.014432706870138645, quantized loss: 0.001000154297798872  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_367_pretrain_beads_001_roi004_original_depth]|![JNet_367_pretrain_beads_001_roi004_output_depth]|![JNet_367_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 0.0, MSE: 0.009960031136870384, quantized loss: 0.000995916547253728  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_367_pretrain_beads_002_roi000_original_depth]|![JNet_367_pretrain_beads_002_roi000_output_depth]|![JNet_367_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 0.0, MSE: 0.011220037937164307, quantized loss: 0.0009962625335901976  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_367_pretrain_beads_002_roi001_original_depth]|![JNet_367_pretrain_beads_002_roi001_output_depth]|![JNet_367_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 0.0, MSE: 0.009367246180772781, quantized loss: 0.0009957829024642706  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_367_pretrain_beads_002_roi002_original_depth]|![JNet_367_pretrain_beads_002_roi002_output_depth]|![JNet_367_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 0.0, MSE: 0.010098973289132118, quantized loss: 0.000995615147985518  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_368_beads_001_roi000_original_depth]|![JNet_368_beads_001_roi000_output_depth]|![JNet_368_beads_001_roi000_reconst_depth]|
  
volume: 0.0, MSE: 0.008520037867128849, quantized loss: 0.0  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_368_beads_001_roi001_original_depth]|![JNet_368_beads_001_roi001_output_depth]|![JNet_368_beads_001_roi001_reconst_depth]|
  
volume: 0.0, MSE: 0.013647239655256271, quantized loss: 0.0  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_368_beads_001_roi002_original_depth]|![JNet_368_beads_001_roi002_output_depth]|![JNet_368_beads_001_roi002_reconst_depth]|
  
volume: 0.0, MSE: 0.008329926058650017, quantized loss: 0.0  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_368_beads_001_roi003_original_depth]|![JNet_368_beads_001_roi003_output_depth]|![JNet_368_beads_001_roi003_reconst_depth]|
  
volume: 0.0, MSE: 0.014731120318174362, quantized loss: 0.0  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_368_beads_001_roi004_original_depth]|![JNet_368_beads_001_roi004_output_depth]|![JNet_368_beads_001_roi004_reconst_depth]|
  
volume: 0.0, MSE: 0.010127575136721134, quantized loss: 0.0  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_368_beads_002_roi000_original_depth]|![JNet_368_beads_002_roi000_output_depth]|![JNet_368_beads_002_roi000_reconst_depth]|
  
volume: 0.0, MSE: 0.011404185555875301, quantized loss: 0.0  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_368_beads_002_roi001_original_depth]|![JNet_368_beads_002_roi001_output_depth]|![JNet_368_beads_002_roi001_reconst_depth]|
  
volume: 0.0, MSE: 0.009525389410555363, quantized loss: 0.0  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_368_beads_002_roi002_original_depth]|![JNet_368_beads_002_roi002_output_depth]|![JNet_368_beads_002_roi002_reconst_depth]|
  
volume: 0.0, MSE: 0.010266602970659733, quantized loss: 0.0  

|pre|post|
| :---: | :---: |
|![JNet_368_psf_pre]|![JNet_368_psf_post]|
  



[JNet_367_pretrain_0_label_depth]: /experiments/images/JNet_367_pretrain_0_label_depth.png
[JNet_367_pretrain_0_label_plane]: /experiments/images/JNet_367_pretrain_0_label_plane.png
[JNet_367_pretrain_0_original_depth]: /experiments/images/JNet_367_pretrain_0_original_depth.png
[JNet_367_pretrain_0_original_plane]: /experiments/images/JNet_367_pretrain_0_original_plane.png
[JNet_367_pretrain_0_output_depth]: /experiments/images/JNet_367_pretrain_0_output_depth.png
[JNet_367_pretrain_0_output_plane]: /experiments/images/JNet_367_pretrain_0_output_plane.png
[JNet_367_pretrain_1_label_depth]: /experiments/images/JNet_367_pretrain_1_label_depth.png
[JNet_367_pretrain_1_label_plane]: /experiments/images/JNet_367_pretrain_1_label_plane.png
[JNet_367_pretrain_1_original_depth]: /experiments/images/JNet_367_pretrain_1_original_depth.png
[JNet_367_pretrain_1_original_plane]: /experiments/images/JNet_367_pretrain_1_original_plane.png
[JNet_367_pretrain_1_output_depth]: /experiments/images/JNet_367_pretrain_1_output_depth.png
[JNet_367_pretrain_1_output_plane]: /experiments/images/JNet_367_pretrain_1_output_plane.png
[JNet_367_pretrain_2_label_depth]: /experiments/images/JNet_367_pretrain_2_label_depth.png
[JNet_367_pretrain_2_label_plane]: /experiments/images/JNet_367_pretrain_2_label_plane.png
[JNet_367_pretrain_2_original_depth]: /experiments/images/JNet_367_pretrain_2_original_depth.png
[JNet_367_pretrain_2_original_plane]: /experiments/images/JNet_367_pretrain_2_original_plane.png
[JNet_367_pretrain_2_output_depth]: /experiments/images/JNet_367_pretrain_2_output_depth.png
[JNet_367_pretrain_2_output_plane]: /experiments/images/JNet_367_pretrain_2_output_plane.png
[JNet_367_pretrain_3_label_depth]: /experiments/images/JNet_367_pretrain_3_label_depth.png
[JNet_367_pretrain_3_label_plane]: /experiments/images/JNet_367_pretrain_3_label_plane.png
[JNet_367_pretrain_3_original_depth]: /experiments/images/JNet_367_pretrain_3_original_depth.png
[JNet_367_pretrain_3_original_plane]: /experiments/images/JNet_367_pretrain_3_original_plane.png
[JNet_367_pretrain_3_output_depth]: /experiments/images/JNet_367_pretrain_3_output_depth.png
[JNet_367_pretrain_3_output_plane]: /experiments/images/JNet_367_pretrain_3_output_plane.png
[JNet_367_pretrain_4_label_depth]: /experiments/images/JNet_367_pretrain_4_label_depth.png
[JNet_367_pretrain_4_label_plane]: /experiments/images/JNet_367_pretrain_4_label_plane.png
[JNet_367_pretrain_4_original_depth]: /experiments/images/JNet_367_pretrain_4_original_depth.png
[JNet_367_pretrain_4_original_plane]: /experiments/images/JNet_367_pretrain_4_original_plane.png
[JNet_367_pretrain_4_output_depth]: /experiments/images/JNet_367_pretrain_4_output_depth.png
[JNet_367_pretrain_4_output_plane]: /experiments/images/JNet_367_pretrain_4_output_plane.png
[JNet_367_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi000_original_depth.png
[JNet_367_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi000_output_depth.png
[JNet_367_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi000_reconst_depth.png
[JNet_367_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi001_original_depth.png
[JNet_367_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi001_output_depth.png
[JNet_367_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi001_reconst_depth.png
[JNet_367_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi002_original_depth.png
[JNet_367_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi002_output_depth.png
[JNet_367_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi002_reconst_depth.png
[JNet_367_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi003_original_depth.png
[JNet_367_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi003_output_depth.png
[JNet_367_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi003_reconst_depth.png
[JNet_367_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi004_original_depth.png
[JNet_367_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi004_output_depth.png
[JNet_367_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_367_pretrain_beads_001_roi004_reconst_depth.png
[JNet_367_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_367_pretrain_beads_002_roi000_original_depth.png
[JNet_367_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_367_pretrain_beads_002_roi000_output_depth.png
[JNet_367_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_367_pretrain_beads_002_roi000_reconst_depth.png
[JNet_367_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_367_pretrain_beads_002_roi001_original_depth.png
[JNet_367_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_367_pretrain_beads_002_roi001_output_depth.png
[JNet_367_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_367_pretrain_beads_002_roi001_reconst_depth.png
[JNet_367_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_367_pretrain_beads_002_roi002_original_depth.png
[JNet_367_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_367_pretrain_beads_002_roi002_output_depth.png
[JNet_367_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_367_pretrain_beads_002_roi002_reconst_depth.png
[JNet_368_0_label_depth]: /experiments/images/JNet_368_0_label_depth.png
[JNet_368_0_label_plane]: /experiments/images/JNet_368_0_label_plane.png
[JNet_368_0_original_depth]: /experiments/images/JNet_368_0_original_depth.png
[JNet_368_0_original_plane]: /experiments/images/JNet_368_0_original_plane.png
[JNet_368_0_output_depth]: /experiments/images/JNet_368_0_output_depth.png
[JNet_368_0_output_plane]: /experiments/images/JNet_368_0_output_plane.png
[JNet_368_1_label_depth]: /experiments/images/JNet_368_1_label_depth.png
[JNet_368_1_label_plane]: /experiments/images/JNet_368_1_label_plane.png
[JNet_368_1_original_depth]: /experiments/images/JNet_368_1_original_depth.png
[JNet_368_1_original_plane]: /experiments/images/JNet_368_1_original_plane.png
[JNet_368_1_output_depth]: /experiments/images/JNet_368_1_output_depth.png
[JNet_368_1_output_plane]: /experiments/images/JNet_368_1_output_plane.png
[JNet_368_2_label_depth]: /experiments/images/JNet_368_2_label_depth.png
[JNet_368_2_label_plane]: /experiments/images/JNet_368_2_label_plane.png
[JNet_368_2_original_depth]: /experiments/images/JNet_368_2_original_depth.png
[JNet_368_2_original_plane]: /experiments/images/JNet_368_2_original_plane.png
[JNet_368_2_output_depth]: /experiments/images/JNet_368_2_output_depth.png
[JNet_368_2_output_plane]: /experiments/images/JNet_368_2_output_plane.png
[JNet_368_3_label_depth]: /experiments/images/JNet_368_3_label_depth.png
[JNet_368_3_label_plane]: /experiments/images/JNet_368_3_label_plane.png
[JNet_368_3_original_depth]: /experiments/images/JNet_368_3_original_depth.png
[JNet_368_3_original_plane]: /experiments/images/JNet_368_3_original_plane.png
[JNet_368_3_output_depth]: /experiments/images/JNet_368_3_output_depth.png
[JNet_368_3_output_plane]: /experiments/images/JNet_368_3_output_plane.png
[JNet_368_4_label_depth]: /experiments/images/JNet_368_4_label_depth.png
[JNet_368_4_label_plane]: /experiments/images/JNet_368_4_label_plane.png
[JNet_368_4_original_depth]: /experiments/images/JNet_368_4_original_depth.png
[JNet_368_4_original_plane]: /experiments/images/JNet_368_4_original_plane.png
[JNet_368_4_output_depth]: /experiments/images/JNet_368_4_output_depth.png
[JNet_368_4_output_plane]: /experiments/images/JNet_368_4_output_plane.png
[JNet_368_beads_001_roi000_original_depth]: /experiments/images/JNet_368_beads_001_roi000_original_depth.png
[JNet_368_beads_001_roi000_output_depth]: /experiments/images/JNet_368_beads_001_roi000_output_depth.png
[JNet_368_beads_001_roi000_reconst_depth]: /experiments/images/JNet_368_beads_001_roi000_reconst_depth.png
[JNet_368_beads_001_roi001_original_depth]: /experiments/images/JNet_368_beads_001_roi001_original_depth.png
[JNet_368_beads_001_roi001_output_depth]: /experiments/images/JNet_368_beads_001_roi001_output_depth.png
[JNet_368_beads_001_roi001_reconst_depth]: /experiments/images/JNet_368_beads_001_roi001_reconst_depth.png
[JNet_368_beads_001_roi002_original_depth]: /experiments/images/JNet_368_beads_001_roi002_original_depth.png
[JNet_368_beads_001_roi002_output_depth]: /experiments/images/JNet_368_beads_001_roi002_output_depth.png
[JNet_368_beads_001_roi002_reconst_depth]: /experiments/images/JNet_368_beads_001_roi002_reconst_depth.png
[JNet_368_beads_001_roi003_original_depth]: /experiments/images/JNet_368_beads_001_roi003_original_depth.png
[JNet_368_beads_001_roi003_output_depth]: /experiments/images/JNet_368_beads_001_roi003_output_depth.png
[JNet_368_beads_001_roi003_reconst_depth]: /experiments/images/JNet_368_beads_001_roi003_reconst_depth.png
[JNet_368_beads_001_roi004_original_depth]: /experiments/images/JNet_368_beads_001_roi004_original_depth.png
[JNet_368_beads_001_roi004_output_depth]: /experiments/images/JNet_368_beads_001_roi004_output_depth.png
[JNet_368_beads_001_roi004_reconst_depth]: /experiments/images/JNet_368_beads_001_roi004_reconst_depth.png
[JNet_368_beads_002_roi000_original_depth]: /experiments/images/JNet_368_beads_002_roi000_original_depth.png
[JNet_368_beads_002_roi000_output_depth]: /experiments/images/JNet_368_beads_002_roi000_output_depth.png
[JNet_368_beads_002_roi000_reconst_depth]: /experiments/images/JNet_368_beads_002_roi000_reconst_depth.png
[JNet_368_beads_002_roi001_original_depth]: /experiments/images/JNet_368_beads_002_roi001_original_depth.png
[JNet_368_beads_002_roi001_output_depth]: /experiments/images/JNet_368_beads_002_roi001_output_depth.png
[JNet_368_beads_002_roi001_reconst_depth]: /experiments/images/JNet_368_beads_002_roi001_reconst_depth.png
[JNet_368_beads_002_roi002_original_depth]: /experiments/images/JNet_368_beads_002_roi002_original_depth.png
[JNet_368_beads_002_roi002_output_depth]: /experiments/images/JNet_368_beads_002_roi002_output_depth.png
[JNet_368_beads_002_roi002_reconst_depth]: /experiments/images/JNet_368_beads_002_roi002_reconst_depth.png
[JNet_368_psf_post]: /experiments/images/JNet_368_psf_post.png
[JNet_368_psf_pre]: /experiments/images/JNet_368_psf_pre.png
[finetuned]: /experiments/tmp/JNet_368_train.png
[pretrained_model]: /experiments/tmp/JNet_367_pretrain_train.png
