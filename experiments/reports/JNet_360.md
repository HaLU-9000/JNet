



# JNet_360 Report
  
the parameters to replicate the results of JNet_360. noise added.  
pretrained model : JNet_359_pretrain
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

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.02673286199569702, mean BCE: 0.09563720226287842
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_359_pretrain_0_original_plane]|![JNet_359_pretrain_0_output_plane]|![JNet_359_pretrain_0_label_plane]|
  
MSE: 0.02378159947693348, BCE: 0.08451894670724869  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_359_pretrain_0_original_depth]|![JNet_359_pretrain_0_output_depth]|![JNet_359_pretrain_0_label_depth]|
  
MSE: 0.02378159947693348, BCE: 0.08451894670724869  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_359_pretrain_1_original_plane]|![JNet_359_pretrain_1_output_plane]|![JNet_359_pretrain_1_label_plane]|
  
MSE: 0.02558640018105507, BCE: 0.09156189113855362  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_359_pretrain_1_original_depth]|![JNet_359_pretrain_1_output_depth]|![JNet_359_pretrain_1_label_depth]|
  
MSE: 0.02558640018105507, BCE: 0.09156189113855362  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_359_pretrain_2_original_plane]|![JNet_359_pretrain_2_output_plane]|![JNet_359_pretrain_2_label_plane]|
  
MSE: 0.01994306407868862, BCE: 0.072027787566185  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_359_pretrain_2_original_depth]|![JNet_359_pretrain_2_output_depth]|![JNet_359_pretrain_2_label_depth]|
  
MSE: 0.01994306407868862, BCE: 0.072027787566185  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_359_pretrain_3_original_plane]|![JNet_359_pretrain_3_output_plane]|![JNet_359_pretrain_3_label_plane]|
  
MSE: 0.04114470258355141, BCE: 0.1461484581232071  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_359_pretrain_3_original_depth]|![JNet_359_pretrain_3_output_depth]|![JNet_359_pretrain_3_label_depth]|
  
MSE: 0.04114470258355141, BCE: 0.1461484581232071  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_359_pretrain_4_original_plane]|![JNet_359_pretrain_4_output_plane]|![JNet_359_pretrain_4_label_plane]|
  
MSE: 0.02320854552090168, BCE: 0.08392893522977829  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_359_pretrain_4_original_depth]|![JNet_359_pretrain_4_output_depth]|![JNet_359_pretrain_4_label_depth]|
  
MSE: 0.02320854552090168, BCE: 0.08392893522977829  
  
mean MSE: 0.03360358625650406, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_360_0_original_plane]|![JNet_360_0_output_plane]|![JNet_360_0_label_plane]|
  
MSE: 0.037226419895887375, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_360_0_original_depth]|![JNet_360_0_output_depth]|![JNet_360_0_label_depth]|
  
MSE: 0.037226419895887375, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_360_1_original_plane]|![JNet_360_1_output_plane]|![JNet_360_1_label_plane]|
  
MSE: 0.03876720741391182, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_360_1_original_depth]|![JNet_360_1_output_depth]|![JNet_360_1_label_depth]|
  
MSE: 0.03876720741391182, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_360_2_original_plane]|![JNet_360_2_output_plane]|![JNet_360_2_label_plane]|
  
MSE: 0.030803918838500977, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_360_2_original_depth]|![JNet_360_2_output_depth]|![JNet_360_2_label_depth]|
  
MSE: 0.030803918838500977, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_360_3_original_plane]|![JNet_360_3_output_plane]|![JNet_360_3_label_plane]|
  
MSE: 0.03111104853451252, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_360_3_original_depth]|![JNet_360_3_output_depth]|![JNet_360_3_label_depth]|
  
MSE: 0.03111104853451252, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_360_4_original_plane]|![JNet_360_4_output_plane]|![JNet_360_4_label_plane]|
  
MSE: 0.0301093477755785, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_360_4_original_depth]|![JNet_360_4_output_depth]|![JNet_360_4_label_depth]|
  
MSE: 0.0301093477755785, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_359_pretrain_beads_001_roi000_original_depth]|![JNet_359_pretrain_beads_001_roi000_output_depth]|![JNet_359_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 15.838875000000003, MSE: 0.0031456761062145233, quantized loss: 0.00397492665797472  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_359_pretrain_beads_001_roi001_original_depth]|![JNet_359_pretrain_beads_001_roi001_output_depth]|![JNet_359_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 24.761500000000005, MSE: 0.004946870729327202, quantized loss: 0.004903579596430063  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_359_pretrain_beads_001_roi002_original_depth]|![JNet_359_pretrain_beads_001_roi002_output_depth]|![JNet_359_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 16.094500000000004, MSE: 0.003204982029274106, quantized loss: 0.004451529588550329  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_359_pretrain_beads_001_roi003_original_depth]|![JNet_359_pretrain_beads_001_roi003_output_depth]|![JNet_359_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 26.109500000000008, MSE: 0.004797038156539202, quantized loss: 0.004984049126505852  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_359_pretrain_beads_001_roi004_original_depth]|![JNet_359_pretrain_beads_001_roi004_output_depth]|![JNet_359_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 17.185125000000003, MSE: 0.0038723060861229897, quantized loss: 0.00431645754724741  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_359_pretrain_beads_002_roi000_original_depth]|![JNet_359_pretrain_beads_002_roi000_output_depth]|![JNet_359_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 18.368875000000003, MSE: 0.004349065478891134, quantized loss: 0.00472352234646678  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_359_pretrain_beads_002_roi001_original_depth]|![JNet_359_pretrain_beads_002_roi001_output_depth]|![JNet_359_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 17.670000000000005, MSE: 0.003559614298865199, quantized loss: 0.004888023715466261  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_359_pretrain_beads_002_roi002_original_depth]|![JNet_359_pretrain_beads_002_roi002_output_depth]|![JNet_359_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 17.415875000000003, MSE: 0.0038492947351187468, quantized loss: 0.004360219929367304  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_360_beads_001_roi000_original_depth]|![JNet_360_beads_001_roi000_output_depth]|![JNet_360_beads_001_roi000_reconst_depth]|
  
volume: 9.874375000000002, MSE: 0.00028677392401732504, quantized loss: 8.748757863941137e-06  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_360_beads_001_roi001_original_depth]|![JNet_360_beads_001_roi001_output_depth]|![JNet_360_beads_001_roi001_reconst_depth]|
  
volume: 15.006125000000004, MSE: 0.0007692772778682411, quantized loss: 1.3529033822123893e-05  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_360_beads_001_roi002_original_depth]|![JNet_360_beads_001_roi002_output_depth]|![JNet_360_beads_001_roi002_reconst_depth]|
  
volume: 9.862750000000002, MSE: 0.00020690153178293258, quantized loss: 7.742641173535958e-06  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_360_beads_001_roi003_original_depth]|![JNet_360_beads_001_roi003_output_depth]|![JNet_360_beads_001_roi003_reconst_depth]|
  
volume: 16.318375000000003, MSE: 0.00046230119187384844, quantized loss: 1.3573907381214667e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_360_beads_001_roi004_original_depth]|![JNet_360_beads_001_roi004_output_depth]|![JNet_360_beads_001_roi004_reconst_depth]|
  
volume: 10.792250000000003, MSE: 0.00016767151828389615, quantized loss: 8.645506568427663e-06  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_360_beads_002_roi000_original_depth]|![JNet_360_beads_002_roi000_output_depth]|![JNet_360_beads_002_roi000_reconst_depth]|
  
volume: 11.527000000000003, MSE: 0.00015657106996513903, quantized loss: 9.573788702255115e-06  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_360_beads_002_roi001_original_depth]|![JNet_360_beads_002_roi001_output_depth]|![JNet_360_beads_002_roi001_reconst_depth]|
  
volume: 10.513625000000003, MSE: 0.00018471053044777364, quantized loss: 8.98726375453407e-06  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_360_beads_002_roi002_original_depth]|![JNet_360_beads_002_roi002_output_depth]|![JNet_360_beads_002_roi002_reconst_depth]|
  
volume: 10.930375000000003, MSE: 0.00016655279614496976, quantized loss: 8.548046025680378e-06  

|pre|post|
| :---: | :---: |
|![JNet_360_psf_pre]|![JNet_360_psf_post]|
  



[JNet_359_pretrain_0_label_depth]: /experiments/images/JNet_359_pretrain_0_label_depth.png
[JNet_359_pretrain_0_label_plane]: /experiments/images/JNet_359_pretrain_0_label_plane.png
[JNet_359_pretrain_0_original_depth]: /experiments/images/JNet_359_pretrain_0_original_depth.png
[JNet_359_pretrain_0_original_plane]: /experiments/images/JNet_359_pretrain_0_original_plane.png
[JNet_359_pretrain_0_output_depth]: /experiments/images/JNet_359_pretrain_0_output_depth.png
[JNet_359_pretrain_0_output_plane]: /experiments/images/JNet_359_pretrain_0_output_plane.png
[JNet_359_pretrain_1_label_depth]: /experiments/images/JNet_359_pretrain_1_label_depth.png
[JNet_359_pretrain_1_label_plane]: /experiments/images/JNet_359_pretrain_1_label_plane.png
[JNet_359_pretrain_1_original_depth]: /experiments/images/JNet_359_pretrain_1_original_depth.png
[JNet_359_pretrain_1_original_plane]: /experiments/images/JNet_359_pretrain_1_original_plane.png
[JNet_359_pretrain_1_output_depth]: /experiments/images/JNet_359_pretrain_1_output_depth.png
[JNet_359_pretrain_1_output_plane]: /experiments/images/JNet_359_pretrain_1_output_plane.png
[JNet_359_pretrain_2_label_depth]: /experiments/images/JNet_359_pretrain_2_label_depth.png
[JNet_359_pretrain_2_label_plane]: /experiments/images/JNet_359_pretrain_2_label_plane.png
[JNet_359_pretrain_2_original_depth]: /experiments/images/JNet_359_pretrain_2_original_depth.png
[JNet_359_pretrain_2_original_plane]: /experiments/images/JNet_359_pretrain_2_original_plane.png
[JNet_359_pretrain_2_output_depth]: /experiments/images/JNet_359_pretrain_2_output_depth.png
[JNet_359_pretrain_2_output_plane]: /experiments/images/JNet_359_pretrain_2_output_plane.png
[JNet_359_pretrain_3_label_depth]: /experiments/images/JNet_359_pretrain_3_label_depth.png
[JNet_359_pretrain_3_label_plane]: /experiments/images/JNet_359_pretrain_3_label_plane.png
[JNet_359_pretrain_3_original_depth]: /experiments/images/JNet_359_pretrain_3_original_depth.png
[JNet_359_pretrain_3_original_plane]: /experiments/images/JNet_359_pretrain_3_original_plane.png
[JNet_359_pretrain_3_output_depth]: /experiments/images/JNet_359_pretrain_3_output_depth.png
[JNet_359_pretrain_3_output_plane]: /experiments/images/JNet_359_pretrain_3_output_plane.png
[JNet_359_pretrain_4_label_depth]: /experiments/images/JNet_359_pretrain_4_label_depth.png
[JNet_359_pretrain_4_label_plane]: /experiments/images/JNet_359_pretrain_4_label_plane.png
[JNet_359_pretrain_4_original_depth]: /experiments/images/JNet_359_pretrain_4_original_depth.png
[JNet_359_pretrain_4_original_plane]: /experiments/images/JNet_359_pretrain_4_original_plane.png
[JNet_359_pretrain_4_output_depth]: /experiments/images/JNet_359_pretrain_4_output_depth.png
[JNet_359_pretrain_4_output_plane]: /experiments/images/JNet_359_pretrain_4_output_plane.png
[JNet_359_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi000_original_depth.png
[JNet_359_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi000_output_depth.png
[JNet_359_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi000_reconst_depth.png
[JNet_359_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi001_original_depth.png
[JNet_359_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi001_output_depth.png
[JNet_359_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi001_reconst_depth.png
[JNet_359_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi002_original_depth.png
[JNet_359_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi002_output_depth.png
[JNet_359_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi002_reconst_depth.png
[JNet_359_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi003_original_depth.png
[JNet_359_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi003_output_depth.png
[JNet_359_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi003_reconst_depth.png
[JNet_359_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi004_original_depth.png
[JNet_359_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi004_output_depth.png
[JNet_359_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_359_pretrain_beads_001_roi004_reconst_depth.png
[JNet_359_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_359_pretrain_beads_002_roi000_original_depth.png
[JNet_359_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_359_pretrain_beads_002_roi000_output_depth.png
[JNet_359_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_359_pretrain_beads_002_roi000_reconst_depth.png
[JNet_359_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_359_pretrain_beads_002_roi001_original_depth.png
[JNet_359_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_359_pretrain_beads_002_roi001_output_depth.png
[JNet_359_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_359_pretrain_beads_002_roi001_reconst_depth.png
[JNet_359_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_359_pretrain_beads_002_roi002_original_depth.png
[JNet_359_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_359_pretrain_beads_002_roi002_output_depth.png
[JNet_359_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_359_pretrain_beads_002_roi002_reconst_depth.png
[JNet_360_0_label_depth]: /experiments/images/JNet_360_0_label_depth.png
[JNet_360_0_label_plane]: /experiments/images/JNet_360_0_label_plane.png
[JNet_360_0_original_depth]: /experiments/images/JNet_360_0_original_depth.png
[JNet_360_0_original_plane]: /experiments/images/JNet_360_0_original_plane.png
[JNet_360_0_output_depth]: /experiments/images/JNet_360_0_output_depth.png
[JNet_360_0_output_plane]: /experiments/images/JNet_360_0_output_plane.png
[JNet_360_1_label_depth]: /experiments/images/JNet_360_1_label_depth.png
[JNet_360_1_label_plane]: /experiments/images/JNet_360_1_label_plane.png
[JNet_360_1_original_depth]: /experiments/images/JNet_360_1_original_depth.png
[JNet_360_1_original_plane]: /experiments/images/JNet_360_1_original_plane.png
[JNet_360_1_output_depth]: /experiments/images/JNet_360_1_output_depth.png
[JNet_360_1_output_plane]: /experiments/images/JNet_360_1_output_plane.png
[JNet_360_2_label_depth]: /experiments/images/JNet_360_2_label_depth.png
[JNet_360_2_label_plane]: /experiments/images/JNet_360_2_label_plane.png
[JNet_360_2_original_depth]: /experiments/images/JNet_360_2_original_depth.png
[JNet_360_2_original_plane]: /experiments/images/JNet_360_2_original_plane.png
[JNet_360_2_output_depth]: /experiments/images/JNet_360_2_output_depth.png
[JNet_360_2_output_plane]: /experiments/images/JNet_360_2_output_plane.png
[JNet_360_3_label_depth]: /experiments/images/JNet_360_3_label_depth.png
[JNet_360_3_label_plane]: /experiments/images/JNet_360_3_label_plane.png
[JNet_360_3_original_depth]: /experiments/images/JNet_360_3_original_depth.png
[JNet_360_3_original_plane]: /experiments/images/JNet_360_3_original_plane.png
[JNet_360_3_output_depth]: /experiments/images/JNet_360_3_output_depth.png
[JNet_360_3_output_plane]: /experiments/images/JNet_360_3_output_plane.png
[JNet_360_4_label_depth]: /experiments/images/JNet_360_4_label_depth.png
[JNet_360_4_label_plane]: /experiments/images/JNet_360_4_label_plane.png
[JNet_360_4_original_depth]: /experiments/images/JNet_360_4_original_depth.png
[JNet_360_4_original_plane]: /experiments/images/JNet_360_4_original_plane.png
[JNet_360_4_output_depth]: /experiments/images/JNet_360_4_output_depth.png
[JNet_360_4_output_plane]: /experiments/images/JNet_360_4_output_plane.png
[JNet_360_beads_001_roi000_original_depth]: /experiments/images/JNet_360_beads_001_roi000_original_depth.png
[JNet_360_beads_001_roi000_output_depth]: /experiments/images/JNet_360_beads_001_roi000_output_depth.png
[JNet_360_beads_001_roi000_reconst_depth]: /experiments/images/JNet_360_beads_001_roi000_reconst_depth.png
[JNet_360_beads_001_roi001_original_depth]: /experiments/images/JNet_360_beads_001_roi001_original_depth.png
[JNet_360_beads_001_roi001_output_depth]: /experiments/images/JNet_360_beads_001_roi001_output_depth.png
[JNet_360_beads_001_roi001_reconst_depth]: /experiments/images/JNet_360_beads_001_roi001_reconst_depth.png
[JNet_360_beads_001_roi002_original_depth]: /experiments/images/JNet_360_beads_001_roi002_original_depth.png
[JNet_360_beads_001_roi002_output_depth]: /experiments/images/JNet_360_beads_001_roi002_output_depth.png
[JNet_360_beads_001_roi002_reconst_depth]: /experiments/images/JNet_360_beads_001_roi002_reconst_depth.png
[JNet_360_beads_001_roi003_original_depth]: /experiments/images/JNet_360_beads_001_roi003_original_depth.png
[JNet_360_beads_001_roi003_output_depth]: /experiments/images/JNet_360_beads_001_roi003_output_depth.png
[JNet_360_beads_001_roi003_reconst_depth]: /experiments/images/JNet_360_beads_001_roi003_reconst_depth.png
[JNet_360_beads_001_roi004_original_depth]: /experiments/images/JNet_360_beads_001_roi004_original_depth.png
[JNet_360_beads_001_roi004_output_depth]: /experiments/images/JNet_360_beads_001_roi004_output_depth.png
[JNet_360_beads_001_roi004_reconst_depth]: /experiments/images/JNet_360_beads_001_roi004_reconst_depth.png
[JNet_360_beads_002_roi000_original_depth]: /experiments/images/JNet_360_beads_002_roi000_original_depth.png
[JNet_360_beads_002_roi000_output_depth]: /experiments/images/JNet_360_beads_002_roi000_output_depth.png
[JNet_360_beads_002_roi000_reconst_depth]: /experiments/images/JNet_360_beads_002_roi000_reconst_depth.png
[JNet_360_beads_002_roi001_original_depth]: /experiments/images/JNet_360_beads_002_roi001_original_depth.png
[JNet_360_beads_002_roi001_output_depth]: /experiments/images/JNet_360_beads_002_roi001_output_depth.png
[JNet_360_beads_002_roi001_reconst_depth]: /experiments/images/JNet_360_beads_002_roi001_reconst_depth.png
[JNet_360_beads_002_roi002_original_depth]: /experiments/images/JNet_360_beads_002_roi002_original_depth.png
[JNet_360_beads_002_roi002_output_depth]: /experiments/images/JNet_360_beads_002_roi002_output_depth.png
[JNet_360_beads_002_roi002_reconst_depth]: /experiments/images/JNet_360_beads_002_roi002_reconst_depth.png
[JNet_360_psf_post]: /experiments/images/JNet_360_psf_post.png
[JNet_360_psf_pre]: /experiments/images/JNet_360_psf_pre.png
[finetuned]: /experiments/tmp/JNet_360_train.png
[pretrained_model]: /experiments/tmp/JNet_359_pretrain_train.png
