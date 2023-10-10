



# JNet_350_full_physics Report
  
the parameters to replicate the results of JNet_350  
pretrained model : JNet_349_pretrain
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

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.019873296841979027, mean BCE: 0.07586642354726791
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_349_pretrain_0_original_plane]|![JNet_349_pretrain_0_output_plane]|![JNet_349_pretrain_0_label_plane]|
  
MSE: 0.02481347694993019, BCE: 0.1102108582854271  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_349_pretrain_0_original_depth]|![JNet_349_pretrain_0_output_depth]|![JNet_349_pretrain_0_label_depth]|
  
MSE: 0.02481347694993019, BCE: 0.1102108582854271  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_349_pretrain_1_original_plane]|![JNet_349_pretrain_1_output_plane]|![JNet_349_pretrain_1_label_plane]|
  
MSE: 0.017249174416065216, BCE: 0.062210652977228165  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_349_pretrain_1_original_depth]|![JNet_349_pretrain_1_output_depth]|![JNet_349_pretrain_1_label_depth]|
  
MSE: 0.017249174416065216, BCE: 0.062210652977228165  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_349_pretrain_2_original_plane]|![JNet_349_pretrain_2_output_plane]|![JNet_349_pretrain_2_label_plane]|
  
MSE: 0.020704779773950577, BCE: 0.07845329493284225  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_349_pretrain_2_original_depth]|![JNet_349_pretrain_2_output_depth]|![JNet_349_pretrain_2_label_depth]|
  
MSE: 0.020704779773950577, BCE: 0.07845329493284225  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_349_pretrain_3_original_plane]|![JNet_349_pretrain_3_output_plane]|![JNet_349_pretrain_3_label_plane]|
  
MSE: 0.014767303131520748, BCE: 0.05138268321752548  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_349_pretrain_3_original_depth]|![JNet_349_pretrain_3_output_depth]|![JNet_349_pretrain_3_label_depth]|
  
MSE: 0.014767303131520748, BCE: 0.05138268321752548  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_349_pretrain_4_original_plane]|![JNet_349_pretrain_4_output_plane]|![JNet_349_pretrain_4_label_plane]|
  
MSE: 0.021831754595041275, BCE: 0.07707460224628448  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_349_pretrain_4_original_depth]|![JNet_349_pretrain_4_output_depth]|![JNet_349_pretrain_4_label_depth]|
  
MSE: 0.021831754595041275, BCE: 0.07707460224628448  
  
mean MSE: 0.03056936524808407, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_350_full_physics_0_original_plane]|![JNet_350_full_physics_0_output_plane]|![JNet_350_full_physics_0_label_plane]|
  
MSE: 0.023425210267305374, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_350_full_physics_0_original_depth]|![JNet_350_full_physics_0_output_depth]|![JNet_350_full_physics_0_label_depth]|
  
MSE: 0.023425210267305374, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_350_full_physics_1_original_plane]|![JNet_350_full_physics_1_output_plane]|![JNet_350_full_physics_1_label_plane]|
  
MSE: 0.032218512147665024, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_350_full_physics_1_original_depth]|![JNet_350_full_physics_1_output_depth]|![JNet_350_full_physics_1_label_depth]|
  
MSE: 0.032218512147665024, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_350_full_physics_2_original_plane]|![JNet_350_full_physics_2_output_plane]|![JNet_350_full_physics_2_label_plane]|
  
MSE: 0.023318253457546234, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_350_full_physics_2_original_depth]|![JNet_350_full_physics_2_output_depth]|![JNet_350_full_physics_2_label_depth]|
  
MSE: 0.023318253457546234, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_350_full_physics_3_original_plane]|![JNet_350_full_physics_3_output_plane]|![JNet_350_full_physics_3_label_plane]|
  
MSE: 0.03080722503364086, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_350_full_physics_3_original_depth]|![JNet_350_full_physics_3_output_depth]|![JNet_350_full_physics_3_label_depth]|
  
MSE: 0.03080722503364086, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_350_full_physics_4_original_plane]|![JNet_350_full_physics_4_output_plane]|![JNet_350_full_physics_4_label_plane]|
  
MSE: 0.04307762160897255, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_350_full_physics_4_original_depth]|![JNet_350_full_physics_4_output_depth]|![JNet_350_full_physics_4_label_depth]|
  
MSE: 0.04307762160897255, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_349_pretrain_beads_001_roi000_original_depth]|![JNet_349_pretrain_beads_001_roi000_output_depth]|![JNet_349_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 14.668375000000003, MSE: 0.0012739795492962003, quantized loss: 0.0017105714650824666  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_349_pretrain_beads_001_roi001_original_depth]|![JNet_349_pretrain_beads_001_roi001_output_depth]|![JNet_349_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 21.243000000000006, MSE: 0.002225303091108799, quantized loss: 0.0021745599806308746  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_349_pretrain_beads_001_roi002_original_depth]|![JNet_349_pretrain_beads_001_roi002_output_depth]|![JNet_349_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 14.081625000000003, MSE: 0.001209041103720665, quantized loss: 0.0015950495144352317  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_349_pretrain_beads_001_roi003_original_depth]|![JNet_349_pretrain_beads_001_roi003_output_depth]|![JNet_349_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 23.342125000000006, MSE: 0.0020235225092619658, quantized loss: 0.002378493780270219  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_349_pretrain_beads_001_roi004_original_depth]|![JNet_349_pretrain_beads_001_roi004_output_depth]|![JNet_349_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 15.641625000000003, MSE: 0.0015588415553793311, quantized loss: 0.0017652035458013415  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_349_pretrain_beads_002_roi000_original_depth]|![JNet_349_pretrain_beads_002_roi000_output_depth]|![JNet_349_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 16.377625000000005, MSE: 0.001700330525636673, quantized loss: 0.0017835918115451932  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_349_pretrain_beads_002_roi001_original_depth]|![JNet_349_pretrain_beads_002_roi001_output_depth]|![JNet_349_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 15.139250000000004, MSE: 0.0012852427316829562, quantized loss: 0.0016124690882861614  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_349_pretrain_beads_002_roi002_original_depth]|![JNet_349_pretrain_beads_002_roi002_output_depth]|![JNet_349_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 15.506000000000004, MSE: 0.0014945593429729342, quantized loss: 0.00167764478828758  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_350_full_physics_beads_001_roi000_original_depth]|![JNet_350_full_physics_beads_001_roi000_output_depth]|![JNet_350_full_physics_beads_001_roi000_reconst_depth]|
  
volume: 10.067500000000003, MSE: 0.0004030589770991355, quantized loss: 4.2092495277756825e-05  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_350_full_physics_beads_001_roi001_original_depth]|![JNet_350_full_physics_beads_001_roi001_output_depth]|![JNet_350_full_physics_beads_001_roi001_reconst_depth]|
  
volume: 16.029250000000005, MSE: 0.0008094462100416422, quantized loss: 6.097833829699084e-05  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_350_full_physics_beads_001_roi002_original_depth]|![JNet_350_full_physics_beads_001_roi002_output_depth]|![JNet_350_full_physics_beads_001_roi002_reconst_depth]|
  
volume: 10.091875000000002, MSE: 0.000292979326331988, quantized loss: 4.231282582622953e-05  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_350_full_physics_beads_001_roi003_original_depth]|![JNet_350_full_physics_beads_001_roi003_output_depth]|![JNet_350_full_physics_beads_001_roi003_reconst_depth]|
  
volume: 16.619125000000004, MSE: 0.0006046192138455808, quantized loss: 6.181728531373665e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_350_full_physics_beads_001_roi004_original_depth]|![JNet_350_full_physics_beads_001_roi004_output_depth]|![JNet_350_full_physics_beads_001_roi004_reconst_depth]|
  
volume: 11.042500000000002, MSE: 0.00029620385612361133, quantized loss: 4.026446913485415e-05  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_350_full_physics_beads_002_roi000_original_depth]|![JNet_350_full_physics_beads_002_roi000_output_depth]|![JNet_350_full_physics_beads_002_roi000_reconst_depth]|
  
volume: 11.832625000000002, MSE: 0.0003028359788004309, quantized loss: 4.1795781726250425e-05  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_350_full_physics_beads_002_roi001_original_depth]|![JNet_350_full_physics_beads_002_roi001_output_depth]|![JNet_350_full_physics_beads_002_roi001_reconst_depth]|
  
volume: 10.804750000000002, MSE: 0.00030398997478187084, quantized loss: 4.160175012657419e-05  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_350_full_physics_beads_002_roi002_original_depth]|![JNet_350_full_physics_beads_002_roi002_output_depth]|![JNet_350_full_physics_beads_002_roi002_reconst_depth]|
  
volume: 11.232750000000003, MSE: 0.00029029863071627915, quantized loss: 4.04145430366043e-05  

|pre|post|
| :---: | :---: |
|![JNet_350_full_physics_psf_pre]|![JNet_350_full_physics_psf_post]|
  



[JNet_349_pretrain_0_label_depth]: /experiments/images/JNet_349_pretrain_0_label_depth.png
[JNet_349_pretrain_0_label_plane]: /experiments/images/JNet_349_pretrain_0_label_plane.png
[JNet_349_pretrain_0_original_depth]: /experiments/images/JNet_349_pretrain_0_original_depth.png
[JNet_349_pretrain_0_original_plane]: /experiments/images/JNet_349_pretrain_0_original_plane.png
[JNet_349_pretrain_0_output_depth]: /experiments/images/JNet_349_pretrain_0_output_depth.png
[JNet_349_pretrain_0_output_plane]: /experiments/images/JNet_349_pretrain_0_output_plane.png
[JNet_349_pretrain_1_label_depth]: /experiments/images/JNet_349_pretrain_1_label_depth.png
[JNet_349_pretrain_1_label_plane]: /experiments/images/JNet_349_pretrain_1_label_plane.png
[JNet_349_pretrain_1_original_depth]: /experiments/images/JNet_349_pretrain_1_original_depth.png
[JNet_349_pretrain_1_original_plane]: /experiments/images/JNet_349_pretrain_1_original_plane.png
[JNet_349_pretrain_1_output_depth]: /experiments/images/JNet_349_pretrain_1_output_depth.png
[JNet_349_pretrain_1_output_plane]: /experiments/images/JNet_349_pretrain_1_output_plane.png
[JNet_349_pretrain_2_label_depth]: /experiments/images/JNet_349_pretrain_2_label_depth.png
[JNet_349_pretrain_2_label_plane]: /experiments/images/JNet_349_pretrain_2_label_plane.png
[JNet_349_pretrain_2_original_depth]: /experiments/images/JNet_349_pretrain_2_original_depth.png
[JNet_349_pretrain_2_original_plane]: /experiments/images/JNet_349_pretrain_2_original_plane.png
[JNet_349_pretrain_2_output_depth]: /experiments/images/JNet_349_pretrain_2_output_depth.png
[JNet_349_pretrain_2_output_plane]: /experiments/images/JNet_349_pretrain_2_output_plane.png
[JNet_349_pretrain_3_label_depth]: /experiments/images/JNet_349_pretrain_3_label_depth.png
[JNet_349_pretrain_3_label_plane]: /experiments/images/JNet_349_pretrain_3_label_plane.png
[JNet_349_pretrain_3_original_depth]: /experiments/images/JNet_349_pretrain_3_original_depth.png
[JNet_349_pretrain_3_original_plane]: /experiments/images/JNet_349_pretrain_3_original_plane.png
[JNet_349_pretrain_3_output_depth]: /experiments/images/JNet_349_pretrain_3_output_depth.png
[JNet_349_pretrain_3_output_plane]: /experiments/images/JNet_349_pretrain_3_output_plane.png
[JNet_349_pretrain_4_label_depth]: /experiments/images/JNet_349_pretrain_4_label_depth.png
[JNet_349_pretrain_4_label_plane]: /experiments/images/JNet_349_pretrain_4_label_plane.png
[JNet_349_pretrain_4_original_depth]: /experiments/images/JNet_349_pretrain_4_original_depth.png
[JNet_349_pretrain_4_original_plane]: /experiments/images/JNet_349_pretrain_4_original_plane.png
[JNet_349_pretrain_4_output_depth]: /experiments/images/JNet_349_pretrain_4_output_depth.png
[JNet_349_pretrain_4_output_plane]: /experiments/images/JNet_349_pretrain_4_output_plane.png
[JNet_349_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi000_original_depth.png
[JNet_349_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi000_output_depth.png
[JNet_349_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi000_reconst_depth.png
[JNet_349_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi001_original_depth.png
[JNet_349_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi001_output_depth.png
[JNet_349_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi001_reconst_depth.png
[JNet_349_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi002_original_depth.png
[JNet_349_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi002_output_depth.png
[JNet_349_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi002_reconst_depth.png
[JNet_349_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi003_original_depth.png
[JNet_349_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi003_output_depth.png
[JNet_349_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi003_reconst_depth.png
[JNet_349_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi004_original_depth.png
[JNet_349_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi004_output_depth.png
[JNet_349_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_349_pretrain_beads_001_roi004_reconst_depth.png
[JNet_349_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_349_pretrain_beads_002_roi000_original_depth.png
[JNet_349_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_349_pretrain_beads_002_roi000_output_depth.png
[JNet_349_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_349_pretrain_beads_002_roi000_reconst_depth.png
[JNet_349_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_349_pretrain_beads_002_roi001_original_depth.png
[JNet_349_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_349_pretrain_beads_002_roi001_output_depth.png
[JNet_349_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_349_pretrain_beads_002_roi001_reconst_depth.png
[JNet_349_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_349_pretrain_beads_002_roi002_original_depth.png
[JNet_349_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_349_pretrain_beads_002_roi002_output_depth.png
[JNet_349_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_349_pretrain_beads_002_roi002_reconst_depth.png
[JNet_350_full_physics_0_label_depth]: /experiments/images/JNet_350_full_physics_0_label_depth.png
[JNet_350_full_physics_0_label_plane]: /experiments/images/JNet_350_full_physics_0_label_plane.png
[JNet_350_full_physics_0_original_depth]: /experiments/images/JNet_350_full_physics_0_original_depth.png
[JNet_350_full_physics_0_original_plane]: /experiments/images/JNet_350_full_physics_0_original_plane.png
[JNet_350_full_physics_0_output_depth]: /experiments/images/JNet_350_full_physics_0_output_depth.png
[JNet_350_full_physics_0_output_plane]: /experiments/images/JNet_350_full_physics_0_output_plane.png
[JNet_350_full_physics_1_label_depth]: /experiments/images/JNet_350_full_physics_1_label_depth.png
[JNet_350_full_physics_1_label_plane]: /experiments/images/JNet_350_full_physics_1_label_plane.png
[JNet_350_full_physics_1_original_depth]: /experiments/images/JNet_350_full_physics_1_original_depth.png
[JNet_350_full_physics_1_original_plane]: /experiments/images/JNet_350_full_physics_1_original_plane.png
[JNet_350_full_physics_1_output_depth]: /experiments/images/JNet_350_full_physics_1_output_depth.png
[JNet_350_full_physics_1_output_plane]: /experiments/images/JNet_350_full_physics_1_output_plane.png
[JNet_350_full_physics_2_label_depth]: /experiments/images/JNet_350_full_physics_2_label_depth.png
[JNet_350_full_physics_2_label_plane]: /experiments/images/JNet_350_full_physics_2_label_plane.png
[JNet_350_full_physics_2_original_depth]: /experiments/images/JNet_350_full_physics_2_original_depth.png
[JNet_350_full_physics_2_original_plane]: /experiments/images/JNet_350_full_physics_2_original_plane.png
[JNet_350_full_physics_2_output_depth]: /experiments/images/JNet_350_full_physics_2_output_depth.png
[JNet_350_full_physics_2_output_plane]: /experiments/images/JNet_350_full_physics_2_output_plane.png
[JNet_350_full_physics_3_label_depth]: /experiments/images/JNet_350_full_physics_3_label_depth.png
[JNet_350_full_physics_3_label_plane]: /experiments/images/JNet_350_full_physics_3_label_plane.png
[JNet_350_full_physics_3_original_depth]: /experiments/images/JNet_350_full_physics_3_original_depth.png
[JNet_350_full_physics_3_original_plane]: /experiments/images/JNet_350_full_physics_3_original_plane.png
[JNet_350_full_physics_3_output_depth]: /experiments/images/JNet_350_full_physics_3_output_depth.png
[JNet_350_full_physics_3_output_plane]: /experiments/images/JNet_350_full_physics_3_output_plane.png
[JNet_350_full_physics_4_label_depth]: /experiments/images/JNet_350_full_physics_4_label_depth.png
[JNet_350_full_physics_4_label_plane]: /experiments/images/JNet_350_full_physics_4_label_plane.png
[JNet_350_full_physics_4_original_depth]: /experiments/images/JNet_350_full_physics_4_original_depth.png
[JNet_350_full_physics_4_original_plane]: /experiments/images/JNet_350_full_physics_4_original_plane.png
[JNet_350_full_physics_4_output_depth]: /experiments/images/JNet_350_full_physics_4_output_depth.png
[JNet_350_full_physics_4_output_plane]: /experiments/images/JNet_350_full_physics_4_output_plane.png
[JNet_350_full_physics_beads_001_roi000_original_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi000_original_depth.png
[JNet_350_full_physics_beads_001_roi000_output_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi000_output_depth.png
[JNet_350_full_physics_beads_001_roi000_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi000_reconst_depth.png
[JNet_350_full_physics_beads_001_roi001_original_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi001_original_depth.png
[JNet_350_full_physics_beads_001_roi001_output_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi001_output_depth.png
[JNet_350_full_physics_beads_001_roi001_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi001_reconst_depth.png
[JNet_350_full_physics_beads_001_roi002_original_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi002_original_depth.png
[JNet_350_full_physics_beads_001_roi002_output_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi002_output_depth.png
[JNet_350_full_physics_beads_001_roi002_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi002_reconst_depth.png
[JNet_350_full_physics_beads_001_roi003_original_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi003_original_depth.png
[JNet_350_full_physics_beads_001_roi003_output_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi003_output_depth.png
[JNet_350_full_physics_beads_001_roi003_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi003_reconst_depth.png
[JNet_350_full_physics_beads_001_roi004_original_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi004_original_depth.png
[JNet_350_full_physics_beads_001_roi004_output_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi004_output_depth.png
[JNet_350_full_physics_beads_001_roi004_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi004_reconst_depth.png
[JNet_350_full_physics_beads_002_roi000_original_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi000_original_depth.png
[JNet_350_full_physics_beads_002_roi000_output_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi000_output_depth.png
[JNet_350_full_physics_beads_002_roi000_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi000_reconst_depth.png
[JNet_350_full_physics_beads_002_roi001_original_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi001_original_depth.png
[JNet_350_full_physics_beads_002_roi001_output_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi001_output_depth.png
[JNet_350_full_physics_beads_002_roi001_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi001_reconst_depth.png
[JNet_350_full_physics_beads_002_roi002_original_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi002_original_depth.png
[JNet_350_full_physics_beads_002_roi002_output_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi002_output_depth.png
[JNet_350_full_physics_beads_002_roi002_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi002_reconst_depth.png
[JNet_350_full_physics_psf_post]: /experiments/images/JNet_350_full_physics_psf_post.png
[JNet_350_full_physics_psf_pre]: /experiments/images/JNet_350_full_physics_psf_pre.png
[finetuned]: /experiments/tmp/JNet_350_full_physics_train.png
[pretrained_model]: /experiments/tmp/JNet_349_pretrain_train.png
