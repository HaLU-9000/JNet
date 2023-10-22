



# JNet_374 Report
  
the parameters to replicate the results of JNet_374. noise added  
pretrained model : JNet_373_pretrain
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
|background|0.001||
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
  
mean MSE: 0.026402929797768593, mean BCE: 0.11744366586208344
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_373_pretrain_0_original_plane]|![JNet_373_pretrain_0_output_plane]|![JNet_373_pretrain_0_label_plane]|
  
MSE: 0.032738443464040756, BCE: 0.17509086430072784  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_373_pretrain_0_original_depth]|![JNet_373_pretrain_0_output_depth]|![JNet_373_pretrain_0_label_depth]|
  
MSE: 0.032738443464040756, BCE: 0.17509086430072784  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_373_pretrain_1_original_plane]|![JNet_373_pretrain_1_output_plane]|![JNet_373_pretrain_1_label_plane]|
  
MSE: 0.021003369241952896, BCE: 0.07386370003223419  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_373_pretrain_1_original_depth]|![JNet_373_pretrain_1_output_depth]|![JNet_373_pretrain_1_label_depth]|
  
MSE: 0.021003369241952896, BCE: 0.07386370003223419  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_373_pretrain_2_original_plane]|![JNet_373_pretrain_2_output_plane]|![JNet_373_pretrain_2_label_plane]|
  
MSE: 0.02139219455420971, BCE: 0.0771569311618805  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_373_pretrain_2_original_depth]|![JNet_373_pretrain_2_output_depth]|![JNet_373_pretrain_2_label_depth]|
  
MSE: 0.02139219455420971, BCE: 0.0771569311618805  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_373_pretrain_3_original_plane]|![JNet_373_pretrain_3_output_plane]|![JNet_373_pretrain_3_label_plane]|
  
MSE: 0.035734258592128754, BCE: 0.18727128207683563  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_373_pretrain_3_original_depth]|![JNet_373_pretrain_3_output_depth]|![JNet_373_pretrain_3_label_depth]|
  
MSE: 0.035734258592128754, BCE: 0.18727128207683563  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_373_pretrain_4_original_plane]|![JNet_373_pretrain_4_output_plane]|![JNet_373_pretrain_4_label_plane]|
  
MSE: 0.021146375685930252, BCE: 0.07383556663990021  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_373_pretrain_4_original_depth]|![JNet_373_pretrain_4_output_depth]|![JNet_373_pretrain_4_label_depth]|
  
MSE: 0.021146375685930252, BCE: 0.07383556663990021  
  
mean MSE: 0.036701612174510956, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_374_0_original_plane]|![JNet_374_0_output_plane]|![JNet_374_0_label_plane]|
  
MSE: 0.03568306192755699, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_374_0_original_depth]|![JNet_374_0_output_depth]|![JNet_374_0_label_depth]|
  
MSE: 0.03568306192755699, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_374_1_original_plane]|![JNet_374_1_output_plane]|![JNet_374_1_label_plane]|
  
MSE: 0.027188628911972046, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_374_1_original_depth]|![JNet_374_1_output_depth]|![JNet_374_1_label_depth]|
  
MSE: 0.027188628911972046, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_374_2_original_plane]|![JNet_374_2_output_plane]|![JNet_374_2_label_plane]|
  
MSE: 0.04530054330825806, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_374_2_original_depth]|![JNet_374_2_output_depth]|![JNet_374_2_label_depth]|
  
MSE: 0.04530054330825806, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_374_3_original_plane]|![JNet_374_3_output_plane]|![JNet_374_3_label_plane]|
  
MSE: 0.03367147594690323, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_374_3_original_depth]|![JNet_374_3_output_depth]|![JNet_374_3_label_depth]|
  
MSE: 0.03367147594690323, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_374_4_original_plane]|![JNet_374_4_output_plane]|![JNet_374_4_label_plane]|
  
MSE: 0.04166434332728386, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_374_4_original_depth]|![JNet_374_4_output_depth]|![JNet_374_4_label_depth]|
  
MSE: 0.04166434332728386, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_373_pretrain_beads_001_roi000_original_depth]|![JNet_373_pretrain_beads_001_roi000_output_depth]|![JNet_373_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 18.303375000000003, MSE: 0.0018873753724619746, quantized loss: 0.002416016301140189  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_373_pretrain_beads_001_roi001_original_depth]|![JNet_373_pretrain_beads_001_roi001_output_depth]|![JNet_373_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 27.402125000000005, MSE: 0.0026711218524724245, quantized loss: 0.0034918389283120632  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_373_pretrain_beads_001_roi002_original_depth]|![JNet_373_pretrain_beads_001_roi002_output_depth]|![JNet_373_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 17.882250000000003, MSE: 0.001866805017925799, quantized loss: 0.0023854251485317945  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_373_pretrain_beads_001_roi003_original_depth]|![JNet_373_pretrain_beads_001_roi003_output_depth]|![JNet_373_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 27.278875000000006, MSE: 0.002588223433122039, quantized loss: 0.003461409593001008  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_373_pretrain_beads_001_roi004_original_depth]|![JNet_373_pretrain_beads_001_roi004_output_depth]|![JNet_373_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 19.562125000000005, MSE: 0.0022467339877039194, quantized loss: 0.0025272828061133623  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_373_pretrain_beads_002_roi000_original_depth]|![JNet_373_pretrain_beads_002_roi000_output_depth]|![JNet_373_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 20.504750000000005, MSE: 0.00236139097250998, quantized loss: 0.002534235594794154  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_373_pretrain_beads_002_roi001_original_depth]|![JNet_373_pretrain_beads_002_roi001_output_depth]|![JNet_373_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 18.596750000000004, MSE: 0.0019067522371187806, quantized loss: 0.0025136503390967846  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_373_pretrain_beads_002_roi002_original_depth]|![JNet_373_pretrain_beads_002_roi002_output_depth]|![JNet_373_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 19.523500000000006, MSE: 0.0021361580584198236, quantized loss: 0.002531774342060089  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_374_beads_001_roi000_original_depth]|![JNet_374_beads_001_roi000_output_depth]|![JNet_374_beads_001_roi000_reconst_depth]|
  
volume: 0.0, MSE: 0.009335086680948734, quantized loss: 0.0  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_374_beads_001_roi001_original_depth]|![JNet_374_beads_001_roi001_output_depth]|![JNet_374_beads_001_roi001_reconst_depth]|
  
volume: 0.0, MSE: 0.014853984117507935, quantized loss: 0.0  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_374_beads_001_roi002_original_depth]|![JNet_374_beads_001_roi002_output_depth]|![JNet_374_beads_001_roi002_reconst_depth]|
  
volume: 0.0, MSE: 0.009056191891431808, quantized loss: 0.0  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_374_beads_001_roi003_original_depth]|![JNet_374_beads_001_roi003_output_depth]|![JNet_374_beads_001_roi003_reconst_depth]|
  
volume: 0.0, MSE: 0.015940722078084946, quantized loss: 0.0  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_374_beads_001_roi004_original_depth]|![JNet_374_beads_001_roi004_output_depth]|![JNet_374_beads_001_roi004_reconst_depth]|
  
volume: 0.0, MSE: 0.010938753373920918, quantized loss: 0.0  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_374_beads_002_roi000_original_depth]|![JNet_374_beads_002_roi000_output_depth]|![JNet_374_beads_002_roi000_reconst_depth]|
  
volume: 0.0, MSE: 0.012265910394489765, quantized loss: 0.0  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_374_beads_002_roi001_original_depth]|![JNet_374_beads_002_roi001_output_depth]|![JNet_374_beads_002_roi001_reconst_depth]|
  
volume: 0.0, MSE: 0.010307948105037212, quantized loss: 0.0  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_374_beads_002_roi002_original_depth]|![JNet_374_beads_002_roi002_output_depth]|![JNet_374_beads_002_roi002_reconst_depth]|
  
volume: 0.0, MSE: 0.011078033596277237, quantized loss: 0.0  

|pre|post|
| :---: | :---: |
|![JNet_374_psf_pre]|![JNet_374_psf_post]|
  



[JNet_373_pretrain_0_label_depth]: /experiments/images/JNet_373_pretrain_0_label_depth.png
[JNet_373_pretrain_0_label_plane]: /experiments/images/JNet_373_pretrain_0_label_plane.png
[JNet_373_pretrain_0_original_depth]: /experiments/images/JNet_373_pretrain_0_original_depth.png
[JNet_373_pretrain_0_original_plane]: /experiments/images/JNet_373_pretrain_0_original_plane.png
[JNet_373_pretrain_0_output_depth]: /experiments/images/JNet_373_pretrain_0_output_depth.png
[JNet_373_pretrain_0_output_plane]: /experiments/images/JNet_373_pretrain_0_output_plane.png
[JNet_373_pretrain_1_label_depth]: /experiments/images/JNet_373_pretrain_1_label_depth.png
[JNet_373_pretrain_1_label_plane]: /experiments/images/JNet_373_pretrain_1_label_plane.png
[JNet_373_pretrain_1_original_depth]: /experiments/images/JNet_373_pretrain_1_original_depth.png
[JNet_373_pretrain_1_original_plane]: /experiments/images/JNet_373_pretrain_1_original_plane.png
[JNet_373_pretrain_1_output_depth]: /experiments/images/JNet_373_pretrain_1_output_depth.png
[JNet_373_pretrain_1_output_plane]: /experiments/images/JNet_373_pretrain_1_output_plane.png
[JNet_373_pretrain_2_label_depth]: /experiments/images/JNet_373_pretrain_2_label_depth.png
[JNet_373_pretrain_2_label_plane]: /experiments/images/JNet_373_pretrain_2_label_plane.png
[JNet_373_pretrain_2_original_depth]: /experiments/images/JNet_373_pretrain_2_original_depth.png
[JNet_373_pretrain_2_original_plane]: /experiments/images/JNet_373_pretrain_2_original_plane.png
[JNet_373_pretrain_2_output_depth]: /experiments/images/JNet_373_pretrain_2_output_depth.png
[JNet_373_pretrain_2_output_plane]: /experiments/images/JNet_373_pretrain_2_output_plane.png
[JNet_373_pretrain_3_label_depth]: /experiments/images/JNet_373_pretrain_3_label_depth.png
[JNet_373_pretrain_3_label_plane]: /experiments/images/JNet_373_pretrain_3_label_plane.png
[JNet_373_pretrain_3_original_depth]: /experiments/images/JNet_373_pretrain_3_original_depth.png
[JNet_373_pretrain_3_original_plane]: /experiments/images/JNet_373_pretrain_3_original_plane.png
[JNet_373_pretrain_3_output_depth]: /experiments/images/JNet_373_pretrain_3_output_depth.png
[JNet_373_pretrain_3_output_plane]: /experiments/images/JNet_373_pretrain_3_output_plane.png
[JNet_373_pretrain_4_label_depth]: /experiments/images/JNet_373_pretrain_4_label_depth.png
[JNet_373_pretrain_4_label_plane]: /experiments/images/JNet_373_pretrain_4_label_plane.png
[JNet_373_pretrain_4_original_depth]: /experiments/images/JNet_373_pretrain_4_original_depth.png
[JNet_373_pretrain_4_original_plane]: /experiments/images/JNet_373_pretrain_4_original_plane.png
[JNet_373_pretrain_4_output_depth]: /experiments/images/JNet_373_pretrain_4_output_depth.png
[JNet_373_pretrain_4_output_plane]: /experiments/images/JNet_373_pretrain_4_output_plane.png
[JNet_373_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi000_original_depth.png
[JNet_373_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi000_output_depth.png
[JNet_373_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi000_reconst_depth.png
[JNet_373_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi001_original_depth.png
[JNet_373_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi001_output_depth.png
[JNet_373_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi001_reconst_depth.png
[JNet_373_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi002_original_depth.png
[JNet_373_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi002_output_depth.png
[JNet_373_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi002_reconst_depth.png
[JNet_373_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi003_original_depth.png
[JNet_373_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi003_output_depth.png
[JNet_373_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi003_reconst_depth.png
[JNet_373_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi004_original_depth.png
[JNet_373_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi004_output_depth.png
[JNet_373_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_373_pretrain_beads_001_roi004_reconst_depth.png
[JNet_373_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_373_pretrain_beads_002_roi000_original_depth.png
[JNet_373_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_373_pretrain_beads_002_roi000_output_depth.png
[JNet_373_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_373_pretrain_beads_002_roi000_reconst_depth.png
[JNet_373_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_373_pretrain_beads_002_roi001_original_depth.png
[JNet_373_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_373_pretrain_beads_002_roi001_output_depth.png
[JNet_373_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_373_pretrain_beads_002_roi001_reconst_depth.png
[JNet_373_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_373_pretrain_beads_002_roi002_original_depth.png
[JNet_373_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_373_pretrain_beads_002_roi002_output_depth.png
[JNet_373_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_373_pretrain_beads_002_roi002_reconst_depth.png
[JNet_374_0_label_depth]: /experiments/images/JNet_374_0_label_depth.png
[JNet_374_0_label_plane]: /experiments/images/JNet_374_0_label_plane.png
[JNet_374_0_original_depth]: /experiments/images/JNet_374_0_original_depth.png
[JNet_374_0_original_plane]: /experiments/images/JNet_374_0_original_plane.png
[JNet_374_0_output_depth]: /experiments/images/JNet_374_0_output_depth.png
[JNet_374_0_output_plane]: /experiments/images/JNet_374_0_output_plane.png
[JNet_374_1_label_depth]: /experiments/images/JNet_374_1_label_depth.png
[JNet_374_1_label_plane]: /experiments/images/JNet_374_1_label_plane.png
[JNet_374_1_original_depth]: /experiments/images/JNet_374_1_original_depth.png
[JNet_374_1_original_plane]: /experiments/images/JNet_374_1_original_plane.png
[JNet_374_1_output_depth]: /experiments/images/JNet_374_1_output_depth.png
[JNet_374_1_output_plane]: /experiments/images/JNet_374_1_output_plane.png
[JNet_374_2_label_depth]: /experiments/images/JNet_374_2_label_depth.png
[JNet_374_2_label_plane]: /experiments/images/JNet_374_2_label_plane.png
[JNet_374_2_original_depth]: /experiments/images/JNet_374_2_original_depth.png
[JNet_374_2_original_plane]: /experiments/images/JNet_374_2_original_plane.png
[JNet_374_2_output_depth]: /experiments/images/JNet_374_2_output_depth.png
[JNet_374_2_output_plane]: /experiments/images/JNet_374_2_output_plane.png
[JNet_374_3_label_depth]: /experiments/images/JNet_374_3_label_depth.png
[JNet_374_3_label_plane]: /experiments/images/JNet_374_3_label_plane.png
[JNet_374_3_original_depth]: /experiments/images/JNet_374_3_original_depth.png
[JNet_374_3_original_plane]: /experiments/images/JNet_374_3_original_plane.png
[JNet_374_3_output_depth]: /experiments/images/JNet_374_3_output_depth.png
[JNet_374_3_output_plane]: /experiments/images/JNet_374_3_output_plane.png
[JNet_374_4_label_depth]: /experiments/images/JNet_374_4_label_depth.png
[JNet_374_4_label_plane]: /experiments/images/JNet_374_4_label_plane.png
[JNet_374_4_original_depth]: /experiments/images/JNet_374_4_original_depth.png
[JNet_374_4_original_plane]: /experiments/images/JNet_374_4_original_plane.png
[JNet_374_4_output_depth]: /experiments/images/JNet_374_4_output_depth.png
[JNet_374_4_output_plane]: /experiments/images/JNet_374_4_output_plane.png
[JNet_374_beads_001_roi000_original_depth]: /experiments/images/JNet_374_beads_001_roi000_original_depth.png
[JNet_374_beads_001_roi000_output_depth]: /experiments/images/JNet_374_beads_001_roi000_output_depth.png
[JNet_374_beads_001_roi000_reconst_depth]: /experiments/images/JNet_374_beads_001_roi000_reconst_depth.png
[JNet_374_beads_001_roi001_original_depth]: /experiments/images/JNet_374_beads_001_roi001_original_depth.png
[JNet_374_beads_001_roi001_output_depth]: /experiments/images/JNet_374_beads_001_roi001_output_depth.png
[JNet_374_beads_001_roi001_reconst_depth]: /experiments/images/JNet_374_beads_001_roi001_reconst_depth.png
[JNet_374_beads_001_roi002_original_depth]: /experiments/images/JNet_374_beads_001_roi002_original_depth.png
[JNet_374_beads_001_roi002_output_depth]: /experiments/images/JNet_374_beads_001_roi002_output_depth.png
[JNet_374_beads_001_roi002_reconst_depth]: /experiments/images/JNet_374_beads_001_roi002_reconst_depth.png
[JNet_374_beads_001_roi003_original_depth]: /experiments/images/JNet_374_beads_001_roi003_original_depth.png
[JNet_374_beads_001_roi003_output_depth]: /experiments/images/JNet_374_beads_001_roi003_output_depth.png
[JNet_374_beads_001_roi003_reconst_depth]: /experiments/images/JNet_374_beads_001_roi003_reconst_depth.png
[JNet_374_beads_001_roi004_original_depth]: /experiments/images/JNet_374_beads_001_roi004_original_depth.png
[JNet_374_beads_001_roi004_output_depth]: /experiments/images/JNet_374_beads_001_roi004_output_depth.png
[JNet_374_beads_001_roi004_reconst_depth]: /experiments/images/JNet_374_beads_001_roi004_reconst_depth.png
[JNet_374_beads_002_roi000_original_depth]: /experiments/images/JNet_374_beads_002_roi000_original_depth.png
[JNet_374_beads_002_roi000_output_depth]: /experiments/images/JNet_374_beads_002_roi000_output_depth.png
[JNet_374_beads_002_roi000_reconst_depth]: /experiments/images/JNet_374_beads_002_roi000_reconst_depth.png
[JNet_374_beads_002_roi001_original_depth]: /experiments/images/JNet_374_beads_002_roi001_original_depth.png
[JNet_374_beads_002_roi001_output_depth]: /experiments/images/JNet_374_beads_002_roi001_output_depth.png
[JNet_374_beads_002_roi001_reconst_depth]: /experiments/images/JNet_374_beads_002_roi001_reconst_depth.png
[JNet_374_beads_002_roi002_original_depth]: /experiments/images/JNet_374_beads_002_roi002_original_depth.png
[JNet_374_beads_002_roi002_output_depth]: /experiments/images/JNet_374_beads_002_roi002_output_depth.png
[JNet_374_beads_002_roi002_reconst_depth]: /experiments/images/JNet_374_beads_002_roi002_reconst_depth.png
[JNet_374_psf_post]: /experiments/images/JNet_374_psf_post.png
[JNet_374_psf_pre]: /experiments/images/JNet_374_psf_pre.png
[finetuned]: /experiments/tmp/JNet_374_train.png
[pretrained_model]: /experiments/tmp/JNet_373_pretrain_train.png
