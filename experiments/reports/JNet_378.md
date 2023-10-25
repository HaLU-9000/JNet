



# JNet_378 Report
  
the parameters to replicate the results of JNet_378. deterministic background simulation training. no  noise.  
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
  
mean MSE: 0.018587002530694008, mean BCE: 0.06831841915845871
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_0_original_plane]|![JNet_377_pretrain_0_output_plane]|![JNet_377_pretrain_0_label_plane]|
  
MSE: 0.018141984939575195, BCE: 0.0655575841665268  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_0_original_depth]|![JNet_377_pretrain_0_output_depth]|![JNet_377_pretrain_0_label_depth]|
  
MSE: 0.018141984939575195, BCE: 0.0655575841665268  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_1_original_plane]|![JNet_377_pretrain_1_output_plane]|![JNet_377_pretrain_1_label_plane]|
  
MSE: 0.018350714817643166, BCE: 0.06601055711507797  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_1_original_depth]|![JNet_377_pretrain_1_output_depth]|![JNet_377_pretrain_1_label_depth]|
  
MSE: 0.018350714817643166, BCE: 0.06601055711507797  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_2_original_plane]|![JNet_377_pretrain_2_output_plane]|![JNet_377_pretrain_2_label_plane]|
  
MSE: 0.015432728454470634, BCE: 0.062320590019226074  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_2_original_depth]|![JNet_377_pretrain_2_output_depth]|![JNet_377_pretrain_2_label_depth]|
  
MSE: 0.015432728454470634, BCE: 0.062320590019226074  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_3_original_plane]|![JNet_377_pretrain_3_output_plane]|![JNet_377_pretrain_3_label_plane]|
  
MSE: 0.01729421131312847, BCE: 0.06296217441558838  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_3_original_depth]|![JNet_377_pretrain_3_output_depth]|![JNet_377_pretrain_3_label_depth]|
  
MSE: 0.01729421131312847, BCE: 0.06296217441558838  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_4_original_plane]|![JNet_377_pretrain_4_output_plane]|![JNet_377_pretrain_4_label_plane]|
  
MSE: 0.023715365678071976, BCE: 0.08474119752645493  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_377_pretrain_4_original_depth]|![JNet_377_pretrain_4_output_depth]|![JNet_377_pretrain_4_label_depth]|
  
MSE: 0.023715365678071976, BCE: 0.08474119752645493  
  
mean MSE: 0.03766091912984848, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_378_0_original_plane]|![JNet_378_0_output_plane]|![JNet_378_0_label_plane]|
  
MSE: 0.04493239149451256, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_378_0_original_depth]|![JNet_378_0_output_depth]|![JNet_378_0_label_depth]|
  
MSE: 0.04493239149451256, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_378_1_original_plane]|![JNet_378_1_output_plane]|![JNet_378_1_label_plane]|
  
MSE: 0.03393702208995819, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_378_1_original_depth]|![JNet_378_1_output_depth]|![JNet_378_1_label_depth]|
  
MSE: 0.03393702208995819, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_378_2_original_plane]|![JNet_378_2_output_plane]|![JNet_378_2_label_plane]|
  
MSE: 0.025977561250329018, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_378_2_original_depth]|![JNet_378_2_output_depth]|![JNet_378_2_label_depth]|
  
MSE: 0.025977561250329018, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_378_3_original_plane]|![JNet_378_3_output_plane]|![JNet_378_3_label_plane]|
  
MSE: 0.042627014219760895, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_378_3_original_depth]|![JNet_378_3_output_depth]|![JNet_378_3_label_depth]|
  
MSE: 0.042627014219760895, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_378_4_original_plane]|![JNet_378_4_output_plane]|![JNet_378_4_label_plane]|
  
MSE: 0.040830593556165695, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_378_4_original_depth]|![JNet_378_4_output_depth]|![JNet_378_4_label_depth]|
  
MSE: 0.040830593556165695, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_001_roi000_original_depth]|![JNet_377_pretrain_beads_001_roi000_output_depth]|![JNet_377_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 14.563750000000004, MSE: 0.002526022493839264, quantized loss: 0.0027920580469071865  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_001_roi001_original_depth]|![JNet_377_pretrain_beads_001_roi001_output_depth]|![JNet_377_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 20.781375000000004, MSE: 0.00411616126075387, quantized loss: 0.0032781800255179405  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_001_roi002_original_depth]|![JNet_377_pretrain_beads_001_roi002_output_depth]|![JNet_377_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 14.183125000000004, MSE: 0.0027253988664597273, quantized loss: 0.00265832943841815  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_001_roi003_original_depth]|![JNet_377_pretrain_beads_001_roi003_output_depth]|![JNet_377_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 21.054000000000006, MSE: 0.004568076692521572, quantized loss: 0.003507738932967186  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_001_roi004_original_depth]|![JNet_377_pretrain_beads_001_roi004_output_depth]|![JNet_377_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 15.082125000000003, MSE: 0.003346625017002225, quantized loss: 0.0028067613020539284  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_002_roi000_original_depth]|![JNet_377_pretrain_beads_002_roi000_output_depth]|![JNet_377_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 16.877500000000005, MSE: 0.003746275557205081, quantized loss: 0.003338450798764825  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_002_roi001_original_depth]|![JNet_377_pretrain_beads_002_roi001_output_depth]|![JNet_377_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 15.599125000000004, MSE: 0.003044930286705494, quantized loss: 0.003032325068488717  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_377_pretrain_beads_002_roi002_original_depth]|![JNet_377_pretrain_beads_002_roi002_output_depth]|![JNet_377_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 15.850500000000004, MSE: 0.00334676424972713, quantized loss: 0.003152257762849331  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_378_beads_001_roi000_original_depth]|![JNet_378_beads_001_roi000_output_depth]|![JNet_378_beads_001_roi000_reconst_depth]|
  
volume: 9.331875000000002, MSE: 0.0001701426226645708, quantized loss: 4.498805083130719e-06  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_378_beads_001_roi001_original_depth]|![JNet_378_beads_001_roi001_output_depth]|![JNet_378_beads_001_roi001_reconst_depth]|
  
volume: 14.311875000000004, MSE: 0.0006428944761864841, quantized loss: 6.033163117535878e-06  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_378_beads_001_roi002_original_depth]|![JNet_378_beads_001_roi002_output_depth]|![JNet_378_beads_001_roi002_reconst_depth]|
  
volume: 9.361125000000003, MSE: 0.00011677921429509297, quantized loss: 3.9399055822286755e-06  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_378_beads_001_roi003_original_depth]|![JNet_378_beads_001_roi003_output_depth]|![JNet_378_beads_001_roi003_reconst_depth]|
  
volume: 14.977000000000004, MSE: 0.0003804079897236079, quantized loss: 5.901460554014193e-06  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_378_beads_001_roi004_original_depth]|![JNet_378_beads_001_roi004_output_depth]|![JNet_378_beads_001_roi004_reconst_depth]|
  
volume: 10.136750000000003, MSE: 9.167270036414266e-05, quantized loss: 3.7011793665442383e-06  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_378_beads_002_roi000_original_depth]|![JNet_378_beads_002_roi000_output_depth]|![JNet_378_beads_002_roi000_reconst_depth]|
  
volume: 10.765750000000002, MSE: 9.579392644809559e-05, quantized loss: 3.9410069803125225e-06  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_378_beads_002_roi001_original_depth]|![JNet_378_beads_002_roi001_output_depth]|![JNet_378_beads_002_roi001_reconst_depth]|
  
volume: 9.779250000000003, MSE: 0.00010004083014791831, quantized loss: 4.412760972627439e-06  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_378_beads_002_roi002_original_depth]|![JNet_378_beads_002_roi002_output_depth]|![JNet_378_beads_002_roi002_reconst_depth]|
  
volume: 10.151000000000002, MSE: 9.089052764466032e-05, quantized loss: 3.43215015163878e-06  

|pre|post|
| :---: | :---: |
|![JNet_378_psf_pre]|![JNet_378_psf_post]|
  



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
[JNet_378_0_label_depth]: /experiments/images/JNet_378_0_label_depth.png
[JNet_378_0_label_plane]: /experiments/images/JNet_378_0_label_plane.png
[JNet_378_0_original_depth]: /experiments/images/JNet_378_0_original_depth.png
[JNet_378_0_original_plane]: /experiments/images/JNet_378_0_original_plane.png
[JNet_378_0_output_depth]: /experiments/images/JNet_378_0_output_depth.png
[JNet_378_0_output_plane]: /experiments/images/JNet_378_0_output_plane.png
[JNet_378_1_label_depth]: /experiments/images/JNet_378_1_label_depth.png
[JNet_378_1_label_plane]: /experiments/images/JNet_378_1_label_plane.png
[JNet_378_1_original_depth]: /experiments/images/JNet_378_1_original_depth.png
[JNet_378_1_original_plane]: /experiments/images/JNet_378_1_original_plane.png
[JNet_378_1_output_depth]: /experiments/images/JNet_378_1_output_depth.png
[JNet_378_1_output_plane]: /experiments/images/JNet_378_1_output_plane.png
[JNet_378_2_label_depth]: /experiments/images/JNet_378_2_label_depth.png
[JNet_378_2_label_plane]: /experiments/images/JNet_378_2_label_plane.png
[JNet_378_2_original_depth]: /experiments/images/JNet_378_2_original_depth.png
[JNet_378_2_original_plane]: /experiments/images/JNet_378_2_original_plane.png
[JNet_378_2_output_depth]: /experiments/images/JNet_378_2_output_depth.png
[JNet_378_2_output_plane]: /experiments/images/JNet_378_2_output_plane.png
[JNet_378_3_label_depth]: /experiments/images/JNet_378_3_label_depth.png
[JNet_378_3_label_plane]: /experiments/images/JNet_378_3_label_plane.png
[JNet_378_3_original_depth]: /experiments/images/JNet_378_3_original_depth.png
[JNet_378_3_original_plane]: /experiments/images/JNet_378_3_original_plane.png
[JNet_378_3_output_depth]: /experiments/images/JNet_378_3_output_depth.png
[JNet_378_3_output_plane]: /experiments/images/JNet_378_3_output_plane.png
[JNet_378_4_label_depth]: /experiments/images/JNet_378_4_label_depth.png
[JNet_378_4_label_plane]: /experiments/images/JNet_378_4_label_plane.png
[JNet_378_4_original_depth]: /experiments/images/JNet_378_4_original_depth.png
[JNet_378_4_original_plane]: /experiments/images/JNet_378_4_original_plane.png
[JNet_378_4_output_depth]: /experiments/images/JNet_378_4_output_depth.png
[JNet_378_4_output_plane]: /experiments/images/JNet_378_4_output_plane.png
[JNet_378_beads_001_roi000_original_depth]: /experiments/images/JNet_378_beads_001_roi000_original_depth.png
[JNet_378_beads_001_roi000_output_depth]: /experiments/images/JNet_378_beads_001_roi000_output_depth.png
[JNet_378_beads_001_roi000_reconst_depth]: /experiments/images/JNet_378_beads_001_roi000_reconst_depth.png
[JNet_378_beads_001_roi001_original_depth]: /experiments/images/JNet_378_beads_001_roi001_original_depth.png
[JNet_378_beads_001_roi001_output_depth]: /experiments/images/JNet_378_beads_001_roi001_output_depth.png
[JNet_378_beads_001_roi001_reconst_depth]: /experiments/images/JNet_378_beads_001_roi001_reconst_depth.png
[JNet_378_beads_001_roi002_original_depth]: /experiments/images/JNet_378_beads_001_roi002_original_depth.png
[JNet_378_beads_001_roi002_output_depth]: /experiments/images/JNet_378_beads_001_roi002_output_depth.png
[JNet_378_beads_001_roi002_reconst_depth]: /experiments/images/JNet_378_beads_001_roi002_reconst_depth.png
[JNet_378_beads_001_roi003_original_depth]: /experiments/images/JNet_378_beads_001_roi003_original_depth.png
[JNet_378_beads_001_roi003_output_depth]: /experiments/images/JNet_378_beads_001_roi003_output_depth.png
[JNet_378_beads_001_roi003_reconst_depth]: /experiments/images/JNet_378_beads_001_roi003_reconst_depth.png
[JNet_378_beads_001_roi004_original_depth]: /experiments/images/JNet_378_beads_001_roi004_original_depth.png
[JNet_378_beads_001_roi004_output_depth]: /experiments/images/JNet_378_beads_001_roi004_output_depth.png
[JNet_378_beads_001_roi004_reconst_depth]: /experiments/images/JNet_378_beads_001_roi004_reconst_depth.png
[JNet_378_beads_002_roi000_original_depth]: /experiments/images/JNet_378_beads_002_roi000_original_depth.png
[JNet_378_beads_002_roi000_output_depth]: /experiments/images/JNet_378_beads_002_roi000_output_depth.png
[JNet_378_beads_002_roi000_reconst_depth]: /experiments/images/JNet_378_beads_002_roi000_reconst_depth.png
[JNet_378_beads_002_roi001_original_depth]: /experiments/images/JNet_378_beads_002_roi001_original_depth.png
[JNet_378_beads_002_roi001_output_depth]: /experiments/images/JNet_378_beads_002_roi001_output_depth.png
[JNet_378_beads_002_roi001_reconst_depth]: /experiments/images/JNet_378_beads_002_roi001_reconst_depth.png
[JNet_378_beads_002_roi002_original_depth]: /experiments/images/JNet_378_beads_002_roi002_original_depth.png
[JNet_378_beads_002_roi002_output_depth]: /experiments/images/JNet_378_beads_002_roi002_output_depth.png
[JNet_378_beads_002_roi002_reconst_depth]: /experiments/images/JNet_378_beads_002_roi002_reconst_depth.png
[JNet_378_psf_post]: /experiments/images/JNet_378_psf_post.png
[JNet_378_psf_pre]: /experiments/images/JNet_378_psf_pre.png
[finetuned]: /experiments/tmp/JNet_378_train.png
[pretrained_model]: /experiments/tmp/JNet_377_pretrain_train.png
