



# JNet_354 Report
  
the parameters to replicate the results of JNet_354. large psf and noise  
pretrained model : JNet_353_pretrain
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
|sig_eps|0.05||
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
  
mean MSE: 0.024658912792801857, mean BCE: 0.09102842956781387
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_353_pretrain_0_original_plane]|![JNet_353_pretrain_0_output_plane]|![JNet_353_pretrain_0_label_plane]|
  
MSE: 0.027237150818109512, BCE: 0.09740278869867325  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_353_pretrain_0_original_depth]|![JNet_353_pretrain_0_output_depth]|![JNet_353_pretrain_0_label_depth]|
  
MSE: 0.027237150818109512, BCE: 0.09740278869867325  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_353_pretrain_1_original_plane]|![JNet_353_pretrain_1_output_plane]|![JNet_353_pretrain_1_label_plane]|
  
MSE: 0.02686794474720955, BCE: 0.1007482185959816  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_353_pretrain_1_original_depth]|![JNet_353_pretrain_1_output_depth]|![JNet_353_pretrain_1_label_depth]|
  
MSE: 0.02686794474720955, BCE: 0.1007482185959816  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_353_pretrain_2_original_plane]|![JNet_353_pretrain_2_output_plane]|![JNet_353_pretrain_2_label_plane]|
  
MSE: 0.021674808114767075, BCE: 0.07811866700649261  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_353_pretrain_2_original_depth]|![JNet_353_pretrain_2_output_depth]|![JNet_353_pretrain_2_label_depth]|
  
MSE: 0.021674808114767075, BCE: 0.07811866700649261  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_353_pretrain_3_original_plane]|![JNet_353_pretrain_3_output_plane]|![JNet_353_pretrain_3_label_plane]|
  
MSE: 0.028139758855104446, BCE: 0.10416919738054276  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_353_pretrain_3_original_depth]|![JNet_353_pretrain_3_output_depth]|![JNet_353_pretrain_3_label_depth]|
  
MSE: 0.028139758855104446, BCE: 0.10416919738054276  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_353_pretrain_4_original_plane]|![JNet_353_pretrain_4_output_plane]|![JNet_353_pretrain_4_label_plane]|
  
MSE: 0.019374905154109, BCE: 0.07470327615737915  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_353_pretrain_4_original_depth]|![JNet_353_pretrain_4_output_depth]|![JNet_353_pretrain_4_label_depth]|
  
MSE: 0.019374905154109, BCE: 0.07470327615737915  
  
mean MSE: 0.03427289426326752, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_354_0_original_plane]|![JNet_354_0_output_plane]|![JNet_354_0_label_plane]|
  
MSE: 0.04089584946632385, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_354_0_original_depth]|![JNet_354_0_output_depth]|![JNet_354_0_label_depth]|
  
MSE: 0.04089584946632385, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_354_1_original_plane]|![JNet_354_1_output_plane]|![JNet_354_1_label_plane]|
  
MSE: 0.026606013998389244, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_354_1_original_depth]|![JNet_354_1_output_depth]|![JNet_354_1_label_depth]|
  
MSE: 0.026606013998389244, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_354_2_original_plane]|![JNet_354_2_output_plane]|![JNet_354_2_label_plane]|
  
MSE: 0.031445447355508804, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_354_2_original_depth]|![JNet_354_2_output_depth]|![JNet_354_2_label_depth]|
  
MSE: 0.031445447355508804, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_354_3_original_plane]|![JNet_354_3_output_plane]|![JNet_354_3_label_plane]|
  
MSE: 0.03737223148345947, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_354_3_original_depth]|![JNet_354_3_output_depth]|![JNet_354_3_label_depth]|
  
MSE: 0.03737223148345947, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_354_4_original_plane]|![JNet_354_4_output_plane]|![JNet_354_4_label_plane]|
  
MSE: 0.03504492715001106, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_354_4_original_depth]|![JNet_354_4_output_depth]|![JNet_354_4_label_depth]|
  
MSE: 0.03504492715001106, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_353_pretrain_beads_001_roi000_original_depth]|![JNet_353_pretrain_beads_001_roi000_output_depth]|![JNet_353_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 16.845250000000004, MSE: 0.0021902502048760653, quantized loss: 0.001725951675325632  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_353_pretrain_beads_001_roi001_original_depth]|![JNet_353_pretrain_beads_001_roi001_output_depth]|![JNet_353_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 25.614125000000005, MSE: 0.0037123810034245253, quantized loss: 0.0024056504480540752  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_353_pretrain_beads_001_roi002_original_depth]|![JNet_353_pretrain_beads_001_roi002_output_depth]|![JNet_353_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 16.370750000000005, MSE: 0.002289404394105077, quantized loss: 0.0016366838244721293  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_353_pretrain_beads_001_roi003_original_depth]|![JNet_353_pretrain_beads_001_roi003_output_depth]|![JNet_353_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 26.755125000000007, MSE: 0.003839424578472972, quantized loss: 0.0024874808732420206  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_353_pretrain_beads_001_roi004_original_depth]|![JNet_353_pretrain_beads_001_roi004_output_depth]|![JNet_353_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 17.630125000000003, MSE: 0.002848595380783081, quantized loss: 0.0016571703599765897  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_353_pretrain_beads_002_roi000_original_depth]|![JNet_353_pretrain_beads_002_roi000_output_depth]|![JNet_353_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 18.756875000000004, MSE: 0.003190448274835944, quantized loss: 0.0016940133646130562  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_353_pretrain_beads_002_roi001_original_depth]|![JNet_353_pretrain_beads_002_roi001_output_depth]|![JNet_353_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 17.424375000000005, MSE: 0.0024986129719763994, quantized loss: 0.001677307765930891  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_353_pretrain_beads_002_roi002_original_depth]|![JNet_353_pretrain_beads_002_roi002_output_depth]|![JNet_353_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 18.081625000000006, MSE: 0.0028229595627635717, quantized loss: 0.0017246475908905268  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_354_beads_001_roi000_original_depth]|![JNet_354_beads_001_roi000_output_depth]|![JNet_354_beads_001_roi000_reconst_depth]|
  
volume: 7.058000000000002, MSE: 0.0005639226292259991, quantized loss: 9.04593980521895e-06  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_354_beads_001_roi001_original_depth]|![JNet_354_beads_001_roi001_output_depth]|![JNet_354_beads_001_roi001_reconst_depth]|
  
volume: 12.168375000000003, MSE: 0.000911931274458766, quantized loss: 1.4019495210959576e-05  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_354_beads_001_roi002_original_depth]|![JNet_354_beads_001_roi002_output_depth]|![JNet_354_beads_001_roi002_reconst_depth]|
  
volume: 7.2105000000000015, MSE: 0.00040344081935472786, quantized loss: 1.0322841262677684e-05  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_354_beads_001_roi003_original_depth]|![JNet_354_beads_001_roi003_output_depth]|![JNet_354_beads_001_roi003_reconst_depth]|
  
volume: 11.117625000000002, MSE: 0.0017272615805268288, quantized loss: 1.3223791938798968e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_354_beads_001_roi004_original_depth]|![JNet_354_beads_001_roi004_output_depth]|![JNet_354_beads_001_roi004_reconst_depth]|
  
volume: 7.957000000000002, MSE: 0.0004465441743377596, quantized loss: 9.495644917478785e-06  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_354_beads_002_roi000_original_depth]|![JNet_354_beads_002_roi000_output_depth]|![JNet_354_beads_002_roi000_reconst_depth]|
  
volume: 8.519000000000002, MSE: 0.0004744856560137123, quantized loss: 1.0439141988172196e-05  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_354_beads_002_roi001_original_depth]|![JNet_354_beads_002_roi001_output_depth]|![JNet_354_beads_002_roi001_reconst_depth]|
  
volume: 7.9837500000000015, MSE: 0.00035019719507545233, quantized loss: 9.557844350638334e-06  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_354_beads_002_roi002_original_depth]|![JNet_354_beads_002_roi002_output_depth]|![JNet_354_beads_002_roi002_reconst_depth]|
  
volume: 8.026625000000003, MSE: 0.00044220825657248497, quantized loss: 9.199286978400778e-06  

|pre|post|
| :---: | :---: |
|![JNet_354_psf_pre]|![JNet_354_psf_post]|
  



[JNet_353_pretrain_0_label_depth]: /experiments/images/JNet_353_pretrain_0_label_depth.png
[JNet_353_pretrain_0_label_plane]: /experiments/images/JNet_353_pretrain_0_label_plane.png
[JNet_353_pretrain_0_original_depth]: /experiments/images/JNet_353_pretrain_0_original_depth.png
[JNet_353_pretrain_0_original_plane]: /experiments/images/JNet_353_pretrain_0_original_plane.png
[JNet_353_pretrain_0_output_depth]: /experiments/images/JNet_353_pretrain_0_output_depth.png
[JNet_353_pretrain_0_output_plane]: /experiments/images/JNet_353_pretrain_0_output_plane.png
[JNet_353_pretrain_1_label_depth]: /experiments/images/JNet_353_pretrain_1_label_depth.png
[JNet_353_pretrain_1_label_plane]: /experiments/images/JNet_353_pretrain_1_label_plane.png
[JNet_353_pretrain_1_original_depth]: /experiments/images/JNet_353_pretrain_1_original_depth.png
[JNet_353_pretrain_1_original_plane]: /experiments/images/JNet_353_pretrain_1_original_plane.png
[JNet_353_pretrain_1_output_depth]: /experiments/images/JNet_353_pretrain_1_output_depth.png
[JNet_353_pretrain_1_output_plane]: /experiments/images/JNet_353_pretrain_1_output_plane.png
[JNet_353_pretrain_2_label_depth]: /experiments/images/JNet_353_pretrain_2_label_depth.png
[JNet_353_pretrain_2_label_plane]: /experiments/images/JNet_353_pretrain_2_label_plane.png
[JNet_353_pretrain_2_original_depth]: /experiments/images/JNet_353_pretrain_2_original_depth.png
[JNet_353_pretrain_2_original_plane]: /experiments/images/JNet_353_pretrain_2_original_plane.png
[JNet_353_pretrain_2_output_depth]: /experiments/images/JNet_353_pretrain_2_output_depth.png
[JNet_353_pretrain_2_output_plane]: /experiments/images/JNet_353_pretrain_2_output_plane.png
[JNet_353_pretrain_3_label_depth]: /experiments/images/JNet_353_pretrain_3_label_depth.png
[JNet_353_pretrain_3_label_plane]: /experiments/images/JNet_353_pretrain_3_label_plane.png
[JNet_353_pretrain_3_original_depth]: /experiments/images/JNet_353_pretrain_3_original_depth.png
[JNet_353_pretrain_3_original_plane]: /experiments/images/JNet_353_pretrain_3_original_plane.png
[JNet_353_pretrain_3_output_depth]: /experiments/images/JNet_353_pretrain_3_output_depth.png
[JNet_353_pretrain_3_output_plane]: /experiments/images/JNet_353_pretrain_3_output_plane.png
[JNet_353_pretrain_4_label_depth]: /experiments/images/JNet_353_pretrain_4_label_depth.png
[JNet_353_pretrain_4_label_plane]: /experiments/images/JNet_353_pretrain_4_label_plane.png
[JNet_353_pretrain_4_original_depth]: /experiments/images/JNet_353_pretrain_4_original_depth.png
[JNet_353_pretrain_4_original_plane]: /experiments/images/JNet_353_pretrain_4_original_plane.png
[JNet_353_pretrain_4_output_depth]: /experiments/images/JNet_353_pretrain_4_output_depth.png
[JNet_353_pretrain_4_output_plane]: /experiments/images/JNet_353_pretrain_4_output_plane.png
[JNet_353_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi000_original_depth.png
[JNet_353_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi000_output_depth.png
[JNet_353_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi000_reconst_depth.png
[JNet_353_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi001_original_depth.png
[JNet_353_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi001_output_depth.png
[JNet_353_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi001_reconst_depth.png
[JNet_353_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi002_original_depth.png
[JNet_353_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi002_output_depth.png
[JNet_353_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi002_reconst_depth.png
[JNet_353_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi003_original_depth.png
[JNet_353_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi003_output_depth.png
[JNet_353_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi003_reconst_depth.png
[JNet_353_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi004_original_depth.png
[JNet_353_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi004_output_depth.png
[JNet_353_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_353_pretrain_beads_001_roi004_reconst_depth.png
[JNet_353_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_353_pretrain_beads_002_roi000_original_depth.png
[JNet_353_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_353_pretrain_beads_002_roi000_output_depth.png
[JNet_353_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_353_pretrain_beads_002_roi000_reconst_depth.png
[JNet_353_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_353_pretrain_beads_002_roi001_original_depth.png
[JNet_353_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_353_pretrain_beads_002_roi001_output_depth.png
[JNet_353_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_353_pretrain_beads_002_roi001_reconst_depth.png
[JNet_353_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_353_pretrain_beads_002_roi002_original_depth.png
[JNet_353_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_353_pretrain_beads_002_roi002_output_depth.png
[JNet_353_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_353_pretrain_beads_002_roi002_reconst_depth.png
[JNet_354_0_label_depth]: /experiments/images/JNet_354_0_label_depth.png
[JNet_354_0_label_plane]: /experiments/images/JNet_354_0_label_plane.png
[JNet_354_0_original_depth]: /experiments/images/JNet_354_0_original_depth.png
[JNet_354_0_original_plane]: /experiments/images/JNet_354_0_original_plane.png
[JNet_354_0_output_depth]: /experiments/images/JNet_354_0_output_depth.png
[JNet_354_0_output_plane]: /experiments/images/JNet_354_0_output_plane.png
[JNet_354_1_label_depth]: /experiments/images/JNet_354_1_label_depth.png
[JNet_354_1_label_plane]: /experiments/images/JNet_354_1_label_plane.png
[JNet_354_1_original_depth]: /experiments/images/JNet_354_1_original_depth.png
[JNet_354_1_original_plane]: /experiments/images/JNet_354_1_original_plane.png
[JNet_354_1_output_depth]: /experiments/images/JNet_354_1_output_depth.png
[JNet_354_1_output_plane]: /experiments/images/JNet_354_1_output_plane.png
[JNet_354_2_label_depth]: /experiments/images/JNet_354_2_label_depth.png
[JNet_354_2_label_plane]: /experiments/images/JNet_354_2_label_plane.png
[JNet_354_2_original_depth]: /experiments/images/JNet_354_2_original_depth.png
[JNet_354_2_original_plane]: /experiments/images/JNet_354_2_original_plane.png
[JNet_354_2_output_depth]: /experiments/images/JNet_354_2_output_depth.png
[JNet_354_2_output_plane]: /experiments/images/JNet_354_2_output_plane.png
[JNet_354_3_label_depth]: /experiments/images/JNet_354_3_label_depth.png
[JNet_354_3_label_plane]: /experiments/images/JNet_354_3_label_plane.png
[JNet_354_3_original_depth]: /experiments/images/JNet_354_3_original_depth.png
[JNet_354_3_original_plane]: /experiments/images/JNet_354_3_original_plane.png
[JNet_354_3_output_depth]: /experiments/images/JNet_354_3_output_depth.png
[JNet_354_3_output_plane]: /experiments/images/JNet_354_3_output_plane.png
[JNet_354_4_label_depth]: /experiments/images/JNet_354_4_label_depth.png
[JNet_354_4_label_plane]: /experiments/images/JNet_354_4_label_plane.png
[JNet_354_4_original_depth]: /experiments/images/JNet_354_4_original_depth.png
[JNet_354_4_original_plane]: /experiments/images/JNet_354_4_original_plane.png
[JNet_354_4_output_depth]: /experiments/images/JNet_354_4_output_depth.png
[JNet_354_4_output_plane]: /experiments/images/JNet_354_4_output_plane.png
[JNet_354_beads_001_roi000_original_depth]: /experiments/images/JNet_354_beads_001_roi000_original_depth.png
[JNet_354_beads_001_roi000_output_depth]: /experiments/images/JNet_354_beads_001_roi000_output_depth.png
[JNet_354_beads_001_roi000_reconst_depth]: /experiments/images/JNet_354_beads_001_roi000_reconst_depth.png
[JNet_354_beads_001_roi001_original_depth]: /experiments/images/JNet_354_beads_001_roi001_original_depth.png
[JNet_354_beads_001_roi001_output_depth]: /experiments/images/JNet_354_beads_001_roi001_output_depth.png
[JNet_354_beads_001_roi001_reconst_depth]: /experiments/images/JNet_354_beads_001_roi001_reconst_depth.png
[JNet_354_beads_001_roi002_original_depth]: /experiments/images/JNet_354_beads_001_roi002_original_depth.png
[JNet_354_beads_001_roi002_output_depth]: /experiments/images/JNet_354_beads_001_roi002_output_depth.png
[JNet_354_beads_001_roi002_reconst_depth]: /experiments/images/JNet_354_beads_001_roi002_reconst_depth.png
[JNet_354_beads_001_roi003_original_depth]: /experiments/images/JNet_354_beads_001_roi003_original_depth.png
[JNet_354_beads_001_roi003_output_depth]: /experiments/images/JNet_354_beads_001_roi003_output_depth.png
[JNet_354_beads_001_roi003_reconst_depth]: /experiments/images/JNet_354_beads_001_roi003_reconst_depth.png
[JNet_354_beads_001_roi004_original_depth]: /experiments/images/JNet_354_beads_001_roi004_original_depth.png
[JNet_354_beads_001_roi004_output_depth]: /experiments/images/JNet_354_beads_001_roi004_output_depth.png
[JNet_354_beads_001_roi004_reconst_depth]: /experiments/images/JNet_354_beads_001_roi004_reconst_depth.png
[JNet_354_beads_002_roi000_original_depth]: /experiments/images/JNet_354_beads_002_roi000_original_depth.png
[JNet_354_beads_002_roi000_output_depth]: /experiments/images/JNet_354_beads_002_roi000_output_depth.png
[JNet_354_beads_002_roi000_reconst_depth]: /experiments/images/JNet_354_beads_002_roi000_reconst_depth.png
[JNet_354_beads_002_roi001_original_depth]: /experiments/images/JNet_354_beads_002_roi001_original_depth.png
[JNet_354_beads_002_roi001_output_depth]: /experiments/images/JNet_354_beads_002_roi001_output_depth.png
[JNet_354_beads_002_roi001_reconst_depth]: /experiments/images/JNet_354_beads_002_roi001_reconst_depth.png
[JNet_354_beads_002_roi002_original_depth]: /experiments/images/JNet_354_beads_002_roi002_original_depth.png
[JNet_354_beads_002_roi002_output_depth]: /experiments/images/JNet_354_beads_002_roi002_output_depth.png
[JNet_354_beads_002_roi002_reconst_depth]: /experiments/images/JNet_354_beads_002_roi002_reconst_depth.png
[JNet_354_psf_post]: /experiments/images/JNet_354_psf_post.png
[JNet_354_psf_pre]: /experiments/images/JNet_354_psf_pre.png
[finetuned]: /experiments/tmp/JNet_354_train.png
[pretrained_model]: /experiments/tmp/JNet_353_pretrain_train.png
