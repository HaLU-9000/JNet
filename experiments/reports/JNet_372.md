



# JNet_372 Report
  
the parameters to replicate the results of JNet_372. mask added.  
pretrained model : JNet_371_pretrain
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
  
mean MSE: 0.01817924715578556, mean BCE: 0.06657002866268158
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_371_pretrain_0_original_plane]|![JNet_371_pretrain_0_output_plane]|![JNet_371_pretrain_0_label_plane]|
  
MSE: 0.014182616025209427, BCE: 0.055203843861818314  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_371_pretrain_0_original_depth]|![JNet_371_pretrain_0_output_depth]|![JNet_371_pretrain_0_label_depth]|
  
MSE: 0.014182616025209427, BCE: 0.055203843861818314  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_371_pretrain_1_original_plane]|![JNet_371_pretrain_1_output_plane]|![JNet_371_pretrain_1_label_plane]|
  
MSE: 0.01878265291452408, BCE: 0.06547018140554428  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_371_pretrain_1_original_depth]|![JNet_371_pretrain_1_output_depth]|![JNet_371_pretrain_1_label_depth]|
  
MSE: 0.01878265291452408, BCE: 0.06547018140554428  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_371_pretrain_2_original_plane]|![JNet_371_pretrain_2_output_plane]|![JNet_371_pretrain_2_label_plane]|
  
MSE: 0.01948653720319271, BCE: 0.07503821700811386  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_371_pretrain_2_original_depth]|![JNet_371_pretrain_2_output_depth]|![JNet_371_pretrain_2_label_depth]|
  
MSE: 0.01948653720319271, BCE: 0.07503821700811386  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_371_pretrain_3_original_plane]|![JNet_371_pretrain_3_output_plane]|![JNet_371_pretrain_3_label_plane]|
  
MSE: 0.01443328894674778, BCE: 0.0526619516313076  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_371_pretrain_3_original_depth]|![JNet_371_pretrain_3_output_depth]|![JNet_371_pretrain_3_label_depth]|
  
MSE: 0.01443328894674778, BCE: 0.0526619516313076  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_371_pretrain_4_original_plane]|![JNet_371_pretrain_4_output_plane]|![JNet_371_pretrain_4_label_plane]|
  
MSE: 0.024011144414544106, BCE: 0.08447595685720444  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_371_pretrain_4_original_depth]|![JNet_371_pretrain_4_output_depth]|![JNet_371_pretrain_4_label_depth]|
  
MSE: 0.024011144414544106, BCE: 0.08447595685720444  
  
mean MSE: 0.032513104379177094, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_372_0_original_plane]|![JNet_372_0_output_plane]|![JNet_372_0_label_plane]|
  
MSE: 0.02864583395421505, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_372_0_original_depth]|![JNet_372_0_output_depth]|![JNet_372_0_label_depth]|
  
MSE: 0.02864583395421505, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_372_1_original_plane]|![JNet_372_1_output_plane]|![JNet_372_1_label_plane]|
  
MSE: 0.038252752274274826, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_372_1_original_depth]|![JNet_372_1_output_depth]|![JNet_372_1_label_depth]|
  
MSE: 0.038252752274274826, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_372_2_original_plane]|![JNet_372_2_output_plane]|![JNet_372_2_label_plane]|
  
MSE: 0.030892282724380493, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_372_2_original_depth]|![JNet_372_2_output_depth]|![JNet_372_2_label_depth]|
  
MSE: 0.030892282724380493, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_372_3_original_plane]|![JNet_372_3_output_plane]|![JNet_372_3_label_plane]|
  
MSE: 0.03525390475988388, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_372_3_original_depth]|![JNet_372_3_output_depth]|![JNet_372_3_label_depth]|
  
MSE: 0.03525390475988388, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_372_4_original_plane]|![JNet_372_4_output_plane]|![JNet_372_4_label_plane]|
  
MSE: 0.029520753771066666, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_372_4_original_depth]|![JNet_372_4_output_depth]|![JNet_372_4_label_depth]|
  
MSE: 0.029520753771066666, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_371_pretrain_beads_001_roi000_original_depth]|![JNet_371_pretrain_beads_001_roi000_output_depth]|![JNet_371_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 13.971750000000004, MSE: 0.0014607119373977184, quantized loss: 0.0024394667707383633  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_371_pretrain_beads_001_roi001_original_depth]|![JNet_371_pretrain_beads_001_roi001_output_depth]|![JNet_371_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 20.053875000000005, MSE: 0.002428383333608508, quantized loss: 0.0030369118321686983  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_371_pretrain_beads_001_roi002_original_depth]|![JNet_371_pretrain_beads_001_roi002_output_depth]|![JNet_371_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 13.381250000000003, MSE: 0.001445284578949213, quantized loss: 0.002187148667871952  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_371_pretrain_beads_001_roi003_original_depth]|![JNet_371_pretrain_beads_001_roi003_output_depth]|![JNet_371_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 21.358375000000006, MSE: 0.002429056214168668, quantized loss: 0.0031943076755851507  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_371_pretrain_beads_001_roi004_original_depth]|![JNet_371_pretrain_beads_001_roi004_output_depth]|![JNet_371_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 14.480125000000003, MSE: 0.001836642506532371, quantized loss: 0.0022160899825394154  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_371_pretrain_beads_002_roi000_original_depth]|![JNet_371_pretrain_beads_002_roi000_output_depth]|![JNet_371_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 15.823125000000005, MSE: 0.0020292133558541536, quantized loss: 0.0024555951822549105  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_371_pretrain_beads_002_roi001_original_depth]|![JNet_371_pretrain_beads_002_roi001_output_depth]|![JNet_371_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 14.850500000000004, MSE: 0.0015039942227303982, quantized loss: 0.002317665610462427  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_371_pretrain_beads_002_roi002_original_depth]|![JNet_371_pretrain_beads_002_roi002_output_depth]|![JNet_371_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 14.769875000000004, MSE: 0.0017613248201087117, quantized loss: 0.0023016822524368763  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_372_beads_001_roi000_original_depth]|![JNet_372_beads_001_roi000_output_depth]|![JNet_372_beads_001_roi000_reconst_depth]|
  
volume: 11.677500000000002, MSE: 0.00033785964478738606, quantized loss: 4.912476470053662e-06  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_372_beads_001_roi001_original_depth]|![JNet_372_beads_001_roi001_output_depth]|![JNet_372_beads_001_roi001_reconst_depth]|
  
volume: 18.118625000000005, MSE: 0.0007646717713214457, quantized loss: 7.082441698003095e-06  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_372_beads_001_roi002_original_depth]|![JNet_372_beads_001_roi002_output_depth]|![JNet_372_beads_001_roi002_reconst_depth]|
  
volume: 11.610125000000004, MSE: 0.00027535928529687226, quantized loss: 5.50763161299983e-06  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_372_beads_001_roi003_original_depth]|![JNet_372_beads_001_roi003_output_depth]|![JNet_372_beads_001_roi003_reconst_depth]|
  
volume: 19.127625000000005, MSE: 0.0006500363815575838, quantized loss: 7.614423793711467e-06  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_372_beads_001_roi004_original_depth]|![JNet_372_beads_001_roi004_output_depth]|![JNet_372_beads_001_roi004_reconst_depth]|
  
volume: 12.612250000000003, MSE: 0.00028950805426575243, quantized loss: 3.984283466706984e-06  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_372_beads_002_roi000_original_depth]|![JNet_372_beads_002_roi000_output_depth]|![JNet_372_beads_002_roi000_reconst_depth]|
  
volume: 13.432500000000003, MSE: 0.0003169737756252289, quantized loss: 4.070833711011801e-06  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_372_beads_002_roi001_original_depth]|![JNet_372_beads_002_roi001_output_depth]|![JNet_372_beads_002_roi001_reconst_depth]|
  
volume: 12.321750000000003, MSE: 0.00028500266489572823, quantized loss: 5.4792094488220755e-06  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_372_beads_002_roi002_original_depth]|![JNet_372_beads_002_roi002_output_depth]|![JNet_372_beads_002_roi002_reconst_depth]|
  
volume: 12.808000000000003, MSE: 0.0002857284271158278, quantized loss: 5.977476121188374e-06  

|pre|post|
| :---: | :---: |
|![JNet_372_psf_pre]|![JNet_372_psf_post]|
  



[JNet_371_pretrain_0_label_depth]: /experiments/images/JNet_371_pretrain_0_label_depth.png
[JNet_371_pretrain_0_label_plane]: /experiments/images/JNet_371_pretrain_0_label_plane.png
[JNet_371_pretrain_0_original_depth]: /experiments/images/JNet_371_pretrain_0_original_depth.png
[JNet_371_pretrain_0_original_plane]: /experiments/images/JNet_371_pretrain_0_original_plane.png
[JNet_371_pretrain_0_output_depth]: /experiments/images/JNet_371_pretrain_0_output_depth.png
[JNet_371_pretrain_0_output_plane]: /experiments/images/JNet_371_pretrain_0_output_plane.png
[JNet_371_pretrain_1_label_depth]: /experiments/images/JNet_371_pretrain_1_label_depth.png
[JNet_371_pretrain_1_label_plane]: /experiments/images/JNet_371_pretrain_1_label_plane.png
[JNet_371_pretrain_1_original_depth]: /experiments/images/JNet_371_pretrain_1_original_depth.png
[JNet_371_pretrain_1_original_plane]: /experiments/images/JNet_371_pretrain_1_original_plane.png
[JNet_371_pretrain_1_output_depth]: /experiments/images/JNet_371_pretrain_1_output_depth.png
[JNet_371_pretrain_1_output_plane]: /experiments/images/JNet_371_pretrain_1_output_plane.png
[JNet_371_pretrain_2_label_depth]: /experiments/images/JNet_371_pretrain_2_label_depth.png
[JNet_371_pretrain_2_label_plane]: /experiments/images/JNet_371_pretrain_2_label_plane.png
[JNet_371_pretrain_2_original_depth]: /experiments/images/JNet_371_pretrain_2_original_depth.png
[JNet_371_pretrain_2_original_plane]: /experiments/images/JNet_371_pretrain_2_original_plane.png
[JNet_371_pretrain_2_output_depth]: /experiments/images/JNet_371_pretrain_2_output_depth.png
[JNet_371_pretrain_2_output_plane]: /experiments/images/JNet_371_pretrain_2_output_plane.png
[JNet_371_pretrain_3_label_depth]: /experiments/images/JNet_371_pretrain_3_label_depth.png
[JNet_371_pretrain_3_label_plane]: /experiments/images/JNet_371_pretrain_3_label_plane.png
[JNet_371_pretrain_3_original_depth]: /experiments/images/JNet_371_pretrain_3_original_depth.png
[JNet_371_pretrain_3_original_plane]: /experiments/images/JNet_371_pretrain_3_original_plane.png
[JNet_371_pretrain_3_output_depth]: /experiments/images/JNet_371_pretrain_3_output_depth.png
[JNet_371_pretrain_3_output_plane]: /experiments/images/JNet_371_pretrain_3_output_plane.png
[JNet_371_pretrain_4_label_depth]: /experiments/images/JNet_371_pretrain_4_label_depth.png
[JNet_371_pretrain_4_label_plane]: /experiments/images/JNet_371_pretrain_4_label_plane.png
[JNet_371_pretrain_4_original_depth]: /experiments/images/JNet_371_pretrain_4_original_depth.png
[JNet_371_pretrain_4_original_plane]: /experiments/images/JNet_371_pretrain_4_original_plane.png
[JNet_371_pretrain_4_output_depth]: /experiments/images/JNet_371_pretrain_4_output_depth.png
[JNet_371_pretrain_4_output_plane]: /experiments/images/JNet_371_pretrain_4_output_plane.png
[JNet_371_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi000_original_depth.png
[JNet_371_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi000_output_depth.png
[JNet_371_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi000_reconst_depth.png
[JNet_371_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi001_original_depth.png
[JNet_371_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi001_output_depth.png
[JNet_371_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi001_reconst_depth.png
[JNet_371_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi002_original_depth.png
[JNet_371_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi002_output_depth.png
[JNet_371_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi002_reconst_depth.png
[JNet_371_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi003_original_depth.png
[JNet_371_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi003_output_depth.png
[JNet_371_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi003_reconst_depth.png
[JNet_371_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi004_original_depth.png
[JNet_371_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi004_output_depth.png
[JNet_371_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_371_pretrain_beads_001_roi004_reconst_depth.png
[JNet_371_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_371_pretrain_beads_002_roi000_original_depth.png
[JNet_371_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_371_pretrain_beads_002_roi000_output_depth.png
[JNet_371_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_371_pretrain_beads_002_roi000_reconst_depth.png
[JNet_371_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_371_pretrain_beads_002_roi001_original_depth.png
[JNet_371_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_371_pretrain_beads_002_roi001_output_depth.png
[JNet_371_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_371_pretrain_beads_002_roi001_reconst_depth.png
[JNet_371_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_371_pretrain_beads_002_roi002_original_depth.png
[JNet_371_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_371_pretrain_beads_002_roi002_output_depth.png
[JNet_371_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_371_pretrain_beads_002_roi002_reconst_depth.png
[JNet_372_0_label_depth]: /experiments/images/JNet_372_0_label_depth.png
[JNet_372_0_label_plane]: /experiments/images/JNet_372_0_label_plane.png
[JNet_372_0_original_depth]: /experiments/images/JNet_372_0_original_depth.png
[JNet_372_0_original_plane]: /experiments/images/JNet_372_0_original_plane.png
[JNet_372_0_output_depth]: /experiments/images/JNet_372_0_output_depth.png
[JNet_372_0_output_plane]: /experiments/images/JNet_372_0_output_plane.png
[JNet_372_1_label_depth]: /experiments/images/JNet_372_1_label_depth.png
[JNet_372_1_label_plane]: /experiments/images/JNet_372_1_label_plane.png
[JNet_372_1_original_depth]: /experiments/images/JNet_372_1_original_depth.png
[JNet_372_1_original_plane]: /experiments/images/JNet_372_1_original_plane.png
[JNet_372_1_output_depth]: /experiments/images/JNet_372_1_output_depth.png
[JNet_372_1_output_plane]: /experiments/images/JNet_372_1_output_plane.png
[JNet_372_2_label_depth]: /experiments/images/JNet_372_2_label_depth.png
[JNet_372_2_label_plane]: /experiments/images/JNet_372_2_label_plane.png
[JNet_372_2_original_depth]: /experiments/images/JNet_372_2_original_depth.png
[JNet_372_2_original_plane]: /experiments/images/JNet_372_2_original_plane.png
[JNet_372_2_output_depth]: /experiments/images/JNet_372_2_output_depth.png
[JNet_372_2_output_plane]: /experiments/images/JNet_372_2_output_plane.png
[JNet_372_3_label_depth]: /experiments/images/JNet_372_3_label_depth.png
[JNet_372_3_label_plane]: /experiments/images/JNet_372_3_label_plane.png
[JNet_372_3_original_depth]: /experiments/images/JNet_372_3_original_depth.png
[JNet_372_3_original_plane]: /experiments/images/JNet_372_3_original_plane.png
[JNet_372_3_output_depth]: /experiments/images/JNet_372_3_output_depth.png
[JNet_372_3_output_plane]: /experiments/images/JNet_372_3_output_plane.png
[JNet_372_4_label_depth]: /experiments/images/JNet_372_4_label_depth.png
[JNet_372_4_label_plane]: /experiments/images/JNet_372_4_label_plane.png
[JNet_372_4_original_depth]: /experiments/images/JNet_372_4_original_depth.png
[JNet_372_4_original_plane]: /experiments/images/JNet_372_4_original_plane.png
[JNet_372_4_output_depth]: /experiments/images/JNet_372_4_output_depth.png
[JNet_372_4_output_plane]: /experiments/images/JNet_372_4_output_plane.png
[JNet_372_beads_001_roi000_original_depth]: /experiments/images/JNet_372_beads_001_roi000_original_depth.png
[JNet_372_beads_001_roi000_output_depth]: /experiments/images/JNet_372_beads_001_roi000_output_depth.png
[JNet_372_beads_001_roi000_reconst_depth]: /experiments/images/JNet_372_beads_001_roi000_reconst_depth.png
[JNet_372_beads_001_roi001_original_depth]: /experiments/images/JNet_372_beads_001_roi001_original_depth.png
[JNet_372_beads_001_roi001_output_depth]: /experiments/images/JNet_372_beads_001_roi001_output_depth.png
[JNet_372_beads_001_roi001_reconst_depth]: /experiments/images/JNet_372_beads_001_roi001_reconst_depth.png
[JNet_372_beads_001_roi002_original_depth]: /experiments/images/JNet_372_beads_001_roi002_original_depth.png
[JNet_372_beads_001_roi002_output_depth]: /experiments/images/JNet_372_beads_001_roi002_output_depth.png
[JNet_372_beads_001_roi002_reconst_depth]: /experiments/images/JNet_372_beads_001_roi002_reconst_depth.png
[JNet_372_beads_001_roi003_original_depth]: /experiments/images/JNet_372_beads_001_roi003_original_depth.png
[JNet_372_beads_001_roi003_output_depth]: /experiments/images/JNet_372_beads_001_roi003_output_depth.png
[JNet_372_beads_001_roi003_reconst_depth]: /experiments/images/JNet_372_beads_001_roi003_reconst_depth.png
[JNet_372_beads_001_roi004_original_depth]: /experiments/images/JNet_372_beads_001_roi004_original_depth.png
[JNet_372_beads_001_roi004_output_depth]: /experiments/images/JNet_372_beads_001_roi004_output_depth.png
[JNet_372_beads_001_roi004_reconst_depth]: /experiments/images/JNet_372_beads_001_roi004_reconst_depth.png
[JNet_372_beads_002_roi000_original_depth]: /experiments/images/JNet_372_beads_002_roi000_original_depth.png
[JNet_372_beads_002_roi000_output_depth]: /experiments/images/JNet_372_beads_002_roi000_output_depth.png
[JNet_372_beads_002_roi000_reconst_depth]: /experiments/images/JNet_372_beads_002_roi000_reconst_depth.png
[JNet_372_beads_002_roi001_original_depth]: /experiments/images/JNet_372_beads_002_roi001_original_depth.png
[JNet_372_beads_002_roi001_output_depth]: /experiments/images/JNet_372_beads_002_roi001_output_depth.png
[JNet_372_beads_002_roi001_reconst_depth]: /experiments/images/JNet_372_beads_002_roi001_reconst_depth.png
[JNet_372_beads_002_roi002_original_depth]: /experiments/images/JNet_372_beads_002_roi002_original_depth.png
[JNet_372_beads_002_roi002_output_depth]: /experiments/images/JNet_372_beads_002_roi002_output_depth.png
[JNet_372_beads_002_roi002_reconst_depth]: /experiments/images/JNet_372_beads_002_roi002_reconst_depth.png
[JNet_372_psf_post]: /experiments/images/JNet_372_psf_post.png
[JNet_372_psf_pre]: /experiments/images/JNet_372_psf_pre.png
[finetuned]: /experiments/tmp/JNet_372_train.png
[pretrained_model]: /experiments/tmp/JNet_371_pretrain_train.png
