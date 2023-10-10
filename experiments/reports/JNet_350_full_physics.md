



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
  
mean MSE: 0.01788933575153351, mean BCE: 0.06870049238204956
### 0

|original|output|label|
| :---: | :---: | :---: |
|![0_original_plane]|![0_output_plane]|![0_label_plane]|
  
MSE: 0.025192763656377792, BCE: 0.10603463649749756  

|original|output|label|
| :---: | :---: | :---: |
|![0_original_depth]|![0_output_depth]|![0_label_depth]|
  
MSE: 0.025192763656377792, BCE: 0.10603463649749756  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![1_original_plane]|![1_output_plane]|![1_label_plane]|
  
MSE: 0.011605626903474331, BCE: 0.04644422605633736  

|original|output|label|
| :---: | :---: | :---: |
|![1_original_depth]|![1_output_depth]|![1_label_depth]|
  
MSE: 0.011605626903474331, BCE: 0.04644422605633736  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![2_original_plane]|![2_output_plane]|![2_label_plane]|
  
MSE: 0.018291732296347618, BCE: 0.06483753025531769  

|original|output|label|
| :---: | :---: | :---: |
|![2_original_depth]|![2_output_depth]|![2_label_depth]|
  
MSE: 0.018291732296347618, BCE: 0.06483753025531769  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![3_original_plane]|![3_output_plane]|![3_label_plane]|
  
MSE: 0.016247157007455826, BCE: 0.059216469526290894  

|original|output|label|
| :---: | :---: | :---: |
|![3_original_depth]|![3_output_depth]|![3_label_depth]|
  
MSE: 0.016247157007455826, BCE: 0.059216469526290894  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![4_original_plane]|![4_output_plane]|![4_label_plane]|
  
MSE: 0.01810939610004425, BCE: 0.06696958839893341  

|original|output|label|
| :---: | :---: | :---: |
|![4_original_depth]|![4_output_depth]|![4_label_depth]|
  
MSE: 0.01810939610004425, BCE: 0.06696958839893341  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![beads_001_roi000_original_depth]|![beads_001_roi000_output_depth]|![beads_001_roi000_reconst_depth]|
  
volume: 10.067500000000003, MSE: 0.0004030589770991355, quantized loss: 4.2092495277756825e-05  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![beads_001_roi001_original_depth]|![beads_001_roi001_output_depth]|![beads_001_roi001_reconst_depth]|
  
volume: 16.029250000000005, MSE: 0.0008094462100416422, quantized loss: 6.097833829699084e-05  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![beads_001_roi002_original_depth]|![beads_001_roi002_output_depth]|![beads_001_roi002_reconst_depth]|
  
volume: 10.091875000000002, MSE: 0.000292979326331988, quantized loss: 4.231282582622953e-05  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![beads_001_roi003_original_depth]|![beads_001_roi003_output_depth]|![beads_001_roi003_reconst_depth]|
  
volume: 16.619125000000004, MSE: 0.0006046192138455808, quantized loss: 6.181728531373665e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![beads_001_roi004_original_depth]|![beads_001_roi004_output_depth]|![beads_001_roi004_reconst_depth]|
  
volume: 11.042500000000002, MSE: 0.00029620385612361133, quantized loss: 4.026446913485415e-05  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![beads_002_roi000_original_depth]|![beads_002_roi000_output_depth]|![beads_002_roi000_reconst_depth]|
  
volume: 11.832625000000002, MSE: 0.0003028359788004309, quantized loss: 4.1795781726250425e-05  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![beads_002_roi001_original_depth]|![beads_002_roi001_output_depth]|![beads_002_roi001_reconst_depth]|
  
volume: 10.804750000000002, MSE: 0.00030398997478187084, quantized loss: 4.160175012657419e-05  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![beads_002_roi002_original_depth]|![beads_002_roi002_output_depth]|![beads_002_roi002_reconst_depth]|
  
volume: 11.232750000000003, MSE: 0.00029029863071627915, quantized loss: 4.04145430366043e-05  
  



[0_label_depth]: /experiments/images/JNet_350_full_physics_0_label_depth.png
[0_label_plane]: /experiments/images/JNet_350_full_physics_0_label_plane.png
[0_original_depth]: /experiments/images/JNet_350_full_physics_0_original_depth.png
[0_original_plane]: /experiments/images/JNet_350_full_physics_0_original_plane.png
[0_output_depth]: /experiments/images/JNet_350_full_physics_0_output_depth.png
[0_output_plane]: /experiments/images/JNet_350_full_physics_0_output_plane.png
[1_label_depth]: /experiments/images/JNet_350_full_physics_1_label_depth.png
[1_label_plane]: /experiments/images/JNet_350_full_physics_1_label_plane.png
[1_original_depth]: /experiments/images/JNet_350_full_physics_1_original_depth.png
[1_original_plane]: /experiments/images/JNet_350_full_physics_1_original_plane.png
[1_output_depth]: /experiments/images/JNet_350_full_physics_1_output_depth.png
[1_output_plane]: /experiments/images/JNet_350_full_physics_1_output_plane.png
[2_label_depth]: /experiments/images/JNet_350_full_physics_2_label_depth.png
[2_label_plane]: /experiments/images/JNet_350_full_physics_2_label_plane.png
[2_original_depth]: /experiments/images/JNet_350_full_physics_2_original_depth.png
[2_original_plane]: /experiments/images/JNet_350_full_physics_2_original_plane.png
[2_output_depth]: /experiments/images/JNet_350_full_physics_2_output_depth.png
[2_output_plane]: /experiments/images/JNet_350_full_physics_2_output_plane.png
[3_label_depth]: /experiments/images/JNet_350_full_physics_3_label_depth.png
[3_label_plane]: /experiments/images/JNet_350_full_physics_3_label_plane.png
[3_original_depth]: /experiments/images/JNet_350_full_physics_3_original_depth.png
[3_original_plane]: /experiments/images/JNet_350_full_physics_3_original_plane.png
[3_output_depth]: /experiments/images/JNet_350_full_physics_3_output_depth.png
[3_output_plane]: /experiments/images/JNet_350_full_physics_3_output_plane.png
[4_label_depth]: /experiments/images/JNet_350_full_physics_4_label_depth.png
[4_label_plane]: /experiments/images/JNet_350_full_physics_4_label_plane.png
[4_original_depth]: /experiments/images/JNet_350_full_physics_4_original_depth.png
[4_original_plane]: /experiments/images/JNet_350_full_physics_4_original_plane.png
[4_output_depth]: /experiments/images/JNet_350_full_physics_4_output_depth.png
[4_output_plane]: /experiments/images/JNet_350_full_physics_4_output_plane.png
[beads_001_roi000_original_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi000_original_depth.png
[beads_001_roi000_output_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi000_output_depth.png
[beads_001_roi000_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi000_reconst_depth.png
[beads_001_roi001_original_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi001_original_depth.png
[beads_001_roi001_output_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi001_output_depth.png
[beads_001_roi001_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi001_reconst_depth.png
[beads_001_roi002_original_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi002_original_depth.png
[beads_001_roi002_output_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi002_output_depth.png
[beads_001_roi002_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi002_reconst_depth.png
[beads_001_roi003_original_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi003_original_depth.png
[beads_001_roi003_output_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi003_output_depth.png
[beads_001_roi003_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi003_reconst_depth.png
[beads_001_roi004_original_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi004_original_depth.png
[beads_001_roi004_output_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi004_output_depth.png
[beads_001_roi004_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_001_roi004_reconst_depth.png
[beads_002_roi000_original_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi000_original_depth.png
[beads_002_roi000_output_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi000_output_depth.png
[beads_002_roi000_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi000_reconst_depth.png
[beads_002_roi001_original_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi001_original_depth.png
[beads_002_roi001_output_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi001_output_depth.png
[beads_002_roi001_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi001_reconst_depth.png
[beads_002_roi002_original_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi002_original_depth.png
[beads_002_roi002_output_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi002_output_depth.png
[beads_002_roi002_reconst_depth]: /experiments/images/JNet_350_full_physics_beads_002_roi002_reconst_depth.png
[finetuned]: /experiments/tmp/JNet_350_full_physics_train.png
[pretrained_model]: /experiments/tmp/JNet_349_pretrain_train.png
