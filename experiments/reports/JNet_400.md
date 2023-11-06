



# JNet_400 Report
  
the parameters to replicate the results of JNet_400 vq loss=0.1, all image params fixed  
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
|seed|1106|
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
|qloss_weight|0.1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.022864609956741333, mean BCE: 0.10970525443553925
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_0_original_plane]|![JNet_385_pretrain_0_output_plane]|![JNet_385_pretrain_0_label_plane]|
  
MSE: 0.020246177911758423, BCE: 0.07590215653181076  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_0_original_depth]|![JNet_385_pretrain_0_output_depth]|![JNet_385_pretrain_0_label_depth]|
  
MSE: 0.020246177911758423, BCE: 0.07590215653181076  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_1_original_plane]|![JNet_385_pretrain_1_output_plane]|![JNet_385_pretrain_1_label_plane]|
  
MSE: 0.025456219911575317, BCE: 0.1406187266111374  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_1_original_depth]|![JNet_385_pretrain_1_output_depth]|![JNet_385_pretrain_1_label_depth]|
  
MSE: 0.025456219911575317, BCE: 0.1406187266111374  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_2_original_plane]|![JNet_385_pretrain_2_output_plane]|![JNet_385_pretrain_2_label_plane]|
  
MSE: 0.031118566170334816, BCE: 0.1803697794675827  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_2_original_depth]|![JNet_385_pretrain_2_output_depth]|![JNet_385_pretrain_2_label_depth]|
  
MSE: 0.031118566170334816, BCE: 0.1803697794675827  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_3_original_plane]|![JNet_385_pretrain_3_output_plane]|![JNet_385_pretrain_3_label_plane]|
  
MSE: 0.01665361411869526, BCE: 0.06512904912233353  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_3_original_depth]|![JNet_385_pretrain_3_output_depth]|![JNet_385_pretrain_3_label_depth]|
  
MSE: 0.01665361411869526, BCE: 0.06512904912233353  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_4_original_plane]|![JNet_385_pretrain_4_output_plane]|![JNet_385_pretrain_4_label_plane]|
  
MSE: 0.020848473533988, BCE: 0.08650662004947662  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_4_original_depth]|![JNet_385_pretrain_4_output_depth]|![JNet_385_pretrain_4_label_depth]|
  
MSE: 0.020848473533988, BCE: 0.08650662004947662  
  
mean MSE: 0.03902380168437958, mean BCE: 1.8258047103881836
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_400_0_original_plane]|![JNet_400_0_output_plane]|![JNet_400_0_label_plane]|
  
MSE: 0.028868716210126877, BCE: 1.409218668937683  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_400_0_original_depth]|![JNet_400_0_output_depth]|![JNet_400_0_label_depth]|
  
MSE: 0.028868716210126877, BCE: 1.409218668937683  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_400_1_original_plane]|![JNet_400_1_output_plane]|![JNet_400_1_label_plane]|
  
MSE: 0.04192637652158737, BCE: 1.8257262706756592  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_400_1_original_depth]|![JNet_400_1_output_depth]|![JNet_400_1_label_depth]|
  
MSE: 0.04192637652158737, BCE: 1.8257262706756592  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_400_2_original_plane]|![JNet_400_2_output_plane]|![JNet_400_2_label_plane]|
  
MSE: 0.04311203584074974, BCE: 1.9611995220184326  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_400_2_original_depth]|![JNet_400_2_output_depth]|![JNet_400_2_label_depth]|
  
MSE: 0.04311203584074974, BCE: 1.9611995220184326  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_400_3_original_plane]|![JNet_400_3_output_plane]|![JNet_400_3_label_plane]|
  
MSE: 0.038263656198978424, BCE: 1.8313908576965332  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_400_3_original_depth]|![JNet_400_3_output_depth]|![JNet_400_3_label_depth]|
  
MSE: 0.038263656198978424, BCE: 1.8313908576965332  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_400_4_original_plane]|![JNet_400_4_output_plane]|![JNet_400_4_label_plane]|
  
MSE: 0.04294822737574577, BCE: 2.1014883518218994  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_400_4_original_depth]|![JNet_400_4_output_depth]|![JNet_400_4_label_depth]|
  
MSE: 0.04294822737574577, BCE: 2.1014883518218994  

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
  
volume: 9.589872070312502, MSE: 0.00395852979272604, quantized loss: 0.000565944064874202  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_002_roi001_original_depth]|![JNet_385_pretrain_beads_002_roi001_output_depth]|![JNet_385_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 9.339445312500002, MSE: 0.0029028363060206175, quantized loss: 0.0005865816492587328  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_002_roi002_original_depth]|![JNet_385_pretrain_beads_002_roi002_output_depth]|![JNet_385_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 9.460128906250002, MSE: 0.003445687470957637, quantized loss: 0.0005728724645450711  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_400_beads_001_roi000_original_depth]|![JNet_400_beads_001_roi000_output_depth]|![JNet_400_beads_001_roi000_reconst_depth]|
  
volume: 20.581863281250005, MSE: 0.002988344756886363, quantized loss: 0.0002714259026106447  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_400_beads_001_roi001_original_depth]|![JNet_400_beads_001_roi001_output_depth]|![JNet_400_beads_001_roi001_reconst_depth]|
  
volume: 31.024291015625007, MSE: 0.00389093323610723, quantized loss: 0.0003828795161098242  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_400_beads_001_roi002_original_depth]|![JNet_400_beads_001_roi002_output_depth]|![JNet_400_beads_001_roi002_reconst_depth]|
  
volume: 19.858515625000006, MSE: 0.0029198157135397196, quantized loss: 0.0002460591495037079  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_400_beads_001_roi003_original_depth]|![JNet_400_beads_001_roi003_output_depth]|![JNet_400_beads_001_roi003_reconst_depth]|
  
volume: 32.23156835937501, MSE: 0.0039467341266572475, quantized loss: 0.0004017812025267631  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_400_beads_001_roi004_original_depth]|![JNet_400_beads_001_roi004_output_depth]|![JNet_400_beads_001_roi004_reconst_depth]|
  
volume: 21.010404296875006, MSE: 0.003149406285956502, quantized loss: 0.00025941673084162176  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_400_beads_002_roi000_original_depth]|![JNet_400_beads_002_roi000_output_depth]|![JNet_400_beads_002_roi000_reconst_depth]|
  
volume: 22.325449218750006, MSE: 0.0034237243235111237, quantized loss: 0.00026523612905293703  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_400_beads_002_roi001_original_depth]|![JNet_400_beads_002_roi001_output_depth]|![JNet_400_beads_002_roi001_reconst_depth]|
  
volume: 20.751796875000004, MSE: 0.0029418691992759705, quantized loss: 0.0002589407376945019  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_400_beads_002_roi002_original_depth]|![JNet_400_beads_002_roi002_output_depth]|![JNet_400_beads_002_roi002_reconst_depth]|
  
volume: 21.524951171875006, MSE: 0.0032430950086563826, quantized loss: 0.00025529839331284165  

|pre|post|
| :---: | :---: |
|![JNet_400_psf_pre]|![JNet_400_psf_post]|

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
[JNet_400_0_label_depth]: /experiments/images/JNet_400_0_label_depth.png
[JNet_400_0_label_plane]: /experiments/images/JNet_400_0_label_plane.png
[JNet_400_0_original_depth]: /experiments/images/JNet_400_0_original_depth.png
[JNet_400_0_original_plane]: /experiments/images/JNet_400_0_original_plane.png
[JNet_400_0_output_depth]: /experiments/images/JNet_400_0_output_depth.png
[JNet_400_0_output_plane]: /experiments/images/JNet_400_0_output_plane.png
[JNet_400_1_label_depth]: /experiments/images/JNet_400_1_label_depth.png
[JNet_400_1_label_plane]: /experiments/images/JNet_400_1_label_plane.png
[JNet_400_1_original_depth]: /experiments/images/JNet_400_1_original_depth.png
[JNet_400_1_original_plane]: /experiments/images/JNet_400_1_original_plane.png
[JNet_400_1_output_depth]: /experiments/images/JNet_400_1_output_depth.png
[JNet_400_1_output_plane]: /experiments/images/JNet_400_1_output_plane.png
[JNet_400_2_label_depth]: /experiments/images/JNet_400_2_label_depth.png
[JNet_400_2_label_plane]: /experiments/images/JNet_400_2_label_plane.png
[JNet_400_2_original_depth]: /experiments/images/JNet_400_2_original_depth.png
[JNet_400_2_original_plane]: /experiments/images/JNet_400_2_original_plane.png
[JNet_400_2_output_depth]: /experiments/images/JNet_400_2_output_depth.png
[JNet_400_2_output_plane]: /experiments/images/JNet_400_2_output_plane.png
[JNet_400_3_label_depth]: /experiments/images/JNet_400_3_label_depth.png
[JNet_400_3_label_plane]: /experiments/images/JNet_400_3_label_plane.png
[JNet_400_3_original_depth]: /experiments/images/JNet_400_3_original_depth.png
[JNet_400_3_original_plane]: /experiments/images/JNet_400_3_original_plane.png
[JNet_400_3_output_depth]: /experiments/images/JNet_400_3_output_depth.png
[JNet_400_3_output_plane]: /experiments/images/JNet_400_3_output_plane.png
[JNet_400_4_label_depth]: /experiments/images/JNet_400_4_label_depth.png
[JNet_400_4_label_plane]: /experiments/images/JNet_400_4_label_plane.png
[JNet_400_4_original_depth]: /experiments/images/JNet_400_4_original_depth.png
[JNet_400_4_original_plane]: /experiments/images/JNet_400_4_original_plane.png
[JNet_400_4_output_depth]: /experiments/images/JNet_400_4_output_depth.png
[JNet_400_4_output_plane]: /experiments/images/JNet_400_4_output_plane.png
[JNet_400_beads_001_roi000_original_depth]: /experiments/images/JNet_400_beads_001_roi000_original_depth.png
[JNet_400_beads_001_roi000_output_depth]: /experiments/images/JNet_400_beads_001_roi000_output_depth.png
[JNet_400_beads_001_roi000_reconst_depth]: /experiments/images/JNet_400_beads_001_roi000_reconst_depth.png
[JNet_400_beads_001_roi001_original_depth]: /experiments/images/JNet_400_beads_001_roi001_original_depth.png
[JNet_400_beads_001_roi001_output_depth]: /experiments/images/JNet_400_beads_001_roi001_output_depth.png
[JNet_400_beads_001_roi001_reconst_depth]: /experiments/images/JNet_400_beads_001_roi001_reconst_depth.png
[JNet_400_beads_001_roi002_original_depth]: /experiments/images/JNet_400_beads_001_roi002_original_depth.png
[JNet_400_beads_001_roi002_output_depth]: /experiments/images/JNet_400_beads_001_roi002_output_depth.png
[JNet_400_beads_001_roi002_reconst_depth]: /experiments/images/JNet_400_beads_001_roi002_reconst_depth.png
[JNet_400_beads_001_roi003_original_depth]: /experiments/images/JNet_400_beads_001_roi003_original_depth.png
[JNet_400_beads_001_roi003_output_depth]: /experiments/images/JNet_400_beads_001_roi003_output_depth.png
[JNet_400_beads_001_roi003_reconst_depth]: /experiments/images/JNet_400_beads_001_roi003_reconst_depth.png
[JNet_400_beads_001_roi004_original_depth]: /experiments/images/JNet_400_beads_001_roi004_original_depth.png
[JNet_400_beads_001_roi004_output_depth]: /experiments/images/JNet_400_beads_001_roi004_output_depth.png
[JNet_400_beads_001_roi004_reconst_depth]: /experiments/images/JNet_400_beads_001_roi004_reconst_depth.png
[JNet_400_beads_002_roi000_original_depth]: /experiments/images/JNet_400_beads_002_roi000_original_depth.png
[JNet_400_beads_002_roi000_output_depth]: /experiments/images/JNet_400_beads_002_roi000_output_depth.png
[JNet_400_beads_002_roi000_reconst_depth]: /experiments/images/JNet_400_beads_002_roi000_reconst_depth.png
[JNet_400_beads_002_roi001_original_depth]: /experiments/images/JNet_400_beads_002_roi001_original_depth.png
[JNet_400_beads_002_roi001_output_depth]: /experiments/images/JNet_400_beads_002_roi001_output_depth.png
[JNet_400_beads_002_roi001_reconst_depth]: /experiments/images/JNet_400_beads_002_roi001_reconst_depth.png
[JNet_400_beads_002_roi002_original_depth]: /experiments/images/JNet_400_beads_002_roi002_original_depth.png
[JNet_400_beads_002_roi002_output_depth]: /experiments/images/JNet_400_beads_002_roi002_output_depth.png
[JNet_400_beads_002_roi002_reconst_depth]: /experiments/images/JNet_400_beads_002_roi002_reconst_depth.png
[JNet_400_psf_post]: /experiments/images/JNet_400_psf_post.png
[JNet_400_psf_pre]: /experiments/images/JNet_400_psf_pre.png
[finetuned]: /experiments/tmp/JNet_400_train.png
[pretrained_model]: /experiments/tmp/JNet_385_pretrain_train.png
