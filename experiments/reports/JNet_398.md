



# JNet_398 Report
  
the parameters to replicate the results of JNet_396 train val   
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
|qloss_weight|1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.020958123728632927, mean BCE: 0.10498839616775513
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_0_original_plane]|![JNet_385_pretrain_0_output_plane]|![JNet_385_pretrain_0_label_plane]|
  
MSE: 0.014574169181287289, BCE: 0.05200992152094841  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_0_original_depth]|![JNet_385_pretrain_0_output_depth]|![JNet_385_pretrain_0_label_depth]|
  
MSE: 0.014574169181287289, BCE: 0.05200992152094841  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_1_original_plane]|![JNet_385_pretrain_1_output_plane]|![JNet_385_pretrain_1_label_plane]|
  
MSE: 0.030931448563933372, BCE: 0.16654448211193085  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_1_original_depth]|![JNet_385_pretrain_1_output_depth]|![JNet_385_pretrain_1_label_depth]|
  
MSE: 0.030931448563933372, BCE: 0.16654448211193085  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_2_original_plane]|![JNet_385_pretrain_2_output_plane]|![JNet_385_pretrain_2_label_plane]|
  
MSE: 0.026216361671686172, BCE: 0.1598225235939026  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_2_original_depth]|![JNet_385_pretrain_2_output_depth]|![JNet_385_pretrain_2_label_depth]|
  
MSE: 0.026216361671686172, BCE: 0.1598225235939026  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_3_original_plane]|![JNet_385_pretrain_3_output_plane]|![JNet_385_pretrain_3_label_plane]|
  
MSE: 0.02023870311677456, BCE: 0.08401476591825485  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_3_original_depth]|![JNet_385_pretrain_3_output_depth]|![JNet_385_pretrain_3_label_depth]|
  
MSE: 0.02023870311677456, BCE: 0.08401476591825485  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_4_original_plane]|![JNet_385_pretrain_4_output_plane]|![JNet_385_pretrain_4_label_plane]|
  
MSE: 0.012829935178160667, BCE: 0.06255026906728745  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_4_original_depth]|![JNet_385_pretrain_4_output_depth]|![JNet_385_pretrain_4_label_depth]|
  
MSE: 0.012829935178160667, BCE: 0.06255026906728745  
  
mean MSE: 0.03606511652469635, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_398_0_original_plane]|![JNet_398_0_output_plane]|![JNet_398_0_label_plane]|
  
MSE: 0.03569302707910538, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_398_0_original_depth]|![JNet_398_0_output_depth]|![JNet_398_0_label_depth]|
  
MSE: 0.03569302707910538, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_398_1_original_plane]|![JNet_398_1_output_plane]|![JNet_398_1_label_plane]|
  
MSE: 0.0388558954000473, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_398_1_original_depth]|![JNet_398_1_output_depth]|![JNet_398_1_label_depth]|
  
MSE: 0.0388558954000473, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_398_2_original_plane]|![JNet_398_2_output_plane]|![JNet_398_2_label_plane]|
  
MSE: 0.04792098328471184, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_398_2_original_depth]|![JNet_398_2_output_depth]|![JNet_398_2_label_depth]|
  
MSE: 0.04792098328471184, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_398_3_original_plane]|![JNet_398_3_output_plane]|![JNet_398_3_label_plane]|
  
MSE: 0.028285102918744087, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_398_3_original_depth]|![JNet_398_3_output_depth]|![JNet_398_3_label_depth]|
  
MSE: 0.028285102918744087, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_398_4_original_plane]|![JNet_398_4_output_plane]|![JNet_398_4_label_plane]|
  
MSE: 0.029570577666163445, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_398_4_original_depth]|![JNet_398_4_output_depth]|![JNet_398_4_label_depth]|
  
MSE: 0.029570577666163445, BCE: nan  

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
|![JNet_398_beads_001_roi000_original_depth]|![JNet_398_beads_001_roi000_output_depth]|![JNet_398_beads_001_roi000_reconst_depth]|
  
volume: 8.375613281250002, MSE: 0.0019645069260150194, quantized loss: 2.1711134650104214e-06  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_398_beads_001_roi001_original_depth]|![JNet_398_beads_001_roi001_output_depth]|![JNet_398_beads_001_roi001_reconst_depth]|
  
volume: 13.417898437500003, MSE: 0.0024052737280726433, quantized loss: 3.4577205951791257e-06  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_398_beads_001_roi002_original_depth]|![JNet_398_beads_001_roi002_output_depth]|![JNet_398_beads_001_roi002_reconst_depth]|
  
volume: 8.342371093750002, MSE: 0.0019969355780631304, quantized loss: 2.157666813218384e-06  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_398_beads_001_roi003_original_depth]|![JNet_398_beads_001_roi003_output_depth]|![JNet_398_beads_001_roi003_reconst_depth]|
  
volume: 14.493765625000004, MSE: 0.002913625678047538, quantized loss: 3.3784569950512378e-06  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_398_beads_001_roi004_original_depth]|![JNet_398_beads_001_roi004_output_depth]|![JNet_398_beads_001_roi004_reconst_depth]|
  
volume: 9.676927734375003, MSE: 0.002556939609348774, quantized loss: 2.435724809402018e-06  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_398_beads_002_roi000_original_depth]|![JNet_398_beads_002_roi000_output_depth]|![JNet_398_beads_002_roi000_reconst_depth]|
  
volume: 10.590038085937502, MSE: 0.002904419554397464, quantized loss: 2.800584070428158e-06  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_398_beads_002_roi001_original_depth]|![JNet_398_beads_002_roi001_output_depth]|![JNet_398_beads_002_roi001_reconst_depth]|
  
volume: 9.204428710937503, MSE: 0.0021285098046064377, quantized loss: 3.232494464100455e-06  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_398_beads_002_roi002_original_depth]|![JNet_398_beads_002_roi002_output_depth]|![JNet_398_beads_002_roi002_reconst_depth]|
  
volume: 9.795581054687503, MSE: 0.0025394822005182505, quantized loss: 2.6481188797333743e-06  

|pre|post|
| :---: | :---: |
|![JNet_398_psf_pre]|![JNet_398_psf_post]|

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
[JNet_398_0_label_depth]: /experiments/images/JNet_398_0_label_depth.png
[JNet_398_0_label_plane]: /experiments/images/JNet_398_0_label_plane.png
[JNet_398_0_original_depth]: /experiments/images/JNet_398_0_original_depth.png
[JNet_398_0_original_plane]: /experiments/images/JNet_398_0_original_plane.png
[JNet_398_0_output_depth]: /experiments/images/JNet_398_0_output_depth.png
[JNet_398_0_output_plane]: /experiments/images/JNet_398_0_output_plane.png
[JNet_398_1_label_depth]: /experiments/images/JNet_398_1_label_depth.png
[JNet_398_1_label_plane]: /experiments/images/JNet_398_1_label_plane.png
[JNet_398_1_original_depth]: /experiments/images/JNet_398_1_original_depth.png
[JNet_398_1_original_plane]: /experiments/images/JNet_398_1_original_plane.png
[JNet_398_1_output_depth]: /experiments/images/JNet_398_1_output_depth.png
[JNet_398_1_output_plane]: /experiments/images/JNet_398_1_output_plane.png
[JNet_398_2_label_depth]: /experiments/images/JNet_398_2_label_depth.png
[JNet_398_2_label_plane]: /experiments/images/JNet_398_2_label_plane.png
[JNet_398_2_original_depth]: /experiments/images/JNet_398_2_original_depth.png
[JNet_398_2_original_plane]: /experiments/images/JNet_398_2_original_plane.png
[JNet_398_2_output_depth]: /experiments/images/JNet_398_2_output_depth.png
[JNet_398_2_output_plane]: /experiments/images/JNet_398_2_output_plane.png
[JNet_398_3_label_depth]: /experiments/images/JNet_398_3_label_depth.png
[JNet_398_3_label_plane]: /experiments/images/JNet_398_3_label_plane.png
[JNet_398_3_original_depth]: /experiments/images/JNet_398_3_original_depth.png
[JNet_398_3_original_plane]: /experiments/images/JNet_398_3_original_plane.png
[JNet_398_3_output_depth]: /experiments/images/JNet_398_3_output_depth.png
[JNet_398_3_output_plane]: /experiments/images/JNet_398_3_output_plane.png
[JNet_398_4_label_depth]: /experiments/images/JNet_398_4_label_depth.png
[JNet_398_4_label_plane]: /experiments/images/JNet_398_4_label_plane.png
[JNet_398_4_original_depth]: /experiments/images/JNet_398_4_original_depth.png
[JNet_398_4_original_plane]: /experiments/images/JNet_398_4_original_plane.png
[JNet_398_4_output_depth]: /experiments/images/JNet_398_4_output_depth.png
[JNet_398_4_output_plane]: /experiments/images/JNet_398_4_output_plane.png
[JNet_398_beads_001_roi000_original_depth]: /experiments/images/JNet_398_beads_001_roi000_original_depth.png
[JNet_398_beads_001_roi000_output_depth]: /experiments/images/JNet_398_beads_001_roi000_output_depth.png
[JNet_398_beads_001_roi000_reconst_depth]: /experiments/images/JNet_398_beads_001_roi000_reconst_depth.png
[JNet_398_beads_001_roi001_original_depth]: /experiments/images/JNet_398_beads_001_roi001_original_depth.png
[JNet_398_beads_001_roi001_output_depth]: /experiments/images/JNet_398_beads_001_roi001_output_depth.png
[JNet_398_beads_001_roi001_reconst_depth]: /experiments/images/JNet_398_beads_001_roi001_reconst_depth.png
[JNet_398_beads_001_roi002_original_depth]: /experiments/images/JNet_398_beads_001_roi002_original_depth.png
[JNet_398_beads_001_roi002_output_depth]: /experiments/images/JNet_398_beads_001_roi002_output_depth.png
[JNet_398_beads_001_roi002_reconst_depth]: /experiments/images/JNet_398_beads_001_roi002_reconst_depth.png
[JNet_398_beads_001_roi003_original_depth]: /experiments/images/JNet_398_beads_001_roi003_original_depth.png
[JNet_398_beads_001_roi003_output_depth]: /experiments/images/JNet_398_beads_001_roi003_output_depth.png
[JNet_398_beads_001_roi003_reconst_depth]: /experiments/images/JNet_398_beads_001_roi003_reconst_depth.png
[JNet_398_beads_001_roi004_original_depth]: /experiments/images/JNet_398_beads_001_roi004_original_depth.png
[JNet_398_beads_001_roi004_output_depth]: /experiments/images/JNet_398_beads_001_roi004_output_depth.png
[JNet_398_beads_001_roi004_reconst_depth]: /experiments/images/JNet_398_beads_001_roi004_reconst_depth.png
[JNet_398_beads_002_roi000_original_depth]: /experiments/images/JNet_398_beads_002_roi000_original_depth.png
[JNet_398_beads_002_roi000_output_depth]: /experiments/images/JNet_398_beads_002_roi000_output_depth.png
[JNet_398_beads_002_roi000_reconst_depth]: /experiments/images/JNet_398_beads_002_roi000_reconst_depth.png
[JNet_398_beads_002_roi001_original_depth]: /experiments/images/JNet_398_beads_002_roi001_original_depth.png
[JNet_398_beads_002_roi001_output_depth]: /experiments/images/JNet_398_beads_002_roi001_output_depth.png
[JNet_398_beads_002_roi001_reconst_depth]: /experiments/images/JNet_398_beads_002_roi001_reconst_depth.png
[JNet_398_beads_002_roi002_original_depth]: /experiments/images/JNet_398_beads_002_roi002_original_depth.png
[JNet_398_beads_002_roi002_output_depth]: /experiments/images/JNet_398_beads_002_roi002_output_depth.png
[JNet_398_beads_002_roi002_reconst_depth]: /experiments/images/JNet_398_beads_002_roi002_reconst_depth.png
[JNet_398_psf_post]: /experiments/images/JNet_398_psf_post.png
[JNet_398_psf_pre]: /experiments/images/JNet_398_psf_pre.png
[finetuned]: /experiments/tmp/JNet_398_train.png
[pretrained_model]: /experiments/tmp/JNet_385_pretrain_train.png
