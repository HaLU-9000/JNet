



# JNet_396 Report
  
the parameters to replicate the results of JNet_396 train val   
pretrained model : JNet_385_pretrain
## Model Parameters
  

|Parameter|Value|Comment|
| :--- | :--- | :--- |
|hidden_channels_list|[16, 32, 64, 128, 256]||
|attn_list|[False, False, False, False, False]||
|nblocks|2||
|activation|nn.ReLU(inplace=True)||
|dropout|0.0||
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
|mask|False|
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
|train|True|
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
  
mean MSE: 0.030026143416762352, mean BCE: 0.1356639415025711
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_0_original_plane]|![JNet_385_pretrain_0_output_plane]|![JNet_385_pretrain_0_label_plane]|
  
MSE: 0.03415403142571449, BCE: 0.16422900557518005  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_0_original_depth]|![JNet_385_pretrain_0_output_depth]|![JNet_385_pretrain_0_label_depth]|
  
MSE: 0.03415403142571449, BCE: 0.16422900557518005  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_1_original_plane]|![JNet_385_pretrain_1_output_plane]|![JNet_385_pretrain_1_label_plane]|
  
MSE: 0.028512626886367798, BCE: 0.12015195190906525  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_1_original_depth]|![JNet_385_pretrain_1_output_depth]|![JNet_385_pretrain_1_label_depth]|
  
MSE: 0.028512626886367798, BCE: 0.12015195190906525  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_2_original_plane]|![JNet_385_pretrain_2_output_plane]|![JNet_385_pretrain_2_label_plane]|
  
MSE: 0.023525526747107506, BCE: 0.1035182848572731  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_2_original_depth]|![JNet_385_pretrain_2_output_depth]|![JNet_385_pretrain_2_label_depth]|
  
MSE: 0.023525526747107506, BCE: 0.1035182848572731  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_3_original_plane]|![JNet_385_pretrain_3_output_plane]|![JNet_385_pretrain_3_label_plane]|
  
MSE: 0.04337000846862793, BCE: 0.19795554876327515  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_3_original_depth]|![JNet_385_pretrain_3_output_depth]|![JNet_385_pretrain_3_label_depth]|
  
MSE: 0.04337000846862793, BCE: 0.19795554876327515  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_4_original_plane]|![JNet_385_pretrain_4_output_plane]|![JNet_385_pretrain_4_label_plane]|
  
MSE: 0.020568521693348885, BCE: 0.09246491640806198  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_4_original_depth]|![JNet_385_pretrain_4_output_depth]|![JNet_385_pretrain_4_label_depth]|
  
MSE: 0.020568521693348885, BCE: 0.09246491640806198  
  
mean MSE: 0.027831431478261948, mean BCE: 0.2945743799209595
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_396_0_original_plane]|![JNet_396_0_output_plane]|![JNet_396_0_label_plane]|
  
MSE: 0.035127438604831696, BCE: 0.3743075728416443  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_396_0_original_depth]|![JNet_396_0_output_depth]|![JNet_396_0_label_depth]|
  
MSE: 0.035127438604831696, BCE: 0.3743075728416443  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_396_1_original_plane]|![JNet_396_1_output_plane]|![JNet_396_1_label_plane]|
  
MSE: 0.026197422295808792, BCE: 0.2951500713825226  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_396_1_original_depth]|![JNet_396_1_output_depth]|![JNet_396_1_label_depth]|
  
MSE: 0.026197422295808792, BCE: 0.2951500713825226  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_396_2_original_plane]|![JNet_396_2_output_plane]|![JNet_396_2_label_plane]|
  
MSE: 0.027046672999858856, BCE: 0.2821400761604309  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_396_2_original_depth]|![JNet_396_2_output_depth]|![JNet_396_2_label_depth]|
  
MSE: 0.027046672999858856, BCE: 0.2821400761604309  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_396_3_original_plane]|![JNet_396_3_output_plane]|![JNet_396_3_label_plane]|
  
MSE: 0.022136053070425987, BCE: 0.21749402582645416  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_396_3_original_depth]|![JNet_396_3_output_depth]|![JNet_396_3_label_depth]|
  
MSE: 0.022136053070425987, BCE: 0.21749402582645416  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_396_4_original_plane]|![JNet_396_4_output_plane]|![JNet_396_4_label_plane]|
  
MSE: 0.028649579733610153, BCE: 0.30378010869026184  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_396_4_original_depth]|![JNet_396_4_output_depth]|![JNet_396_4_label_depth]|
  
MSE: 0.028649579733610153, BCE: 0.30378010869026184  

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
|![JNet_396_beads_001_roi000_original_depth]|![JNet_396_beads_001_roi000_output_depth]|![JNet_396_beads_001_roi000_reconst_depth]|
  
volume: 17.016568359375004, MSE: 0.0020395778119564056, quantized loss: 0.00021730545267928392  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_396_beads_001_roi001_original_depth]|![JNet_396_beads_001_roi001_output_depth]|![JNet_396_beads_001_roi001_reconst_depth]|
  
volume: 25.796460937500004, MSE: 0.002905680565163493, quantized loss: 0.0003130545374006033  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_396_beads_001_roi002_original_depth]|![JNet_396_beads_001_roi002_output_depth]|![JNet_396_beads_001_roi002_reconst_depth]|
  
volume: 16.298306640625004, MSE: 0.002019428415223956, quantized loss: 0.00018508532957639545  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_396_beads_001_roi003_original_depth]|![JNet_396_beads_001_roi003_output_depth]|![JNet_396_beads_001_roi003_reconst_depth]|
  
volume: 25.672554687500007, MSE: 0.002740963129326701, quantized loss: 0.0002941408602055162  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_396_beads_001_roi004_original_depth]|![JNet_396_beads_001_roi004_output_depth]|![JNet_396_beads_001_roi004_reconst_depth]|
  
volume: 16.739044921875003, MSE: 0.0022065250668674707, quantized loss: 0.00017608290363568813  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_396_beads_002_roi000_original_depth]|![JNet_396_beads_002_roi000_output_depth]|![JNet_396_beads_002_roi000_reconst_depth]|
  
volume: 17.534896484375004, MSE: 0.0023969372268766165, quantized loss: 0.00017367533291690052  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_396_beads_002_roi001_original_depth]|![JNet_396_beads_002_roi001_output_depth]|![JNet_396_beads_002_roi001_reconst_depth]|
  
volume: 16.888195312500002, MSE: 0.002008745213970542, quantized loss: 0.00017814786406233907  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_396_beads_002_roi002_original_depth]|![JNet_396_beads_002_roi002_output_depth]|![JNet_396_beads_002_roi002_reconst_depth]|
  
volume: 17.221931640625005, MSE: 0.0022372461389750242, quantized loss: 0.00017733518325258046  

|pre|post|
| :---: | :---: |
|![JNet_396_psf_pre]|![JNet_396_psf_post]|

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
      (dropout1): Dropout(p=0.0, inplace=False)  
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
        (dropout1): Dropout(p=0.0, inplace=False)  
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
          (dropout1): Dropout(p=0.0, inplace=False)  
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
            (dropout1): Dropout(p=0.0, inplace=False)  
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
              (dropout1): Dropout(p=0.0, inplace=False)  
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
              (dropout1): Dropout(p=0.0, inplace=False)  
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
            (dropout1): Dropout(p=0.0, inplace=False)  
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
          (dropout1): Dropout(p=0.0, inplace=False)  
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
        (dropout1): Dropout(p=0.0, inplace=False)  
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
      (dropout1): Dropout(p=0.0, inplace=False)  
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
[JNet_396_0_label_depth]: /experiments/images/JNet_396_0_label_depth.png
[JNet_396_0_label_plane]: /experiments/images/JNet_396_0_label_plane.png
[JNet_396_0_original_depth]: /experiments/images/JNet_396_0_original_depth.png
[JNet_396_0_original_plane]: /experiments/images/JNet_396_0_original_plane.png
[JNet_396_0_output_depth]: /experiments/images/JNet_396_0_output_depth.png
[JNet_396_0_output_plane]: /experiments/images/JNet_396_0_output_plane.png
[JNet_396_1_label_depth]: /experiments/images/JNet_396_1_label_depth.png
[JNet_396_1_label_plane]: /experiments/images/JNet_396_1_label_plane.png
[JNet_396_1_original_depth]: /experiments/images/JNet_396_1_original_depth.png
[JNet_396_1_original_plane]: /experiments/images/JNet_396_1_original_plane.png
[JNet_396_1_output_depth]: /experiments/images/JNet_396_1_output_depth.png
[JNet_396_1_output_plane]: /experiments/images/JNet_396_1_output_plane.png
[JNet_396_2_label_depth]: /experiments/images/JNet_396_2_label_depth.png
[JNet_396_2_label_plane]: /experiments/images/JNet_396_2_label_plane.png
[JNet_396_2_original_depth]: /experiments/images/JNet_396_2_original_depth.png
[JNet_396_2_original_plane]: /experiments/images/JNet_396_2_original_plane.png
[JNet_396_2_output_depth]: /experiments/images/JNet_396_2_output_depth.png
[JNet_396_2_output_plane]: /experiments/images/JNet_396_2_output_plane.png
[JNet_396_3_label_depth]: /experiments/images/JNet_396_3_label_depth.png
[JNet_396_3_label_plane]: /experiments/images/JNet_396_3_label_plane.png
[JNet_396_3_original_depth]: /experiments/images/JNet_396_3_original_depth.png
[JNet_396_3_original_plane]: /experiments/images/JNet_396_3_original_plane.png
[JNet_396_3_output_depth]: /experiments/images/JNet_396_3_output_depth.png
[JNet_396_3_output_plane]: /experiments/images/JNet_396_3_output_plane.png
[JNet_396_4_label_depth]: /experiments/images/JNet_396_4_label_depth.png
[JNet_396_4_label_plane]: /experiments/images/JNet_396_4_label_plane.png
[JNet_396_4_original_depth]: /experiments/images/JNet_396_4_original_depth.png
[JNet_396_4_original_plane]: /experiments/images/JNet_396_4_original_plane.png
[JNet_396_4_output_depth]: /experiments/images/JNet_396_4_output_depth.png
[JNet_396_4_output_plane]: /experiments/images/JNet_396_4_output_plane.png
[JNet_396_beads_001_roi000_original_depth]: /experiments/images/JNet_396_beads_001_roi000_original_depth.png
[JNet_396_beads_001_roi000_output_depth]: /experiments/images/JNet_396_beads_001_roi000_output_depth.png
[JNet_396_beads_001_roi000_reconst_depth]: /experiments/images/JNet_396_beads_001_roi000_reconst_depth.png
[JNet_396_beads_001_roi001_original_depth]: /experiments/images/JNet_396_beads_001_roi001_original_depth.png
[JNet_396_beads_001_roi001_output_depth]: /experiments/images/JNet_396_beads_001_roi001_output_depth.png
[JNet_396_beads_001_roi001_reconst_depth]: /experiments/images/JNet_396_beads_001_roi001_reconst_depth.png
[JNet_396_beads_001_roi002_original_depth]: /experiments/images/JNet_396_beads_001_roi002_original_depth.png
[JNet_396_beads_001_roi002_output_depth]: /experiments/images/JNet_396_beads_001_roi002_output_depth.png
[JNet_396_beads_001_roi002_reconst_depth]: /experiments/images/JNet_396_beads_001_roi002_reconst_depth.png
[JNet_396_beads_001_roi003_original_depth]: /experiments/images/JNet_396_beads_001_roi003_original_depth.png
[JNet_396_beads_001_roi003_output_depth]: /experiments/images/JNet_396_beads_001_roi003_output_depth.png
[JNet_396_beads_001_roi003_reconst_depth]: /experiments/images/JNet_396_beads_001_roi003_reconst_depth.png
[JNet_396_beads_001_roi004_original_depth]: /experiments/images/JNet_396_beads_001_roi004_original_depth.png
[JNet_396_beads_001_roi004_output_depth]: /experiments/images/JNet_396_beads_001_roi004_output_depth.png
[JNet_396_beads_001_roi004_reconst_depth]: /experiments/images/JNet_396_beads_001_roi004_reconst_depth.png
[JNet_396_beads_002_roi000_original_depth]: /experiments/images/JNet_396_beads_002_roi000_original_depth.png
[JNet_396_beads_002_roi000_output_depth]: /experiments/images/JNet_396_beads_002_roi000_output_depth.png
[JNet_396_beads_002_roi000_reconst_depth]: /experiments/images/JNet_396_beads_002_roi000_reconst_depth.png
[JNet_396_beads_002_roi001_original_depth]: /experiments/images/JNet_396_beads_002_roi001_original_depth.png
[JNet_396_beads_002_roi001_output_depth]: /experiments/images/JNet_396_beads_002_roi001_output_depth.png
[JNet_396_beads_002_roi001_reconst_depth]: /experiments/images/JNet_396_beads_002_roi001_reconst_depth.png
[JNet_396_beads_002_roi002_original_depth]: /experiments/images/JNet_396_beads_002_roi002_original_depth.png
[JNet_396_beads_002_roi002_output_depth]: /experiments/images/JNet_396_beads_002_roi002_output_depth.png
[JNet_396_beads_002_roi002_reconst_depth]: /experiments/images/JNet_396_beads_002_roi002_reconst_depth.png
[JNet_396_psf_post]: /experiments/images/JNet_396_psf_post.png
[JNet_396_psf_pre]: /experiments/images/JNet_396_psf_pre.png
[finetuned]: /experiments/tmp/JNet_396_train.png
[pretrained_model]: /experiments/tmp/JNet_385_pretrain_train.png
