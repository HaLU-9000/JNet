



# JNet_402 Report
  
the parameters to replicate the results of JNet_402. softmax with temp, all image params fixed  
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
|qloss_weight|0.0|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.024046605452895164, mean BCE: 0.09302738308906555
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_0_original_plane]|![JNet_385_pretrain_0_output_plane]|![JNet_385_pretrain_0_label_plane]|
  
MSE: 0.022824229672551155, BCE: 0.08381031453609467  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_0_original_depth]|![JNet_385_pretrain_0_output_depth]|![JNet_385_pretrain_0_label_depth]|
  
MSE: 0.022824229672551155, BCE: 0.08381031453609467  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_1_original_plane]|![JNet_385_pretrain_1_output_plane]|![JNet_385_pretrain_1_label_plane]|
  
MSE: 0.019501347094774246, BCE: 0.08288253843784332  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_1_original_depth]|![JNet_385_pretrain_1_output_depth]|![JNet_385_pretrain_1_label_depth]|
  
MSE: 0.019501347094774246, BCE: 0.08288253843784332  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_2_original_plane]|![JNet_385_pretrain_2_output_plane]|![JNet_385_pretrain_2_label_plane]|
  
MSE: 0.02154463902115822, BCE: 0.0777846947312355  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_2_original_depth]|![JNet_385_pretrain_2_output_depth]|![JNet_385_pretrain_2_label_depth]|
  
MSE: 0.02154463902115822, BCE: 0.0777846947312355  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_3_original_plane]|![JNet_385_pretrain_3_output_plane]|![JNet_385_pretrain_3_label_plane]|
  
MSE: 0.025530358776450157, BCE: 0.10621960461139679  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_3_original_depth]|![JNet_385_pretrain_3_output_depth]|![JNet_385_pretrain_3_label_depth]|
  
MSE: 0.025530358776450157, BCE: 0.10621960461139679  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_4_original_plane]|![JNet_385_pretrain_4_output_plane]|![JNet_385_pretrain_4_label_plane]|
  
MSE: 0.030832448974251747, BCE: 0.11443974822759628  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_385_pretrain_4_original_depth]|![JNet_385_pretrain_4_output_depth]|![JNet_385_pretrain_4_label_depth]|
  
MSE: 0.030832448974251747, BCE: 0.11443974822759628  
  
mean MSE: 0.027537167072296143, mean BCE: 0.1892939656972885
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_402_0_original_plane]|![JNet_402_0_output_plane]|![JNet_402_0_label_plane]|
  
MSE: 0.0331801138818264, BCE: 0.2572542428970337  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_402_0_original_depth]|![JNet_402_0_output_depth]|![JNet_402_0_label_depth]|
  
MSE: 0.0331801138818264, BCE: 0.2572542428970337  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_402_1_original_plane]|![JNet_402_1_output_plane]|![JNet_402_1_label_plane]|
  
MSE: 0.03394244611263275, BCE: 0.19903036952018738  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_402_1_original_depth]|![JNet_402_1_output_depth]|![JNet_402_1_label_depth]|
  
MSE: 0.03394244611263275, BCE: 0.19903036952018738  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_402_2_original_plane]|![JNet_402_2_output_plane]|![JNet_402_2_label_plane]|
  
MSE: 0.022143583744764328, BCE: 0.15586499869823456  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_402_2_original_depth]|![JNet_402_2_output_depth]|![JNet_402_2_label_depth]|
  
MSE: 0.022143583744764328, BCE: 0.15586499869823456  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_402_3_original_plane]|![JNet_402_3_output_plane]|![JNet_402_3_label_plane]|
  
MSE: 0.017223356291651726, BCE: 0.15975400805473328  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_402_3_original_depth]|![JNet_402_3_output_depth]|![JNet_402_3_label_depth]|
  
MSE: 0.017223356291651726, BCE: 0.15975400805473328  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_402_4_original_plane]|![JNet_402_4_output_plane]|![JNet_402_4_label_plane]|
  
MSE: 0.031196322292089462, BCE: 0.17456619441509247  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_402_4_original_depth]|![JNet_402_4_output_depth]|![JNet_402_4_label_depth]|
  
MSE: 0.031196322292089462, BCE: 0.17456619441509247  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_001_roi000_original_depth]|![JNet_385_pretrain_beads_001_roi000_output_depth]|![JNet_385_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 8.819509765625002, MSE: 0.0026929082814604044, quantized loss: 6.589564145542681e-05  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_001_roi001_original_depth]|![JNet_385_pretrain_beads_001_roi001_output_depth]|![JNet_385_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 13.659982421875004, MSE: 0.004503066185861826, quantized loss: 8.896458894014359e-05  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_001_roi002_original_depth]|![JNet_385_pretrain_beads_001_roi002_output_depth]|![JNet_385_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 8.402117187500002, MSE: 0.0027492335066199303, quantized loss: 5.411385063780472e-05  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_001_roi003_original_depth]|![JNet_385_pretrain_beads_001_roi003_output_depth]|![JNet_385_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 14.222596679687504, MSE: 0.0046167136169970036, quantized loss: 9.446191688766703e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_001_roi004_original_depth]|![JNet_385_pretrain_beads_001_roi004_output_depth]|![JNet_385_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 8.789151367187502, MSE: 0.0035470265429466963, quantized loss: 5.0633909268071875e-05  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_002_roi000_original_depth]|![JNet_385_pretrain_beads_002_roi000_output_depth]|![JNet_385_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 9.326360351562503, MSE: 0.004011465702205896, quantized loss: 5.533065996132791e-05  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_002_roi001_original_depth]|![JNet_385_pretrain_beads_002_roi001_output_depth]|![JNet_385_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 9.058920898437503, MSE: 0.0029490471351891756, quantized loss: 5.659149246639572e-05  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_385_pretrain_beads_002_roi002_original_depth]|![JNet_385_pretrain_beads_002_roi002_output_depth]|![JNet_385_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 9.054398437500002, MSE: 0.003490591188892722, quantized loss: 5.7902892876882106e-05  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_402_beads_001_roi000_original_depth]|![JNet_402_beads_001_roi000_output_depth]|![JNet_402_beads_001_roi000_reconst_depth]|
  
volume: 17.745025390625003, MSE: 0.0021793500054627657, quantized loss: 0.00012629528646357358  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_402_beads_001_roi001_original_depth]|![JNet_402_beads_001_roi001_output_depth]|![JNet_402_beads_001_roi001_reconst_depth]|
  
volume: 27.229455078125007, MSE: 0.003108763135969639, quantized loss: 0.00016525291721336544  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_402_beads_001_roi002_original_depth]|![JNet_402_beads_001_roi002_output_depth]|![JNet_402_beads_001_roi002_reconst_depth]|
  
volume: 17.241781250000003, MSE: 0.0021888611372560263, quantized loss: 0.00010698201367631555  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_402_beads_001_roi003_original_depth]|![JNet_402_beads_001_roi003_output_depth]|![JNet_402_beads_001_roi003_reconst_depth]|
  
volume: 27.143832031250007, MSE: 0.002986305160447955, quantized loss: 0.00015782237460371107  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_402_beads_001_roi004_original_depth]|![JNet_402_beads_001_roi004_output_depth]|![JNet_402_beads_001_roi004_reconst_depth]|
  
volume: 18.110117187500006, MSE: 0.0024400120601058006, quantized loss: 0.00010632016346789896  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_402_beads_002_roi000_original_depth]|![JNet_402_beads_002_roi000_output_depth]|![JNet_402_beads_002_roi000_reconst_depth]|
  
volume: 19.207201171875006, MSE: 0.002670086920261383, quantized loss: 0.00010936976468656212  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_402_beads_002_roi001_original_depth]|![JNet_402_beads_002_roi001_output_depth]|![JNet_402_beads_002_roi001_reconst_depth]|
  
volume: 17.884662109375004, MSE: 0.002195603447034955, quantized loss: 0.00010935468890238553  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_402_beads_002_roi002_original_depth]|![JNet_402_beads_002_roi002_output_depth]|![JNet_402_beads_002_roi002_reconst_depth]|
  
volume: 18.656916015625004, MSE: 0.0024956457782536745, quantized loss: 0.00011109362822026014  

|pre|post|
| :---: | :---: |
|![JNet_402_psf_pre]|![JNet_402_psf_post]|

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
[JNet_402_0_label_depth]: /experiments/images/JNet_402_0_label_depth.png
[JNet_402_0_label_plane]: /experiments/images/JNet_402_0_label_plane.png
[JNet_402_0_original_depth]: /experiments/images/JNet_402_0_original_depth.png
[JNet_402_0_original_plane]: /experiments/images/JNet_402_0_original_plane.png
[JNet_402_0_output_depth]: /experiments/images/JNet_402_0_output_depth.png
[JNet_402_0_output_plane]: /experiments/images/JNet_402_0_output_plane.png
[JNet_402_1_label_depth]: /experiments/images/JNet_402_1_label_depth.png
[JNet_402_1_label_plane]: /experiments/images/JNet_402_1_label_plane.png
[JNet_402_1_original_depth]: /experiments/images/JNet_402_1_original_depth.png
[JNet_402_1_original_plane]: /experiments/images/JNet_402_1_original_plane.png
[JNet_402_1_output_depth]: /experiments/images/JNet_402_1_output_depth.png
[JNet_402_1_output_plane]: /experiments/images/JNet_402_1_output_plane.png
[JNet_402_2_label_depth]: /experiments/images/JNet_402_2_label_depth.png
[JNet_402_2_label_plane]: /experiments/images/JNet_402_2_label_plane.png
[JNet_402_2_original_depth]: /experiments/images/JNet_402_2_original_depth.png
[JNet_402_2_original_plane]: /experiments/images/JNet_402_2_original_plane.png
[JNet_402_2_output_depth]: /experiments/images/JNet_402_2_output_depth.png
[JNet_402_2_output_plane]: /experiments/images/JNet_402_2_output_plane.png
[JNet_402_3_label_depth]: /experiments/images/JNet_402_3_label_depth.png
[JNet_402_3_label_plane]: /experiments/images/JNet_402_3_label_plane.png
[JNet_402_3_original_depth]: /experiments/images/JNet_402_3_original_depth.png
[JNet_402_3_original_plane]: /experiments/images/JNet_402_3_original_plane.png
[JNet_402_3_output_depth]: /experiments/images/JNet_402_3_output_depth.png
[JNet_402_3_output_plane]: /experiments/images/JNet_402_3_output_plane.png
[JNet_402_4_label_depth]: /experiments/images/JNet_402_4_label_depth.png
[JNet_402_4_label_plane]: /experiments/images/JNet_402_4_label_plane.png
[JNet_402_4_original_depth]: /experiments/images/JNet_402_4_original_depth.png
[JNet_402_4_original_plane]: /experiments/images/JNet_402_4_original_plane.png
[JNet_402_4_output_depth]: /experiments/images/JNet_402_4_output_depth.png
[JNet_402_4_output_plane]: /experiments/images/JNet_402_4_output_plane.png
[JNet_402_beads_001_roi000_original_depth]: /experiments/images/JNet_402_beads_001_roi000_original_depth.png
[JNet_402_beads_001_roi000_output_depth]: /experiments/images/JNet_402_beads_001_roi000_output_depth.png
[JNet_402_beads_001_roi000_reconst_depth]: /experiments/images/JNet_402_beads_001_roi000_reconst_depth.png
[JNet_402_beads_001_roi001_original_depth]: /experiments/images/JNet_402_beads_001_roi001_original_depth.png
[JNet_402_beads_001_roi001_output_depth]: /experiments/images/JNet_402_beads_001_roi001_output_depth.png
[JNet_402_beads_001_roi001_reconst_depth]: /experiments/images/JNet_402_beads_001_roi001_reconst_depth.png
[JNet_402_beads_001_roi002_original_depth]: /experiments/images/JNet_402_beads_001_roi002_original_depth.png
[JNet_402_beads_001_roi002_output_depth]: /experiments/images/JNet_402_beads_001_roi002_output_depth.png
[JNet_402_beads_001_roi002_reconst_depth]: /experiments/images/JNet_402_beads_001_roi002_reconst_depth.png
[JNet_402_beads_001_roi003_original_depth]: /experiments/images/JNet_402_beads_001_roi003_original_depth.png
[JNet_402_beads_001_roi003_output_depth]: /experiments/images/JNet_402_beads_001_roi003_output_depth.png
[JNet_402_beads_001_roi003_reconst_depth]: /experiments/images/JNet_402_beads_001_roi003_reconst_depth.png
[JNet_402_beads_001_roi004_original_depth]: /experiments/images/JNet_402_beads_001_roi004_original_depth.png
[JNet_402_beads_001_roi004_output_depth]: /experiments/images/JNet_402_beads_001_roi004_output_depth.png
[JNet_402_beads_001_roi004_reconst_depth]: /experiments/images/JNet_402_beads_001_roi004_reconst_depth.png
[JNet_402_beads_002_roi000_original_depth]: /experiments/images/JNet_402_beads_002_roi000_original_depth.png
[JNet_402_beads_002_roi000_output_depth]: /experiments/images/JNet_402_beads_002_roi000_output_depth.png
[JNet_402_beads_002_roi000_reconst_depth]: /experiments/images/JNet_402_beads_002_roi000_reconst_depth.png
[JNet_402_beads_002_roi001_original_depth]: /experiments/images/JNet_402_beads_002_roi001_original_depth.png
[JNet_402_beads_002_roi001_output_depth]: /experiments/images/JNet_402_beads_002_roi001_output_depth.png
[JNet_402_beads_002_roi001_reconst_depth]: /experiments/images/JNet_402_beads_002_roi001_reconst_depth.png
[JNet_402_beads_002_roi002_original_depth]: /experiments/images/JNet_402_beads_002_roi002_original_depth.png
[JNet_402_beads_002_roi002_output_depth]: /experiments/images/JNet_402_beads_002_roi002_output_depth.png
[JNet_402_beads_002_roi002_reconst_depth]: /experiments/images/JNet_402_beads_002_roi002_reconst_depth.png
[JNet_402_psf_post]: /experiments/images/JNet_402_psf_post.png
[JNet_402_psf_pre]: /experiments/images/JNet_402_psf_pre.png
[finetuned]: /experiments/tmp/JNet_402_train.png
[pretrained_model]: /experiments/tmp/JNet_385_pretrain_train.png
