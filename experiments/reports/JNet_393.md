



# JNet_393 Report
  
the parameters to replicate the results of JNet_393. psf=0.6, realtrain with fixed psf, no vibratation   
pretrained model : JNet_390_pretrain
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
|size_z|239||
|NA|0.6||
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
  
mean MSE: 0.02730676904320717, mean BCE: 0.1017170399427414
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_390_pretrain_0_original_plane]|![JNet_390_pretrain_0_output_plane]|![JNet_390_pretrain_0_label_plane]|
  
MSE: 0.02458835206925869, BCE: 0.09045159816741943  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_390_pretrain_0_original_depth]|![JNet_390_pretrain_0_output_depth]|![JNet_390_pretrain_0_label_depth]|
  
MSE: 0.02458835206925869, BCE: 0.09045159816741943  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_390_pretrain_1_original_plane]|![JNet_390_pretrain_1_output_plane]|![JNet_390_pretrain_1_label_plane]|
  
MSE: 0.027684254571795464, BCE: 0.09981188923120499  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_390_pretrain_1_original_depth]|![JNet_390_pretrain_1_output_depth]|![JNet_390_pretrain_1_label_depth]|
  
MSE: 0.027684254571795464, BCE: 0.09981188923120499  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_390_pretrain_2_original_plane]|![JNet_390_pretrain_2_output_plane]|![JNet_390_pretrain_2_label_plane]|
  
MSE: 0.028344223275780678, BCE: 0.10673341155052185  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_390_pretrain_2_original_depth]|![JNet_390_pretrain_2_output_depth]|![JNet_390_pretrain_2_label_depth]|
  
MSE: 0.028344223275780678, BCE: 0.10673341155052185  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_390_pretrain_3_original_plane]|![JNet_390_pretrain_3_output_plane]|![JNet_390_pretrain_3_label_plane]|
  
MSE: 0.02673322707414627, BCE: 0.10006295889616013  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_390_pretrain_3_original_depth]|![JNet_390_pretrain_3_output_depth]|![JNet_390_pretrain_3_label_depth]|
  
MSE: 0.02673322707414627, BCE: 0.10006295889616013  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_390_pretrain_4_original_plane]|![JNet_390_pretrain_4_output_plane]|![JNet_390_pretrain_4_label_plane]|
  
MSE: 0.029183782637119293, BCE: 0.11152537912130356  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_390_pretrain_4_original_depth]|![JNet_390_pretrain_4_output_depth]|![JNet_390_pretrain_4_label_depth]|
  
MSE: 0.029183782637119293, BCE: 0.11152537912130356  
  
mean MSE: 0.03593716770410538, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_393_0_original_plane]|![JNet_393_0_output_plane]|![JNet_393_0_label_plane]|
  
MSE: 0.04858597740530968, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_393_0_original_depth]|![JNet_393_0_output_depth]|![JNet_393_0_label_depth]|
  
MSE: 0.04858597740530968, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_393_1_original_plane]|![JNet_393_1_output_plane]|![JNet_393_1_label_plane]|
  
MSE: 0.0326942503452301, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_393_1_original_depth]|![JNet_393_1_output_depth]|![JNet_393_1_label_depth]|
  
MSE: 0.0326942503452301, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_393_2_original_plane]|![JNet_393_2_output_plane]|![JNet_393_2_label_plane]|
  
MSE: 0.026280492544174194, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_393_2_original_depth]|![JNet_393_2_output_depth]|![JNet_393_2_label_depth]|
  
MSE: 0.026280492544174194, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_393_3_original_plane]|![JNet_393_3_output_plane]|![JNet_393_3_label_plane]|
  
MSE: 0.03826696798205376, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_393_3_original_depth]|![JNet_393_3_output_depth]|![JNet_393_3_label_depth]|
  
MSE: 0.03826696798205376, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_393_4_original_plane]|![JNet_393_4_output_plane]|![JNet_393_4_label_plane]|
  
MSE: 0.033858153969049454, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_393_4_original_depth]|![JNet_393_4_output_depth]|![JNet_393_4_label_depth]|
  
MSE: 0.033858153969049454, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_390_pretrain_beads_001_roi000_original_depth]|![JNet_390_pretrain_beads_001_roi000_output_depth]|![JNet_390_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 19.009087890625004, MSE: 0.0030195561703294516, quantized loss: 0.002667914144694805  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_390_pretrain_beads_001_roi001_original_depth]|![JNet_390_pretrain_beads_001_roi001_output_depth]|![JNet_390_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 26.132500000000007, MSE: 0.004564464557915926, quantized loss: 0.0033796634525060654  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_390_pretrain_beads_001_roi002_original_depth]|![JNet_390_pretrain_beads_001_roi002_output_depth]|![JNet_390_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 19.279964843750005, MSE: 0.003303835866972804, quantized loss: 0.0030898996628820896  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_390_pretrain_beads_001_roi003_original_depth]|![JNet_390_pretrain_beads_001_roi003_output_depth]|![JNet_390_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 28.396339843750006, MSE: 0.00492035411298275, quantized loss: 0.003777140285819769  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_390_pretrain_beads_001_roi004_original_depth]|![JNet_390_pretrain_beads_001_roi004_output_depth]|![JNet_390_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 20.320158203125004, MSE: 0.0038717358838766813, quantized loss: 0.003144889837130904  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_390_pretrain_beads_002_roi000_original_depth]|![JNet_390_pretrain_beads_002_roi000_output_depth]|![JNet_390_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 22.579222656250007, MSE: 0.004497770685702562, quantized loss: 0.003682052483782172  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_390_pretrain_beads_002_roi001_original_depth]|![JNet_390_pretrain_beads_002_roi001_output_depth]|![JNet_390_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 19.973757812500004, MSE: 0.0035375163424760103, quantized loss: 0.003180434461683035  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_390_pretrain_beads_002_roi002_original_depth]|![JNet_390_pretrain_beads_002_roi002_output_depth]|![JNet_390_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 21.017960937500003, MSE: 0.003999006934463978, quantized loss: 0.0033663343638181686  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_393_beads_001_roi000_original_depth]|![JNet_393_beads_001_roi000_output_depth]|![JNet_393_beads_001_roi000_reconst_depth]|
  
volume: 19.417585937500004, MSE: 0.0029654456302523613, quantized loss: 1.925368997035548e-05  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_393_beads_001_roi001_original_depth]|![JNet_393_beads_001_roi001_output_depth]|![JNet_393_beads_001_roi001_reconst_depth]|
  
volume: 29.41635351562501, MSE: 0.004199237562716007, quantized loss: 2.5001958420034498e-05  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_393_beads_001_roi002_original_depth]|![JNet_393_beads_001_roi002_output_depth]|![JNet_393_beads_001_roi002_reconst_depth]|
  
volume: 18.590427734375005, MSE: 0.002931013470515609, quantized loss: 1.676889951340854e-05  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_393_beads_001_roi003_original_depth]|![JNet_393_beads_001_roi003_output_depth]|![JNet_393_beads_001_roi003_reconst_depth]|
  
volume: 29.897400390625005, MSE: 0.004258614964783192, quantized loss: 2.593063072708901e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_393_beads_001_roi004_original_depth]|![JNet_393_beads_001_roi004_output_depth]|![JNet_393_beads_001_roi004_reconst_depth]|
  
volume: 19.838492187500005, MSE: 0.0033569829538464546, quantized loss: 1.9546820112736896e-05  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_393_beads_002_roi000_original_depth]|![JNet_393_beads_002_roi000_output_depth]|![JNet_393_beads_002_roi000_reconst_depth]|
  
volume: 21.089824218750007, MSE: 0.003684747964143753, quantized loss: 2.018075429077726e-05  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_393_beads_002_roi001_original_depth]|![JNet_393_beads_002_roi001_output_depth]|![JNet_393_beads_002_roi001_reconst_depth]|
  
volume: 19.617175781250005, MSE: 0.003061818890273571, quantized loss: 1.9292296201456338e-05  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_393_beads_002_roi002_original_depth]|![JNet_393_beads_002_roi002_output_depth]|![JNet_393_beads_002_roi002_reconst_depth]|
  
volume: 20.305203125000006, MSE: 0.0034240351524204016, quantized loss: 1.7737389498506673e-05  

|pre|post|
| :---: | :---: |
|![JNet_393_psf_pre]|![JNet_393_psf_post]|

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
  



[JNet_390_pretrain_0_label_depth]: /experiments/images/JNet_390_pretrain_0_label_depth.png
[JNet_390_pretrain_0_label_plane]: /experiments/images/JNet_390_pretrain_0_label_plane.png
[JNet_390_pretrain_0_original_depth]: /experiments/images/JNet_390_pretrain_0_original_depth.png
[JNet_390_pretrain_0_original_plane]: /experiments/images/JNet_390_pretrain_0_original_plane.png
[JNet_390_pretrain_0_output_depth]: /experiments/images/JNet_390_pretrain_0_output_depth.png
[JNet_390_pretrain_0_output_plane]: /experiments/images/JNet_390_pretrain_0_output_plane.png
[JNet_390_pretrain_1_label_depth]: /experiments/images/JNet_390_pretrain_1_label_depth.png
[JNet_390_pretrain_1_label_plane]: /experiments/images/JNet_390_pretrain_1_label_plane.png
[JNet_390_pretrain_1_original_depth]: /experiments/images/JNet_390_pretrain_1_original_depth.png
[JNet_390_pretrain_1_original_plane]: /experiments/images/JNet_390_pretrain_1_original_plane.png
[JNet_390_pretrain_1_output_depth]: /experiments/images/JNet_390_pretrain_1_output_depth.png
[JNet_390_pretrain_1_output_plane]: /experiments/images/JNet_390_pretrain_1_output_plane.png
[JNet_390_pretrain_2_label_depth]: /experiments/images/JNet_390_pretrain_2_label_depth.png
[JNet_390_pretrain_2_label_plane]: /experiments/images/JNet_390_pretrain_2_label_plane.png
[JNet_390_pretrain_2_original_depth]: /experiments/images/JNet_390_pretrain_2_original_depth.png
[JNet_390_pretrain_2_original_plane]: /experiments/images/JNet_390_pretrain_2_original_plane.png
[JNet_390_pretrain_2_output_depth]: /experiments/images/JNet_390_pretrain_2_output_depth.png
[JNet_390_pretrain_2_output_plane]: /experiments/images/JNet_390_pretrain_2_output_plane.png
[JNet_390_pretrain_3_label_depth]: /experiments/images/JNet_390_pretrain_3_label_depth.png
[JNet_390_pretrain_3_label_plane]: /experiments/images/JNet_390_pretrain_3_label_plane.png
[JNet_390_pretrain_3_original_depth]: /experiments/images/JNet_390_pretrain_3_original_depth.png
[JNet_390_pretrain_3_original_plane]: /experiments/images/JNet_390_pretrain_3_original_plane.png
[JNet_390_pretrain_3_output_depth]: /experiments/images/JNet_390_pretrain_3_output_depth.png
[JNet_390_pretrain_3_output_plane]: /experiments/images/JNet_390_pretrain_3_output_plane.png
[JNet_390_pretrain_4_label_depth]: /experiments/images/JNet_390_pretrain_4_label_depth.png
[JNet_390_pretrain_4_label_plane]: /experiments/images/JNet_390_pretrain_4_label_plane.png
[JNet_390_pretrain_4_original_depth]: /experiments/images/JNet_390_pretrain_4_original_depth.png
[JNet_390_pretrain_4_original_plane]: /experiments/images/JNet_390_pretrain_4_original_plane.png
[JNet_390_pretrain_4_output_depth]: /experiments/images/JNet_390_pretrain_4_output_depth.png
[JNet_390_pretrain_4_output_plane]: /experiments/images/JNet_390_pretrain_4_output_plane.png
[JNet_390_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi000_original_depth.png
[JNet_390_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi000_output_depth.png
[JNet_390_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi000_reconst_depth.png
[JNet_390_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi001_original_depth.png
[JNet_390_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi001_output_depth.png
[JNet_390_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi001_reconst_depth.png
[JNet_390_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi002_original_depth.png
[JNet_390_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi002_output_depth.png
[JNet_390_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi002_reconst_depth.png
[JNet_390_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi003_original_depth.png
[JNet_390_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi003_output_depth.png
[JNet_390_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi003_reconst_depth.png
[JNet_390_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi004_original_depth.png
[JNet_390_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi004_output_depth.png
[JNet_390_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_390_pretrain_beads_001_roi004_reconst_depth.png
[JNet_390_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_390_pretrain_beads_002_roi000_original_depth.png
[JNet_390_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_390_pretrain_beads_002_roi000_output_depth.png
[JNet_390_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_390_pretrain_beads_002_roi000_reconst_depth.png
[JNet_390_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_390_pretrain_beads_002_roi001_original_depth.png
[JNet_390_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_390_pretrain_beads_002_roi001_output_depth.png
[JNet_390_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_390_pretrain_beads_002_roi001_reconst_depth.png
[JNet_390_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_390_pretrain_beads_002_roi002_original_depth.png
[JNet_390_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_390_pretrain_beads_002_roi002_output_depth.png
[JNet_390_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_390_pretrain_beads_002_roi002_reconst_depth.png
[JNet_393_0_label_depth]: /experiments/images/JNet_393_0_label_depth.png
[JNet_393_0_label_plane]: /experiments/images/JNet_393_0_label_plane.png
[JNet_393_0_original_depth]: /experiments/images/JNet_393_0_original_depth.png
[JNet_393_0_original_plane]: /experiments/images/JNet_393_0_original_plane.png
[JNet_393_0_output_depth]: /experiments/images/JNet_393_0_output_depth.png
[JNet_393_0_output_plane]: /experiments/images/JNet_393_0_output_plane.png
[JNet_393_1_label_depth]: /experiments/images/JNet_393_1_label_depth.png
[JNet_393_1_label_plane]: /experiments/images/JNet_393_1_label_plane.png
[JNet_393_1_original_depth]: /experiments/images/JNet_393_1_original_depth.png
[JNet_393_1_original_plane]: /experiments/images/JNet_393_1_original_plane.png
[JNet_393_1_output_depth]: /experiments/images/JNet_393_1_output_depth.png
[JNet_393_1_output_plane]: /experiments/images/JNet_393_1_output_plane.png
[JNet_393_2_label_depth]: /experiments/images/JNet_393_2_label_depth.png
[JNet_393_2_label_plane]: /experiments/images/JNet_393_2_label_plane.png
[JNet_393_2_original_depth]: /experiments/images/JNet_393_2_original_depth.png
[JNet_393_2_original_plane]: /experiments/images/JNet_393_2_original_plane.png
[JNet_393_2_output_depth]: /experiments/images/JNet_393_2_output_depth.png
[JNet_393_2_output_plane]: /experiments/images/JNet_393_2_output_plane.png
[JNet_393_3_label_depth]: /experiments/images/JNet_393_3_label_depth.png
[JNet_393_3_label_plane]: /experiments/images/JNet_393_3_label_plane.png
[JNet_393_3_original_depth]: /experiments/images/JNet_393_3_original_depth.png
[JNet_393_3_original_plane]: /experiments/images/JNet_393_3_original_plane.png
[JNet_393_3_output_depth]: /experiments/images/JNet_393_3_output_depth.png
[JNet_393_3_output_plane]: /experiments/images/JNet_393_3_output_plane.png
[JNet_393_4_label_depth]: /experiments/images/JNet_393_4_label_depth.png
[JNet_393_4_label_plane]: /experiments/images/JNet_393_4_label_plane.png
[JNet_393_4_original_depth]: /experiments/images/JNet_393_4_original_depth.png
[JNet_393_4_original_plane]: /experiments/images/JNet_393_4_original_plane.png
[JNet_393_4_output_depth]: /experiments/images/JNet_393_4_output_depth.png
[JNet_393_4_output_plane]: /experiments/images/JNet_393_4_output_plane.png
[JNet_393_beads_001_roi000_original_depth]: /experiments/images/JNet_393_beads_001_roi000_original_depth.png
[JNet_393_beads_001_roi000_output_depth]: /experiments/images/JNet_393_beads_001_roi000_output_depth.png
[JNet_393_beads_001_roi000_reconst_depth]: /experiments/images/JNet_393_beads_001_roi000_reconst_depth.png
[JNet_393_beads_001_roi001_original_depth]: /experiments/images/JNet_393_beads_001_roi001_original_depth.png
[JNet_393_beads_001_roi001_output_depth]: /experiments/images/JNet_393_beads_001_roi001_output_depth.png
[JNet_393_beads_001_roi001_reconst_depth]: /experiments/images/JNet_393_beads_001_roi001_reconst_depth.png
[JNet_393_beads_001_roi002_original_depth]: /experiments/images/JNet_393_beads_001_roi002_original_depth.png
[JNet_393_beads_001_roi002_output_depth]: /experiments/images/JNet_393_beads_001_roi002_output_depth.png
[JNet_393_beads_001_roi002_reconst_depth]: /experiments/images/JNet_393_beads_001_roi002_reconst_depth.png
[JNet_393_beads_001_roi003_original_depth]: /experiments/images/JNet_393_beads_001_roi003_original_depth.png
[JNet_393_beads_001_roi003_output_depth]: /experiments/images/JNet_393_beads_001_roi003_output_depth.png
[JNet_393_beads_001_roi003_reconst_depth]: /experiments/images/JNet_393_beads_001_roi003_reconst_depth.png
[JNet_393_beads_001_roi004_original_depth]: /experiments/images/JNet_393_beads_001_roi004_original_depth.png
[JNet_393_beads_001_roi004_output_depth]: /experiments/images/JNet_393_beads_001_roi004_output_depth.png
[JNet_393_beads_001_roi004_reconst_depth]: /experiments/images/JNet_393_beads_001_roi004_reconst_depth.png
[JNet_393_beads_002_roi000_original_depth]: /experiments/images/JNet_393_beads_002_roi000_original_depth.png
[JNet_393_beads_002_roi000_output_depth]: /experiments/images/JNet_393_beads_002_roi000_output_depth.png
[JNet_393_beads_002_roi000_reconst_depth]: /experiments/images/JNet_393_beads_002_roi000_reconst_depth.png
[JNet_393_beads_002_roi001_original_depth]: /experiments/images/JNet_393_beads_002_roi001_original_depth.png
[JNet_393_beads_002_roi001_output_depth]: /experiments/images/JNet_393_beads_002_roi001_output_depth.png
[JNet_393_beads_002_roi001_reconst_depth]: /experiments/images/JNet_393_beads_002_roi001_reconst_depth.png
[JNet_393_beads_002_roi002_original_depth]: /experiments/images/JNet_393_beads_002_roi002_original_depth.png
[JNet_393_beads_002_roi002_output_depth]: /experiments/images/JNet_393_beads_002_roi002_output_depth.png
[JNet_393_beads_002_roi002_reconst_depth]: /experiments/images/JNet_393_beads_002_roi002_reconst_depth.png
[JNet_393_psf_post]: /experiments/images/JNet_393_psf_post.png
[JNet_393_psf_pre]: /experiments/images/JNet_393_psf_pre.png
[finetuned]: /experiments/tmp/JNet_393_train.png
[pretrained_model]: /experiments/tmp/JNet_390_pretrain_train.png
