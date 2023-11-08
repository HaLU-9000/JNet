



# JNet_408 Report
  
the parameters to replicate the results of JNet_408. res_axial 0.5 test.  
pretrained model : JNet_407_pretrain
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
|res_axial|0.5|microns|
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
|loss_fn|nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.6], device=params['device']))|
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
|is_vibrate|True|
|loss_weight|1|
|qloss_weight|0.1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.016434287652373314, mean BCE: 0.059210341423749924
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_0_original_plane]|![JNet_407_pretrain_0_output_plane]|![JNet_407_pretrain_0_label_plane]|
  
MSE: 0.020364848896861076, BCE: 0.07280116528272629  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_0_original_depth]|![JNet_407_pretrain_0_output_depth]|![JNet_407_pretrain_0_label_depth]|
  
MSE: 0.020364848896861076, BCE: 0.07280116528272629  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_1_original_plane]|![JNet_407_pretrain_1_output_plane]|![JNet_407_pretrain_1_label_plane]|
  
MSE: 0.016077842563390732, BCE: 0.056497711688280106  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_1_original_depth]|![JNet_407_pretrain_1_output_depth]|![JNet_407_pretrain_1_label_depth]|
  
MSE: 0.016077842563390732, BCE: 0.056497711688280106  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_2_original_plane]|![JNet_407_pretrain_2_output_plane]|![JNet_407_pretrain_2_label_plane]|
  
MSE: 0.012925716117024422, BCE: 0.04814667999744415  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_2_original_depth]|![JNet_407_pretrain_2_output_depth]|![JNet_407_pretrain_2_label_depth]|
  
MSE: 0.012925716117024422, BCE: 0.04814667999744415  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_3_original_plane]|![JNet_407_pretrain_3_output_plane]|![JNet_407_pretrain_3_label_plane]|
  
MSE: 0.018424347043037415, BCE: 0.06734953820705414  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_3_original_depth]|![JNet_407_pretrain_3_output_depth]|![JNet_407_pretrain_3_label_depth]|
  
MSE: 0.018424347043037415, BCE: 0.06734953820705414  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_4_original_plane]|![JNet_407_pretrain_4_output_plane]|![JNet_407_pretrain_4_label_plane]|
  
MSE: 0.014378692023456097, BCE: 0.051256608217954636  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_4_original_depth]|![JNet_407_pretrain_4_output_depth]|![JNet_407_pretrain_4_label_depth]|
  
MSE: 0.014378692023456097, BCE: 0.051256608217954636  
  
mean MSE: 0.029965508729219437, mean BCE: 0.15980707108974457
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_408_0_original_plane]|![JNet_408_0_output_plane]|![JNet_408_0_label_plane]|
  
MSE: 0.036653805524110794, BCE: 0.2393578290939331  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_408_0_original_depth]|![JNet_408_0_output_depth]|![JNet_408_0_label_depth]|
  
MSE: 0.036653805524110794, BCE: 0.2393578290939331  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_408_1_original_plane]|![JNet_408_1_output_plane]|![JNet_408_1_label_plane]|
  
MSE: 0.023526335135102272, BCE: 0.12033962458372116  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_408_1_original_depth]|![JNet_408_1_output_depth]|![JNet_408_1_label_depth]|
  
MSE: 0.023526335135102272, BCE: 0.12033962458372116  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_408_2_original_plane]|![JNet_408_2_output_plane]|![JNet_408_2_label_plane]|
  
MSE: 0.027599109336733818, BCE: 0.11923719197511673  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_408_2_original_depth]|![JNet_408_2_output_depth]|![JNet_408_2_label_depth]|
  
MSE: 0.027599109336733818, BCE: 0.11923719197511673  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_408_3_original_plane]|![JNet_408_3_output_plane]|![JNet_408_3_label_plane]|
  
MSE: 0.03432098403573036, BCE: 0.15421821177005768  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_408_3_original_depth]|![JNet_408_3_output_depth]|![JNet_408_3_label_depth]|
  
MSE: 0.03432098403573036, BCE: 0.15421821177005768  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_408_4_original_plane]|![JNet_408_4_output_plane]|![JNet_408_4_label_plane]|
  
MSE: 0.027727298438549042, BCE: 0.16588252782821655  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_408_4_original_depth]|![JNet_408_4_output_depth]|![JNet_408_4_label_depth]|
  
MSE: 0.027727298438549042, BCE: 0.16588252782821655  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_001_roi000_original_depth]|![JNet_407_pretrain_beads_001_roi000_output_depth]|![JNet_407_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 11.305373535156253, MSE: 0.002964101731777191, quantized loss: 0.0004109727160539478  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_001_roi001_original_depth]|![JNet_407_pretrain_beads_001_roi001_output_depth]|![JNet_407_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 16.433638916015628, MSE: 0.005106337834149599, quantized loss: 0.0007220489205792546  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_001_roi002_original_depth]|![JNet_407_pretrain_beads_001_roi002_output_depth]|![JNet_407_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 14.448996582031253, MSE: 0.002319271443411708, quantized loss: 0.0006344459834508598  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_001_roi003_original_depth]|![JNet_407_pretrain_beads_001_roi003_output_depth]|![JNet_407_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 21.272116699218753, MSE: 0.005650651175528765, quantized loss: 0.0009322650148533285  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_001_roi004_original_depth]|![JNet_407_pretrain_beads_001_roi004_output_depth]|![JNet_407_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 15.132409667968753, MSE: 0.0028237912338227034, quantized loss: 0.0006969032110646367  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_002_roi000_original_depth]|![JNet_407_pretrain_beads_002_roi000_output_depth]|![JNet_407_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 16.135328369140627, MSE: 0.0032083916012197733, quantized loss: 0.0007405178621411324  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_002_roi001_original_depth]|![JNet_407_pretrain_beads_002_roi001_output_depth]|![JNet_407_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 16.988107910156252, MSE: 0.0021884909365326166, quantized loss: 0.0007907944964244962  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_002_roi002_original_depth]|![JNet_407_pretrain_beads_002_roi002_output_depth]|![JNet_407_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 15.992633056640628, MSE: 0.002832418540492654, quantized loss: 0.0007267890032380819  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_408_beads_001_roi000_original_depth]|![JNet_408_beads_001_roi000_output_depth]|![JNet_408_beads_001_roi000_reconst_depth]|
  
volume: 11.310432128906251, MSE: 0.0013911441201344132, quantized loss: 0.0005659024463966489  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_408_beads_001_roi001_original_depth]|![JNet_408_beads_001_roi001_output_depth]|![JNet_408_beads_001_roi001_reconst_depth]|
  
volume: 19.66059204101563, MSE: 0.0017755426233634353, quantized loss: 0.0008623991161584854  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_408_beads_001_roi002_original_depth]|![JNet_408_beads_001_roi002_output_depth]|![JNet_408_beads_001_roi002_reconst_depth]|
  
volume: 11.851219482421877, MSE: 0.0013092024018988013, quantized loss: 0.000514246872626245  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_408_beads_001_roi003_original_depth]|![JNet_408_beads_001_roi003_output_depth]|![JNet_408_beads_001_roi003_reconst_depth]|
  
volume: 20.619062500000005, MSE: 0.0018501720624044538, quantized loss: 0.0007271353388205171  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_408_beads_001_roi004_original_depth]|![JNet_408_beads_001_roi004_output_depth]|![JNet_408_beads_001_roi004_reconst_depth]|
  
volume: 13.495635986328127, MSE: 0.0015488527715206146, quantized loss: 0.0004856628365814686  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_408_beads_002_roi000_original_depth]|![JNet_408_beads_002_roi000_output_depth]|![JNet_408_beads_002_roi000_reconst_depth]|
  
volume: 14.817954101562503, MSE: 0.0017079365206882358, quantized loss: 0.00048143655294552445  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_408_beads_002_roi001_original_depth]|![JNet_408_beads_002_roi001_output_depth]|![JNet_408_beads_002_roi001_reconst_depth]|
  
volume: 13.175388183593753, MSE: 0.0012701658997684717, quantized loss: 0.0004450188425835222  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_408_beads_002_roi002_original_depth]|![JNet_408_beads_002_roi002_output_depth]|![JNet_408_beads_002_roi002_reconst_depth]|
  
volume: 13.810339355468752, MSE: 0.001536877709440887, quantized loss: 0.0005001764511689544  

|pre|post|
| :---: | :---: |
|![JNet_408_psf_pre]|![JNet_408_psf_post]|

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
    (conv): Conv3d(16, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
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
  



[JNet_407_pretrain_0_label_depth]: /experiments/images/JNet_407_pretrain_0_label_depth.png
[JNet_407_pretrain_0_label_plane]: /experiments/images/JNet_407_pretrain_0_label_plane.png
[JNet_407_pretrain_0_original_depth]: /experiments/images/JNet_407_pretrain_0_original_depth.png
[JNet_407_pretrain_0_original_plane]: /experiments/images/JNet_407_pretrain_0_original_plane.png
[JNet_407_pretrain_0_output_depth]: /experiments/images/JNet_407_pretrain_0_output_depth.png
[JNet_407_pretrain_0_output_plane]: /experiments/images/JNet_407_pretrain_0_output_plane.png
[JNet_407_pretrain_1_label_depth]: /experiments/images/JNet_407_pretrain_1_label_depth.png
[JNet_407_pretrain_1_label_plane]: /experiments/images/JNet_407_pretrain_1_label_plane.png
[JNet_407_pretrain_1_original_depth]: /experiments/images/JNet_407_pretrain_1_original_depth.png
[JNet_407_pretrain_1_original_plane]: /experiments/images/JNet_407_pretrain_1_original_plane.png
[JNet_407_pretrain_1_output_depth]: /experiments/images/JNet_407_pretrain_1_output_depth.png
[JNet_407_pretrain_1_output_plane]: /experiments/images/JNet_407_pretrain_1_output_plane.png
[JNet_407_pretrain_2_label_depth]: /experiments/images/JNet_407_pretrain_2_label_depth.png
[JNet_407_pretrain_2_label_plane]: /experiments/images/JNet_407_pretrain_2_label_plane.png
[JNet_407_pretrain_2_original_depth]: /experiments/images/JNet_407_pretrain_2_original_depth.png
[JNet_407_pretrain_2_original_plane]: /experiments/images/JNet_407_pretrain_2_original_plane.png
[JNet_407_pretrain_2_output_depth]: /experiments/images/JNet_407_pretrain_2_output_depth.png
[JNet_407_pretrain_2_output_plane]: /experiments/images/JNet_407_pretrain_2_output_plane.png
[JNet_407_pretrain_3_label_depth]: /experiments/images/JNet_407_pretrain_3_label_depth.png
[JNet_407_pretrain_3_label_plane]: /experiments/images/JNet_407_pretrain_3_label_plane.png
[JNet_407_pretrain_3_original_depth]: /experiments/images/JNet_407_pretrain_3_original_depth.png
[JNet_407_pretrain_3_original_plane]: /experiments/images/JNet_407_pretrain_3_original_plane.png
[JNet_407_pretrain_3_output_depth]: /experiments/images/JNet_407_pretrain_3_output_depth.png
[JNet_407_pretrain_3_output_plane]: /experiments/images/JNet_407_pretrain_3_output_plane.png
[JNet_407_pretrain_4_label_depth]: /experiments/images/JNet_407_pretrain_4_label_depth.png
[JNet_407_pretrain_4_label_plane]: /experiments/images/JNet_407_pretrain_4_label_plane.png
[JNet_407_pretrain_4_original_depth]: /experiments/images/JNet_407_pretrain_4_original_depth.png
[JNet_407_pretrain_4_original_plane]: /experiments/images/JNet_407_pretrain_4_original_plane.png
[JNet_407_pretrain_4_output_depth]: /experiments/images/JNet_407_pretrain_4_output_depth.png
[JNet_407_pretrain_4_output_plane]: /experiments/images/JNet_407_pretrain_4_output_plane.png
[JNet_407_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi000_original_depth.png
[JNet_407_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi000_output_depth.png
[JNet_407_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi000_reconst_depth.png
[JNet_407_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi001_original_depth.png
[JNet_407_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi001_output_depth.png
[JNet_407_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi001_reconst_depth.png
[JNet_407_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi002_original_depth.png
[JNet_407_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi002_output_depth.png
[JNet_407_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi002_reconst_depth.png
[JNet_407_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi003_original_depth.png
[JNet_407_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi003_output_depth.png
[JNet_407_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi003_reconst_depth.png
[JNet_407_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi004_original_depth.png
[JNet_407_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi004_output_depth.png
[JNet_407_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_407_pretrain_beads_001_roi004_reconst_depth.png
[JNet_407_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_407_pretrain_beads_002_roi000_original_depth.png
[JNet_407_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_407_pretrain_beads_002_roi000_output_depth.png
[JNet_407_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_407_pretrain_beads_002_roi000_reconst_depth.png
[JNet_407_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_407_pretrain_beads_002_roi001_original_depth.png
[JNet_407_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_407_pretrain_beads_002_roi001_output_depth.png
[JNet_407_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_407_pretrain_beads_002_roi001_reconst_depth.png
[JNet_407_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_407_pretrain_beads_002_roi002_original_depth.png
[JNet_407_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_407_pretrain_beads_002_roi002_output_depth.png
[JNet_407_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_407_pretrain_beads_002_roi002_reconst_depth.png
[JNet_408_0_label_depth]: /experiments/images/JNet_408_0_label_depth.png
[JNet_408_0_label_plane]: /experiments/images/JNet_408_0_label_plane.png
[JNet_408_0_original_depth]: /experiments/images/JNet_408_0_original_depth.png
[JNet_408_0_original_plane]: /experiments/images/JNet_408_0_original_plane.png
[JNet_408_0_output_depth]: /experiments/images/JNet_408_0_output_depth.png
[JNet_408_0_output_plane]: /experiments/images/JNet_408_0_output_plane.png
[JNet_408_1_label_depth]: /experiments/images/JNet_408_1_label_depth.png
[JNet_408_1_label_plane]: /experiments/images/JNet_408_1_label_plane.png
[JNet_408_1_original_depth]: /experiments/images/JNet_408_1_original_depth.png
[JNet_408_1_original_plane]: /experiments/images/JNet_408_1_original_plane.png
[JNet_408_1_output_depth]: /experiments/images/JNet_408_1_output_depth.png
[JNet_408_1_output_plane]: /experiments/images/JNet_408_1_output_plane.png
[JNet_408_2_label_depth]: /experiments/images/JNet_408_2_label_depth.png
[JNet_408_2_label_plane]: /experiments/images/JNet_408_2_label_plane.png
[JNet_408_2_original_depth]: /experiments/images/JNet_408_2_original_depth.png
[JNet_408_2_original_plane]: /experiments/images/JNet_408_2_original_plane.png
[JNet_408_2_output_depth]: /experiments/images/JNet_408_2_output_depth.png
[JNet_408_2_output_plane]: /experiments/images/JNet_408_2_output_plane.png
[JNet_408_3_label_depth]: /experiments/images/JNet_408_3_label_depth.png
[JNet_408_3_label_plane]: /experiments/images/JNet_408_3_label_plane.png
[JNet_408_3_original_depth]: /experiments/images/JNet_408_3_original_depth.png
[JNet_408_3_original_plane]: /experiments/images/JNet_408_3_original_plane.png
[JNet_408_3_output_depth]: /experiments/images/JNet_408_3_output_depth.png
[JNet_408_3_output_plane]: /experiments/images/JNet_408_3_output_plane.png
[JNet_408_4_label_depth]: /experiments/images/JNet_408_4_label_depth.png
[JNet_408_4_label_plane]: /experiments/images/JNet_408_4_label_plane.png
[JNet_408_4_original_depth]: /experiments/images/JNet_408_4_original_depth.png
[JNet_408_4_original_plane]: /experiments/images/JNet_408_4_original_plane.png
[JNet_408_4_output_depth]: /experiments/images/JNet_408_4_output_depth.png
[JNet_408_4_output_plane]: /experiments/images/JNet_408_4_output_plane.png
[JNet_408_beads_001_roi000_original_depth]: /experiments/images/JNet_408_beads_001_roi000_original_depth.png
[JNet_408_beads_001_roi000_output_depth]: /experiments/images/JNet_408_beads_001_roi000_output_depth.png
[JNet_408_beads_001_roi000_reconst_depth]: /experiments/images/JNet_408_beads_001_roi000_reconst_depth.png
[JNet_408_beads_001_roi001_original_depth]: /experiments/images/JNet_408_beads_001_roi001_original_depth.png
[JNet_408_beads_001_roi001_output_depth]: /experiments/images/JNet_408_beads_001_roi001_output_depth.png
[JNet_408_beads_001_roi001_reconst_depth]: /experiments/images/JNet_408_beads_001_roi001_reconst_depth.png
[JNet_408_beads_001_roi002_original_depth]: /experiments/images/JNet_408_beads_001_roi002_original_depth.png
[JNet_408_beads_001_roi002_output_depth]: /experiments/images/JNet_408_beads_001_roi002_output_depth.png
[JNet_408_beads_001_roi002_reconst_depth]: /experiments/images/JNet_408_beads_001_roi002_reconst_depth.png
[JNet_408_beads_001_roi003_original_depth]: /experiments/images/JNet_408_beads_001_roi003_original_depth.png
[JNet_408_beads_001_roi003_output_depth]: /experiments/images/JNet_408_beads_001_roi003_output_depth.png
[JNet_408_beads_001_roi003_reconst_depth]: /experiments/images/JNet_408_beads_001_roi003_reconst_depth.png
[JNet_408_beads_001_roi004_original_depth]: /experiments/images/JNet_408_beads_001_roi004_original_depth.png
[JNet_408_beads_001_roi004_output_depth]: /experiments/images/JNet_408_beads_001_roi004_output_depth.png
[JNet_408_beads_001_roi004_reconst_depth]: /experiments/images/JNet_408_beads_001_roi004_reconst_depth.png
[JNet_408_beads_002_roi000_original_depth]: /experiments/images/JNet_408_beads_002_roi000_original_depth.png
[JNet_408_beads_002_roi000_output_depth]: /experiments/images/JNet_408_beads_002_roi000_output_depth.png
[JNet_408_beads_002_roi000_reconst_depth]: /experiments/images/JNet_408_beads_002_roi000_reconst_depth.png
[JNet_408_beads_002_roi001_original_depth]: /experiments/images/JNet_408_beads_002_roi001_original_depth.png
[JNet_408_beads_002_roi001_output_depth]: /experiments/images/JNet_408_beads_002_roi001_output_depth.png
[JNet_408_beads_002_roi001_reconst_depth]: /experiments/images/JNet_408_beads_002_roi001_reconst_depth.png
[JNet_408_beads_002_roi002_original_depth]: /experiments/images/JNet_408_beads_002_roi002_original_depth.png
[JNet_408_beads_002_roi002_output_depth]: /experiments/images/JNet_408_beads_002_roi002_output_depth.png
[JNet_408_beads_002_roi002_reconst_depth]: /experiments/images/JNet_408_beads_002_roi002_reconst_depth.png
[JNet_408_psf_post]: /experiments/images/JNet_408_psf_post.png
[JNet_408_psf_pre]: /experiments/images/JNet_408_psf_pre.png
[finetuned]: /experiments/tmp/JNet_408_train.png
[pretrained_model]: /experiments/tmp/JNet_407_pretrain_train.png
