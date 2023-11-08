



# JNet_412 Report
  
the parameters to replicate the results of JNet_412. nearest interp of PSF, logit loss = 1., NA = 0.7  
pretrained model : JNet_411_pretrain
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
|loss_fn|nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=params['device']))|
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
  
mean MSE: 0.02589581348001957, mean BCE: 0.10284918546676636
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_411_pretrain_0_original_plane]|![JNet_411_pretrain_0_output_plane]|![JNet_411_pretrain_0_label_plane]|
  
MSE: 0.02176993526518345, BCE: 0.08029066771268845  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_411_pretrain_0_original_depth]|![JNet_411_pretrain_0_output_depth]|![JNet_411_pretrain_0_label_depth]|
  
MSE: 0.02176993526518345, BCE: 0.08029066771268845  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_411_pretrain_1_original_plane]|![JNet_411_pretrain_1_output_plane]|![JNet_411_pretrain_1_label_plane]|
  
MSE: 0.03007151000201702, BCE: 0.10623566806316376  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_411_pretrain_1_original_depth]|![JNet_411_pretrain_1_output_depth]|![JNet_411_pretrain_1_label_depth]|
  
MSE: 0.03007151000201702, BCE: 0.10623566806316376  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_411_pretrain_2_original_plane]|![JNet_411_pretrain_2_output_plane]|![JNet_411_pretrain_2_label_plane]|
  
MSE: 0.03140329197049141, BCE: 0.11683665215969086  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_411_pretrain_2_original_depth]|![JNet_411_pretrain_2_output_depth]|![JNet_411_pretrain_2_label_depth]|
  
MSE: 0.03140329197049141, BCE: 0.11683665215969086  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_411_pretrain_3_original_plane]|![JNet_411_pretrain_3_output_plane]|![JNet_411_pretrain_3_label_plane]|
  
MSE: 0.019193226471543312, BCE: 0.07669878751039505  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_411_pretrain_3_original_depth]|![JNet_411_pretrain_3_output_depth]|![JNet_411_pretrain_3_label_depth]|
  
MSE: 0.019193226471543312, BCE: 0.07669878751039505  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_411_pretrain_4_original_plane]|![JNet_411_pretrain_4_output_plane]|![JNet_411_pretrain_4_label_plane]|
  
MSE: 0.027041101828217506, BCE: 0.13418418169021606  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_411_pretrain_4_original_depth]|![JNet_411_pretrain_4_output_depth]|![JNet_411_pretrain_4_label_depth]|
  
MSE: 0.027041101828217506, BCE: 0.13418418169021606  
  
mean MSE: 0.035251665860414505, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_412_0_original_plane]|![JNet_412_0_output_plane]|![JNet_412_0_label_plane]|
  
MSE: 0.04543552175164223, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_412_0_original_depth]|![JNet_412_0_output_depth]|![JNet_412_0_label_depth]|
  
MSE: 0.04543552175164223, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_412_1_original_plane]|![JNet_412_1_output_plane]|![JNet_412_1_label_plane]|
  
MSE: 0.031409043818712234, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_412_1_original_depth]|![JNet_412_1_output_depth]|![JNet_412_1_label_depth]|
  
MSE: 0.031409043818712234, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_412_2_original_plane]|![JNet_412_2_output_plane]|![JNet_412_2_label_plane]|
  
MSE: 0.031929321587085724, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_412_2_original_depth]|![JNet_412_2_output_depth]|![JNet_412_2_label_depth]|
  
MSE: 0.031929321587085724, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_412_3_original_plane]|![JNet_412_3_output_plane]|![JNet_412_3_label_plane]|
  
MSE: 0.04350656270980835, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_412_3_original_depth]|![JNet_412_3_output_depth]|![JNet_412_3_label_depth]|
  
MSE: 0.04350656270980835, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_412_4_original_plane]|![JNet_412_4_output_plane]|![JNet_412_4_label_plane]|
  
MSE: 0.023977873846888542, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_412_4_original_depth]|![JNet_412_4_output_depth]|![JNet_412_4_label_depth]|
  
MSE: 0.023977873846888542, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_411_pretrain_beads_001_roi000_original_depth]|![JNet_411_pretrain_beads_001_roi000_output_depth]|![JNet_411_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 1.9539985351562505, MSE: 0.002801923081278801, quantized loss: 0.0007171472534537315  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_411_pretrain_beads_001_roi001_original_depth]|![JNet_411_pretrain_beads_001_roi001_output_depth]|![JNet_411_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 2.6736718750000006, MSE: 0.004444709047675133, quantized loss: 0.001405365881510079  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_411_pretrain_beads_001_roi002_original_depth]|![JNet_411_pretrain_beads_001_roi002_output_depth]|![JNet_411_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 2.2711496582031256, MSE: 0.003264711005613208, quantized loss: 0.0009717885404825211  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_411_pretrain_beads_001_roi003_original_depth]|![JNet_411_pretrain_beads_001_roi003_output_depth]|![JNet_411_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 3.0995410156250007, MSE: 0.007831131108105183, quantized loss: 0.001475087134167552  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_411_pretrain_beads_001_roi004_original_depth]|![JNet_411_pretrain_beads_001_roi004_output_depth]|![JNet_411_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 2.5034934082031257, MSE: 0.004172184970229864, quantized loss: 0.0011191056109964848  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_411_pretrain_beads_002_roi000_original_depth]|![JNet_411_pretrain_beads_002_roi000_output_depth]|![JNet_411_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 2.7122663574218757, MSE: 0.00481260335072875, quantized loss: 0.0012436346150934696  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_411_pretrain_beads_002_roi001_original_depth]|![JNet_411_pretrain_beads_002_roi001_output_depth]|![JNet_411_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 2.5568193359375004, MSE: 0.004156888462603092, quantized loss: 0.001205655513331294  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_411_pretrain_beads_002_roi002_original_depth]|![JNet_411_pretrain_beads_002_roi002_output_depth]|![JNet_411_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 2.5651875000000004, MSE: 0.0042531718499958515, quantized loss: 0.0011780920904129744  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_412_beads_001_roi000_original_depth]|![JNet_412_beads_001_roi000_output_depth]|![JNet_412_beads_001_roi000_reconst_depth]|
  
volume: 1.527868041992188, MSE: 0.00027396820951253176, quantized loss: 2.0610448700608686e-05  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_412_beads_001_roi001_original_depth]|![JNet_412_beads_001_roi001_output_depth]|![JNet_412_beads_001_roi001_reconst_depth]|
  
volume: 2.1717211914062506, MSE: 0.0006773409550078213, quantized loss: 2.467403101036325e-05  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_412_beads_001_roi002_original_depth]|![JNet_412_beads_001_roi002_output_depth]|![JNet_412_beads_001_roi002_reconst_depth]|
  
volume: 1.510346313476563, MSE: 0.00023700612655375153, quantized loss: 1.9377328499103896e-05  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_412_beads_001_roi003_original_depth]|![JNet_412_beads_001_roi003_output_depth]|![JNet_412_beads_001_roi003_reconst_depth]|
  
volume: 2.2848918457031258, MSE: 0.0004390246467664838, quantized loss: 2.7415708245825954e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_412_beads_001_roi004_original_depth]|![JNet_412_beads_001_roi004_output_depth]|![JNet_412_beads_001_roi004_reconst_depth]|
  
volume: 1.5822016601562503, MSE: 0.00014901856775395572, quantized loss: 1.9700039047165774e-05  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_412_beads_002_roi000_original_depth]|![JNet_412_beads_002_roi000_output_depth]|![JNet_412_beads_002_roi000_reconst_depth]|
  
volume: 1.657910766601563, MSE: 0.0001362384791718796, quantized loss: 2.1520743757719174e-05  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_412_beads_002_roi001_original_depth]|![JNet_412_beads_002_roi001_output_depth]|![JNet_412_beads_002_roi001_reconst_depth]|
  
volume: 1.505567260742188, MSE: 0.0001241907011717558, quantized loss: 2.011851938732434e-05  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_412_beads_002_roi002_original_depth]|![JNet_412_beads_002_roi002_output_depth]|![JNet_412_beads_002_roi002_reconst_depth]|
  
volume: 1.6063825683593753, MSE: 0.0001464748493162915, quantized loss: 1.8433651348459534e-05  

|pre|post|
| :---: | :---: |
|![JNet_412_psf_pre]|![JNet_412_psf_post]|

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
  



[JNet_411_pretrain_0_label_depth]: /experiments/images/JNet_411_pretrain_0_label_depth.png
[JNet_411_pretrain_0_label_plane]: /experiments/images/JNet_411_pretrain_0_label_plane.png
[JNet_411_pretrain_0_original_depth]: /experiments/images/JNet_411_pretrain_0_original_depth.png
[JNet_411_pretrain_0_original_plane]: /experiments/images/JNet_411_pretrain_0_original_plane.png
[JNet_411_pretrain_0_output_depth]: /experiments/images/JNet_411_pretrain_0_output_depth.png
[JNet_411_pretrain_0_output_plane]: /experiments/images/JNet_411_pretrain_0_output_plane.png
[JNet_411_pretrain_1_label_depth]: /experiments/images/JNet_411_pretrain_1_label_depth.png
[JNet_411_pretrain_1_label_plane]: /experiments/images/JNet_411_pretrain_1_label_plane.png
[JNet_411_pretrain_1_original_depth]: /experiments/images/JNet_411_pretrain_1_original_depth.png
[JNet_411_pretrain_1_original_plane]: /experiments/images/JNet_411_pretrain_1_original_plane.png
[JNet_411_pretrain_1_output_depth]: /experiments/images/JNet_411_pretrain_1_output_depth.png
[JNet_411_pretrain_1_output_plane]: /experiments/images/JNet_411_pretrain_1_output_plane.png
[JNet_411_pretrain_2_label_depth]: /experiments/images/JNet_411_pretrain_2_label_depth.png
[JNet_411_pretrain_2_label_plane]: /experiments/images/JNet_411_pretrain_2_label_plane.png
[JNet_411_pretrain_2_original_depth]: /experiments/images/JNet_411_pretrain_2_original_depth.png
[JNet_411_pretrain_2_original_plane]: /experiments/images/JNet_411_pretrain_2_original_plane.png
[JNet_411_pretrain_2_output_depth]: /experiments/images/JNet_411_pretrain_2_output_depth.png
[JNet_411_pretrain_2_output_plane]: /experiments/images/JNet_411_pretrain_2_output_plane.png
[JNet_411_pretrain_3_label_depth]: /experiments/images/JNet_411_pretrain_3_label_depth.png
[JNet_411_pretrain_3_label_plane]: /experiments/images/JNet_411_pretrain_3_label_plane.png
[JNet_411_pretrain_3_original_depth]: /experiments/images/JNet_411_pretrain_3_original_depth.png
[JNet_411_pretrain_3_original_plane]: /experiments/images/JNet_411_pretrain_3_original_plane.png
[JNet_411_pretrain_3_output_depth]: /experiments/images/JNet_411_pretrain_3_output_depth.png
[JNet_411_pretrain_3_output_plane]: /experiments/images/JNet_411_pretrain_3_output_plane.png
[JNet_411_pretrain_4_label_depth]: /experiments/images/JNet_411_pretrain_4_label_depth.png
[JNet_411_pretrain_4_label_plane]: /experiments/images/JNet_411_pretrain_4_label_plane.png
[JNet_411_pretrain_4_original_depth]: /experiments/images/JNet_411_pretrain_4_original_depth.png
[JNet_411_pretrain_4_original_plane]: /experiments/images/JNet_411_pretrain_4_original_plane.png
[JNet_411_pretrain_4_output_depth]: /experiments/images/JNet_411_pretrain_4_output_depth.png
[JNet_411_pretrain_4_output_plane]: /experiments/images/JNet_411_pretrain_4_output_plane.png
[JNet_411_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi000_original_depth.png
[JNet_411_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi000_output_depth.png
[JNet_411_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi000_reconst_depth.png
[JNet_411_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi001_original_depth.png
[JNet_411_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi001_output_depth.png
[JNet_411_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi001_reconst_depth.png
[JNet_411_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi002_original_depth.png
[JNet_411_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi002_output_depth.png
[JNet_411_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi002_reconst_depth.png
[JNet_411_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi003_original_depth.png
[JNet_411_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi003_output_depth.png
[JNet_411_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi003_reconst_depth.png
[JNet_411_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi004_original_depth.png
[JNet_411_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi004_output_depth.png
[JNet_411_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_411_pretrain_beads_001_roi004_reconst_depth.png
[JNet_411_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_411_pretrain_beads_002_roi000_original_depth.png
[JNet_411_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_411_pretrain_beads_002_roi000_output_depth.png
[JNet_411_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_411_pretrain_beads_002_roi000_reconst_depth.png
[JNet_411_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_411_pretrain_beads_002_roi001_original_depth.png
[JNet_411_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_411_pretrain_beads_002_roi001_output_depth.png
[JNet_411_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_411_pretrain_beads_002_roi001_reconst_depth.png
[JNet_411_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_411_pretrain_beads_002_roi002_original_depth.png
[JNet_411_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_411_pretrain_beads_002_roi002_output_depth.png
[JNet_411_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_411_pretrain_beads_002_roi002_reconst_depth.png
[JNet_412_0_label_depth]: /experiments/images/JNet_412_0_label_depth.png
[JNet_412_0_label_plane]: /experiments/images/JNet_412_0_label_plane.png
[JNet_412_0_original_depth]: /experiments/images/JNet_412_0_original_depth.png
[JNet_412_0_original_plane]: /experiments/images/JNet_412_0_original_plane.png
[JNet_412_0_output_depth]: /experiments/images/JNet_412_0_output_depth.png
[JNet_412_0_output_plane]: /experiments/images/JNet_412_0_output_plane.png
[JNet_412_1_label_depth]: /experiments/images/JNet_412_1_label_depth.png
[JNet_412_1_label_plane]: /experiments/images/JNet_412_1_label_plane.png
[JNet_412_1_original_depth]: /experiments/images/JNet_412_1_original_depth.png
[JNet_412_1_original_plane]: /experiments/images/JNet_412_1_original_plane.png
[JNet_412_1_output_depth]: /experiments/images/JNet_412_1_output_depth.png
[JNet_412_1_output_plane]: /experiments/images/JNet_412_1_output_plane.png
[JNet_412_2_label_depth]: /experiments/images/JNet_412_2_label_depth.png
[JNet_412_2_label_plane]: /experiments/images/JNet_412_2_label_plane.png
[JNet_412_2_original_depth]: /experiments/images/JNet_412_2_original_depth.png
[JNet_412_2_original_plane]: /experiments/images/JNet_412_2_original_plane.png
[JNet_412_2_output_depth]: /experiments/images/JNet_412_2_output_depth.png
[JNet_412_2_output_plane]: /experiments/images/JNet_412_2_output_plane.png
[JNet_412_3_label_depth]: /experiments/images/JNet_412_3_label_depth.png
[JNet_412_3_label_plane]: /experiments/images/JNet_412_3_label_plane.png
[JNet_412_3_original_depth]: /experiments/images/JNet_412_3_original_depth.png
[JNet_412_3_original_plane]: /experiments/images/JNet_412_3_original_plane.png
[JNet_412_3_output_depth]: /experiments/images/JNet_412_3_output_depth.png
[JNet_412_3_output_plane]: /experiments/images/JNet_412_3_output_plane.png
[JNet_412_4_label_depth]: /experiments/images/JNet_412_4_label_depth.png
[JNet_412_4_label_plane]: /experiments/images/JNet_412_4_label_plane.png
[JNet_412_4_original_depth]: /experiments/images/JNet_412_4_original_depth.png
[JNet_412_4_original_plane]: /experiments/images/JNet_412_4_original_plane.png
[JNet_412_4_output_depth]: /experiments/images/JNet_412_4_output_depth.png
[JNet_412_4_output_plane]: /experiments/images/JNet_412_4_output_plane.png
[JNet_412_beads_001_roi000_original_depth]: /experiments/images/JNet_412_beads_001_roi000_original_depth.png
[JNet_412_beads_001_roi000_output_depth]: /experiments/images/JNet_412_beads_001_roi000_output_depth.png
[JNet_412_beads_001_roi000_reconst_depth]: /experiments/images/JNet_412_beads_001_roi000_reconst_depth.png
[JNet_412_beads_001_roi001_original_depth]: /experiments/images/JNet_412_beads_001_roi001_original_depth.png
[JNet_412_beads_001_roi001_output_depth]: /experiments/images/JNet_412_beads_001_roi001_output_depth.png
[JNet_412_beads_001_roi001_reconst_depth]: /experiments/images/JNet_412_beads_001_roi001_reconst_depth.png
[JNet_412_beads_001_roi002_original_depth]: /experiments/images/JNet_412_beads_001_roi002_original_depth.png
[JNet_412_beads_001_roi002_output_depth]: /experiments/images/JNet_412_beads_001_roi002_output_depth.png
[JNet_412_beads_001_roi002_reconst_depth]: /experiments/images/JNet_412_beads_001_roi002_reconst_depth.png
[JNet_412_beads_001_roi003_original_depth]: /experiments/images/JNet_412_beads_001_roi003_original_depth.png
[JNet_412_beads_001_roi003_output_depth]: /experiments/images/JNet_412_beads_001_roi003_output_depth.png
[JNet_412_beads_001_roi003_reconst_depth]: /experiments/images/JNet_412_beads_001_roi003_reconst_depth.png
[JNet_412_beads_001_roi004_original_depth]: /experiments/images/JNet_412_beads_001_roi004_original_depth.png
[JNet_412_beads_001_roi004_output_depth]: /experiments/images/JNet_412_beads_001_roi004_output_depth.png
[JNet_412_beads_001_roi004_reconst_depth]: /experiments/images/JNet_412_beads_001_roi004_reconst_depth.png
[JNet_412_beads_002_roi000_original_depth]: /experiments/images/JNet_412_beads_002_roi000_original_depth.png
[JNet_412_beads_002_roi000_output_depth]: /experiments/images/JNet_412_beads_002_roi000_output_depth.png
[JNet_412_beads_002_roi000_reconst_depth]: /experiments/images/JNet_412_beads_002_roi000_reconst_depth.png
[JNet_412_beads_002_roi001_original_depth]: /experiments/images/JNet_412_beads_002_roi001_original_depth.png
[JNet_412_beads_002_roi001_output_depth]: /experiments/images/JNet_412_beads_002_roi001_output_depth.png
[JNet_412_beads_002_roi001_reconst_depth]: /experiments/images/JNet_412_beads_002_roi001_reconst_depth.png
[JNet_412_beads_002_roi002_original_depth]: /experiments/images/JNet_412_beads_002_roi002_original_depth.png
[JNet_412_beads_002_roi002_output_depth]: /experiments/images/JNet_412_beads_002_roi002_output_depth.png
[JNet_412_beads_002_roi002_reconst_depth]: /experiments/images/JNet_412_beads_002_roi002_reconst_depth.png
[JNet_412_psf_post]: /experiments/images/JNet_412_psf_post.png
[JNet_412_psf_pre]: /experiments/images/JNet_412_psf_pre.png
[finetuned]: /experiments/tmp/JNet_412_train.png
[pretrained_model]: /experiments/tmp/JNet_411_pretrain_train.png
