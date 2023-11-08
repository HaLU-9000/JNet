



# JNet_409 Report
  
the parameters to replicate the results of JNet_409. same as 408.  
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
  
mean MSE: 0.021496422588825226, mean BCE: 0.07958251237869263
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_0_original_plane]|![JNet_407_pretrain_0_output_plane]|![JNet_407_pretrain_0_label_plane]|
  
MSE: 0.029069814831018448, BCE: 0.1127784475684166  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_0_original_depth]|![JNet_407_pretrain_0_output_depth]|![JNet_407_pretrain_0_label_depth]|
  
MSE: 0.029069814831018448, BCE: 0.1127784475684166  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_1_original_plane]|![JNet_407_pretrain_1_output_plane]|![JNet_407_pretrain_1_label_plane]|
  
MSE: 0.017249977216124535, BCE: 0.06585695594549179  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_1_original_depth]|![JNet_407_pretrain_1_output_depth]|![JNet_407_pretrain_1_label_depth]|
  
MSE: 0.017249977216124535, BCE: 0.06585695594549179  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_2_original_plane]|![JNet_407_pretrain_2_output_plane]|![JNet_407_pretrain_2_label_plane]|
  
MSE: 0.025774924084544182, BCE: 0.09126245230436325  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_2_original_depth]|![JNet_407_pretrain_2_output_depth]|![JNet_407_pretrain_2_label_depth]|
  
MSE: 0.025774924084544182, BCE: 0.09126245230436325  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_3_original_plane]|![JNet_407_pretrain_3_output_plane]|![JNet_407_pretrain_3_label_plane]|
  
MSE: 0.022937582805752754, BCE: 0.08346625417470932  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_3_original_depth]|![JNet_407_pretrain_3_output_depth]|![JNet_407_pretrain_3_label_depth]|
  
MSE: 0.022937582805752754, BCE: 0.08346625417470932  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_4_original_plane]|![JNet_407_pretrain_4_output_plane]|![JNet_407_pretrain_4_label_plane]|
  
MSE: 0.012449813075363636, BCE: 0.04454844072461128  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_407_pretrain_4_original_depth]|![JNet_407_pretrain_4_output_depth]|![JNet_407_pretrain_4_label_depth]|
  
MSE: 0.012449813075363636, BCE: 0.04454844072461128  
  
mean MSE: 0.030876491218805313, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_409_0_original_plane]|![JNet_409_0_output_plane]|![JNet_409_0_label_plane]|
  
MSE: 0.03046300634741783, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_409_0_original_depth]|![JNet_409_0_output_depth]|![JNet_409_0_label_depth]|
  
MSE: 0.03046300634741783, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_409_1_original_plane]|![JNet_409_1_output_plane]|![JNet_409_1_label_plane]|
  
MSE: 0.03617783263325691, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_409_1_original_depth]|![JNet_409_1_output_depth]|![JNet_409_1_label_depth]|
  
MSE: 0.03617783263325691, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_409_2_original_plane]|![JNet_409_2_output_plane]|![JNet_409_2_label_plane]|
  
MSE: 0.03451692312955856, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_409_2_original_depth]|![JNet_409_2_output_depth]|![JNet_409_2_label_depth]|
  
MSE: 0.03451692312955856, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_409_3_original_plane]|![JNet_409_3_output_plane]|![JNet_409_3_label_plane]|
  
MSE: 0.02868836559355259, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_409_3_original_depth]|![JNet_409_3_output_depth]|![JNet_409_3_label_depth]|
  
MSE: 0.02868836559355259, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_409_4_original_plane]|![JNet_409_4_output_plane]|![JNet_409_4_label_plane]|
  
MSE: 0.02453632466495037, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_409_4_original_depth]|![JNet_409_4_output_depth]|![JNet_409_4_label_depth]|
  
MSE: 0.02453632466495037, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_001_roi000_original_depth]|![JNet_407_pretrain_beads_001_roi000_output_depth]|![JNet_407_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 1.1305373535156253, MSE: 0.002964101731777191, quantized loss: 0.0004109727160539478  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_001_roi001_original_depth]|![JNet_407_pretrain_beads_001_roi001_output_depth]|![JNet_407_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 1.643363891601563, MSE: 0.005106337834149599, quantized loss: 0.0007220489205792546  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_001_roi002_original_depth]|![JNet_407_pretrain_beads_001_roi002_output_depth]|![JNet_407_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 1.4448996582031253, MSE: 0.002319271443411708, quantized loss: 0.0006344459834508598  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_001_roi003_original_depth]|![JNet_407_pretrain_beads_001_roi003_output_depth]|![JNet_407_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 2.1272116699218757, MSE: 0.005650651175528765, quantized loss: 0.0009322650148533285  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_001_roi004_original_depth]|![JNet_407_pretrain_beads_001_roi004_output_depth]|![JNet_407_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 1.5132409667968754, MSE: 0.0028237912338227034, quantized loss: 0.0006969032110646367  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_002_roi000_original_depth]|![JNet_407_pretrain_beads_002_roi000_output_depth]|![JNet_407_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 1.613532836914063, MSE: 0.0032083916012197733, quantized loss: 0.0007405178621411324  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_002_roi001_original_depth]|![JNet_407_pretrain_beads_002_roi001_output_depth]|![JNet_407_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 1.6988107910156254, MSE: 0.0021884909365326166, quantized loss: 0.0007907944964244962  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_407_pretrain_beads_002_roi002_original_depth]|![JNet_407_pretrain_beads_002_roi002_output_depth]|![JNet_407_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 1.5992633056640628, MSE: 0.002832418540492654, quantized loss: 0.0007267890032380819  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_409_beads_001_roi000_original_depth]|![JNet_409_beads_001_roi000_output_depth]|![JNet_409_beads_001_roi000_reconst_depth]|
  
volume: 1.2990638427734378, MSE: 0.0001652841892791912, quantized loss: 6.657418998656794e-05  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_409_beads_001_roi001_original_depth]|![JNet_409_beads_001_roi001_output_depth]|![JNet_409_beads_001_roi001_reconst_depth]|
  
volume: 1.930450561523438, MSE: 0.0006262995302677155, quantized loss: 0.00010031417332356796  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_409_beads_001_roi002_original_depth]|![JNet_409_beads_001_roi002_output_depth]|![JNet_409_beads_001_roi002_reconst_depth]|
  
volume: 1.2849794921875004, MSE: 0.00015892021474428475, quantized loss: 7.075255416566506e-05  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_409_beads_001_roi003_original_depth]|![JNet_409_beads_001_roi003_output_depth]|![JNet_409_beads_001_roi003_reconst_depth]|
  
volume: 2.0185246582031255, MSE: 0.0004743396712001413, quantized loss: 8.85284025571309e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_409_beads_001_roi004_original_depth]|![JNet_409_beads_001_roi004_output_depth]|![JNet_409_beads_001_roi004_reconst_depth]|
  
volume: 1.3950136718750004, MSE: 0.0001282692392123863, quantized loss: 5.756153041147627e-05  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_409_beads_002_roi000_original_depth]|![JNet_409_beads_002_roi000_output_depth]|![JNet_409_beads_002_roi000_reconst_depth]|
  
volume: 1.5053344726562503, MSE: 0.00013028363173361868, quantized loss: 6.54910909361206e-05  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_409_beads_002_roi001_original_depth]|![JNet_409_beads_002_roi001_output_depth]|![JNet_409_beads_002_roi001_reconst_depth]|
  
volume: 1.3419661865234378, MSE: 0.00012181862257421017, quantized loss: 6.841536378487945e-05  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_409_beads_002_roi002_original_depth]|![JNet_409_beads_002_roi002_output_depth]|![JNet_409_beads_002_roi002_reconst_depth]|
  
volume: 1.4244260253906253, MSE: 0.00014181456936057657, quantized loss: 7.077044574543834e-05  

|pre|post|
| :---: | :---: |
|![JNet_409_psf_pre]|![JNet_409_psf_post]|

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
[JNet_409_0_label_depth]: /experiments/images/JNet_409_0_label_depth.png
[JNet_409_0_label_plane]: /experiments/images/JNet_409_0_label_plane.png
[JNet_409_0_original_depth]: /experiments/images/JNet_409_0_original_depth.png
[JNet_409_0_original_plane]: /experiments/images/JNet_409_0_original_plane.png
[JNet_409_0_output_depth]: /experiments/images/JNet_409_0_output_depth.png
[JNet_409_0_output_plane]: /experiments/images/JNet_409_0_output_plane.png
[JNet_409_1_label_depth]: /experiments/images/JNet_409_1_label_depth.png
[JNet_409_1_label_plane]: /experiments/images/JNet_409_1_label_plane.png
[JNet_409_1_original_depth]: /experiments/images/JNet_409_1_original_depth.png
[JNet_409_1_original_plane]: /experiments/images/JNet_409_1_original_plane.png
[JNet_409_1_output_depth]: /experiments/images/JNet_409_1_output_depth.png
[JNet_409_1_output_plane]: /experiments/images/JNet_409_1_output_plane.png
[JNet_409_2_label_depth]: /experiments/images/JNet_409_2_label_depth.png
[JNet_409_2_label_plane]: /experiments/images/JNet_409_2_label_plane.png
[JNet_409_2_original_depth]: /experiments/images/JNet_409_2_original_depth.png
[JNet_409_2_original_plane]: /experiments/images/JNet_409_2_original_plane.png
[JNet_409_2_output_depth]: /experiments/images/JNet_409_2_output_depth.png
[JNet_409_2_output_plane]: /experiments/images/JNet_409_2_output_plane.png
[JNet_409_3_label_depth]: /experiments/images/JNet_409_3_label_depth.png
[JNet_409_3_label_plane]: /experiments/images/JNet_409_3_label_plane.png
[JNet_409_3_original_depth]: /experiments/images/JNet_409_3_original_depth.png
[JNet_409_3_original_plane]: /experiments/images/JNet_409_3_original_plane.png
[JNet_409_3_output_depth]: /experiments/images/JNet_409_3_output_depth.png
[JNet_409_3_output_plane]: /experiments/images/JNet_409_3_output_plane.png
[JNet_409_4_label_depth]: /experiments/images/JNet_409_4_label_depth.png
[JNet_409_4_label_plane]: /experiments/images/JNet_409_4_label_plane.png
[JNet_409_4_original_depth]: /experiments/images/JNet_409_4_original_depth.png
[JNet_409_4_original_plane]: /experiments/images/JNet_409_4_original_plane.png
[JNet_409_4_output_depth]: /experiments/images/JNet_409_4_output_depth.png
[JNet_409_4_output_plane]: /experiments/images/JNet_409_4_output_plane.png
[JNet_409_beads_001_roi000_original_depth]: /experiments/images/JNet_409_beads_001_roi000_original_depth.png
[JNet_409_beads_001_roi000_output_depth]: /experiments/images/JNet_409_beads_001_roi000_output_depth.png
[JNet_409_beads_001_roi000_reconst_depth]: /experiments/images/JNet_409_beads_001_roi000_reconst_depth.png
[JNet_409_beads_001_roi001_original_depth]: /experiments/images/JNet_409_beads_001_roi001_original_depth.png
[JNet_409_beads_001_roi001_output_depth]: /experiments/images/JNet_409_beads_001_roi001_output_depth.png
[JNet_409_beads_001_roi001_reconst_depth]: /experiments/images/JNet_409_beads_001_roi001_reconst_depth.png
[JNet_409_beads_001_roi002_original_depth]: /experiments/images/JNet_409_beads_001_roi002_original_depth.png
[JNet_409_beads_001_roi002_output_depth]: /experiments/images/JNet_409_beads_001_roi002_output_depth.png
[JNet_409_beads_001_roi002_reconst_depth]: /experiments/images/JNet_409_beads_001_roi002_reconst_depth.png
[JNet_409_beads_001_roi003_original_depth]: /experiments/images/JNet_409_beads_001_roi003_original_depth.png
[JNet_409_beads_001_roi003_output_depth]: /experiments/images/JNet_409_beads_001_roi003_output_depth.png
[JNet_409_beads_001_roi003_reconst_depth]: /experiments/images/JNet_409_beads_001_roi003_reconst_depth.png
[JNet_409_beads_001_roi004_original_depth]: /experiments/images/JNet_409_beads_001_roi004_original_depth.png
[JNet_409_beads_001_roi004_output_depth]: /experiments/images/JNet_409_beads_001_roi004_output_depth.png
[JNet_409_beads_001_roi004_reconst_depth]: /experiments/images/JNet_409_beads_001_roi004_reconst_depth.png
[JNet_409_beads_002_roi000_original_depth]: /experiments/images/JNet_409_beads_002_roi000_original_depth.png
[JNet_409_beads_002_roi000_output_depth]: /experiments/images/JNet_409_beads_002_roi000_output_depth.png
[JNet_409_beads_002_roi000_reconst_depth]: /experiments/images/JNet_409_beads_002_roi000_reconst_depth.png
[JNet_409_beads_002_roi001_original_depth]: /experiments/images/JNet_409_beads_002_roi001_original_depth.png
[JNet_409_beads_002_roi001_output_depth]: /experiments/images/JNet_409_beads_002_roi001_output_depth.png
[JNet_409_beads_002_roi001_reconst_depth]: /experiments/images/JNet_409_beads_002_roi001_reconst_depth.png
[JNet_409_beads_002_roi002_original_depth]: /experiments/images/JNet_409_beads_002_roi002_original_depth.png
[JNet_409_beads_002_roi002_output_depth]: /experiments/images/JNet_409_beads_002_roi002_output_depth.png
[JNet_409_beads_002_roi002_reconst_depth]: /experiments/images/JNet_409_beads_002_roi002_reconst_depth.png
[JNet_409_psf_post]: /experiments/images/JNet_409_psf_post.png
[JNet_409_psf_pre]: /experiments/images/JNet_409_psf_pre.png
[finetuned]: /experiments/tmp/JNet_409_train.png
[pretrained_model]: /experiments/tmp/JNet_407_pretrain_train.png
