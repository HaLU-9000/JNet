



# JNet_416 Report
  
the parameters to replicate the results of JNet_416. nearest interp of PSF, logit loss = 1.0, NA = 0.8  
pretrained model : JNet_415_pretrain
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
  
mean MSE: 0.023406393826007843, mean BCE: 0.09314142912626266
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_415_pretrain_0_original_plane]|![JNet_415_pretrain_0_output_plane]|![JNet_415_pretrain_0_label_plane]|
  
MSE: 0.01890525221824646, BCE: 0.0677155926823616  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_415_pretrain_0_original_depth]|![JNet_415_pretrain_0_output_depth]|![JNet_415_pretrain_0_label_depth]|
  
MSE: 0.01890525221824646, BCE: 0.0677155926823616  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_415_pretrain_1_original_plane]|![JNet_415_pretrain_1_output_plane]|![JNet_415_pretrain_1_label_plane]|
  
MSE: 0.03348435461521149, BCE: 0.15472964942455292  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_415_pretrain_1_original_depth]|![JNet_415_pretrain_1_output_depth]|![JNet_415_pretrain_1_label_depth]|
  
MSE: 0.03348435461521149, BCE: 0.15472964942455292  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_415_pretrain_2_original_plane]|![JNet_415_pretrain_2_output_plane]|![JNet_415_pretrain_2_label_plane]|
  
MSE: 0.018531566485762596, BCE: 0.06646449863910675  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_415_pretrain_2_original_depth]|![JNet_415_pretrain_2_output_depth]|![JNet_415_pretrain_2_label_depth]|
  
MSE: 0.018531566485762596, BCE: 0.06646449863910675  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_415_pretrain_3_original_plane]|![JNet_415_pretrain_3_output_plane]|![JNet_415_pretrain_3_label_plane]|
  
MSE: 0.025907762348651886, BCE: 0.10485467314720154  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_415_pretrain_3_original_depth]|![JNet_415_pretrain_3_output_depth]|![JNet_415_pretrain_3_label_depth]|
  
MSE: 0.025907762348651886, BCE: 0.10485467314720154  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_415_pretrain_4_original_plane]|![JNet_415_pretrain_4_output_plane]|![JNet_415_pretrain_4_label_plane]|
  
MSE: 0.020203029736876488, BCE: 0.07194273918867111  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_415_pretrain_4_original_depth]|![JNet_415_pretrain_4_output_depth]|![JNet_415_pretrain_4_label_depth]|
  
MSE: 0.020203029736876488, BCE: 0.07194273918867111  
  
mean MSE: 0.05604792386293411, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_416_0_original_plane]|![JNet_416_0_output_plane]|![JNet_416_0_label_plane]|
  
MSE: 0.04426819831132889, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_416_0_original_depth]|![JNet_416_0_output_depth]|![JNet_416_0_label_depth]|
  
MSE: 0.04426819831132889, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_416_1_original_plane]|![JNet_416_1_output_plane]|![JNet_416_1_label_plane]|
  
MSE: 0.08281330019235611, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_416_1_original_depth]|![JNet_416_1_output_depth]|![JNet_416_1_label_depth]|
  
MSE: 0.08281330019235611, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_416_2_original_plane]|![JNet_416_2_output_plane]|![JNet_416_2_label_plane]|
  
MSE: 0.08914121985435486, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_416_2_original_depth]|![JNet_416_2_output_depth]|![JNet_416_2_label_depth]|
  
MSE: 0.08914121985435486, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_416_3_original_plane]|![JNet_416_3_output_plane]|![JNet_416_3_label_plane]|
  
MSE: 0.04327268898487091, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_416_3_original_depth]|![JNet_416_3_output_depth]|![JNet_416_3_label_depth]|
  
MSE: 0.04327268898487091, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_416_4_original_plane]|![JNet_416_4_output_plane]|![JNet_416_4_label_plane]|
  
MSE: 0.020744197070598602, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_416_4_original_depth]|![JNet_416_4_output_depth]|![JNet_416_4_label_depth]|
  
MSE: 0.020744197070598602, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_415_pretrain_beads_001_roi000_original_depth]|![JNet_415_pretrain_beads_001_roi000_output_depth]|![JNet_415_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 2.2697729492187504, MSE: 0.0023025572299957275, quantized loss: 0.0007045672973617911  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_415_pretrain_beads_001_roi001_original_depth]|![JNet_415_pretrain_beads_001_roi001_output_depth]|![JNet_415_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 2.8623540039062507, MSE: 0.003003098303452134, quantized loss: 0.001078323693946004  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_415_pretrain_beads_001_roi002_original_depth]|![JNet_415_pretrain_beads_001_roi002_output_depth]|![JNet_415_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 2.2674047851562507, MSE: 0.0024411126505583525, quantized loss: 0.0008285228977911174  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_415_pretrain_beads_001_roi003_original_depth]|![JNet_415_pretrain_beads_001_roi003_output_depth]|![JNet_415_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 2.9030720214843755, MSE: 0.004163907375186682, quantized loss: 0.0011435606284067035  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_415_pretrain_beads_001_roi004_original_depth]|![JNet_415_pretrain_beads_001_roi004_output_depth]|![JNet_415_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 2.5026425781250006, MSE: 0.00271610077470541, quantized loss: 0.0009212670847773552  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_415_pretrain_beads_002_roi000_original_depth]|![JNet_415_pretrain_beads_002_roi000_output_depth]|![JNet_415_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 2.6437436523437507, MSE: 0.002949229208752513, quantized loss: 0.0009671046864241362  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_415_pretrain_beads_002_roi001_original_depth]|![JNet_415_pretrain_beads_002_roi001_output_depth]|![JNet_415_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 2.3930056152343755, MSE: 0.002384454943239689, quantized loss: 0.0008987230830825865  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_415_pretrain_beads_002_roi002_original_depth]|![JNet_415_pretrain_beads_002_roi002_output_depth]|![JNet_415_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 2.4639306640625005, MSE: 0.00266215275041759, quantized loss: 0.0009113173000514507  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_416_beads_001_roi000_original_depth]|![JNet_416_beads_001_roi000_output_depth]|![JNet_416_beads_001_roi000_reconst_depth]|
  
volume: 1.2872192382812504, MSE: 0.0001912888983497396, quantized loss: 6.90172828399227e-06  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_416_beads_001_roi001_original_depth]|![JNet_416_beads_001_roi001_output_depth]|![JNet_416_beads_001_roi001_reconst_depth]|
  
volume: 1.9500783691406254, MSE: 0.0006683667306788266, quantized loss: 9.925687663780991e-06  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_416_beads_001_roi002_original_depth]|![JNet_416_beads_001_roi002_output_depth]|![JNet_416_beads_001_roi002_reconst_depth]|
  
volume: 1.280926147460938, MSE: 0.00011157347034895793, quantized loss: 6.894134912727168e-06  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_416_beads_001_roi003_original_depth]|![JNet_416_beads_001_roi003_output_depth]|![JNet_416_beads_001_roi003_reconst_depth]|
  
volume: 2.024602172851563, MSE: 0.0005331120337359607, quantized loss: 1.011991571431281e-05  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_416_beads_001_roi004_original_depth]|![JNet_416_beads_001_roi004_output_depth]|![JNet_416_beads_001_roi004_reconst_depth]|
  
volume: 1.4076506347656252, MSE: 0.00014356222527567297, quantized loss: 7.6247520155448e-06  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_416_beads_002_roi000_original_depth]|![JNet_416_beads_002_roi000_output_depth]|![JNet_416_beads_002_roi000_reconst_depth]|
  
volume: 1.4817080078125004, MSE: 0.00015764852287247777, quantized loss: 6.869179742352571e-06  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_416_beads_002_roi001_original_depth]|![JNet_416_beads_002_roi001_output_depth]|![JNet_416_beads_002_roi001_reconst_depth]|
  
volume: 1.3096298828125004, MSE: 0.00015442575386259705, quantized loss: 7.322787041630363e-06  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_416_beads_002_roi002_original_depth]|![JNet_416_beads_002_roi002_output_depth]|![JNet_416_beads_002_roi002_reconst_depth]|
  
volume: 1.3441877441406254, MSE: 0.00021305955306161195, quantized loss: 6.786408448533621e-06  

|pre|post|
| :---: | :---: |
|![JNet_416_psf_pre]|![JNet_416_psf_post]|

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
  



[JNet_415_pretrain_0_label_depth]: /experiments/images/JNet_415_pretrain_0_label_depth.png
[JNet_415_pretrain_0_label_plane]: /experiments/images/JNet_415_pretrain_0_label_plane.png
[JNet_415_pretrain_0_original_depth]: /experiments/images/JNet_415_pretrain_0_original_depth.png
[JNet_415_pretrain_0_original_plane]: /experiments/images/JNet_415_pretrain_0_original_plane.png
[JNet_415_pretrain_0_output_depth]: /experiments/images/JNet_415_pretrain_0_output_depth.png
[JNet_415_pretrain_0_output_plane]: /experiments/images/JNet_415_pretrain_0_output_plane.png
[JNet_415_pretrain_1_label_depth]: /experiments/images/JNet_415_pretrain_1_label_depth.png
[JNet_415_pretrain_1_label_plane]: /experiments/images/JNet_415_pretrain_1_label_plane.png
[JNet_415_pretrain_1_original_depth]: /experiments/images/JNet_415_pretrain_1_original_depth.png
[JNet_415_pretrain_1_original_plane]: /experiments/images/JNet_415_pretrain_1_original_plane.png
[JNet_415_pretrain_1_output_depth]: /experiments/images/JNet_415_pretrain_1_output_depth.png
[JNet_415_pretrain_1_output_plane]: /experiments/images/JNet_415_pretrain_1_output_plane.png
[JNet_415_pretrain_2_label_depth]: /experiments/images/JNet_415_pretrain_2_label_depth.png
[JNet_415_pretrain_2_label_plane]: /experiments/images/JNet_415_pretrain_2_label_plane.png
[JNet_415_pretrain_2_original_depth]: /experiments/images/JNet_415_pretrain_2_original_depth.png
[JNet_415_pretrain_2_original_plane]: /experiments/images/JNet_415_pretrain_2_original_plane.png
[JNet_415_pretrain_2_output_depth]: /experiments/images/JNet_415_pretrain_2_output_depth.png
[JNet_415_pretrain_2_output_plane]: /experiments/images/JNet_415_pretrain_2_output_plane.png
[JNet_415_pretrain_3_label_depth]: /experiments/images/JNet_415_pretrain_3_label_depth.png
[JNet_415_pretrain_3_label_plane]: /experiments/images/JNet_415_pretrain_3_label_plane.png
[JNet_415_pretrain_3_original_depth]: /experiments/images/JNet_415_pretrain_3_original_depth.png
[JNet_415_pretrain_3_original_plane]: /experiments/images/JNet_415_pretrain_3_original_plane.png
[JNet_415_pretrain_3_output_depth]: /experiments/images/JNet_415_pretrain_3_output_depth.png
[JNet_415_pretrain_3_output_plane]: /experiments/images/JNet_415_pretrain_3_output_plane.png
[JNet_415_pretrain_4_label_depth]: /experiments/images/JNet_415_pretrain_4_label_depth.png
[JNet_415_pretrain_4_label_plane]: /experiments/images/JNet_415_pretrain_4_label_plane.png
[JNet_415_pretrain_4_original_depth]: /experiments/images/JNet_415_pretrain_4_original_depth.png
[JNet_415_pretrain_4_original_plane]: /experiments/images/JNet_415_pretrain_4_original_plane.png
[JNet_415_pretrain_4_output_depth]: /experiments/images/JNet_415_pretrain_4_output_depth.png
[JNet_415_pretrain_4_output_plane]: /experiments/images/JNet_415_pretrain_4_output_plane.png
[JNet_415_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi000_original_depth.png
[JNet_415_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi000_output_depth.png
[JNet_415_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi000_reconst_depth.png
[JNet_415_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi001_original_depth.png
[JNet_415_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi001_output_depth.png
[JNet_415_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi001_reconst_depth.png
[JNet_415_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi002_original_depth.png
[JNet_415_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi002_output_depth.png
[JNet_415_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi002_reconst_depth.png
[JNet_415_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi003_original_depth.png
[JNet_415_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi003_output_depth.png
[JNet_415_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi003_reconst_depth.png
[JNet_415_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi004_original_depth.png
[JNet_415_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi004_output_depth.png
[JNet_415_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_415_pretrain_beads_001_roi004_reconst_depth.png
[JNet_415_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_415_pretrain_beads_002_roi000_original_depth.png
[JNet_415_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_415_pretrain_beads_002_roi000_output_depth.png
[JNet_415_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_415_pretrain_beads_002_roi000_reconst_depth.png
[JNet_415_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_415_pretrain_beads_002_roi001_original_depth.png
[JNet_415_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_415_pretrain_beads_002_roi001_output_depth.png
[JNet_415_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_415_pretrain_beads_002_roi001_reconst_depth.png
[JNet_415_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_415_pretrain_beads_002_roi002_original_depth.png
[JNet_415_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_415_pretrain_beads_002_roi002_output_depth.png
[JNet_415_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_415_pretrain_beads_002_roi002_reconst_depth.png
[JNet_416_0_label_depth]: /experiments/images/JNet_416_0_label_depth.png
[JNet_416_0_label_plane]: /experiments/images/JNet_416_0_label_plane.png
[JNet_416_0_original_depth]: /experiments/images/JNet_416_0_original_depth.png
[JNet_416_0_original_plane]: /experiments/images/JNet_416_0_original_plane.png
[JNet_416_0_output_depth]: /experiments/images/JNet_416_0_output_depth.png
[JNet_416_0_output_plane]: /experiments/images/JNet_416_0_output_plane.png
[JNet_416_1_label_depth]: /experiments/images/JNet_416_1_label_depth.png
[JNet_416_1_label_plane]: /experiments/images/JNet_416_1_label_plane.png
[JNet_416_1_original_depth]: /experiments/images/JNet_416_1_original_depth.png
[JNet_416_1_original_plane]: /experiments/images/JNet_416_1_original_plane.png
[JNet_416_1_output_depth]: /experiments/images/JNet_416_1_output_depth.png
[JNet_416_1_output_plane]: /experiments/images/JNet_416_1_output_plane.png
[JNet_416_2_label_depth]: /experiments/images/JNet_416_2_label_depth.png
[JNet_416_2_label_plane]: /experiments/images/JNet_416_2_label_plane.png
[JNet_416_2_original_depth]: /experiments/images/JNet_416_2_original_depth.png
[JNet_416_2_original_plane]: /experiments/images/JNet_416_2_original_plane.png
[JNet_416_2_output_depth]: /experiments/images/JNet_416_2_output_depth.png
[JNet_416_2_output_plane]: /experiments/images/JNet_416_2_output_plane.png
[JNet_416_3_label_depth]: /experiments/images/JNet_416_3_label_depth.png
[JNet_416_3_label_plane]: /experiments/images/JNet_416_3_label_plane.png
[JNet_416_3_original_depth]: /experiments/images/JNet_416_3_original_depth.png
[JNet_416_3_original_plane]: /experiments/images/JNet_416_3_original_plane.png
[JNet_416_3_output_depth]: /experiments/images/JNet_416_3_output_depth.png
[JNet_416_3_output_plane]: /experiments/images/JNet_416_3_output_plane.png
[JNet_416_4_label_depth]: /experiments/images/JNet_416_4_label_depth.png
[JNet_416_4_label_plane]: /experiments/images/JNet_416_4_label_plane.png
[JNet_416_4_original_depth]: /experiments/images/JNet_416_4_original_depth.png
[JNet_416_4_original_plane]: /experiments/images/JNet_416_4_original_plane.png
[JNet_416_4_output_depth]: /experiments/images/JNet_416_4_output_depth.png
[JNet_416_4_output_plane]: /experiments/images/JNet_416_4_output_plane.png
[JNet_416_beads_001_roi000_original_depth]: /experiments/images/JNet_416_beads_001_roi000_original_depth.png
[JNet_416_beads_001_roi000_output_depth]: /experiments/images/JNet_416_beads_001_roi000_output_depth.png
[JNet_416_beads_001_roi000_reconst_depth]: /experiments/images/JNet_416_beads_001_roi000_reconst_depth.png
[JNet_416_beads_001_roi001_original_depth]: /experiments/images/JNet_416_beads_001_roi001_original_depth.png
[JNet_416_beads_001_roi001_output_depth]: /experiments/images/JNet_416_beads_001_roi001_output_depth.png
[JNet_416_beads_001_roi001_reconst_depth]: /experiments/images/JNet_416_beads_001_roi001_reconst_depth.png
[JNet_416_beads_001_roi002_original_depth]: /experiments/images/JNet_416_beads_001_roi002_original_depth.png
[JNet_416_beads_001_roi002_output_depth]: /experiments/images/JNet_416_beads_001_roi002_output_depth.png
[JNet_416_beads_001_roi002_reconst_depth]: /experiments/images/JNet_416_beads_001_roi002_reconst_depth.png
[JNet_416_beads_001_roi003_original_depth]: /experiments/images/JNet_416_beads_001_roi003_original_depth.png
[JNet_416_beads_001_roi003_output_depth]: /experiments/images/JNet_416_beads_001_roi003_output_depth.png
[JNet_416_beads_001_roi003_reconst_depth]: /experiments/images/JNet_416_beads_001_roi003_reconst_depth.png
[JNet_416_beads_001_roi004_original_depth]: /experiments/images/JNet_416_beads_001_roi004_original_depth.png
[JNet_416_beads_001_roi004_output_depth]: /experiments/images/JNet_416_beads_001_roi004_output_depth.png
[JNet_416_beads_001_roi004_reconst_depth]: /experiments/images/JNet_416_beads_001_roi004_reconst_depth.png
[JNet_416_beads_002_roi000_original_depth]: /experiments/images/JNet_416_beads_002_roi000_original_depth.png
[JNet_416_beads_002_roi000_output_depth]: /experiments/images/JNet_416_beads_002_roi000_output_depth.png
[JNet_416_beads_002_roi000_reconst_depth]: /experiments/images/JNet_416_beads_002_roi000_reconst_depth.png
[JNet_416_beads_002_roi001_original_depth]: /experiments/images/JNet_416_beads_002_roi001_original_depth.png
[JNet_416_beads_002_roi001_output_depth]: /experiments/images/JNet_416_beads_002_roi001_output_depth.png
[JNet_416_beads_002_roi001_reconst_depth]: /experiments/images/JNet_416_beads_002_roi001_reconst_depth.png
[JNet_416_beads_002_roi002_original_depth]: /experiments/images/JNet_416_beads_002_roi002_original_depth.png
[JNet_416_beads_002_roi002_output_depth]: /experiments/images/JNet_416_beads_002_roi002_output_depth.png
[JNet_416_beads_002_roi002_reconst_depth]: /experiments/images/JNet_416_beads_002_roi002_reconst_depth.png
[JNet_416_psf_post]: /experiments/images/JNet_416_psf_post.png
[JNet_416_psf_pre]: /experiments/images/JNet_416_psf_pre.png
[finetuned]: /experiments/tmp/JNet_416_train.png
[pretrained_model]: /experiments/tmp/JNet_415_pretrain_train.png
