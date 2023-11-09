



# JNet_423 Report
  
the parameters to replicate the results of JNet_423. nearest interp of PSF, logit loss = 1.0, NA = 1.0 vq loss 1 psf upsampling by 5  
pretrained model : JNet_422_pretrain
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
|mu_z|0.3||
|sig_z|0.3||
|blur_mode|gibsonlanni|`gaussian` or `gibsonlanni`|
|size_x|51||
|size_y|51||
|size_z|201||
|NA|0.9||
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
|es_patience|10|
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
|es_patience|10|
|reconstruct|True|
|is_instantblur|False|
|is_vibrate|True|
|loss_weight|1|
|qloss_weight|1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.01897640898823738, mean BCE: 0.07054482400417328
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_422_pretrain_0_original_plane]|![JNet_422_pretrain_0_output_plane]|![JNet_422_pretrain_0_label_plane]|
  
MSE: 0.018091652542352676, BCE: 0.06512095034122467  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_422_pretrain_0_original_depth]|![JNet_422_pretrain_0_output_depth]|![JNet_422_pretrain_0_label_depth]|
  
MSE: 0.018091652542352676, BCE: 0.06512095034122467  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_422_pretrain_1_original_plane]|![JNet_422_pretrain_1_output_plane]|![JNet_422_pretrain_1_label_plane]|
  
MSE: 0.014536092057824135, BCE: 0.050080470740795135  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_422_pretrain_1_original_depth]|![JNet_422_pretrain_1_output_depth]|![JNet_422_pretrain_1_label_depth]|
  
MSE: 0.014536092057824135, BCE: 0.050080470740795135  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_422_pretrain_2_original_plane]|![JNet_422_pretrain_2_output_plane]|![JNet_422_pretrain_2_label_plane]|
  
MSE: 0.014665154740214348, BCE: 0.05297047644853592  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_422_pretrain_2_original_depth]|![JNet_422_pretrain_2_output_depth]|![JNet_422_pretrain_2_label_depth]|
  
MSE: 0.014665154740214348, BCE: 0.05297047644853592  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_422_pretrain_3_original_plane]|![JNet_422_pretrain_3_output_plane]|![JNet_422_pretrain_3_label_plane]|
  
MSE: 0.02549968659877777, BCE: 0.10032468289136887  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_422_pretrain_3_original_depth]|![JNet_422_pretrain_3_output_depth]|![JNet_422_pretrain_3_label_depth]|
  
MSE: 0.02549968659877777, BCE: 0.10032468289136887  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_422_pretrain_4_original_plane]|![JNet_422_pretrain_4_output_plane]|![JNet_422_pretrain_4_label_plane]|
  
MSE: 0.02208944782614708, BCE: 0.08422752469778061  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_422_pretrain_4_original_depth]|![JNet_422_pretrain_4_output_depth]|![JNet_422_pretrain_4_label_depth]|
  
MSE: 0.02208944782614708, BCE: 0.08422752469778061  
  
mean MSE: 0.03055829368531704, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_423_0_original_plane]|![JNet_423_0_output_plane]|![JNet_423_0_label_plane]|
  
MSE: 0.030235569924116135, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_423_0_original_depth]|![JNet_423_0_output_depth]|![JNet_423_0_label_depth]|
  
MSE: 0.030235569924116135, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_423_1_original_plane]|![JNet_423_1_output_plane]|![JNet_423_1_label_plane]|
  
MSE: 0.027779217809438705, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_423_1_original_depth]|![JNet_423_1_output_depth]|![JNet_423_1_label_depth]|
  
MSE: 0.027779217809438705, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_423_2_original_plane]|![JNet_423_2_output_plane]|![JNet_423_2_label_plane]|
  
MSE: 0.03692129999399185, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_423_2_original_depth]|![JNet_423_2_output_depth]|![JNet_423_2_label_depth]|
  
MSE: 0.03692129999399185, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_423_3_original_plane]|![JNet_423_3_output_plane]|![JNet_423_3_label_plane]|
  
MSE: 0.014670692384243011, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_423_3_original_depth]|![JNet_423_3_output_depth]|![JNet_423_3_label_depth]|
  
MSE: 0.014670692384243011, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_423_4_original_plane]|![JNet_423_4_output_plane]|![JNet_423_4_label_plane]|
  
MSE: 0.04318469762802124, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_423_4_original_depth]|![JNet_423_4_output_depth]|![JNet_423_4_label_depth]|
  
MSE: 0.04318469762802124, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_422_pretrain_beads_001_roi000_original_depth]|![JNet_422_pretrain_beads_001_roi000_output_depth]|![JNet_422_pretrain_beads_001_roi000_reconst_depth]|![JNet_422_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 15.926258789062503, MSE: 0.002453467808663845, quantized loss: 0.0017437051283195615  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_422_pretrain_beads_001_roi001_original_depth]|![JNet_422_pretrain_beads_001_roi001_output_depth]|![JNet_422_pretrain_beads_001_roi001_reconst_depth]|![JNet_422_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 24.164746093750004, MSE: 0.0033800334203988314, quantized loss: 0.0022753095254302025  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_422_pretrain_beads_001_roi002_original_depth]|![JNet_422_pretrain_beads_001_roi002_output_depth]|![JNet_422_pretrain_beads_001_roi002_reconst_depth]|![JNet_422_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 15.491249023437504, MSE: 0.0024060800205916166, quantized loss: 0.0013978165807202458  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_422_pretrain_beads_001_roi003_original_depth]|![JNet_422_pretrain_beads_001_roi003_output_depth]|![JNet_422_pretrain_beads_001_roi003_reconst_depth]|![JNet_422_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 24.665201171875005, MSE: 0.0032091746106743813, quantized loss: 0.002173936227336526  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_422_pretrain_beads_001_roi004_original_depth]|![JNet_422_pretrain_beads_001_roi004_output_depth]|![JNet_422_pretrain_beads_001_roi004_reconst_depth]|![JNet_422_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 16.222515625000003, MSE: 0.0023092441260814667, quantized loss: 0.0014048423618078232  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_422_pretrain_beads_002_roi000_original_depth]|![JNet_422_pretrain_beads_002_roi000_output_depth]|![JNet_422_pretrain_beads_002_roi000_reconst_depth]|![JNet_422_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 17.066839843750003, MSE: 0.002362616127356887, quantized loss: 0.001438973704352975  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_422_pretrain_beads_002_roi001_original_depth]|![JNet_422_pretrain_beads_002_roi001_output_depth]|![JNet_422_pretrain_beads_002_roi001_reconst_depth]|![JNet_422_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 16.446652343750003, MSE: 0.0025929086841642857, quantized loss: 0.001498921075835824  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_422_pretrain_beads_002_roi002_original_depth]|![JNet_422_pretrain_beads_002_roi002_output_depth]|![JNet_422_pretrain_beads_002_roi002_reconst_depth]|![JNet_422_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 16.600148437500003, MSE: 0.002428607316687703, quantized loss: 0.0014159672427922487  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_423_beads_001_roi000_original_depth]|![JNet_423_beads_001_roi000_output_depth]|![JNet_423_beads_001_roi000_reconst_depth]|![JNet_423_beads_001_roi000_heatmap_depth]|
  
volume: 8.825821289062501, MSE: 0.00016593259351793677, quantized loss: 4.407803317008074e-06  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_423_beads_001_roi001_original_depth]|![JNet_423_beads_001_roi001_output_depth]|![JNet_423_beads_001_roi001_reconst_depth]|![JNet_423_beads_001_roi001_heatmap_depth]|
  
volume: 14.069916015625003, MSE: 0.0005545264575630426, quantized loss: 7.739713510090951e-06  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_423_beads_001_roi002_original_depth]|![JNet_423_beads_001_roi002_output_depth]|![JNet_423_beads_001_roi002_reconst_depth]|![JNet_423_beads_001_roi002_heatmap_depth]|
  
volume: 8.917185546875002, MSE: 0.00010000148176914081, quantized loss: 5.157266969035845e-06  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_423_beads_001_roi003_original_depth]|![JNet_423_beads_001_roi003_output_depth]|![JNet_423_beads_001_roi003_reconst_depth]|![JNet_423_beads_001_roi003_heatmap_depth]|
  
volume: 15.131326171875003, MSE: 0.00031600959482602775, quantized loss: 8.334084668604191e-06  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_423_beads_001_roi004_original_depth]|![JNet_423_beads_001_roi004_output_depth]|![JNet_423_beads_001_roi004_reconst_depth]|![JNet_423_beads_001_roi004_heatmap_depth]|
  
volume: 9.857721679687502, MSE: 7.801004539942369e-05, quantized loss: 5.284471626509912e-06  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_423_beads_002_roi000_original_depth]|![JNet_423_beads_002_roi000_output_depth]|![JNet_423_beads_002_roi000_reconst_depth]|![JNet_423_beads_002_roi000_heatmap_depth]|
  
volume: 10.579934570312503, MSE: 7.82278148108162e-05, quantized loss: 4.380239261081442e-06  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_423_beads_002_roi001_original_depth]|![JNet_423_beads_002_roi001_output_depth]|![JNet_423_beads_002_roi001_reconst_depth]|![JNet_423_beads_002_roi001_heatmap_depth]|
  
volume: 9.532247070312502, MSE: 9.245416731573641e-05, quantized loss: 5.0033172556140926e-06  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_423_beads_002_roi002_original_depth]|![JNet_423_beads_002_roi002_output_depth]|![JNet_423_beads_002_roi002_reconst_depth]|![JNet_423_beads_002_roi002_heatmap_depth]|
  
volume: 9.981240234375003, MSE: 8.942047861637548e-05, quantized loss: 4.891350272373529e-06  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_423_psf_pre]|![JNet_423_psf_post]|

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
  



[JNet_422_pretrain_0_label_depth]: /experiments/images/JNet_422_pretrain_0_label_depth.png
[JNet_422_pretrain_0_label_plane]: /experiments/images/JNet_422_pretrain_0_label_plane.png
[JNet_422_pretrain_0_original_depth]: /experiments/images/JNet_422_pretrain_0_original_depth.png
[JNet_422_pretrain_0_original_plane]: /experiments/images/JNet_422_pretrain_0_original_plane.png
[JNet_422_pretrain_0_output_depth]: /experiments/images/JNet_422_pretrain_0_output_depth.png
[JNet_422_pretrain_0_output_plane]: /experiments/images/JNet_422_pretrain_0_output_plane.png
[JNet_422_pretrain_1_label_depth]: /experiments/images/JNet_422_pretrain_1_label_depth.png
[JNet_422_pretrain_1_label_plane]: /experiments/images/JNet_422_pretrain_1_label_plane.png
[JNet_422_pretrain_1_original_depth]: /experiments/images/JNet_422_pretrain_1_original_depth.png
[JNet_422_pretrain_1_original_plane]: /experiments/images/JNet_422_pretrain_1_original_plane.png
[JNet_422_pretrain_1_output_depth]: /experiments/images/JNet_422_pretrain_1_output_depth.png
[JNet_422_pretrain_1_output_plane]: /experiments/images/JNet_422_pretrain_1_output_plane.png
[JNet_422_pretrain_2_label_depth]: /experiments/images/JNet_422_pretrain_2_label_depth.png
[JNet_422_pretrain_2_label_plane]: /experiments/images/JNet_422_pretrain_2_label_plane.png
[JNet_422_pretrain_2_original_depth]: /experiments/images/JNet_422_pretrain_2_original_depth.png
[JNet_422_pretrain_2_original_plane]: /experiments/images/JNet_422_pretrain_2_original_plane.png
[JNet_422_pretrain_2_output_depth]: /experiments/images/JNet_422_pretrain_2_output_depth.png
[JNet_422_pretrain_2_output_plane]: /experiments/images/JNet_422_pretrain_2_output_plane.png
[JNet_422_pretrain_3_label_depth]: /experiments/images/JNet_422_pretrain_3_label_depth.png
[JNet_422_pretrain_3_label_plane]: /experiments/images/JNet_422_pretrain_3_label_plane.png
[JNet_422_pretrain_3_original_depth]: /experiments/images/JNet_422_pretrain_3_original_depth.png
[JNet_422_pretrain_3_original_plane]: /experiments/images/JNet_422_pretrain_3_original_plane.png
[JNet_422_pretrain_3_output_depth]: /experiments/images/JNet_422_pretrain_3_output_depth.png
[JNet_422_pretrain_3_output_plane]: /experiments/images/JNet_422_pretrain_3_output_plane.png
[JNet_422_pretrain_4_label_depth]: /experiments/images/JNet_422_pretrain_4_label_depth.png
[JNet_422_pretrain_4_label_plane]: /experiments/images/JNet_422_pretrain_4_label_plane.png
[JNet_422_pretrain_4_original_depth]: /experiments/images/JNet_422_pretrain_4_original_depth.png
[JNet_422_pretrain_4_original_plane]: /experiments/images/JNet_422_pretrain_4_original_plane.png
[JNet_422_pretrain_4_output_depth]: /experiments/images/JNet_422_pretrain_4_output_depth.png
[JNet_422_pretrain_4_output_plane]: /experiments/images/JNet_422_pretrain_4_output_plane.png
[JNet_422_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_422_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi000_original_depth.png
[JNet_422_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi000_output_depth.png
[JNet_422_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi000_reconst_depth.png
[JNet_422_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_422_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi001_original_depth.png
[JNet_422_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi001_output_depth.png
[JNet_422_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi001_reconst_depth.png
[JNet_422_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_422_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi002_original_depth.png
[JNet_422_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi002_output_depth.png
[JNet_422_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi002_reconst_depth.png
[JNet_422_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_422_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi003_original_depth.png
[JNet_422_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi003_output_depth.png
[JNet_422_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi003_reconst_depth.png
[JNet_422_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_422_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi004_original_depth.png
[JNet_422_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi004_output_depth.png
[JNet_422_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_422_pretrain_beads_001_roi004_reconst_depth.png
[JNet_422_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_422_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi000_original_depth.png
[JNet_422_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi000_output_depth.png
[JNet_422_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi000_reconst_depth.png
[JNet_422_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_422_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi001_original_depth.png
[JNet_422_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi001_output_depth.png
[JNet_422_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi001_reconst_depth.png
[JNet_422_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_422_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi002_original_depth.png
[JNet_422_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi002_output_depth.png
[JNet_422_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_422_pretrain_beads_002_roi002_reconst_depth.png
[JNet_423_0_label_depth]: /experiments/images/JNet_423_0_label_depth.png
[JNet_423_0_label_plane]: /experiments/images/JNet_423_0_label_plane.png
[JNet_423_0_original_depth]: /experiments/images/JNet_423_0_original_depth.png
[JNet_423_0_original_plane]: /experiments/images/JNet_423_0_original_plane.png
[JNet_423_0_output_depth]: /experiments/images/JNet_423_0_output_depth.png
[JNet_423_0_output_plane]: /experiments/images/JNet_423_0_output_plane.png
[JNet_423_1_label_depth]: /experiments/images/JNet_423_1_label_depth.png
[JNet_423_1_label_plane]: /experiments/images/JNet_423_1_label_plane.png
[JNet_423_1_original_depth]: /experiments/images/JNet_423_1_original_depth.png
[JNet_423_1_original_plane]: /experiments/images/JNet_423_1_original_plane.png
[JNet_423_1_output_depth]: /experiments/images/JNet_423_1_output_depth.png
[JNet_423_1_output_plane]: /experiments/images/JNet_423_1_output_plane.png
[JNet_423_2_label_depth]: /experiments/images/JNet_423_2_label_depth.png
[JNet_423_2_label_plane]: /experiments/images/JNet_423_2_label_plane.png
[JNet_423_2_original_depth]: /experiments/images/JNet_423_2_original_depth.png
[JNet_423_2_original_plane]: /experiments/images/JNet_423_2_original_plane.png
[JNet_423_2_output_depth]: /experiments/images/JNet_423_2_output_depth.png
[JNet_423_2_output_plane]: /experiments/images/JNet_423_2_output_plane.png
[JNet_423_3_label_depth]: /experiments/images/JNet_423_3_label_depth.png
[JNet_423_3_label_plane]: /experiments/images/JNet_423_3_label_plane.png
[JNet_423_3_original_depth]: /experiments/images/JNet_423_3_original_depth.png
[JNet_423_3_original_plane]: /experiments/images/JNet_423_3_original_plane.png
[JNet_423_3_output_depth]: /experiments/images/JNet_423_3_output_depth.png
[JNet_423_3_output_plane]: /experiments/images/JNet_423_3_output_plane.png
[JNet_423_4_label_depth]: /experiments/images/JNet_423_4_label_depth.png
[JNet_423_4_label_plane]: /experiments/images/JNet_423_4_label_plane.png
[JNet_423_4_original_depth]: /experiments/images/JNet_423_4_original_depth.png
[JNet_423_4_original_plane]: /experiments/images/JNet_423_4_original_plane.png
[JNet_423_4_output_depth]: /experiments/images/JNet_423_4_output_depth.png
[JNet_423_4_output_plane]: /experiments/images/JNet_423_4_output_plane.png
[JNet_423_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_423_beads_001_roi000_heatmap_depth.png
[JNet_423_beads_001_roi000_original_depth]: /experiments/images/JNet_423_beads_001_roi000_original_depth.png
[JNet_423_beads_001_roi000_output_depth]: /experiments/images/JNet_423_beads_001_roi000_output_depth.png
[JNet_423_beads_001_roi000_reconst_depth]: /experiments/images/JNet_423_beads_001_roi000_reconst_depth.png
[JNet_423_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_423_beads_001_roi001_heatmap_depth.png
[JNet_423_beads_001_roi001_original_depth]: /experiments/images/JNet_423_beads_001_roi001_original_depth.png
[JNet_423_beads_001_roi001_output_depth]: /experiments/images/JNet_423_beads_001_roi001_output_depth.png
[JNet_423_beads_001_roi001_reconst_depth]: /experiments/images/JNet_423_beads_001_roi001_reconst_depth.png
[JNet_423_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_423_beads_001_roi002_heatmap_depth.png
[JNet_423_beads_001_roi002_original_depth]: /experiments/images/JNet_423_beads_001_roi002_original_depth.png
[JNet_423_beads_001_roi002_output_depth]: /experiments/images/JNet_423_beads_001_roi002_output_depth.png
[JNet_423_beads_001_roi002_reconst_depth]: /experiments/images/JNet_423_beads_001_roi002_reconst_depth.png
[JNet_423_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_423_beads_001_roi003_heatmap_depth.png
[JNet_423_beads_001_roi003_original_depth]: /experiments/images/JNet_423_beads_001_roi003_original_depth.png
[JNet_423_beads_001_roi003_output_depth]: /experiments/images/JNet_423_beads_001_roi003_output_depth.png
[JNet_423_beads_001_roi003_reconst_depth]: /experiments/images/JNet_423_beads_001_roi003_reconst_depth.png
[JNet_423_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_423_beads_001_roi004_heatmap_depth.png
[JNet_423_beads_001_roi004_original_depth]: /experiments/images/JNet_423_beads_001_roi004_original_depth.png
[JNet_423_beads_001_roi004_output_depth]: /experiments/images/JNet_423_beads_001_roi004_output_depth.png
[JNet_423_beads_001_roi004_reconst_depth]: /experiments/images/JNet_423_beads_001_roi004_reconst_depth.png
[JNet_423_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_423_beads_002_roi000_heatmap_depth.png
[JNet_423_beads_002_roi000_original_depth]: /experiments/images/JNet_423_beads_002_roi000_original_depth.png
[JNet_423_beads_002_roi000_output_depth]: /experiments/images/JNet_423_beads_002_roi000_output_depth.png
[JNet_423_beads_002_roi000_reconst_depth]: /experiments/images/JNet_423_beads_002_roi000_reconst_depth.png
[JNet_423_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_423_beads_002_roi001_heatmap_depth.png
[JNet_423_beads_002_roi001_original_depth]: /experiments/images/JNet_423_beads_002_roi001_original_depth.png
[JNet_423_beads_002_roi001_output_depth]: /experiments/images/JNet_423_beads_002_roi001_output_depth.png
[JNet_423_beads_002_roi001_reconst_depth]: /experiments/images/JNet_423_beads_002_roi001_reconst_depth.png
[JNet_423_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_423_beads_002_roi002_heatmap_depth.png
[JNet_423_beads_002_roi002_original_depth]: /experiments/images/JNet_423_beads_002_roi002_original_depth.png
[JNet_423_beads_002_roi002_output_depth]: /experiments/images/JNet_423_beads_002_roi002_output_depth.png
[JNet_423_beads_002_roi002_reconst_depth]: /experiments/images/JNet_423_beads_002_roi002_reconst_depth.png
[JNet_423_psf_post]: /experiments/images/JNet_423_psf_post.png
[JNet_423_psf_pre]: /experiments/images/JNet_423_psf_pre.png
[finetuned]: /experiments/tmp/JNet_423_train.png
[pretrained_model]: /experiments/tmp/JNet_422_pretrain_train.png
