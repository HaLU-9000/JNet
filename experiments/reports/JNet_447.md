



# JNet_447 Report
  
the parameters to replicate the results of JNet_447. no vibrate, NA=0.7, mu_z = 1.2, sig_z = 1.27  
pretrained model : JNet_445_pretrain
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
|mu_z|1.2||
|sig_z|0.3||
|blur_mode|gaussian|`gaussian` or `gibsonlanni`|
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
|bet_z|5.0||
|bet_xy|10.0||
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
|is_vibrate|False|
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
  
mean MSE: 0.017077622935175896, mean BCE: 0.05996549874544144
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_0_original_plane]|![JNet_445_pretrain_0_output_plane]|![JNet_445_pretrain_0_label_plane]|
  
MSE: 0.018719740211963654, BCE: 0.06555832177400589  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_0_original_depth]|![JNet_445_pretrain_0_output_depth]|![JNet_445_pretrain_0_label_depth]|
  
MSE: 0.018719740211963654, BCE: 0.06555832177400589  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_1_original_plane]|![JNet_445_pretrain_1_output_plane]|![JNet_445_pretrain_1_label_plane]|
  
MSE: 0.018428780138492584, BCE: 0.06466378271579742  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_1_original_depth]|![JNet_445_pretrain_1_output_depth]|![JNet_445_pretrain_1_label_depth]|
  
MSE: 0.018428780138492584, BCE: 0.06466378271579742  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_2_original_plane]|![JNet_445_pretrain_2_output_plane]|![JNet_445_pretrain_2_label_plane]|
  
MSE: 0.014938317239284515, BCE: 0.052801601588726044  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_2_original_depth]|![JNet_445_pretrain_2_output_depth]|![JNet_445_pretrain_2_label_depth]|
  
MSE: 0.014938317239284515, BCE: 0.052801601588726044  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_3_original_plane]|![JNet_445_pretrain_3_output_plane]|![JNet_445_pretrain_3_label_plane]|
  
MSE: 0.01876642182469368, BCE: 0.06482703983783722  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_3_original_depth]|![JNet_445_pretrain_3_output_depth]|![JNet_445_pretrain_3_label_depth]|
  
MSE: 0.01876642182469368, BCE: 0.06482703983783722  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_4_original_plane]|![JNet_445_pretrain_4_output_plane]|![JNet_445_pretrain_4_label_plane]|
  
MSE: 0.014534851536154747, BCE: 0.051976725459098816  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_4_original_depth]|![JNet_445_pretrain_4_output_depth]|![JNet_445_pretrain_4_label_depth]|
  
MSE: 0.014534851536154747, BCE: 0.051976725459098816  
  
mean MSE: 0.038791242986917496, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_447_0_original_plane]|![JNet_447_0_output_plane]|![JNet_447_0_label_plane]|
  
MSE: 0.033923257142305374, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_447_0_original_depth]|![JNet_447_0_output_depth]|![JNet_447_0_label_depth]|
  
MSE: 0.033923257142305374, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_447_1_original_plane]|![JNet_447_1_output_plane]|![JNet_447_1_label_plane]|
  
MSE: 0.039241477847099304, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_447_1_original_depth]|![JNet_447_1_output_depth]|![JNet_447_1_label_depth]|
  
MSE: 0.039241477847099304, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_447_2_original_plane]|![JNet_447_2_output_plane]|![JNet_447_2_label_plane]|
  
MSE: 0.047783467918634415, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_447_2_original_depth]|![JNet_447_2_output_depth]|![JNet_447_2_label_depth]|
  
MSE: 0.047783467918634415, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_447_3_original_plane]|![JNet_447_3_output_plane]|![JNet_447_3_label_plane]|
  
MSE: 0.03722430393099785, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_447_3_original_depth]|![JNet_447_3_output_depth]|![JNet_447_3_label_depth]|
  
MSE: 0.03722430393099785, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_447_4_original_plane]|![JNet_447_4_output_plane]|![JNet_447_4_label_plane]|
  
MSE: 0.03578370809555054, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_447_4_original_depth]|![JNet_447_4_output_depth]|![JNet_447_4_label_depth]|
  
MSE: 0.03578370809555054, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_445_pretrain_beads_001_roi000_original_depth]|![JNet_445_pretrain_beads_001_roi000_output_depth]|![JNet_445_pretrain_beads_001_roi000_reconst_depth]|![JNet_445_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 9.789972656250002, MSE: 0.00044052605517208576, quantized loss: 0.0014370380667969584  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_445_pretrain_beads_001_roi001_original_depth]|![JNet_445_pretrain_beads_001_roi001_output_depth]|![JNet_445_pretrain_beads_001_roi001_reconst_depth]|![JNet_445_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 14.213304687500003, MSE: 0.0009615654707886279, quantized loss: 0.0017466736026108265  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_445_pretrain_beads_001_roi002_original_depth]|![JNet_445_pretrain_beads_001_roi002_output_depth]|![JNet_445_pretrain_beads_001_roi002_reconst_depth]|![JNet_445_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 8.804541015625002, MSE: 0.00037771163624711335, quantized loss: 0.0010129434522241354  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_445_pretrain_beads_001_roi003_original_depth]|![JNet_445_pretrain_beads_001_roi003_output_depth]|![JNet_445_pretrain_beads_001_roi003_reconst_depth]|![JNet_445_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 14.677488281250003, MSE: 0.000725846563000232, quantized loss: 0.0019496731692925096  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_445_pretrain_beads_001_roi004_original_depth]|![JNet_445_pretrain_beads_001_roi004_output_depth]|![JNet_445_pretrain_beads_001_roi004_reconst_depth]|![JNet_445_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 9.205800781250002, MSE: 0.0004384561616461724, quantized loss: 0.0009677806519903243  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_445_pretrain_beads_002_roi000_original_depth]|![JNet_445_pretrain_beads_002_roi000_output_depth]|![JNet_445_pretrain_beads_002_roi000_reconst_depth]|![JNet_445_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 9.803613281250003, MSE: 0.0005221799365244806, quantized loss: 0.0011422712123021483  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_445_pretrain_beads_002_roi001_original_depth]|![JNet_445_pretrain_beads_002_roi001_output_depth]|![JNet_445_pretrain_beads_002_roi001_reconst_depth]|![JNet_445_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 9.459628906250002, MSE: 0.0003708935109898448, quantized loss: 0.0011202795431017876  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_445_pretrain_beads_002_roi002_original_depth]|![JNet_445_pretrain_beads_002_roi002_output_depth]|![JNet_445_pretrain_beads_002_roi002_reconst_depth]|![JNet_445_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 9.483640625000001, MSE: 0.00043759451364167035, quantized loss: 0.0011047007283195853  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_447_beads_001_roi000_original_depth]|![JNet_447_beads_001_roi000_output_depth]|![JNet_447_beads_001_roi000_reconst_depth]|![JNet_447_beads_001_roi000_heatmap_depth]|
  
volume: 7.295301757812502, MSE: 0.00016563817916903645, quantized loss: 4.580646418617107e-06  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_447_beads_001_roi001_original_depth]|![JNet_447_beads_001_roi001_output_depth]|![JNet_447_beads_001_roi001_reconst_depth]|![JNet_447_beads_001_roi001_heatmap_depth]|
  
volume: 11.149063476562503, MSE: 0.0008989798370748758, quantized loss: 7.1902095442055725e-06  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_447_beads_001_roi002_original_depth]|![JNet_447_beads_001_roi002_output_depth]|![JNet_447_beads_001_roi002_reconst_depth]|![JNet_447_beads_001_roi002_heatmap_depth]|
  
volume: 7.493996093750002, MSE: 0.00013779803703073412, quantized loss: 5.159653937880648e-06  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_447_beads_001_roi003_original_depth]|![JNet_447_beads_001_roi003_output_depth]|![JNet_447_beads_001_roi003_reconst_depth]|![JNet_447_beads_001_roi003_heatmap_depth]|
  
volume: 12.305451171875003, MSE: 0.00044768527732230723, quantized loss: 7.016747986199334e-06  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_447_beads_001_roi004_original_depth]|![JNet_447_beads_001_roi004_output_depth]|![JNet_447_beads_001_roi004_reconst_depth]|![JNet_447_beads_001_roi004_heatmap_depth]|
  
volume: 7.917214843750002, MSE: 0.00017400736396666616, quantized loss: 4.38026881965925e-06  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_447_beads_002_roi000_original_depth]|![JNet_447_beads_002_roi000_output_depth]|![JNet_447_beads_002_roi000_reconst_depth]|![JNet_447_beads_002_roi000_heatmap_depth]|
  
volume: 8.356400390625002, MSE: 0.0001936279732035473, quantized loss: 4.1874100134009495e-06  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_447_beads_002_roi001_original_depth]|![JNet_447_beads_002_roi001_output_depth]|![JNet_447_beads_002_roi001_reconst_depth]|![JNet_447_beads_002_roi001_heatmap_depth]|
  
volume: 7.8944995117187515, MSE: 0.00014587062469217926, quantized loss: 3.9005853977869265e-06  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_447_beads_002_roi002_original_depth]|![JNet_447_beads_002_roi002_output_depth]|![JNet_447_beads_002_roi002_reconst_depth]|![JNet_447_beads_002_roi002_heatmap_depth]|
  
volume: 8.036529785156253, MSE: 0.00016768602654337883, quantized loss: 3.970142188336467e-06  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_447_psf_pre]|![JNet_447_psf_post]|

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
  



[JNet_445_pretrain_0_label_depth]: /experiments/images/JNet_445_pretrain_0_label_depth.png
[JNet_445_pretrain_0_label_plane]: /experiments/images/JNet_445_pretrain_0_label_plane.png
[JNet_445_pretrain_0_original_depth]: /experiments/images/JNet_445_pretrain_0_original_depth.png
[JNet_445_pretrain_0_original_plane]: /experiments/images/JNet_445_pretrain_0_original_plane.png
[JNet_445_pretrain_0_output_depth]: /experiments/images/JNet_445_pretrain_0_output_depth.png
[JNet_445_pretrain_0_output_plane]: /experiments/images/JNet_445_pretrain_0_output_plane.png
[JNet_445_pretrain_1_label_depth]: /experiments/images/JNet_445_pretrain_1_label_depth.png
[JNet_445_pretrain_1_label_plane]: /experiments/images/JNet_445_pretrain_1_label_plane.png
[JNet_445_pretrain_1_original_depth]: /experiments/images/JNet_445_pretrain_1_original_depth.png
[JNet_445_pretrain_1_original_plane]: /experiments/images/JNet_445_pretrain_1_original_plane.png
[JNet_445_pretrain_1_output_depth]: /experiments/images/JNet_445_pretrain_1_output_depth.png
[JNet_445_pretrain_1_output_plane]: /experiments/images/JNet_445_pretrain_1_output_plane.png
[JNet_445_pretrain_2_label_depth]: /experiments/images/JNet_445_pretrain_2_label_depth.png
[JNet_445_pretrain_2_label_plane]: /experiments/images/JNet_445_pretrain_2_label_plane.png
[JNet_445_pretrain_2_original_depth]: /experiments/images/JNet_445_pretrain_2_original_depth.png
[JNet_445_pretrain_2_original_plane]: /experiments/images/JNet_445_pretrain_2_original_plane.png
[JNet_445_pretrain_2_output_depth]: /experiments/images/JNet_445_pretrain_2_output_depth.png
[JNet_445_pretrain_2_output_plane]: /experiments/images/JNet_445_pretrain_2_output_plane.png
[JNet_445_pretrain_3_label_depth]: /experiments/images/JNet_445_pretrain_3_label_depth.png
[JNet_445_pretrain_3_label_plane]: /experiments/images/JNet_445_pretrain_3_label_plane.png
[JNet_445_pretrain_3_original_depth]: /experiments/images/JNet_445_pretrain_3_original_depth.png
[JNet_445_pretrain_3_original_plane]: /experiments/images/JNet_445_pretrain_3_original_plane.png
[JNet_445_pretrain_3_output_depth]: /experiments/images/JNet_445_pretrain_3_output_depth.png
[JNet_445_pretrain_3_output_plane]: /experiments/images/JNet_445_pretrain_3_output_plane.png
[JNet_445_pretrain_4_label_depth]: /experiments/images/JNet_445_pretrain_4_label_depth.png
[JNet_445_pretrain_4_label_plane]: /experiments/images/JNet_445_pretrain_4_label_plane.png
[JNet_445_pretrain_4_original_depth]: /experiments/images/JNet_445_pretrain_4_original_depth.png
[JNet_445_pretrain_4_original_plane]: /experiments/images/JNet_445_pretrain_4_original_plane.png
[JNet_445_pretrain_4_output_depth]: /experiments/images/JNet_445_pretrain_4_output_depth.png
[JNet_445_pretrain_4_output_plane]: /experiments/images/JNet_445_pretrain_4_output_plane.png
[JNet_445_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_445_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi000_original_depth.png
[JNet_445_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi000_output_depth.png
[JNet_445_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi000_reconst_depth.png
[JNet_445_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_445_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi001_original_depth.png
[JNet_445_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi001_output_depth.png
[JNet_445_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi001_reconst_depth.png
[JNet_445_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_445_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi002_original_depth.png
[JNet_445_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi002_output_depth.png
[JNet_445_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi002_reconst_depth.png
[JNet_445_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_445_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi003_original_depth.png
[JNet_445_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi003_output_depth.png
[JNet_445_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi003_reconst_depth.png
[JNet_445_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_445_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi004_original_depth.png
[JNet_445_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi004_output_depth.png
[JNet_445_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_445_pretrain_beads_001_roi004_reconst_depth.png
[JNet_445_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_445_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi000_original_depth.png
[JNet_445_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi000_output_depth.png
[JNet_445_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi000_reconst_depth.png
[JNet_445_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_445_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi001_original_depth.png
[JNet_445_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi001_output_depth.png
[JNet_445_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi001_reconst_depth.png
[JNet_445_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_445_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi002_original_depth.png
[JNet_445_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi002_output_depth.png
[JNet_445_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_445_pretrain_beads_002_roi002_reconst_depth.png
[JNet_447_0_label_depth]: /experiments/images/JNet_447_0_label_depth.png
[JNet_447_0_label_plane]: /experiments/images/JNet_447_0_label_plane.png
[JNet_447_0_original_depth]: /experiments/images/JNet_447_0_original_depth.png
[JNet_447_0_original_plane]: /experiments/images/JNet_447_0_original_plane.png
[JNet_447_0_output_depth]: /experiments/images/JNet_447_0_output_depth.png
[JNet_447_0_output_plane]: /experiments/images/JNet_447_0_output_plane.png
[JNet_447_1_label_depth]: /experiments/images/JNet_447_1_label_depth.png
[JNet_447_1_label_plane]: /experiments/images/JNet_447_1_label_plane.png
[JNet_447_1_original_depth]: /experiments/images/JNet_447_1_original_depth.png
[JNet_447_1_original_plane]: /experiments/images/JNet_447_1_original_plane.png
[JNet_447_1_output_depth]: /experiments/images/JNet_447_1_output_depth.png
[JNet_447_1_output_plane]: /experiments/images/JNet_447_1_output_plane.png
[JNet_447_2_label_depth]: /experiments/images/JNet_447_2_label_depth.png
[JNet_447_2_label_plane]: /experiments/images/JNet_447_2_label_plane.png
[JNet_447_2_original_depth]: /experiments/images/JNet_447_2_original_depth.png
[JNet_447_2_original_plane]: /experiments/images/JNet_447_2_original_plane.png
[JNet_447_2_output_depth]: /experiments/images/JNet_447_2_output_depth.png
[JNet_447_2_output_plane]: /experiments/images/JNet_447_2_output_plane.png
[JNet_447_3_label_depth]: /experiments/images/JNet_447_3_label_depth.png
[JNet_447_3_label_plane]: /experiments/images/JNet_447_3_label_plane.png
[JNet_447_3_original_depth]: /experiments/images/JNet_447_3_original_depth.png
[JNet_447_3_original_plane]: /experiments/images/JNet_447_3_original_plane.png
[JNet_447_3_output_depth]: /experiments/images/JNet_447_3_output_depth.png
[JNet_447_3_output_plane]: /experiments/images/JNet_447_3_output_plane.png
[JNet_447_4_label_depth]: /experiments/images/JNet_447_4_label_depth.png
[JNet_447_4_label_plane]: /experiments/images/JNet_447_4_label_plane.png
[JNet_447_4_original_depth]: /experiments/images/JNet_447_4_original_depth.png
[JNet_447_4_original_plane]: /experiments/images/JNet_447_4_original_plane.png
[JNet_447_4_output_depth]: /experiments/images/JNet_447_4_output_depth.png
[JNet_447_4_output_plane]: /experiments/images/JNet_447_4_output_plane.png
[JNet_447_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_447_beads_001_roi000_heatmap_depth.png
[JNet_447_beads_001_roi000_original_depth]: /experiments/images/JNet_447_beads_001_roi000_original_depth.png
[JNet_447_beads_001_roi000_output_depth]: /experiments/images/JNet_447_beads_001_roi000_output_depth.png
[JNet_447_beads_001_roi000_reconst_depth]: /experiments/images/JNet_447_beads_001_roi000_reconst_depth.png
[JNet_447_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_447_beads_001_roi001_heatmap_depth.png
[JNet_447_beads_001_roi001_original_depth]: /experiments/images/JNet_447_beads_001_roi001_original_depth.png
[JNet_447_beads_001_roi001_output_depth]: /experiments/images/JNet_447_beads_001_roi001_output_depth.png
[JNet_447_beads_001_roi001_reconst_depth]: /experiments/images/JNet_447_beads_001_roi001_reconst_depth.png
[JNet_447_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_447_beads_001_roi002_heatmap_depth.png
[JNet_447_beads_001_roi002_original_depth]: /experiments/images/JNet_447_beads_001_roi002_original_depth.png
[JNet_447_beads_001_roi002_output_depth]: /experiments/images/JNet_447_beads_001_roi002_output_depth.png
[JNet_447_beads_001_roi002_reconst_depth]: /experiments/images/JNet_447_beads_001_roi002_reconst_depth.png
[JNet_447_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_447_beads_001_roi003_heatmap_depth.png
[JNet_447_beads_001_roi003_original_depth]: /experiments/images/JNet_447_beads_001_roi003_original_depth.png
[JNet_447_beads_001_roi003_output_depth]: /experiments/images/JNet_447_beads_001_roi003_output_depth.png
[JNet_447_beads_001_roi003_reconst_depth]: /experiments/images/JNet_447_beads_001_roi003_reconst_depth.png
[JNet_447_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_447_beads_001_roi004_heatmap_depth.png
[JNet_447_beads_001_roi004_original_depth]: /experiments/images/JNet_447_beads_001_roi004_original_depth.png
[JNet_447_beads_001_roi004_output_depth]: /experiments/images/JNet_447_beads_001_roi004_output_depth.png
[JNet_447_beads_001_roi004_reconst_depth]: /experiments/images/JNet_447_beads_001_roi004_reconst_depth.png
[JNet_447_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_447_beads_002_roi000_heatmap_depth.png
[JNet_447_beads_002_roi000_original_depth]: /experiments/images/JNet_447_beads_002_roi000_original_depth.png
[JNet_447_beads_002_roi000_output_depth]: /experiments/images/JNet_447_beads_002_roi000_output_depth.png
[JNet_447_beads_002_roi000_reconst_depth]: /experiments/images/JNet_447_beads_002_roi000_reconst_depth.png
[JNet_447_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_447_beads_002_roi001_heatmap_depth.png
[JNet_447_beads_002_roi001_original_depth]: /experiments/images/JNet_447_beads_002_roi001_original_depth.png
[JNet_447_beads_002_roi001_output_depth]: /experiments/images/JNet_447_beads_002_roi001_output_depth.png
[JNet_447_beads_002_roi001_reconst_depth]: /experiments/images/JNet_447_beads_002_roi001_reconst_depth.png
[JNet_447_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_447_beads_002_roi002_heatmap_depth.png
[JNet_447_beads_002_roi002_original_depth]: /experiments/images/JNet_447_beads_002_roi002_original_depth.png
[JNet_447_beads_002_roi002_output_depth]: /experiments/images/JNet_447_beads_002_roi002_output_depth.png
[JNet_447_beads_002_roi002_reconst_depth]: /experiments/images/JNet_447_beads_002_roi002_reconst_depth.png
[JNet_447_psf_post]: /experiments/images/JNet_447_psf_post.png
[JNet_447_psf_pre]: /experiments/images/JNet_447_psf_pre.png
[finetuned]: /experiments/tmp/JNet_447_train.png
[pretrained_model]: /experiments/tmp/JNet_445_pretrain_train.png
