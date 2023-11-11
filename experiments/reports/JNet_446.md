



# JNet_446 Report
  
the parameters to replicate the results of JNet_446. no vibrate, NA=0.7, mu_z = 1.2, sig_z = 1.27  
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
  
mean MSE: 0.01575380750000477, mean BCE: 0.055797673761844635
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_0_original_plane]|![JNet_445_pretrain_0_output_plane]|![JNet_445_pretrain_0_label_plane]|
  
MSE: 0.016598962247371674, BCE: 0.05875512585043907  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_0_original_depth]|![JNet_445_pretrain_0_output_depth]|![JNet_445_pretrain_0_label_depth]|
  
MSE: 0.016598962247371674, BCE: 0.05875512585043907  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_1_original_plane]|![JNet_445_pretrain_1_output_plane]|![JNet_445_pretrain_1_label_plane]|
  
MSE: 0.016344817355275154, BCE: 0.05701267346739769  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_1_original_depth]|![JNet_445_pretrain_1_output_depth]|![JNet_445_pretrain_1_label_depth]|
  
MSE: 0.016344817355275154, BCE: 0.05701267346739769  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_2_original_plane]|![JNet_445_pretrain_2_output_plane]|![JNet_445_pretrain_2_label_plane]|
  
MSE: 0.014952465891838074, BCE: 0.05394945666193962  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_2_original_depth]|![JNet_445_pretrain_2_output_depth]|![JNet_445_pretrain_2_label_depth]|
  
MSE: 0.014952465891838074, BCE: 0.05394945666193962  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_3_original_plane]|![JNet_445_pretrain_3_output_plane]|![JNet_445_pretrain_3_label_plane]|
  
MSE: 0.014537470415234566, BCE: 0.05193939059972763  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_3_original_depth]|![JNet_445_pretrain_3_output_depth]|![JNet_445_pretrain_3_label_depth]|
  
MSE: 0.014537470415234566, BCE: 0.05193939059972763  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_4_original_plane]|![JNet_445_pretrain_4_output_plane]|![JNet_445_pretrain_4_label_plane]|
  
MSE: 0.016335321590304375, BCE: 0.057331714779138565  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_445_pretrain_4_original_depth]|![JNet_445_pretrain_4_output_depth]|![JNet_445_pretrain_4_label_depth]|
  
MSE: 0.016335321590304375, BCE: 0.057331714779138565  
  
mean MSE: 0.02530321478843689, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_446_0_original_plane]|![JNet_446_0_output_plane]|![JNet_446_0_label_plane]|
  
MSE: 0.022976448759436607, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_446_0_original_depth]|![JNet_446_0_output_depth]|![JNet_446_0_label_depth]|
  
MSE: 0.022976448759436607, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_446_1_original_plane]|![JNet_446_1_output_plane]|![JNet_446_1_label_plane]|
  
MSE: 0.024339813739061356, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_446_1_original_depth]|![JNet_446_1_output_depth]|![JNet_446_1_label_depth]|
  
MSE: 0.024339813739061356, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_446_2_original_plane]|![JNet_446_2_output_plane]|![JNet_446_2_label_plane]|
  
MSE: 0.02441423200070858, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_446_2_original_depth]|![JNet_446_2_output_depth]|![JNet_446_2_label_depth]|
  
MSE: 0.02441423200070858, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_446_3_original_plane]|![JNet_446_3_output_plane]|![JNet_446_3_label_plane]|
  
MSE: 0.028444040566682816, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_446_3_original_depth]|![JNet_446_3_output_depth]|![JNet_446_3_label_depth]|
  
MSE: 0.028444040566682816, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_446_4_original_plane]|![JNet_446_4_output_plane]|![JNet_446_4_label_plane]|
  
MSE: 0.02634153887629509, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_446_4_original_depth]|![JNet_446_4_output_depth]|![JNet_446_4_label_depth]|
  
MSE: 0.02634153887629509, BCE: nan  

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
|![JNet_446_beads_001_roi000_original_depth]|![JNet_446_beads_001_roi000_output_depth]|![JNet_446_beads_001_roi000_reconst_depth]|![JNet_446_beads_001_roi000_heatmap_depth]|
  
volume: 3.688173583984376, MSE: 0.00021716312039643526, quantized loss: 1.1860955964948516e-05  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_446_beads_001_roi001_original_depth]|![JNet_446_beads_001_roi001_output_depth]|![JNet_446_beads_001_roi001_reconst_depth]|![JNet_446_beads_001_roi001_heatmap_depth]|
  
volume: 6.155327148437501, MSE: 0.000631375762168318, quantized loss: 1.737072307150811e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_446_beads_001_roi002_original_depth]|![JNet_446_beads_001_roi002_output_depth]|![JNet_446_beads_001_roi002_reconst_depth]|![JNet_446_beads_001_roi002_heatmap_depth]|
  
volume: 3.7796208496093757, MSE: 0.00014963772264309227, quantized loss: 9.422121365787461e-06  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_446_beads_001_roi003_original_depth]|![JNet_446_beads_001_roi003_output_depth]|![JNet_446_beads_001_roi003_reconst_depth]|![JNet_446_beads_001_roi003_heatmap_depth]|
  
volume: 6.128815429687501, MSE: 0.0003158613108098507, quantized loss: 1.3265842426335439e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_446_beads_001_roi004_original_depth]|![JNet_446_beads_001_roi004_output_depth]|![JNet_446_beads_001_roi004_reconst_depth]|![JNet_446_beads_001_roi004_heatmap_depth]|
  
volume: 3.808742431640626, MSE: 0.00016407582734245807, quantized loss: 8.66918981046183e-06  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_446_beads_002_roi000_original_depth]|![JNet_446_beads_002_roi000_output_depth]|![JNet_446_beads_002_roi000_reconst_depth]|![JNet_446_beads_002_roi000_heatmap_depth]|
  
volume: 3.917823974609376, MSE: 0.00017097355157602578, quantized loss: 8.53056917549111e-06  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_446_beads_002_roi001_original_depth]|![JNet_446_beads_002_roi001_output_depth]|![JNet_446_beads_002_roi001_reconst_depth]|![JNet_446_beads_002_roi001_heatmap_depth]|
  
volume: 3.838820312500001, MSE: 0.00013452126586344093, quantized loss: 9.25658423511777e-06  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_446_beads_002_roi002_original_depth]|![JNet_446_beads_002_roi002_output_depth]|![JNet_446_beads_002_roi002_reconst_depth]|![JNet_446_beads_002_roi002_heatmap_depth]|
  
volume: 3.882716552734376, MSE: 0.0001533353206468746, quantized loss: 8.78594073583372e-06  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_446_psf_pre]|![JNet_446_psf_post]|

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
[JNet_446_0_label_depth]: /experiments/images/JNet_446_0_label_depth.png
[JNet_446_0_label_plane]: /experiments/images/JNet_446_0_label_plane.png
[JNet_446_0_original_depth]: /experiments/images/JNet_446_0_original_depth.png
[JNet_446_0_original_plane]: /experiments/images/JNet_446_0_original_plane.png
[JNet_446_0_output_depth]: /experiments/images/JNet_446_0_output_depth.png
[JNet_446_0_output_plane]: /experiments/images/JNet_446_0_output_plane.png
[JNet_446_1_label_depth]: /experiments/images/JNet_446_1_label_depth.png
[JNet_446_1_label_plane]: /experiments/images/JNet_446_1_label_plane.png
[JNet_446_1_original_depth]: /experiments/images/JNet_446_1_original_depth.png
[JNet_446_1_original_plane]: /experiments/images/JNet_446_1_original_plane.png
[JNet_446_1_output_depth]: /experiments/images/JNet_446_1_output_depth.png
[JNet_446_1_output_plane]: /experiments/images/JNet_446_1_output_plane.png
[JNet_446_2_label_depth]: /experiments/images/JNet_446_2_label_depth.png
[JNet_446_2_label_plane]: /experiments/images/JNet_446_2_label_plane.png
[JNet_446_2_original_depth]: /experiments/images/JNet_446_2_original_depth.png
[JNet_446_2_original_plane]: /experiments/images/JNet_446_2_original_plane.png
[JNet_446_2_output_depth]: /experiments/images/JNet_446_2_output_depth.png
[JNet_446_2_output_plane]: /experiments/images/JNet_446_2_output_plane.png
[JNet_446_3_label_depth]: /experiments/images/JNet_446_3_label_depth.png
[JNet_446_3_label_plane]: /experiments/images/JNet_446_3_label_plane.png
[JNet_446_3_original_depth]: /experiments/images/JNet_446_3_original_depth.png
[JNet_446_3_original_plane]: /experiments/images/JNet_446_3_original_plane.png
[JNet_446_3_output_depth]: /experiments/images/JNet_446_3_output_depth.png
[JNet_446_3_output_plane]: /experiments/images/JNet_446_3_output_plane.png
[JNet_446_4_label_depth]: /experiments/images/JNet_446_4_label_depth.png
[JNet_446_4_label_plane]: /experiments/images/JNet_446_4_label_plane.png
[JNet_446_4_original_depth]: /experiments/images/JNet_446_4_original_depth.png
[JNet_446_4_original_plane]: /experiments/images/JNet_446_4_original_plane.png
[JNet_446_4_output_depth]: /experiments/images/JNet_446_4_output_depth.png
[JNet_446_4_output_plane]: /experiments/images/JNet_446_4_output_plane.png
[JNet_446_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_446_beads_001_roi000_heatmap_depth.png
[JNet_446_beads_001_roi000_original_depth]: /experiments/images/JNet_446_beads_001_roi000_original_depth.png
[JNet_446_beads_001_roi000_output_depth]: /experiments/images/JNet_446_beads_001_roi000_output_depth.png
[JNet_446_beads_001_roi000_reconst_depth]: /experiments/images/JNet_446_beads_001_roi000_reconst_depth.png
[JNet_446_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_446_beads_001_roi001_heatmap_depth.png
[JNet_446_beads_001_roi001_original_depth]: /experiments/images/JNet_446_beads_001_roi001_original_depth.png
[JNet_446_beads_001_roi001_output_depth]: /experiments/images/JNet_446_beads_001_roi001_output_depth.png
[JNet_446_beads_001_roi001_reconst_depth]: /experiments/images/JNet_446_beads_001_roi001_reconst_depth.png
[JNet_446_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_446_beads_001_roi002_heatmap_depth.png
[JNet_446_beads_001_roi002_original_depth]: /experiments/images/JNet_446_beads_001_roi002_original_depth.png
[JNet_446_beads_001_roi002_output_depth]: /experiments/images/JNet_446_beads_001_roi002_output_depth.png
[JNet_446_beads_001_roi002_reconst_depth]: /experiments/images/JNet_446_beads_001_roi002_reconst_depth.png
[JNet_446_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_446_beads_001_roi003_heatmap_depth.png
[JNet_446_beads_001_roi003_original_depth]: /experiments/images/JNet_446_beads_001_roi003_original_depth.png
[JNet_446_beads_001_roi003_output_depth]: /experiments/images/JNet_446_beads_001_roi003_output_depth.png
[JNet_446_beads_001_roi003_reconst_depth]: /experiments/images/JNet_446_beads_001_roi003_reconst_depth.png
[JNet_446_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_446_beads_001_roi004_heatmap_depth.png
[JNet_446_beads_001_roi004_original_depth]: /experiments/images/JNet_446_beads_001_roi004_original_depth.png
[JNet_446_beads_001_roi004_output_depth]: /experiments/images/JNet_446_beads_001_roi004_output_depth.png
[JNet_446_beads_001_roi004_reconst_depth]: /experiments/images/JNet_446_beads_001_roi004_reconst_depth.png
[JNet_446_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_446_beads_002_roi000_heatmap_depth.png
[JNet_446_beads_002_roi000_original_depth]: /experiments/images/JNet_446_beads_002_roi000_original_depth.png
[JNet_446_beads_002_roi000_output_depth]: /experiments/images/JNet_446_beads_002_roi000_output_depth.png
[JNet_446_beads_002_roi000_reconst_depth]: /experiments/images/JNet_446_beads_002_roi000_reconst_depth.png
[JNet_446_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_446_beads_002_roi001_heatmap_depth.png
[JNet_446_beads_002_roi001_original_depth]: /experiments/images/JNet_446_beads_002_roi001_original_depth.png
[JNet_446_beads_002_roi001_output_depth]: /experiments/images/JNet_446_beads_002_roi001_output_depth.png
[JNet_446_beads_002_roi001_reconst_depth]: /experiments/images/JNet_446_beads_002_roi001_reconst_depth.png
[JNet_446_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_446_beads_002_roi002_heatmap_depth.png
[JNet_446_beads_002_roi002_original_depth]: /experiments/images/JNet_446_beads_002_roi002_original_depth.png
[JNet_446_beads_002_roi002_output_depth]: /experiments/images/JNet_446_beads_002_roi002_output_depth.png
[JNet_446_beads_002_roi002_reconst_depth]: /experiments/images/JNet_446_beads_002_roi002_reconst_depth.png
[JNet_446_psf_post]: /experiments/images/JNet_446_psf_post.png
[JNet_446_psf_pre]: /experiments/images/JNet_446_psf_pre.png
[finetuned]: /experiments/tmp/JNet_446_train.png
[pretrained_model]: /experiments/tmp/JNet_445_pretrain_train.png
