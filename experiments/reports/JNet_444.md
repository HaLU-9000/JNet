



# JNet_444 Report
  
the parameters to replicate the results of JNet_445. vibrate, gaussian. NA=0.7, mu_z = 1.2, sig_z = 1.27  
pretrained model : JNet_442_pretrain
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
  
mean MSE: 0.02153816632926464, mean BCE: 0.0836089625954628
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_442_pretrain_0_original_plane]|![JNet_442_pretrain_0_output_plane]|![JNet_442_pretrain_0_label_plane]|
  
MSE: 0.018816262483596802, BCE: 0.07101569324731827  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_442_pretrain_0_original_depth]|![JNet_442_pretrain_0_output_depth]|![JNet_442_pretrain_0_label_depth]|
  
MSE: 0.018816262483596802, BCE: 0.07101569324731827  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_442_pretrain_1_original_plane]|![JNet_442_pretrain_1_output_plane]|![JNet_442_pretrain_1_label_plane]|
  
MSE: 0.0232292078435421, BCE: 0.08617259562015533  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_442_pretrain_1_original_depth]|![JNet_442_pretrain_1_output_depth]|![JNet_442_pretrain_1_label_depth]|
  
MSE: 0.0232292078435421, BCE: 0.08617259562015533  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_442_pretrain_2_original_plane]|![JNet_442_pretrain_2_output_plane]|![JNet_442_pretrain_2_label_plane]|
  
MSE: 0.0197904035449028, BCE: 0.0761687159538269  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_442_pretrain_2_original_depth]|![JNet_442_pretrain_2_output_depth]|![JNet_442_pretrain_2_label_depth]|
  
MSE: 0.0197904035449028, BCE: 0.0761687159538269  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_442_pretrain_3_original_plane]|![JNet_442_pretrain_3_output_plane]|![JNet_442_pretrain_3_label_plane]|
  
MSE: 0.023153061047196388, BCE: 0.08527017384767532  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_442_pretrain_3_original_depth]|![JNet_442_pretrain_3_output_depth]|![JNet_442_pretrain_3_label_depth]|
  
MSE: 0.023153061047196388, BCE: 0.08527017384767532  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_442_pretrain_4_original_plane]|![JNet_442_pretrain_4_output_plane]|![JNet_442_pretrain_4_label_plane]|
  
MSE: 0.022701894864439964, BCE: 0.09941764175891876  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_442_pretrain_4_original_depth]|![JNet_442_pretrain_4_output_depth]|![JNet_442_pretrain_4_label_depth]|
  
MSE: 0.022701894864439964, BCE: 0.09941764175891876  
  
mean MSE: 0.03360932320356369, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_444_0_original_plane]|![JNet_444_0_output_plane]|![JNet_444_0_label_plane]|
  
MSE: 0.04425397142767906, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_444_0_original_depth]|![JNet_444_0_output_depth]|![JNet_444_0_label_depth]|
  
MSE: 0.04425397142767906, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_444_1_original_plane]|![JNet_444_1_output_plane]|![JNet_444_1_label_plane]|
  
MSE: 0.03617434948682785, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_444_1_original_depth]|![JNet_444_1_output_depth]|![JNet_444_1_label_depth]|
  
MSE: 0.03617434948682785, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_444_2_original_plane]|![JNet_444_2_output_plane]|![JNet_444_2_label_plane]|
  
MSE: 0.03812522813677788, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_444_2_original_depth]|![JNet_444_2_output_depth]|![JNet_444_2_label_depth]|
  
MSE: 0.03812522813677788, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_444_3_original_plane]|![JNet_444_3_output_plane]|![JNet_444_3_label_plane]|
  
MSE: 0.027035269886255264, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_444_3_original_depth]|![JNet_444_3_output_depth]|![JNet_444_3_label_depth]|
  
MSE: 0.027035269886255264, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_444_4_original_plane]|![JNet_444_4_output_plane]|![JNet_444_4_label_plane]|
  
MSE: 0.022457802668213844, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_444_4_original_depth]|![JNet_444_4_output_depth]|![JNet_444_4_label_depth]|
  
MSE: 0.022457802668213844, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_442_pretrain_beads_001_roi000_original_depth]|![JNet_442_pretrain_beads_001_roi000_output_depth]|![JNet_442_pretrain_beads_001_roi000_reconst_depth]|![JNet_442_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 11.882860351562503, MSE: 0.0005124936578795314, quantized loss: 0.0016298257978633046  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_442_pretrain_beads_001_roi001_original_depth]|![JNet_442_pretrain_beads_001_roi001_output_depth]|![JNet_442_pretrain_beads_001_roi001_reconst_depth]|![JNet_442_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 16.840541015625003, MSE: 0.0011752358404919505, quantized loss: 0.001993205165490508  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_442_pretrain_beads_001_roi002_original_depth]|![JNet_442_pretrain_beads_001_roi002_output_depth]|![JNet_442_pretrain_beads_001_roi002_reconst_depth]|![JNet_442_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 10.688013671875003, MSE: 0.00043122199713252485, quantized loss: 0.0012505807681009173  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_442_pretrain_beads_001_roi003_original_depth]|![JNet_442_pretrain_beads_001_roi003_output_depth]|![JNet_442_pretrain_beads_001_roi003_reconst_depth]|![JNet_442_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 17.562056640625006, MSE: 0.0008420750382356346, quantized loss: 0.0021433245856314898  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_442_pretrain_beads_001_roi004_original_depth]|![JNet_442_pretrain_beads_001_roi004_output_depth]|![JNet_442_pretrain_beads_001_roi004_reconst_depth]|![JNet_442_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 11.633230468750003, MSE: 0.0005613792454823852, quantized loss: 0.001315408037044108  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_442_pretrain_beads_002_roi000_original_depth]|![JNet_442_pretrain_beads_002_roi000_output_depth]|![JNet_442_pretrain_beads_002_roi000_reconst_depth]|![JNet_442_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 12.550719726562503, MSE: 0.0006510337116196752, quantized loss: 0.0014215260744094849  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_442_pretrain_beads_002_roi001_original_depth]|![JNet_442_pretrain_beads_002_roi001_output_depth]|![JNet_442_pretrain_beads_002_roi001_reconst_depth]|![JNet_442_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 11.446823242187502, MSE: 0.00045054161455482244, quantized loss: 0.0013332206290215254  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_442_pretrain_beads_002_roi002_original_depth]|![JNet_442_pretrain_beads_002_roi002_output_depth]|![JNet_442_pretrain_beads_002_roi002_reconst_depth]|![JNet_442_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 11.947891601562503, MSE: 0.0005538011901080608, quantized loss: 0.0013866854133084416  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_444_beads_001_roi000_original_depth]|![JNet_444_beads_001_roi000_output_depth]|![JNet_444_beads_001_roi000_reconst_depth]|![JNet_444_beads_001_roi000_heatmap_depth]|
  
volume: 15.603808593750003, MSE: 0.00030055982642807066, quantized loss: 2.5477002054685727e-06  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_444_beads_001_roi001_original_depth]|![JNet_444_beads_001_roi001_output_depth]|![JNet_444_beads_001_roi001_reconst_depth]|![JNet_444_beads_001_roi001_heatmap_depth]|
  
volume: 23.934974609375004, MSE: 0.0008556490647606552, quantized loss: 3.823345650744159e-06  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_444_beads_001_roi002_original_depth]|![JNet_444_beads_001_roi002_output_depth]|![JNet_444_beads_001_roi002_reconst_depth]|![JNet_444_beads_001_roi002_heatmap_depth]|
  
volume: 15.239514648437504, MSE: 0.00022895469737704843, quantized loss: 2.16457988244656e-06  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_444_beads_001_roi003_original_depth]|![JNet_444_beads_001_roi003_output_depth]|![JNet_444_beads_001_roi003_reconst_depth]|![JNet_444_beads_001_roi003_heatmap_depth]|
  
volume: 24.045064453125004, MSE: 0.0007078968337737024, quantized loss: 2.8319459488557186e-06  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_444_beads_001_roi004_original_depth]|![JNet_444_beads_001_roi004_output_depth]|![JNet_444_beads_001_roi004_reconst_depth]|![JNet_444_beads_001_roi004_heatmap_depth]|
  
volume: 16.260728515625004, MSE: 0.00036114317481406033, quantized loss: 2.5880835892166942e-06  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_444_beads_002_roi000_original_depth]|![JNet_444_beads_002_roi000_output_depth]|![JNet_444_beads_002_roi000_reconst_depth]|![JNet_444_beads_002_roi000_heatmap_depth]|
  
volume: 17.313173828125006, MSE: 0.0004645214357879013, quantized loss: 2.399092863925034e-06  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_444_beads_002_roi001_original_depth]|![JNet_444_beads_002_roi001_output_depth]|![JNet_444_beads_002_roi001_reconst_depth]|![JNet_444_beads_002_roi001_heatmap_depth]|
  
volume: 16.158395507812504, MSE: 0.0003262779500801116, quantized loss: 2.5606643703213194e-06  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_444_beads_002_roi002_original_depth]|![JNet_444_beads_002_roi002_output_depth]|![JNet_444_beads_002_roi002_reconst_depth]|![JNet_444_beads_002_roi002_heatmap_depth]|
  
volume: 16.649904296875004, MSE: 0.00037061236798763275, quantized loss: 2.572636731201783e-06  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_444_psf_pre]|![JNet_444_psf_post]|

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
  



[JNet_442_pretrain_0_label_depth]: /experiments/images/JNet_442_pretrain_0_label_depth.png
[JNet_442_pretrain_0_label_plane]: /experiments/images/JNet_442_pretrain_0_label_plane.png
[JNet_442_pretrain_0_original_depth]: /experiments/images/JNet_442_pretrain_0_original_depth.png
[JNet_442_pretrain_0_original_plane]: /experiments/images/JNet_442_pretrain_0_original_plane.png
[JNet_442_pretrain_0_output_depth]: /experiments/images/JNet_442_pretrain_0_output_depth.png
[JNet_442_pretrain_0_output_plane]: /experiments/images/JNet_442_pretrain_0_output_plane.png
[JNet_442_pretrain_1_label_depth]: /experiments/images/JNet_442_pretrain_1_label_depth.png
[JNet_442_pretrain_1_label_plane]: /experiments/images/JNet_442_pretrain_1_label_plane.png
[JNet_442_pretrain_1_original_depth]: /experiments/images/JNet_442_pretrain_1_original_depth.png
[JNet_442_pretrain_1_original_plane]: /experiments/images/JNet_442_pretrain_1_original_plane.png
[JNet_442_pretrain_1_output_depth]: /experiments/images/JNet_442_pretrain_1_output_depth.png
[JNet_442_pretrain_1_output_plane]: /experiments/images/JNet_442_pretrain_1_output_plane.png
[JNet_442_pretrain_2_label_depth]: /experiments/images/JNet_442_pretrain_2_label_depth.png
[JNet_442_pretrain_2_label_plane]: /experiments/images/JNet_442_pretrain_2_label_plane.png
[JNet_442_pretrain_2_original_depth]: /experiments/images/JNet_442_pretrain_2_original_depth.png
[JNet_442_pretrain_2_original_plane]: /experiments/images/JNet_442_pretrain_2_original_plane.png
[JNet_442_pretrain_2_output_depth]: /experiments/images/JNet_442_pretrain_2_output_depth.png
[JNet_442_pretrain_2_output_plane]: /experiments/images/JNet_442_pretrain_2_output_plane.png
[JNet_442_pretrain_3_label_depth]: /experiments/images/JNet_442_pretrain_3_label_depth.png
[JNet_442_pretrain_3_label_plane]: /experiments/images/JNet_442_pretrain_3_label_plane.png
[JNet_442_pretrain_3_original_depth]: /experiments/images/JNet_442_pretrain_3_original_depth.png
[JNet_442_pretrain_3_original_plane]: /experiments/images/JNet_442_pretrain_3_original_plane.png
[JNet_442_pretrain_3_output_depth]: /experiments/images/JNet_442_pretrain_3_output_depth.png
[JNet_442_pretrain_3_output_plane]: /experiments/images/JNet_442_pretrain_3_output_plane.png
[JNet_442_pretrain_4_label_depth]: /experiments/images/JNet_442_pretrain_4_label_depth.png
[JNet_442_pretrain_4_label_plane]: /experiments/images/JNet_442_pretrain_4_label_plane.png
[JNet_442_pretrain_4_original_depth]: /experiments/images/JNet_442_pretrain_4_original_depth.png
[JNet_442_pretrain_4_original_plane]: /experiments/images/JNet_442_pretrain_4_original_plane.png
[JNet_442_pretrain_4_output_depth]: /experiments/images/JNet_442_pretrain_4_output_depth.png
[JNet_442_pretrain_4_output_plane]: /experiments/images/JNet_442_pretrain_4_output_plane.png
[JNet_442_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_442_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi000_original_depth.png
[JNet_442_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi000_output_depth.png
[JNet_442_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi000_reconst_depth.png
[JNet_442_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_442_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi001_original_depth.png
[JNet_442_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi001_output_depth.png
[JNet_442_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi001_reconst_depth.png
[JNet_442_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_442_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi002_original_depth.png
[JNet_442_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi002_output_depth.png
[JNet_442_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi002_reconst_depth.png
[JNet_442_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_442_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi003_original_depth.png
[JNet_442_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi003_output_depth.png
[JNet_442_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi003_reconst_depth.png
[JNet_442_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_442_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi004_original_depth.png
[JNet_442_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi004_output_depth.png
[JNet_442_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_442_pretrain_beads_001_roi004_reconst_depth.png
[JNet_442_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_442_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi000_original_depth.png
[JNet_442_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi000_output_depth.png
[JNet_442_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi000_reconst_depth.png
[JNet_442_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_442_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi001_original_depth.png
[JNet_442_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi001_output_depth.png
[JNet_442_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi001_reconst_depth.png
[JNet_442_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_442_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi002_original_depth.png
[JNet_442_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi002_output_depth.png
[JNet_442_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_442_pretrain_beads_002_roi002_reconst_depth.png
[JNet_444_0_label_depth]: /experiments/images/JNet_444_0_label_depth.png
[JNet_444_0_label_plane]: /experiments/images/JNet_444_0_label_plane.png
[JNet_444_0_original_depth]: /experiments/images/JNet_444_0_original_depth.png
[JNet_444_0_original_plane]: /experiments/images/JNet_444_0_original_plane.png
[JNet_444_0_output_depth]: /experiments/images/JNet_444_0_output_depth.png
[JNet_444_0_output_plane]: /experiments/images/JNet_444_0_output_plane.png
[JNet_444_1_label_depth]: /experiments/images/JNet_444_1_label_depth.png
[JNet_444_1_label_plane]: /experiments/images/JNet_444_1_label_plane.png
[JNet_444_1_original_depth]: /experiments/images/JNet_444_1_original_depth.png
[JNet_444_1_original_plane]: /experiments/images/JNet_444_1_original_plane.png
[JNet_444_1_output_depth]: /experiments/images/JNet_444_1_output_depth.png
[JNet_444_1_output_plane]: /experiments/images/JNet_444_1_output_plane.png
[JNet_444_2_label_depth]: /experiments/images/JNet_444_2_label_depth.png
[JNet_444_2_label_plane]: /experiments/images/JNet_444_2_label_plane.png
[JNet_444_2_original_depth]: /experiments/images/JNet_444_2_original_depth.png
[JNet_444_2_original_plane]: /experiments/images/JNet_444_2_original_plane.png
[JNet_444_2_output_depth]: /experiments/images/JNet_444_2_output_depth.png
[JNet_444_2_output_plane]: /experiments/images/JNet_444_2_output_plane.png
[JNet_444_3_label_depth]: /experiments/images/JNet_444_3_label_depth.png
[JNet_444_3_label_plane]: /experiments/images/JNet_444_3_label_plane.png
[JNet_444_3_original_depth]: /experiments/images/JNet_444_3_original_depth.png
[JNet_444_3_original_plane]: /experiments/images/JNet_444_3_original_plane.png
[JNet_444_3_output_depth]: /experiments/images/JNet_444_3_output_depth.png
[JNet_444_3_output_plane]: /experiments/images/JNet_444_3_output_plane.png
[JNet_444_4_label_depth]: /experiments/images/JNet_444_4_label_depth.png
[JNet_444_4_label_plane]: /experiments/images/JNet_444_4_label_plane.png
[JNet_444_4_original_depth]: /experiments/images/JNet_444_4_original_depth.png
[JNet_444_4_original_plane]: /experiments/images/JNet_444_4_original_plane.png
[JNet_444_4_output_depth]: /experiments/images/JNet_444_4_output_depth.png
[JNet_444_4_output_plane]: /experiments/images/JNet_444_4_output_plane.png
[JNet_444_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_444_beads_001_roi000_heatmap_depth.png
[JNet_444_beads_001_roi000_original_depth]: /experiments/images/JNet_444_beads_001_roi000_original_depth.png
[JNet_444_beads_001_roi000_output_depth]: /experiments/images/JNet_444_beads_001_roi000_output_depth.png
[JNet_444_beads_001_roi000_reconst_depth]: /experiments/images/JNet_444_beads_001_roi000_reconst_depth.png
[JNet_444_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_444_beads_001_roi001_heatmap_depth.png
[JNet_444_beads_001_roi001_original_depth]: /experiments/images/JNet_444_beads_001_roi001_original_depth.png
[JNet_444_beads_001_roi001_output_depth]: /experiments/images/JNet_444_beads_001_roi001_output_depth.png
[JNet_444_beads_001_roi001_reconst_depth]: /experiments/images/JNet_444_beads_001_roi001_reconst_depth.png
[JNet_444_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_444_beads_001_roi002_heatmap_depth.png
[JNet_444_beads_001_roi002_original_depth]: /experiments/images/JNet_444_beads_001_roi002_original_depth.png
[JNet_444_beads_001_roi002_output_depth]: /experiments/images/JNet_444_beads_001_roi002_output_depth.png
[JNet_444_beads_001_roi002_reconst_depth]: /experiments/images/JNet_444_beads_001_roi002_reconst_depth.png
[JNet_444_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_444_beads_001_roi003_heatmap_depth.png
[JNet_444_beads_001_roi003_original_depth]: /experiments/images/JNet_444_beads_001_roi003_original_depth.png
[JNet_444_beads_001_roi003_output_depth]: /experiments/images/JNet_444_beads_001_roi003_output_depth.png
[JNet_444_beads_001_roi003_reconst_depth]: /experiments/images/JNet_444_beads_001_roi003_reconst_depth.png
[JNet_444_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_444_beads_001_roi004_heatmap_depth.png
[JNet_444_beads_001_roi004_original_depth]: /experiments/images/JNet_444_beads_001_roi004_original_depth.png
[JNet_444_beads_001_roi004_output_depth]: /experiments/images/JNet_444_beads_001_roi004_output_depth.png
[JNet_444_beads_001_roi004_reconst_depth]: /experiments/images/JNet_444_beads_001_roi004_reconst_depth.png
[JNet_444_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_444_beads_002_roi000_heatmap_depth.png
[JNet_444_beads_002_roi000_original_depth]: /experiments/images/JNet_444_beads_002_roi000_original_depth.png
[JNet_444_beads_002_roi000_output_depth]: /experiments/images/JNet_444_beads_002_roi000_output_depth.png
[JNet_444_beads_002_roi000_reconst_depth]: /experiments/images/JNet_444_beads_002_roi000_reconst_depth.png
[JNet_444_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_444_beads_002_roi001_heatmap_depth.png
[JNet_444_beads_002_roi001_original_depth]: /experiments/images/JNet_444_beads_002_roi001_original_depth.png
[JNet_444_beads_002_roi001_output_depth]: /experiments/images/JNet_444_beads_002_roi001_output_depth.png
[JNet_444_beads_002_roi001_reconst_depth]: /experiments/images/JNet_444_beads_002_roi001_reconst_depth.png
[JNet_444_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_444_beads_002_roi002_heatmap_depth.png
[JNet_444_beads_002_roi002_original_depth]: /experiments/images/JNet_444_beads_002_roi002_original_depth.png
[JNet_444_beads_002_roi002_output_depth]: /experiments/images/JNet_444_beads_002_roi002_output_depth.png
[JNet_444_beads_002_roi002_reconst_depth]: /experiments/images/JNet_444_beads_002_roi002_reconst_depth.png
[JNet_444_psf_post]: /experiments/images/JNet_444_psf_post.png
[JNet_444_psf_pre]: /experiments/images/JNet_444_psf_pre.png
[finetuned]: /experiments/tmp/JNet_444_train.png
[pretrained_model]: /experiments/tmp/JNet_442_pretrain_train.png
