



# JNet_435 Report
  
the parameters to replicate the results of JNet_427. nearest interp of PSF, NA=0.7, mu_z = 0.3, sig_z = 1.27  
pretrained model : JNet_434_pretrain
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
|sig_z|1.27||
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
  
mean MSE: 0.024101920425891876, mean BCE: 0.08591405302286148
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_434_pretrain_0_original_plane]|![JNet_434_pretrain_0_output_plane]|![JNet_434_pretrain_0_label_plane]|
  
MSE: 0.024383295327425003, BCE: 0.08849699050188065  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_434_pretrain_0_original_depth]|![JNet_434_pretrain_0_output_depth]|![JNet_434_pretrain_0_label_depth]|
  
MSE: 0.024383295327425003, BCE: 0.08849699050188065  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_434_pretrain_1_original_plane]|![JNet_434_pretrain_1_output_plane]|![JNet_434_pretrain_1_label_plane]|
  
MSE: 0.02626889757812023, BCE: 0.092821404337883  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_434_pretrain_1_original_depth]|![JNet_434_pretrain_1_output_depth]|![JNet_434_pretrain_1_label_depth]|
  
MSE: 0.02626889757812023, BCE: 0.092821404337883  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_434_pretrain_2_original_plane]|![JNet_434_pretrain_2_output_plane]|![JNet_434_pretrain_2_label_plane]|
  
MSE: 0.022419653832912445, BCE: 0.07772812247276306  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_434_pretrain_2_original_depth]|![JNet_434_pretrain_2_output_depth]|![JNet_434_pretrain_2_label_depth]|
  
MSE: 0.022419653832912445, BCE: 0.07772812247276306  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_434_pretrain_3_original_plane]|![JNet_434_pretrain_3_output_plane]|![JNet_434_pretrain_3_label_plane]|
  
MSE: 0.024628158658742905, BCE: 0.09022310376167297  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_434_pretrain_3_original_depth]|![JNet_434_pretrain_3_output_depth]|![JNet_434_pretrain_3_label_depth]|
  
MSE: 0.024628158658742905, BCE: 0.09022310376167297  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_434_pretrain_4_original_plane]|![JNet_434_pretrain_4_output_plane]|![JNet_434_pretrain_4_label_plane]|
  
MSE: 0.022809604182839394, BCE: 0.08030061423778534  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_434_pretrain_4_original_depth]|![JNet_434_pretrain_4_output_depth]|![JNet_434_pretrain_4_label_depth]|
  
MSE: 0.022809604182839394, BCE: 0.08030061423778534  
  
mean MSE: 0.03061121143400669, mean BCE: 0.12660232186317444
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_435_0_original_plane]|![JNet_435_0_output_plane]|![JNet_435_0_label_plane]|
  
MSE: 0.03603210300207138, BCE: 0.14473284780979156  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_435_0_original_depth]|![JNet_435_0_output_depth]|![JNet_435_0_label_depth]|
  
MSE: 0.03603210300207138, BCE: 0.14473284780979156  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_435_1_original_plane]|![JNet_435_1_output_plane]|![JNet_435_1_label_plane]|
  
MSE: 0.038998305797576904, BCE: 0.1561432033777237  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_435_1_original_depth]|![JNet_435_1_output_depth]|![JNet_435_1_label_depth]|
  
MSE: 0.038998305797576904, BCE: 0.1561432033777237  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_435_2_original_plane]|![JNet_435_2_output_plane]|![JNet_435_2_label_plane]|
  
MSE: 0.02299007773399353, BCE: 0.09948144853115082  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_435_2_original_depth]|![JNet_435_2_output_depth]|![JNet_435_2_label_depth]|
  
MSE: 0.02299007773399353, BCE: 0.09948144853115082  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_435_3_original_plane]|![JNet_435_3_output_plane]|![JNet_435_3_label_plane]|
  
MSE: 0.03146975114941597, BCE: 0.13403593003749847  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_435_3_original_depth]|![JNet_435_3_output_depth]|![JNet_435_3_label_depth]|
  
MSE: 0.03146975114941597, BCE: 0.13403593003749847  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_435_4_original_plane]|![JNet_435_4_output_plane]|![JNet_435_4_label_plane]|
  
MSE: 0.02356581576168537, BCE: 0.09861816465854645  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_435_4_original_depth]|![JNet_435_4_output_depth]|![JNet_435_4_label_depth]|
  
MSE: 0.02356581576168537, BCE: 0.09861816465854645  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_434_pretrain_beads_001_roi000_original_depth]|![JNet_434_pretrain_beads_001_roi000_output_depth]|![JNet_434_pretrain_beads_001_roi000_reconst_depth]|![JNet_434_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 29.742144531250005, MSE: 0.06741415709257126, quantized loss: 0.0033293755259364843  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_434_pretrain_beads_001_roi001_original_depth]|![JNet_434_pretrain_beads_001_roi001_output_depth]|![JNet_434_pretrain_beads_001_roi001_reconst_depth]|![JNet_434_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 41.15752343750001, MSE: 0.08243367820978165, quantized loss: 0.004121509846299887  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_434_pretrain_beads_001_roi002_original_depth]|![JNet_434_pretrain_beads_001_roi002_output_depth]|![JNet_434_pretrain_beads_001_roi002_reconst_depth]|![JNet_434_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 30.52801562500001, MSE: 0.07157910615205765, quantized loss: 0.0032648465130478144  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_434_pretrain_beads_001_roi003_original_depth]|![JNet_434_pretrain_beads_001_roi003_output_depth]|![JNet_434_pretrain_beads_001_roi003_reconst_depth]|![JNet_434_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 42.85167968750001, MSE: 0.08730748295783997, quantized loss: 0.004580452106893063  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_434_pretrain_beads_001_roi004_original_depth]|![JNet_434_pretrain_beads_001_roi004_output_depth]|![JNet_434_pretrain_beads_001_roi004_reconst_depth]|![JNet_434_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 32.038605468750006, MSE: 0.0736350491642952, quantized loss: 0.003286755643785  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_434_pretrain_beads_002_roi000_original_depth]|![JNet_434_pretrain_beads_002_roi000_output_depth]|![JNet_434_pretrain_beads_002_roi000_reconst_depth]|![JNet_434_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 34.10547656250001, MSE: 0.07806777209043503, quantized loss: 0.0035920690279453993  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_434_pretrain_beads_002_roi001_original_depth]|![JNet_434_pretrain_beads_002_roi001_output_depth]|![JNet_434_pretrain_beads_002_roi001_reconst_depth]|![JNet_434_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 31.392554687500006, MSE: 0.07243634015321732, quantized loss: 0.003509917063638568  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_434_pretrain_beads_002_roi002_original_depth]|![JNet_434_pretrain_beads_002_roi002_output_depth]|![JNet_434_pretrain_beads_002_roi002_reconst_depth]|![JNet_434_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 32.56341796875001, MSE: 0.07531020790338516, quantized loss: 0.003513053758069873  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_435_beads_001_roi000_original_depth]|![JNet_435_beads_001_roi000_output_depth]|![JNet_435_beads_001_roi000_reconst_depth]|![JNet_435_beads_001_roi000_heatmap_depth]|
  
volume: 14.997824218750004, MSE: 0.0008637039572931826, quantized loss: 0.0018295154441148043  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_435_beads_001_roi001_original_depth]|![JNet_435_beads_001_roi001_output_depth]|![JNet_435_beads_001_roi001_reconst_depth]|![JNet_435_beads_001_roi001_heatmap_depth]|
  
volume: 20.407392578125005, MSE: 0.0009355624788440764, quantized loss: 0.002762714633718133  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_435_beads_001_roi002_original_depth]|![JNet_435_beads_001_roi002_output_depth]|![JNet_435_beads_001_roi002_reconst_depth]|![JNet_435_beads_001_roi002_heatmap_depth]|
  
volume: 13.562041015625002, MSE: 0.0005707715754397213, quantized loss: 0.0017628467176109552  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_435_beads_001_roi003_original_depth]|![JNet_435_beads_001_roi003_output_depth]|![JNet_435_beads_001_roi003_reconst_depth]|![JNet_435_beads_001_roi003_heatmap_depth]|
  
volume: 19.258890625000003, MSE: 0.0008396151824854314, quantized loss: 0.002614585217088461  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_435_beads_001_roi004_original_depth]|![JNet_435_beads_001_roi004_output_depth]|![JNet_435_beads_001_roi004_reconst_depth]|![JNet_435_beads_001_roi004_heatmap_depth]|
  
volume: 14.411363281250003, MSE: 0.0005268718232400715, quantized loss: 0.0017986688762903214  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_435_beads_002_roi000_original_depth]|![JNet_435_beads_002_roi000_output_depth]|![JNet_435_beads_002_roi000_reconst_depth]|![JNet_435_beads_002_roi000_heatmap_depth]|
  
volume: 15.305972656250004, MSE: 0.0005974919185973704, quantized loss: 0.0018780494574457407  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_435_beads_002_roi001_original_depth]|![JNet_435_beads_002_roi001_output_depth]|![JNet_435_beads_002_roi001_reconst_depth]|![JNet_435_beads_002_roi001_heatmap_depth]|
  
volume: 12.787745117187503, MSE: 0.00045470483019016683, quantized loss: 0.0016801845049485564  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_435_beads_002_roi002_original_depth]|![JNet_435_beads_002_roi002_output_depth]|![JNet_435_beads_002_roi002_reconst_depth]|![JNet_435_beads_002_roi002_heatmap_depth]|
  
volume: 14.130263671875003, MSE: 0.0004885270609520376, quantized loss: 0.001837266841903329  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_435_psf_pre]|![JNet_435_psf_post]|

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
  



[JNet_434_pretrain_0_label_depth]: /experiments/images/JNet_434_pretrain_0_label_depth.png
[JNet_434_pretrain_0_label_plane]: /experiments/images/JNet_434_pretrain_0_label_plane.png
[JNet_434_pretrain_0_original_depth]: /experiments/images/JNet_434_pretrain_0_original_depth.png
[JNet_434_pretrain_0_original_plane]: /experiments/images/JNet_434_pretrain_0_original_plane.png
[JNet_434_pretrain_0_output_depth]: /experiments/images/JNet_434_pretrain_0_output_depth.png
[JNet_434_pretrain_0_output_plane]: /experiments/images/JNet_434_pretrain_0_output_plane.png
[JNet_434_pretrain_1_label_depth]: /experiments/images/JNet_434_pretrain_1_label_depth.png
[JNet_434_pretrain_1_label_plane]: /experiments/images/JNet_434_pretrain_1_label_plane.png
[JNet_434_pretrain_1_original_depth]: /experiments/images/JNet_434_pretrain_1_original_depth.png
[JNet_434_pretrain_1_original_plane]: /experiments/images/JNet_434_pretrain_1_original_plane.png
[JNet_434_pretrain_1_output_depth]: /experiments/images/JNet_434_pretrain_1_output_depth.png
[JNet_434_pretrain_1_output_plane]: /experiments/images/JNet_434_pretrain_1_output_plane.png
[JNet_434_pretrain_2_label_depth]: /experiments/images/JNet_434_pretrain_2_label_depth.png
[JNet_434_pretrain_2_label_plane]: /experiments/images/JNet_434_pretrain_2_label_plane.png
[JNet_434_pretrain_2_original_depth]: /experiments/images/JNet_434_pretrain_2_original_depth.png
[JNet_434_pretrain_2_original_plane]: /experiments/images/JNet_434_pretrain_2_original_plane.png
[JNet_434_pretrain_2_output_depth]: /experiments/images/JNet_434_pretrain_2_output_depth.png
[JNet_434_pretrain_2_output_plane]: /experiments/images/JNet_434_pretrain_2_output_plane.png
[JNet_434_pretrain_3_label_depth]: /experiments/images/JNet_434_pretrain_3_label_depth.png
[JNet_434_pretrain_3_label_plane]: /experiments/images/JNet_434_pretrain_3_label_plane.png
[JNet_434_pretrain_3_original_depth]: /experiments/images/JNet_434_pretrain_3_original_depth.png
[JNet_434_pretrain_3_original_plane]: /experiments/images/JNet_434_pretrain_3_original_plane.png
[JNet_434_pretrain_3_output_depth]: /experiments/images/JNet_434_pretrain_3_output_depth.png
[JNet_434_pretrain_3_output_plane]: /experiments/images/JNet_434_pretrain_3_output_plane.png
[JNet_434_pretrain_4_label_depth]: /experiments/images/JNet_434_pretrain_4_label_depth.png
[JNet_434_pretrain_4_label_plane]: /experiments/images/JNet_434_pretrain_4_label_plane.png
[JNet_434_pretrain_4_original_depth]: /experiments/images/JNet_434_pretrain_4_original_depth.png
[JNet_434_pretrain_4_original_plane]: /experiments/images/JNet_434_pretrain_4_original_plane.png
[JNet_434_pretrain_4_output_depth]: /experiments/images/JNet_434_pretrain_4_output_depth.png
[JNet_434_pretrain_4_output_plane]: /experiments/images/JNet_434_pretrain_4_output_plane.png
[JNet_434_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_434_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi000_original_depth.png
[JNet_434_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi000_output_depth.png
[JNet_434_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi000_reconst_depth.png
[JNet_434_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_434_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi001_original_depth.png
[JNet_434_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi001_output_depth.png
[JNet_434_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi001_reconst_depth.png
[JNet_434_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_434_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi002_original_depth.png
[JNet_434_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi002_output_depth.png
[JNet_434_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi002_reconst_depth.png
[JNet_434_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_434_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi003_original_depth.png
[JNet_434_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi003_output_depth.png
[JNet_434_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi003_reconst_depth.png
[JNet_434_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_434_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi004_original_depth.png
[JNet_434_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi004_output_depth.png
[JNet_434_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_434_pretrain_beads_001_roi004_reconst_depth.png
[JNet_434_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_434_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi000_original_depth.png
[JNet_434_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi000_output_depth.png
[JNet_434_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi000_reconst_depth.png
[JNet_434_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_434_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi001_original_depth.png
[JNet_434_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi001_output_depth.png
[JNet_434_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi001_reconst_depth.png
[JNet_434_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_434_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi002_original_depth.png
[JNet_434_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi002_output_depth.png
[JNet_434_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_434_pretrain_beads_002_roi002_reconst_depth.png
[JNet_435_0_label_depth]: /experiments/images/JNet_435_0_label_depth.png
[JNet_435_0_label_plane]: /experiments/images/JNet_435_0_label_plane.png
[JNet_435_0_original_depth]: /experiments/images/JNet_435_0_original_depth.png
[JNet_435_0_original_plane]: /experiments/images/JNet_435_0_original_plane.png
[JNet_435_0_output_depth]: /experiments/images/JNet_435_0_output_depth.png
[JNet_435_0_output_plane]: /experiments/images/JNet_435_0_output_plane.png
[JNet_435_1_label_depth]: /experiments/images/JNet_435_1_label_depth.png
[JNet_435_1_label_plane]: /experiments/images/JNet_435_1_label_plane.png
[JNet_435_1_original_depth]: /experiments/images/JNet_435_1_original_depth.png
[JNet_435_1_original_plane]: /experiments/images/JNet_435_1_original_plane.png
[JNet_435_1_output_depth]: /experiments/images/JNet_435_1_output_depth.png
[JNet_435_1_output_plane]: /experiments/images/JNet_435_1_output_plane.png
[JNet_435_2_label_depth]: /experiments/images/JNet_435_2_label_depth.png
[JNet_435_2_label_plane]: /experiments/images/JNet_435_2_label_plane.png
[JNet_435_2_original_depth]: /experiments/images/JNet_435_2_original_depth.png
[JNet_435_2_original_plane]: /experiments/images/JNet_435_2_original_plane.png
[JNet_435_2_output_depth]: /experiments/images/JNet_435_2_output_depth.png
[JNet_435_2_output_plane]: /experiments/images/JNet_435_2_output_plane.png
[JNet_435_3_label_depth]: /experiments/images/JNet_435_3_label_depth.png
[JNet_435_3_label_plane]: /experiments/images/JNet_435_3_label_plane.png
[JNet_435_3_original_depth]: /experiments/images/JNet_435_3_original_depth.png
[JNet_435_3_original_plane]: /experiments/images/JNet_435_3_original_plane.png
[JNet_435_3_output_depth]: /experiments/images/JNet_435_3_output_depth.png
[JNet_435_3_output_plane]: /experiments/images/JNet_435_3_output_plane.png
[JNet_435_4_label_depth]: /experiments/images/JNet_435_4_label_depth.png
[JNet_435_4_label_plane]: /experiments/images/JNet_435_4_label_plane.png
[JNet_435_4_original_depth]: /experiments/images/JNet_435_4_original_depth.png
[JNet_435_4_original_plane]: /experiments/images/JNet_435_4_original_plane.png
[JNet_435_4_output_depth]: /experiments/images/JNet_435_4_output_depth.png
[JNet_435_4_output_plane]: /experiments/images/JNet_435_4_output_plane.png
[JNet_435_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_435_beads_001_roi000_heatmap_depth.png
[JNet_435_beads_001_roi000_original_depth]: /experiments/images/JNet_435_beads_001_roi000_original_depth.png
[JNet_435_beads_001_roi000_output_depth]: /experiments/images/JNet_435_beads_001_roi000_output_depth.png
[JNet_435_beads_001_roi000_reconst_depth]: /experiments/images/JNet_435_beads_001_roi000_reconst_depth.png
[JNet_435_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_435_beads_001_roi001_heatmap_depth.png
[JNet_435_beads_001_roi001_original_depth]: /experiments/images/JNet_435_beads_001_roi001_original_depth.png
[JNet_435_beads_001_roi001_output_depth]: /experiments/images/JNet_435_beads_001_roi001_output_depth.png
[JNet_435_beads_001_roi001_reconst_depth]: /experiments/images/JNet_435_beads_001_roi001_reconst_depth.png
[JNet_435_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_435_beads_001_roi002_heatmap_depth.png
[JNet_435_beads_001_roi002_original_depth]: /experiments/images/JNet_435_beads_001_roi002_original_depth.png
[JNet_435_beads_001_roi002_output_depth]: /experiments/images/JNet_435_beads_001_roi002_output_depth.png
[JNet_435_beads_001_roi002_reconst_depth]: /experiments/images/JNet_435_beads_001_roi002_reconst_depth.png
[JNet_435_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_435_beads_001_roi003_heatmap_depth.png
[JNet_435_beads_001_roi003_original_depth]: /experiments/images/JNet_435_beads_001_roi003_original_depth.png
[JNet_435_beads_001_roi003_output_depth]: /experiments/images/JNet_435_beads_001_roi003_output_depth.png
[JNet_435_beads_001_roi003_reconst_depth]: /experiments/images/JNet_435_beads_001_roi003_reconst_depth.png
[JNet_435_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_435_beads_001_roi004_heatmap_depth.png
[JNet_435_beads_001_roi004_original_depth]: /experiments/images/JNet_435_beads_001_roi004_original_depth.png
[JNet_435_beads_001_roi004_output_depth]: /experiments/images/JNet_435_beads_001_roi004_output_depth.png
[JNet_435_beads_001_roi004_reconst_depth]: /experiments/images/JNet_435_beads_001_roi004_reconst_depth.png
[JNet_435_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_435_beads_002_roi000_heatmap_depth.png
[JNet_435_beads_002_roi000_original_depth]: /experiments/images/JNet_435_beads_002_roi000_original_depth.png
[JNet_435_beads_002_roi000_output_depth]: /experiments/images/JNet_435_beads_002_roi000_output_depth.png
[JNet_435_beads_002_roi000_reconst_depth]: /experiments/images/JNet_435_beads_002_roi000_reconst_depth.png
[JNet_435_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_435_beads_002_roi001_heatmap_depth.png
[JNet_435_beads_002_roi001_original_depth]: /experiments/images/JNet_435_beads_002_roi001_original_depth.png
[JNet_435_beads_002_roi001_output_depth]: /experiments/images/JNet_435_beads_002_roi001_output_depth.png
[JNet_435_beads_002_roi001_reconst_depth]: /experiments/images/JNet_435_beads_002_roi001_reconst_depth.png
[JNet_435_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_435_beads_002_roi002_heatmap_depth.png
[JNet_435_beads_002_roi002_original_depth]: /experiments/images/JNet_435_beads_002_roi002_original_depth.png
[JNet_435_beads_002_roi002_output_depth]: /experiments/images/JNet_435_beads_002_roi002_output_depth.png
[JNet_435_beads_002_roi002_reconst_depth]: /experiments/images/JNet_435_beads_002_roi002_reconst_depth.png
[JNet_435_psf_post]: /experiments/images/JNet_435_psf_post.png
[JNet_435_psf_pre]: /experiments/images/JNet_435_psf_pre.png
[finetuned]: /experiments/tmp/JNet_435_train.png
[pretrained_model]: /experiments/tmp/JNet_434_pretrain_train.png
