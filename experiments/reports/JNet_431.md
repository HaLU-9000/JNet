



# JNet_431 Report
  
the parameters to replicate the results of JNet_427. nearest interp of PSF, NA=0.7, mu_z = 1.2  
pretrained model : JNet_430_pretrain
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
  
mean MSE: 0.022880423814058304, mean BCE: 0.09139001369476318
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_0_original_plane]|![JNet_430_pretrain_0_output_plane]|![JNet_430_pretrain_0_label_plane]|
  
MSE: 0.024239668622612953, BCE: 0.09153971076011658  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_0_original_depth]|![JNet_430_pretrain_0_output_depth]|![JNet_430_pretrain_0_label_depth]|
  
MSE: 0.024239668622612953, BCE: 0.09153971076011658  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_1_original_plane]|![JNet_430_pretrain_1_output_plane]|![JNet_430_pretrain_1_label_plane]|
  
MSE: 0.021222727373242378, BCE: 0.07377481460571289  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_1_original_depth]|![JNet_430_pretrain_1_output_depth]|![JNet_430_pretrain_1_label_depth]|
  
MSE: 0.021222727373242378, BCE: 0.07377481460571289  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_2_original_plane]|![JNet_430_pretrain_2_output_plane]|![JNet_430_pretrain_2_label_plane]|
  
MSE: 0.018461715430021286, BCE: 0.0675402358174324  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_2_original_depth]|![JNet_430_pretrain_2_output_depth]|![JNet_430_pretrain_2_label_depth]|
  
MSE: 0.018461715430021286, BCE: 0.0675402358174324  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_3_original_plane]|![JNet_430_pretrain_3_output_plane]|![JNet_430_pretrain_3_label_plane]|
  
MSE: 0.02400103025138378, BCE: 0.08526496589183807  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_3_original_depth]|![JNet_430_pretrain_3_output_depth]|![JNet_430_pretrain_3_label_depth]|
  
MSE: 0.02400103025138378, BCE: 0.08526496589183807  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_4_original_plane]|![JNet_430_pretrain_4_output_plane]|![JNet_430_pretrain_4_label_plane]|
  
MSE: 0.02647697553038597, BCE: 0.13883034884929657  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_430_pretrain_4_original_depth]|![JNet_430_pretrain_4_output_depth]|![JNet_430_pretrain_4_label_depth]|
  
MSE: 0.02647697553038597, BCE: 0.13883034884929657  
  
mean MSE: 0.03591756895184517, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_431_0_original_plane]|![JNet_431_0_output_plane]|![JNet_431_0_label_plane]|
  
MSE: 0.044105082750320435, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_431_0_original_depth]|![JNet_431_0_output_depth]|![JNet_431_0_label_depth]|
  
MSE: 0.044105082750320435, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_431_1_original_plane]|![JNet_431_1_output_plane]|![JNet_431_1_label_plane]|
  
MSE: 0.029643986374139786, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_431_1_original_depth]|![JNet_431_1_output_depth]|![JNet_431_1_label_depth]|
  
MSE: 0.029643986374139786, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_431_2_original_plane]|![JNet_431_2_output_plane]|![JNet_431_2_label_plane]|
  
MSE: 0.0338631346821785, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_431_2_original_depth]|![JNet_431_2_output_depth]|![JNet_431_2_label_depth]|
  
MSE: 0.0338631346821785, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_431_3_original_plane]|![JNet_431_3_output_plane]|![JNet_431_3_label_plane]|
  
MSE: 0.03393122926354408, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_431_3_original_depth]|![JNet_431_3_output_depth]|![JNet_431_3_label_depth]|
  
MSE: 0.03393122926354408, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_431_4_original_plane]|![JNet_431_4_output_plane]|![JNet_431_4_label_plane]|
  
MSE: 0.038044415414333344, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_431_4_original_depth]|![JNet_431_4_output_depth]|![JNet_431_4_label_depth]|
  
MSE: 0.038044415414333344, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_001_roi000_original_depth]|![JNet_430_pretrain_beads_001_roi000_output_depth]|![JNet_430_pretrain_beads_001_roi000_reconst_depth]|![JNet_430_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 15.019928710937503, MSE: 0.02923627384006977, quantized loss: 0.0016075718449428678  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_001_roi001_original_depth]|![JNet_430_pretrain_beads_001_roi001_output_depth]|![JNet_430_pretrain_beads_001_roi001_reconst_depth]|![JNet_430_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 22.582943359375005, MSE: 0.037140484899282455, quantized loss: 0.0023082024417817593  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_001_roi002_original_depth]|![JNet_430_pretrain_beads_001_roi002_output_depth]|![JNet_430_pretrain_beads_001_roi002_reconst_depth]|![JNet_430_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 14.074012695312504, MSE: 0.02678012289106846, quantized loss: 0.0014789028791710734  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_001_roi003_original_depth]|![JNet_430_pretrain_beads_001_roi003_output_depth]|![JNet_430_pretrain_beads_001_roi003_reconst_depth]|![JNet_430_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 23.027904296875004, MSE: 0.03844589367508888, quantized loss: 0.0022499915212392807  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_001_roi004_original_depth]|![JNet_430_pretrain_beads_001_roi004_output_depth]|![JNet_430_pretrain_beads_001_roi004_reconst_depth]|![JNet_430_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 15.297892578125003, MSE: 0.02924150414764881, quantized loss: 0.0015527462819591165  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_002_roi000_original_depth]|![JNet_430_pretrain_beads_002_roi000_output_depth]|![JNet_430_pretrain_beads_002_roi000_reconst_depth]|![JNet_430_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 16.374963867187503, MSE: 0.03151892125606537, quantized loss: 0.0016209491295740008  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_002_roi001_original_depth]|![JNet_430_pretrain_beads_002_roi001_output_depth]|![JNet_430_pretrain_beads_002_roi001_reconst_depth]|![JNet_430_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 15.247485351562503, MSE: 0.02938508801162243, quantized loss: 0.0015378047246485949  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_430_pretrain_beads_002_roi002_original_depth]|![JNet_430_pretrain_beads_002_roi002_output_depth]|![JNet_430_pretrain_beads_002_roi002_reconst_depth]|![JNet_430_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 15.627458984375004, MSE: 0.030016442760825157, quantized loss: 0.0015970974927768111  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_431_beads_001_roi000_original_depth]|![JNet_431_beads_001_roi000_output_depth]|![JNet_431_beads_001_roi000_reconst_depth]|![JNet_431_beads_001_roi000_heatmap_depth]|
  
volume: 3.405829101562501, MSE: 0.0002595653058961034, quantized loss: 2.429141204629559e-05  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_431_beads_001_roi001_original_depth]|![JNet_431_beads_001_roi001_output_depth]|![JNet_431_beads_001_roi001_reconst_depth]|![JNet_431_beads_001_roi001_heatmap_depth]|
  
volume: 5.322004394531251, MSE: 0.0007562876562587917, quantized loss: 3.3343694667564705e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_431_beads_001_roi002_original_depth]|![JNet_431_beads_001_roi002_output_depth]|![JNet_431_beads_001_roi002_reconst_depth]|![JNet_431_beads_001_roi002_heatmap_depth]|
  
volume: 3.3986220703125007, MSE: 0.00021142289915587753, quantized loss: 2.3516362489317544e-05  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_431_beads_001_roi003_original_depth]|![JNet_431_beads_001_roi003_output_depth]|![JNet_431_beads_001_roi003_reconst_depth]|![JNet_431_beads_001_roi003_heatmap_depth]|
  
volume: 5.623069335937501, MSE: 0.0004805831704288721, quantized loss: 3.565754013834521e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_431_beads_001_roi004_original_depth]|![JNet_431_beads_001_roi004_output_depth]|![JNet_431_beads_001_roi004_reconst_depth]|![JNet_431_beads_001_roi004_heatmap_depth]|
  
volume: 3.699795898437501, MSE: 0.00021415877563413233, quantized loss: 2.5388684662175365e-05  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_431_beads_002_roi000_original_depth]|![JNet_431_beads_002_roi000_output_depth]|![JNet_431_beads_002_roi000_reconst_depth]|![JNet_431_beads_002_roi000_heatmap_depth]|
  
volume: 3.960117919921876, MSE: 0.00022061933123040944, quantized loss: 2.5700242986204103e-05  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_431_beads_002_roi001_original_depth]|![JNet_431_beads_002_roi001_output_depth]|![JNet_431_beads_002_roi001_reconst_depth]|![JNet_431_beads_002_roi001_heatmap_depth]|
  
volume: 3.5413090820312507, MSE: 0.0002143570891348645, quantized loss: 2.5495448426227085e-05  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_431_beads_002_roi002_original_depth]|![JNet_431_beads_002_roi002_output_depth]|![JNet_431_beads_002_roi002_reconst_depth]|![JNet_431_beads_002_roi002_heatmap_depth]|
  
volume: 3.771183349609376, MSE: 0.00020645292534027249, quantized loss: 2.7607242373051122e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_431_psf_pre]|![JNet_431_psf_post]|

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
  



[JNet_430_pretrain_0_label_depth]: /experiments/images/JNet_430_pretrain_0_label_depth.png
[JNet_430_pretrain_0_label_plane]: /experiments/images/JNet_430_pretrain_0_label_plane.png
[JNet_430_pretrain_0_original_depth]: /experiments/images/JNet_430_pretrain_0_original_depth.png
[JNet_430_pretrain_0_original_plane]: /experiments/images/JNet_430_pretrain_0_original_plane.png
[JNet_430_pretrain_0_output_depth]: /experiments/images/JNet_430_pretrain_0_output_depth.png
[JNet_430_pretrain_0_output_plane]: /experiments/images/JNet_430_pretrain_0_output_plane.png
[JNet_430_pretrain_1_label_depth]: /experiments/images/JNet_430_pretrain_1_label_depth.png
[JNet_430_pretrain_1_label_plane]: /experiments/images/JNet_430_pretrain_1_label_plane.png
[JNet_430_pretrain_1_original_depth]: /experiments/images/JNet_430_pretrain_1_original_depth.png
[JNet_430_pretrain_1_original_plane]: /experiments/images/JNet_430_pretrain_1_original_plane.png
[JNet_430_pretrain_1_output_depth]: /experiments/images/JNet_430_pretrain_1_output_depth.png
[JNet_430_pretrain_1_output_plane]: /experiments/images/JNet_430_pretrain_1_output_plane.png
[JNet_430_pretrain_2_label_depth]: /experiments/images/JNet_430_pretrain_2_label_depth.png
[JNet_430_pretrain_2_label_plane]: /experiments/images/JNet_430_pretrain_2_label_plane.png
[JNet_430_pretrain_2_original_depth]: /experiments/images/JNet_430_pretrain_2_original_depth.png
[JNet_430_pretrain_2_original_plane]: /experiments/images/JNet_430_pretrain_2_original_plane.png
[JNet_430_pretrain_2_output_depth]: /experiments/images/JNet_430_pretrain_2_output_depth.png
[JNet_430_pretrain_2_output_plane]: /experiments/images/JNet_430_pretrain_2_output_plane.png
[JNet_430_pretrain_3_label_depth]: /experiments/images/JNet_430_pretrain_3_label_depth.png
[JNet_430_pretrain_3_label_plane]: /experiments/images/JNet_430_pretrain_3_label_plane.png
[JNet_430_pretrain_3_original_depth]: /experiments/images/JNet_430_pretrain_3_original_depth.png
[JNet_430_pretrain_3_original_plane]: /experiments/images/JNet_430_pretrain_3_original_plane.png
[JNet_430_pretrain_3_output_depth]: /experiments/images/JNet_430_pretrain_3_output_depth.png
[JNet_430_pretrain_3_output_plane]: /experiments/images/JNet_430_pretrain_3_output_plane.png
[JNet_430_pretrain_4_label_depth]: /experiments/images/JNet_430_pretrain_4_label_depth.png
[JNet_430_pretrain_4_label_plane]: /experiments/images/JNet_430_pretrain_4_label_plane.png
[JNet_430_pretrain_4_original_depth]: /experiments/images/JNet_430_pretrain_4_original_depth.png
[JNet_430_pretrain_4_original_plane]: /experiments/images/JNet_430_pretrain_4_original_plane.png
[JNet_430_pretrain_4_output_depth]: /experiments/images/JNet_430_pretrain_4_output_depth.png
[JNet_430_pretrain_4_output_plane]: /experiments/images/JNet_430_pretrain_4_output_plane.png
[JNet_430_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_430_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi000_original_depth.png
[JNet_430_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi000_output_depth.png
[JNet_430_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi000_reconst_depth.png
[JNet_430_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_430_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi001_original_depth.png
[JNet_430_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi001_output_depth.png
[JNet_430_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi001_reconst_depth.png
[JNet_430_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_430_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi002_original_depth.png
[JNet_430_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi002_output_depth.png
[JNet_430_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi002_reconst_depth.png
[JNet_430_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_430_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi003_original_depth.png
[JNet_430_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi003_output_depth.png
[JNet_430_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi003_reconst_depth.png
[JNet_430_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_430_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi004_original_depth.png
[JNet_430_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi004_output_depth.png
[JNet_430_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_001_roi004_reconst_depth.png
[JNet_430_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_430_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi000_original_depth.png
[JNet_430_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi000_output_depth.png
[JNet_430_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi000_reconst_depth.png
[JNet_430_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_430_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi001_original_depth.png
[JNet_430_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi001_output_depth.png
[JNet_430_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi001_reconst_depth.png
[JNet_430_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_430_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi002_original_depth.png
[JNet_430_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi002_output_depth.png
[JNet_430_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_430_pretrain_beads_002_roi002_reconst_depth.png
[JNet_431_0_label_depth]: /experiments/images/JNet_431_0_label_depth.png
[JNet_431_0_label_plane]: /experiments/images/JNet_431_0_label_plane.png
[JNet_431_0_original_depth]: /experiments/images/JNet_431_0_original_depth.png
[JNet_431_0_original_plane]: /experiments/images/JNet_431_0_original_plane.png
[JNet_431_0_output_depth]: /experiments/images/JNet_431_0_output_depth.png
[JNet_431_0_output_plane]: /experiments/images/JNet_431_0_output_plane.png
[JNet_431_1_label_depth]: /experiments/images/JNet_431_1_label_depth.png
[JNet_431_1_label_plane]: /experiments/images/JNet_431_1_label_plane.png
[JNet_431_1_original_depth]: /experiments/images/JNet_431_1_original_depth.png
[JNet_431_1_original_plane]: /experiments/images/JNet_431_1_original_plane.png
[JNet_431_1_output_depth]: /experiments/images/JNet_431_1_output_depth.png
[JNet_431_1_output_plane]: /experiments/images/JNet_431_1_output_plane.png
[JNet_431_2_label_depth]: /experiments/images/JNet_431_2_label_depth.png
[JNet_431_2_label_plane]: /experiments/images/JNet_431_2_label_plane.png
[JNet_431_2_original_depth]: /experiments/images/JNet_431_2_original_depth.png
[JNet_431_2_original_plane]: /experiments/images/JNet_431_2_original_plane.png
[JNet_431_2_output_depth]: /experiments/images/JNet_431_2_output_depth.png
[JNet_431_2_output_plane]: /experiments/images/JNet_431_2_output_plane.png
[JNet_431_3_label_depth]: /experiments/images/JNet_431_3_label_depth.png
[JNet_431_3_label_plane]: /experiments/images/JNet_431_3_label_plane.png
[JNet_431_3_original_depth]: /experiments/images/JNet_431_3_original_depth.png
[JNet_431_3_original_plane]: /experiments/images/JNet_431_3_original_plane.png
[JNet_431_3_output_depth]: /experiments/images/JNet_431_3_output_depth.png
[JNet_431_3_output_plane]: /experiments/images/JNet_431_3_output_plane.png
[JNet_431_4_label_depth]: /experiments/images/JNet_431_4_label_depth.png
[JNet_431_4_label_plane]: /experiments/images/JNet_431_4_label_plane.png
[JNet_431_4_original_depth]: /experiments/images/JNet_431_4_original_depth.png
[JNet_431_4_original_plane]: /experiments/images/JNet_431_4_original_plane.png
[JNet_431_4_output_depth]: /experiments/images/JNet_431_4_output_depth.png
[JNet_431_4_output_plane]: /experiments/images/JNet_431_4_output_plane.png
[JNet_431_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_431_beads_001_roi000_heatmap_depth.png
[JNet_431_beads_001_roi000_original_depth]: /experiments/images/JNet_431_beads_001_roi000_original_depth.png
[JNet_431_beads_001_roi000_output_depth]: /experiments/images/JNet_431_beads_001_roi000_output_depth.png
[JNet_431_beads_001_roi000_reconst_depth]: /experiments/images/JNet_431_beads_001_roi000_reconst_depth.png
[JNet_431_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_431_beads_001_roi001_heatmap_depth.png
[JNet_431_beads_001_roi001_original_depth]: /experiments/images/JNet_431_beads_001_roi001_original_depth.png
[JNet_431_beads_001_roi001_output_depth]: /experiments/images/JNet_431_beads_001_roi001_output_depth.png
[JNet_431_beads_001_roi001_reconst_depth]: /experiments/images/JNet_431_beads_001_roi001_reconst_depth.png
[JNet_431_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_431_beads_001_roi002_heatmap_depth.png
[JNet_431_beads_001_roi002_original_depth]: /experiments/images/JNet_431_beads_001_roi002_original_depth.png
[JNet_431_beads_001_roi002_output_depth]: /experiments/images/JNet_431_beads_001_roi002_output_depth.png
[JNet_431_beads_001_roi002_reconst_depth]: /experiments/images/JNet_431_beads_001_roi002_reconst_depth.png
[JNet_431_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_431_beads_001_roi003_heatmap_depth.png
[JNet_431_beads_001_roi003_original_depth]: /experiments/images/JNet_431_beads_001_roi003_original_depth.png
[JNet_431_beads_001_roi003_output_depth]: /experiments/images/JNet_431_beads_001_roi003_output_depth.png
[JNet_431_beads_001_roi003_reconst_depth]: /experiments/images/JNet_431_beads_001_roi003_reconst_depth.png
[JNet_431_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_431_beads_001_roi004_heatmap_depth.png
[JNet_431_beads_001_roi004_original_depth]: /experiments/images/JNet_431_beads_001_roi004_original_depth.png
[JNet_431_beads_001_roi004_output_depth]: /experiments/images/JNet_431_beads_001_roi004_output_depth.png
[JNet_431_beads_001_roi004_reconst_depth]: /experiments/images/JNet_431_beads_001_roi004_reconst_depth.png
[JNet_431_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_431_beads_002_roi000_heatmap_depth.png
[JNet_431_beads_002_roi000_original_depth]: /experiments/images/JNet_431_beads_002_roi000_original_depth.png
[JNet_431_beads_002_roi000_output_depth]: /experiments/images/JNet_431_beads_002_roi000_output_depth.png
[JNet_431_beads_002_roi000_reconst_depth]: /experiments/images/JNet_431_beads_002_roi000_reconst_depth.png
[JNet_431_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_431_beads_002_roi001_heatmap_depth.png
[JNet_431_beads_002_roi001_original_depth]: /experiments/images/JNet_431_beads_002_roi001_original_depth.png
[JNet_431_beads_002_roi001_output_depth]: /experiments/images/JNet_431_beads_002_roi001_output_depth.png
[JNet_431_beads_002_roi001_reconst_depth]: /experiments/images/JNet_431_beads_002_roi001_reconst_depth.png
[JNet_431_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_431_beads_002_roi002_heatmap_depth.png
[JNet_431_beads_002_roi002_original_depth]: /experiments/images/JNet_431_beads_002_roi002_original_depth.png
[JNet_431_beads_002_roi002_output_depth]: /experiments/images/JNet_431_beads_002_roi002_output_depth.png
[JNet_431_beads_002_roi002_reconst_depth]: /experiments/images/JNet_431_beads_002_roi002_reconst_depth.png
[JNet_431_psf_post]: /experiments/images/JNet_431_psf_post.png
[JNet_431_psf_pre]: /experiments/images/JNet_431_psf_pre.png
[finetuned]: /experiments/tmp/JNet_431_train.png
[pretrained_model]: /experiments/tmp/JNet_430_pretrain_train.png
