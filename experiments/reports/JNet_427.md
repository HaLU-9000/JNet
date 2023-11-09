



# JNet_427 Report
  
the parameters to replicate the results of JNet_427. nearest interp of PSF, NA=0.7, mu_z = 0.6  
pretrained model : JNet_426_pretrain
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
|mu_z|0.6||
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
  
mean MSE: 0.01973220333456993, mean BCE: 0.07313083112239838
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_426_pretrain_0_original_plane]|![JNet_426_pretrain_0_output_plane]|![JNet_426_pretrain_0_label_plane]|
  
MSE: 0.019639331847429276, BCE: 0.08381210267543793  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_426_pretrain_0_original_depth]|![JNet_426_pretrain_0_output_depth]|![JNet_426_pretrain_0_label_depth]|
  
MSE: 0.019639331847429276, BCE: 0.08381210267543793  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_426_pretrain_1_original_plane]|![JNet_426_pretrain_1_output_plane]|![JNet_426_pretrain_1_label_plane]|
  
MSE: 0.02056802064180374, BCE: 0.07054866850376129  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_426_pretrain_1_original_depth]|![JNet_426_pretrain_1_output_depth]|![JNet_426_pretrain_1_label_depth]|
  
MSE: 0.02056802064180374, BCE: 0.07054866850376129  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_426_pretrain_2_original_plane]|![JNet_426_pretrain_2_output_plane]|![JNet_426_pretrain_2_label_plane]|
  
MSE: 0.01089108269661665, BCE: 0.04042726755142212  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_426_pretrain_2_original_depth]|![JNet_426_pretrain_2_output_depth]|![JNet_426_pretrain_2_label_depth]|
  
MSE: 0.01089108269661665, BCE: 0.04042726755142212  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_426_pretrain_3_original_plane]|![JNet_426_pretrain_3_output_plane]|![JNet_426_pretrain_3_label_plane]|
  
MSE: 0.01999932900071144, BCE: 0.07347824424505234  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_426_pretrain_3_original_depth]|![JNet_426_pretrain_3_output_depth]|![JNet_426_pretrain_3_label_depth]|
  
MSE: 0.01999932900071144, BCE: 0.07347824424505234  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_426_pretrain_4_original_plane]|![JNet_426_pretrain_4_output_plane]|![JNet_426_pretrain_4_label_plane]|
  
MSE: 0.02756325900554657, BCE: 0.09738784283399582  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_426_pretrain_4_original_depth]|![JNet_426_pretrain_4_output_depth]|![JNet_426_pretrain_4_label_depth]|
  
MSE: 0.02756325900554657, BCE: 0.09738784283399582  
  
mean MSE: 0.032627902925014496, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_427_0_original_plane]|![JNet_427_0_output_plane]|![JNet_427_0_label_plane]|
  
MSE: 0.028120530769228935, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_427_0_original_depth]|![JNet_427_0_output_depth]|![JNet_427_0_label_depth]|
  
MSE: 0.028120530769228935, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_427_1_original_plane]|![JNet_427_1_output_plane]|![JNet_427_1_label_plane]|
  
MSE: 0.026908282190561295, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_427_1_original_depth]|![JNet_427_1_output_depth]|![JNet_427_1_label_depth]|
  
MSE: 0.026908282190561295, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_427_2_original_plane]|![JNet_427_2_output_plane]|![JNet_427_2_label_plane]|
  
MSE: 0.03979026526212692, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_427_2_original_depth]|![JNet_427_2_output_depth]|![JNet_427_2_label_depth]|
  
MSE: 0.03979026526212692, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_427_3_original_plane]|![JNet_427_3_output_plane]|![JNet_427_3_label_plane]|
  
MSE: 0.04113918915390968, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_427_3_original_depth]|![JNet_427_3_output_depth]|![JNet_427_3_label_depth]|
  
MSE: 0.04113918915390968, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_427_4_original_plane]|![JNet_427_4_output_plane]|![JNet_427_4_label_plane]|
  
MSE: 0.0271812342107296, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_427_4_original_depth]|![JNet_427_4_output_depth]|![JNet_427_4_label_depth]|
  
MSE: 0.0271812342107296, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_426_pretrain_beads_001_roi000_original_depth]|![JNet_426_pretrain_beads_001_roi000_output_depth]|![JNet_426_pretrain_beads_001_roi000_reconst_depth]|![JNet_426_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 22.326683593750005, MSE: 0.013875111937522888, quantized loss: 0.003278255695477128  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_426_pretrain_beads_001_roi001_original_depth]|![JNet_426_pretrain_beads_001_roi001_output_depth]|![JNet_426_pretrain_beads_001_roi001_reconst_depth]|![JNet_426_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 30.301646484375006, MSE: 0.014433509670197964, quantized loss: 0.003990002907812595  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_426_pretrain_beads_001_roi002_original_depth]|![JNet_426_pretrain_beads_001_roi002_output_depth]|![JNet_426_pretrain_beads_001_roi002_reconst_depth]|![JNet_426_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 21.474115234375006, MSE: 0.012593795545399189, quantized loss: 0.003129382152110338  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_426_pretrain_beads_001_roi003_original_depth]|![JNet_426_pretrain_beads_001_roi003_output_depth]|![JNet_426_pretrain_beads_001_roi003_reconst_depth]|![JNet_426_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 33.075671875000005, MSE: 0.01780557818710804, quantized loss: 0.004754609894007444  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_426_pretrain_beads_001_roi004_original_depth]|![JNet_426_pretrain_beads_001_roi004_output_depth]|![JNet_426_pretrain_beads_001_roi004_reconst_depth]|![JNet_426_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 23.480050781250007, MSE: 0.014666476286947727, quantized loss: 0.0034702124539762735  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_426_pretrain_beads_002_roi000_original_depth]|![JNet_426_pretrain_beads_002_roi000_output_depth]|![JNet_426_pretrain_beads_002_roi000_reconst_depth]|![JNet_426_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 24.756687500000005, MSE: 0.015538465231657028, quantized loss: 0.0037467475049197674  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_426_pretrain_beads_002_roi001_original_depth]|![JNet_426_pretrain_beads_002_roi001_output_depth]|![JNet_426_pretrain_beads_002_roi001_reconst_depth]|![JNet_426_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 22.573757812500006, MSE: 0.013722325675189495, quantized loss: 0.0031965423841029406  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_426_pretrain_beads_002_roi002_original_depth]|![JNet_426_pretrain_beads_002_roi002_output_depth]|![JNet_426_pretrain_beads_002_roi002_reconst_depth]|![JNet_426_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 23.459769531250007, MSE: 0.014226183295249939, quantized loss: 0.0034607001580297947  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_427_beads_001_roi000_original_depth]|![JNet_427_beads_001_roi000_output_depth]|![JNet_427_beads_001_roi000_reconst_depth]|![JNet_427_beads_001_roi000_heatmap_depth]|
  
volume: 7.781605468750002, MSE: 0.00040079126483760774, quantized loss: 1.1512133823998738e-05  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_427_beads_001_roi001_original_depth]|![JNet_427_beads_001_roi001_output_depth]|![JNet_427_beads_001_roi001_reconst_depth]|![JNet_427_beads_001_roi001_heatmap_depth]|
  
volume: 12.252366210937502, MSE: 0.0008556426037102938, quantized loss: 1.7028858565026894e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_427_beads_001_roi002_original_depth]|![JNet_427_beads_001_roi002_output_depth]|![JNet_427_beads_001_roi002_reconst_depth]|![JNet_427_beads_001_roi002_heatmap_depth]|
  
volume: 7.816500488281251, MSE: 0.0003962648333981633, quantized loss: 1.2904295545013156e-05  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_427_beads_001_roi003_original_depth]|![JNet_427_beads_001_roi003_output_depth]|![JNet_427_beads_001_roi003_reconst_depth]|![JNet_427_beads_001_roi003_heatmap_depth]|
  
volume: 12.742735351562503, MSE: 0.0005632371758110821, quantized loss: 1.9170654923073016e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_427_beads_001_roi004_original_depth]|![JNet_427_beads_001_roi004_output_depth]|![JNet_427_beads_001_roi004_reconst_depth]|![JNet_427_beads_001_roi004_heatmap_depth]|
  
volume: 8.282359375000002, MSE: 0.0003018586721736938, quantized loss: 1.2589493053383194e-05  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_427_beads_002_roi000_original_depth]|![JNet_427_beads_002_roi000_output_depth]|![JNet_427_beads_002_roi000_reconst_depth]|![JNet_427_beads_002_roi000_heatmap_depth]|
  
volume: 8.765501953125002, MSE: 0.0002767058613244444, quantized loss: 1.222074388351757e-05  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_427_beads_002_roi001_original_depth]|![JNet_427_beads_002_roi001_output_depth]|![JNet_427_beads_002_roi001_reconst_depth]|![JNet_427_beads_002_roi001_heatmap_depth]|
  
volume: 8.199420898437502, MSE: 0.00035039486829191446, quantized loss: 1.1762635949708056e-05  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_427_beads_002_roi002_original_depth]|![JNet_427_beads_002_roi002_output_depth]|![JNet_427_beads_002_roi002_reconst_depth]|![JNet_427_beads_002_roi002_heatmap_depth]|
  
volume: 8.452925781250002, MSE: 0.000310918694594875, quantized loss: 1.1916399671463296e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_427_psf_pre]|![JNet_427_psf_post]|

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
  



[JNet_426_pretrain_0_label_depth]: /experiments/images/JNet_426_pretrain_0_label_depth.png
[JNet_426_pretrain_0_label_plane]: /experiments/images/JNet_426_pretrain_0_label_plane.png
[JNet_426_pretrain_0_original_depth]: /experiments/images/JNet_426_pretrain_0_original_depth.png
[JNet_426_pretrain_0_original_plane]: /experiments/images/JNet_426_pretrain_0_original_plane.png
[JNet_426_pretrain_0_output_depth]: /experiments/images/JNet_426_pretrain_0_output_depth.png
[JNet_426_pretrain_0_output_plane]: /experiments/images/JNet_426_pretrain_0_output_plane.png
[JNet_426_pretrain_1_label_depth]: /experiments/images/JNet_426_pretrain_1_label_depth.png
[JNet_426_pretrain_1_label_plane]: /experiments/images/JNet_426_pretrain_1_label_plane.png
[JNet_426_pretrain_1_original_depth]: /experiments/images/JNet_426_pretrain_1_original_depth.png
[JNet_426_pretrain_1_original_plane]: /experiments/images/JNet_426_pretrain_1_original_plane.png
[JNet_426_pretrain_1_output_depth]: /experiments/images/JNet_426_pretrain_1_output_depth.png
[JNet_426_pretrain_1_output_plane]: /experiments/images/JNet_426_pretrain_1_output_plane.png
[JNet_426_pretrain_2_label_depth]: /experiments/images/JNet_426_pretrain_2_label_depth.png
[JNet_426_pretrain_2_label_plane]: /experiments/images/JNet_426_pretrain_2_label_plane.png
[JNet_426_pretrain_2_original_depth]: /experiments/images/JNet_426_pretrain_2_original_depth.png
[JNet_426_pretrain_2_original_plane]: /experiments/images/JNet_426_pretrain_2_original_plane.png
[JNet_426_pretrain_2_output_depth]: /experiments/images/JNet_426_pretrain_2_output_depth.png
[JNet_426_pretrain_2_output_plane]: /experiments/images/JNet_426_pretrain_2_output_plane.png
[JNet_426_pretrain_3_label_depth]: /experiments/images/JNet_426_pretrain_3_label_depth.png
[JNet_426_pretrain_3_label_plane]: /experiments/images/JNet_426_pretrain_3_label_plane.png
[JNet_426_pretrain_3_original_depth]: /experiments/images/JNet_426_pretrain_3_original_depth.png
[JNet_426_pretrain_3_original_plane]: /experiments/images/JNet_426_pretrain_3_original_plane.png
[JNet_426_pretrain_3_output_depth]: /experiments/images/JNet_426_pretrain_3_output_depth.png
[JNet_426_pretrain_3_output_plane]: /experiments/images/JNet_426_pretrain_3_output_plane.png
[JNet_426_pretrain_4_label_depth]: /experiments/images/JNet_426_pretrain_4_label_depth.png
[JNet_426_pretrain_4_label_plane]: /experiments/images/JNet_426_pretrain_4_label_plane.png
[JNet_426_pretrain_4_original_depth]: /experiments/images/JNet_426_pretrain_4_original_depth.png
[JNet_426_pretrain_4_original_plane]: /experiments/images/JNet_426_pretrain_4_original_plane.png
[JNet_426_pretrain_4_output_depth]: /experiments/images/JNet_426_pretrain_4_output_depth.png
[JNet_426_pretrain_4_output_plane]: /experiments/images/JNet_426_pretrain_4_output_plane.png
[JNet_426_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_426_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi000_original_depth.png
[JNet_426_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi000_output_depth.png
[JNet_426_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi000_reconst_depth.png
[JNet_426_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_426_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi001_original_depth.png
[JNet_426_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi001_output_depth.png
[JNet_426_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi001_reconst_depth.png
[JNet_426_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_426_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi002_original_depth.png
[JNet_426_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi002_output_depth.png
[JNet_426_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi002_reconst_depth.png
[JNet_426_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_426_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi003_original_depth.png
[JNet_426_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi003_output_depth.png
[JNet_426_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi003_reconst_depth.png
[JNet_426_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_426_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi004_original_depth.png
[JNet_426_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi004_output_depth.png
[JNet_426_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_426_pretrain_beads_001_roi004_reconst_depth.png
[JNet_426_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_426_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi000_original_depth.png
[JNet_426_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi000_output_depth.png
[JNet_426_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi000_reconst_depth.png
[JNet_426_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_426_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi001_original_depth.png
[JNet_426_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi001_output_depth.png
[JNet_426_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi001_reconst_depth.png
[JNet_426_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_426_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi002_original_depth.png
[JNet_426_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi002_output_depth.png
[JNet_426_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_426_pretrain_beads_002_roi002_reconst_depth.png
[JNet_427_0_label_depth]: /experiments/images/JNet_427_0_label_depth.png
[JNet_427_0_label_plane]: /experiments/images/JNet_427_0_label_plane.png
[JNet_427_0_original_depth]: /experiments/images/JNet_427_0_original_depth.png
[JNet_427_0_original_plane]: /experiments/images/JNet_427_0_original_plane.png
[JNet_427_0_output_depth]: /experiments/images/JNet_427_0_output_depth.png
[JNet_427_0_output_plane]: /experiments/images/JNet_427_0_output_plane.png
[JNet_427_1_label_depth]: /experiments/images/JNet_427_1_label_depth.png
[JNet_427_1_label_plane]: /experiments/images/JNet_427_1_label_plane.png
[JNet_427_1_original_depth]: /experiments/images/JNet_427_1_original_depth.png
[JNet_427_1_original_plane]: /experiments/images/JNet_427_1_original_plane.png
[JNet_427_1_output_depth]: /experiments/images/JNet_427_1_output_depth.png
[JNet_427_1_output_plane]: /experiments/images/JNet_427_1_output_plane.png
[JNet_427_2_label_depth]: /experiments/images/JNet_427_2_label_depth.png
[JNet_427_2_label_plane]: /experiments/images/JNet_427_2_label_plane.png
[JNet_427_2_original_depth]: /experiments/images/JNet_427_2_original_depth.png
[JNet_427_2_original_plane]: /experiments/images/JNet_427_2_original_plane.png
[JNet_427_2_output_depth]: /experiments/images/JNet_427_2_output_depth.png
[JNet_427_2_output_plane]: /experiments/images/JNet_427_2_output_plane.png
[JNet_427_3_label_depth]: /experiments/images/JNet_427_3_label_depth.png
[JNet_427_3_label_plane]: /experiments/images/JNet_427_3_label_plane.png
[JNet_427_3_original_depth]: /experiments/images/JNet_427_3_original_depth.png
[JNet_427_3_original_plane]: /experiments/images/JNet_427_3_original_plane.png
[JNet_427_3_output_depth]: /experiments/images/JNet_427_3_output_depth.png
[JNet_427_3_output_plane]: /experiments/images/JNet_427_3_output_plane.png
[JNet_427_4_label_depth]: /experiments/images/JNet_427_4_label_depth.png
[JNet_427_4_label_plane]: /experiments/images/JNet_427_4_label_plane.png
[JNet_427_4_original_depth]: /experiments/images/JNet_427_4_original_depth.png
[JNet_427_4_original_plane]: /experiments/images/JNet_427_4_original_plane.png
[JNet_427_4_output_depth]: /experiments/images/JNet_427_4_output_depth.png
[JNet_427_4_output_plane]: /experiments/images/JNet_427_4_output_plane.png
[JNet_427_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_427_beads_001_roi000_heatmap_depth.png
[JNet_427_beads_001_roi000_original_depth]: /experiments/images/JNet_427_beads_001_roi000_original_depth.png
[JNet_427_beads_001_roi000_output_depth]: /experiments/images/JNet_427_beads_001_roi000_output_depth.png
[JNet_427_beads_001_roi000_reconst_depth]: /experiments/images/JNet_427_beads_001_roi000_reconst_depth.png
[JNet_427_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_427_beads_001_roi001_heatmap_depth.png
[JNet_427_beads_001_roi001_original_depth]: /experiments/images/JNet_427_beads_001_roi001_original_depth.png
[JNet_427_beads_001_roi001_output_depth]: /experiments/images/JNet_427_beads_001_roi001_output_depth.png
[JNet_427_beads_001_roi001_reconst_depth]: /experiments/images/JNet_427_beads_001_roi001_reconst_depth.png
[JNet_427_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_427_beads_001_roi002_heatmap_depth.png
[JNet_427_beads_001_roi002_original_depth]: /experiments/images/JNet_427_beads_001_roi002_original_depth.png
[JNet_427_beads_001_roi002_output_depth]: /experiments/images/JNet_427_beads_001_roi002_output_depth.png
[JNet_427_beads_001_roi002_reconst_depth]: /experiments/images/JNet_427_beads_001_roi002_reconst_depth.png
[JNet_427_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_427_beads_001_roi003_heatmap_depth.png
[JNet_427_beads_001_roi003_original_depth]: /experiments/images/JNet_427_beads_001_roi003_original_depth.png
[JNet_427_beads_001_roi003_output_depth]: /experiments/images/JNet_427_beads_001_roi003_output_depth.png
[JNet_427_beads_001_roi003_reconst_depth]: /experiments/images/JNet_427_beads_001_roi003_reconst_depth.png
[JNet_427_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_427_beads_001_roi004_heatmap_depth.png
[JNet_427_beads_001_roi004_original_depth]: /experiments/images/JNet_427_beads_001_roi004_original_depth.png
[JNet_427_beads_001_roi004_output_depth]: /experiments/images/JNet_427_beads_001_roi004_output_depth.png
[JNet_427_beads_001_roi004_reconst_depth]: /experiments/images/JNet_427_beads_001_roi004_reconst_depth.png
[JNet_427_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_427_beads_002_roi000_heatmap_depth.png
[JNet_427_beads_002_roi000_original_depth]: /experiments/images/JNet_427_beads_002_roi000_original_depth.png
[JNet_427_beads_002_roi000_output_depth]: /experiments/images/JNet_427_beads_002_roi000_output_depth.png
[JNet_427_beads_002_roi000_reconst_depth]: /experiments/images/JNet_427_beads_002_roi000_reconst_depth.png
[JNet_427_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_427_beads_002_roi001_heatmap_depth.png
[JNet_427_beads_002_roi001_original_depth]: /experiments/images/JNet_427_beads_002_roi001_original_depth.png
[JNet_427_beads_002_roi001_output_depth]: /experiments/images/JNet_427_beads_002_roi001_output_depth.png
[JNet_427_beads_002_roi001_reconst_depth]: /experiments/images/JNet_427_beads_002_roi001_reconst_depth.png
[JNet_427_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_427_beads_002_roi002_heatmap_depth.png
[JNet_427_beads_002_roi002_original_depth]: /experiments/images/JNet_427_beads_002_roi002_original_depth.png
[JNet_427_beads_002_roi002_output_depth]: /experiments/images/JNet_427_beads_002_roi002_output_depth.png
[JNet_427_beads_002_roi002_reconst_depth]: /experiments/images/JNet_427_beads_002_roi002_reconst_depth.png
[JNet_427_psf_post]: /experiments/images/JNet_427_psf_post.png
[JNet_427_psf_pre]: /experiments/images/JNet_427_psf_pre.png
[finetuned]: /experiments/tmp/JNet_427_train.png
[pretrained_model]: /experiments/tmp/JNet_426_pretrain_train.png
