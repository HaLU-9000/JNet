



# JNet_470 Report
  
new data generation with more objects. axon deconv  
pretrained model : JNet_469_pretrain
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
|wavelength|2.0|microns|
|M|25|magnification|
|ns|1.4|specimen refractive index (RI)|
|ng0|1.5|coverslip RI design value|
|ng|1.5|coverslip RI experimental value|
|ni0|1.33|immersion medium RI design value|
|ni|1.33|immersion medium RI experimental value|
|ti0|150|microns, working distance (immersion medium thickness) design value|
|tg0|170|microns, coverslip thickness design value|
|tg|170|microns, coverslip thickness experimental value|
|res_lateral|0.16|microns|
|res_axial|1.0|microns|
|pZ|0|microns, particle distance from coverslip|
|bet_z|30.0||
|bet_xy|3.0||
|sig_eps|0.01||
|background|0.01||
|scale|6||
|mid|20|num of NeurIPSF middle channel|
|loss_fn|nn.MSELoss()|loss func for NeurIPSF|
|lr|0.01|lr for pre-training NeurIPSF|
|num_iter_psf_pretrain|1000|epoch for pre-training of NeurIPSF|
|device|cuda||

## Datasets and other training details

### simulation_data_generation

|Parameter|Value|
| :--- | :--- |
|dataset_name|_var_num_realisticdataset|
|train_num|16|
|valid_num|4|
|image_size|[1200, 500, 500]|
|train_object_num_min|2000|
|train_object_num_max|18000|
|valid_object_num_min|6000|
|valid_object_num_max|10000|

### pretrain_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_var_num_realisticdata|
|labelname|_label|
|size|[1200, 500, 500]|
|cropsize|[240, 112, 112]|
|I|200|
|low|0|
|high|16|
|scale|6|
|mask|True|
|mask_size|[1, 10, 10]|
|mask_num|30|
|surround|False|
|surround_size|[32, 4, 4]|

### pretrain_val_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_var_num_realisticdata|
|labelname|_label|
|size|[1200, 500, 500]|
|cropsize|[240, 112, 112]|
|I|20|
|low|16|
|high|20|
|scale|6|
|mask|False|
|mask_size|[1, 10, 10]|
|mask_num|False|
|surround|False|
|surround_size|[32, 4, 4]|
|seed|907|

### train_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_20231208_tsuji_beads_stackreged|
|size|[310, 512, 512]|
|cropsize|[240, 112, 112]|
|I|200|
|scale|10|
|train|True|
|mask|True|
|mask_size|[1, 10, 10]|
|mask_num|10|
|surround|False|
|surround_size|[32, 4, 4]|

### val_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_20231208_tsuji_beads_stackreged|
|size|[310, 512, 512]|
|cropsize|[240, 112, 112]|
|I|20|
|scale|10|
|train|False|
|mask|False|
|mask_size|[1, 10, 10]|
|mask_num|10|
|surround|False|
|surround_size|[32, 4, 4]|
|seed|1204|

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
|partial|None|
|ewc|ewc|
|params|params|
|es_patience|10|
|reconstruct|True|
|is_instantblur|False|
|is_vibrate|True|
|loss_weight|1|
|ewc_weight|100000|
|qloss_weight|1|
|ploss_weight|0.0|

## Results
  
mean MSE: 0.011957434937357903, mean BCE: 0.05360846593976021
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_0_original_plane]|![JNet_469_pretrain_0_output_plane]|![JNet_469_pretrain_0_label_plane]|
  
MSE: 0.013277714140713215, BCE: 0.060848601162433624  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_0_original_depth]|![JNet_469_pretrain_0_output_depth]|![JNet_469_pretrain_0_label_depth]|
  
MSE: 0.013277714140713215, BCE: 0.060848601162433624  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_1_original_plane]|![JNet_469_pretrain_1_output_plane]|![JNet_469_pretrain_1_label_plane]|
  
MSE: 0.01370929554104805, BCE: 0.059450432658195496  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_1_original_depth]|![JNet_469_pretrain_1_output_depth]|![JNet_469_pretrain_1_label_depth]|
  
MSE: 0.01370929554104805, BCE: 0.059450432658195496  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_2_original_plane]|![JNet_469_pretrain_2_output_plane]|![JNet_469_pretrain_2_label_plane]|
  
MSE: 0.013354506343603134, BCE: 0.05504806339740753  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_2_original_depth]|![JNet_469_pretrain_2_output_depth]|![JNet_469_pretrain_2_label_depth]|
  
MSE: 0.013354506343603134, BCE: 0.05504806339740753  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_3_original_plane]|![JNet_469_pretrain_3_output_plane]|![JNet_469_pretrain_3_label_plane]|
  
MSE: 0.007831933908164501, BCE: 0.03907131403684616  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_3_original_depth]|![JNet_469_pretrain_3_output_depth]|![JNet_469_pretrain_3_label_depth]|
  
MSE: 0.007831933908164501, BCE: 0.03907131403684616  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_4_original_plane]|![JNet_469_pretrain_4_output_plane]|![JNet_469_pretrain_4_label_plane]|
  
MSE: 0.011613722890615463, BCE: 0.053623925894498825  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_469_pretrain_4_original_depth]|![JNet_469_pretrain_4_output_depth]|![JNet_469_pretrain_4_label_depth]|
  
MSE: 0.011613722890615463, BCE: 0.053623925894498825  
  
mean MSE: 0.23564119637012482, mean BCE: 0.6644203066825867
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_0_original_plane]|![JNet_470_0_output_plane]|![JNet_470_0_label_plane]|
  
MSE: 0.23557491600513458, BCE: 0.664287269115448  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_0_original_depth]|![JNet_470_0_output_depth]|![JNet_470_0_label_depth]|
  
MSE: 0.23557491600513458, BCE: 0.664287269115448  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_1_original_plane]|![JNet_470_1_output_plane]|![JNet_470_1_label_plane]|
  
MSE: 0.23580025136470795, BCE: 0.6647390723228455  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_1_original_depth]|![JNet_470_1_output_depth]|![JNet_470_1_label_depth]|
  
MSE: 0.23580025136470795, BCE: 0.6647390723228455  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_2_original_plane]|![JNet_470_2_output_plane]|![JNet_470_2_label_plane]|
  
MSE: 0.23558524250984192, BCE: 0.6643079519271851  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_2_original_depth]|![JNet_470_2_output_depth]|![JNet_470_2_label_depth]|
  
MSE: 0.23558524250984192, BCE: 0.6643079519271851  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_3_original_plane]|![JNet_470_3_output_plane]|![JNet_470_3_label_plane]|
  
MSE: 0.23557204008102417, BCE: 0.664281964302063  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_3_original_depth]|![JNet_470_3_output_depth]|![JNet_470_3_label_depth]|
  
MSE: 0.23557204008102417, BCE: 0.664281964302063  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_4_original_plane]|![JNet_470_4_output_plane]|![JNet_470_4_label_plane]|
  
MSE: 0.2356734573841095, BCE: 0.664485514163971  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_470_4_original_depth]|![JNet_470_4_output_depth]|![JNet_470_4_label_depth]|
  
MSE: 0.2356734573841095, BCE: 0.664485514163971  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_001_roi000_original_depth]|![JNet_469_pretrain_beads_001_roi000_output_depth]|![JNet_469_pretrain_beads_001_roi000_reconst_depth]|![JNet_469_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 243.68760000000003, MSE: 0.002504714298993349, quantized loss: 0.0009659665520302951  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_001_roi001_original_depth]|![JNet_469_pretrain_beads_001_roi001_output_depth]|![JNet_469_pretrain_beads_001_roi001_reconst_depth]|![JNet_469_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 330.52281600000003, MSE: 0.00396127812564373, quantized loss: 0.0013532678131014109  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_001_roi002_original_depth]|![JNet_469_pretrain_beads_001_roi002_output_depth]|![JNet_469_pretrain_beads_001_roi002_reconst_depth]|![JNet_469_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 225.63308800000004, MSE: 0.002506928751245141, quantized loss: 0.0008721015183255076  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_001_roi003_original_depth]|![JNet_469_pretrain_beads_001_roi003_output_depth]|![JNet_469_pretrain_beads_001_roi003_reconst_depth]|![JNet_469_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 324.38643200000007, MSE: 0.003752098185941577, quantized loss: 0.001301943906582892  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_001_roi004_original_depth]|![JNet_469_pretrain_beads_001_roi004_output_depth]|![JNet_469_pretrain_beads_001_roi004_reconst_depth]|![JNet_469_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 240.32196800000003, MSE: 0.0030387069564312696, quantized loss: 0.0009117322042584419  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_002_roi000_original_depth]|![JNet_469_pretrain_beads_002_roi000_output_depth]|![JNet_469_pretrain_beads_002_roi000_reconst_depth]|![JNet_469_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 249.14320000000004, MSE: 0.0033665031660348177, quantized loss: 0.0009471529047004879  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_002_roi001_original_depth]|![JNet_469_pretrain_beads_002_roi001_output_depth]|![JNet_469_pretrain_beads_002_roi001_reconst_depth]|![JNet_469_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 216.36432000000005, MSE: 0.002638336271047592, quantized loss: 0.0008794142631813884  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_469_pretrain_beads_002_roi002_original_depth]|![JNet_469_pretrain_beads_002_roi002_output_depth]|![JNet_469_pretrain_beads_002_roi002_reconst_depth]|![JNet_469_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 224.18921600000004, MSE: 0.003014633432030678, quantized loss: 0.0008994840900413692  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_001_roi000_original_depth]|![JNet_470_beads_001_roi000_output_depth]|![JNet_470_beads_001_roi000_reconst_depth]|![JNet_470_beads_001_roi000_heatmap_depth]|
  
volume: 4827.315200000001, MSE: 0.0074240341782569885, quantized loss: 0.24761541187763214  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_001_roi001_original_depth]|![JNet_470_beads_001_roi001_output_depth]|![JNet_470_beads_001_roi001_reconst_depth]|![JNet_470_beads_001_roi001_heatmap_depth]|
  
volume: 4831.522304000001, MSE: 0.012107712216675282, quantized loss: 0.24735896289348602  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_001_roi002_original_depth]|![JNet_470_beads_001_roi002_output_depth]|![JNet_470_beads_001_roi002_reconst_depth]|![JNet_470_beads_001_roi002_heatmap_depth]|
  
volume: 4825.800192000001, MSE: 0.007352085318416357, quantized loss: 0.24755896627902985  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_001_roi003_original_depth]|![JNet_470_beads_001_roi003_output_depth]|![JNet_470_beads_001_roi003_reconst_depth]|![JNet_470_beads_001_roi003_heatmap_depth]|
  
volume: 4835.150336000001, MSE: 0.012862059287726879, quantized loss: 0.24704302847385406  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_001_roi004_original_depth]|![JNet_470_beads_001_roi004_output_depth]|![JNet_470_beads_001_roi004_reconst_depth]|![JNet_470_beads_001_roi004_heatmap_depth]|
  
volume: 4827.08736, MSE: 0.008899264968931675, quantized loss: 0.24747636914253235  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_002_roi000_original_depth]|![JNet_470_beads_002_roi000_output_depth]|![JNet_470_beads_002_roi000_reconst_depth]|![JNet_470_beads_002_roi000_heatmap_depth]|
  
volume: 4827.873792000001, MSE: 0.009970004670321941, quantized loss: 0.2473992258310318  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_002_roi001_original_depth]|![JNet_470_beads_002_roi001_output_depth]|![JNet_470_beads_002_roi001_reconst_depth]|![JNet_470_beads_002_roi001_heatmap_depth]|
  
volume: 4826.656768000001, MSE: 0.008358863182365894, quantized loss: 0.2474968433380127  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_470_beads_002_roi002_original_depth]|![JNet_470_beads_002_roi002_output_depth]|![JNet_470_beads_002_roi002_reconst_depth]|![JNet_470_beads_002_roi002_heatmap_depth]|
  
volume: 4827.219456000001, MSE: 0.008957600221037865, quantized loss: 0.24746070802211761  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_470_psf_pre]|![JNet_470_psf_post]|

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
    (blur): Blur(  
      (neuripsf): NeuralImplicitPSF(  
        (layers): Sequential(  
          (0): BatchNorm1d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
          (1): Linear(in_features=2, out_features=20, bias=True)  
          (2): Sigmoid()  
          (3): BatchNorm1d(20, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)  
          (4): Linear(in_features=20, out_features=1, bias=True)  
          (5): Sigmoid()  
        )  
      )  
    )  
    (noise): Noise()  
    (preprocess): PreProcess()  
  )  
  (upsample): JNetUpsample(  
    (upsample): Upsample(scale_factor=(6.0, 1.0, 1.0), mode='trilinear')  
  )  
  (vq): VectorQuantizer()  
)  
```  
  



[JNet_469_pretrain_0_label_depth]: /experiments/images/JNet_469_pretrain_0_label_depth.png
[JNet_469_pretrain_0_label_plane]: /experiments/images/JNet_469_pretrain_0_label_plane.png
[JNet_469_pretrain_0_original_depth]: /experiments/images/JNet_469_pretrain_0_original_depth.png
[JNet_469_pretrain_0_original_plane]: /experiments/images/JNet_469_pretrain_0_original_plane.png
[JNet_469_pretrain_0_output_depth]: /experiments/images/JNet_469_pretrain_0_output_depth.png
[JNet_469_pretrain_0_output_plane]: /experiments/images/JNet_469_pretrain_0_output_plane.png
[JNet_469_pretrain_1_label_depth]: /experiments/images/JNet_469_pretrain_1_label_depth.png
[JNet_469_pretrain_1_label_plane]: /experiments/images/JNet_469_pretrain_1_label_plane.png
[JNet_469_pretrain_1_original_depth]: /experiments/images/JNet_469_pretrain_1_original_depth.png
[JNet_469_pretrain_1_original_plane]: /experiments/images/JNet_469_pretrain_1_original_plane.png
[JNet_469_pretrain_1_output_depth]: /experiments/images/JNet_469_pretrain_1_output_depth.png
[JNet_469_pretrain_1_output_plane]: /experiments/images/JNet_469_pretrain_1_output_plane.png
[JNet_469_pretrain_2_label_depth]: /experiments/images/JNet_469_pretrain_2_label_depth.png
[JNet_469_pretrain_2_label_plane]: /experiments/images/JNet_469_pretrain_2_label_plane.png
[JNet_469_pretrain_2_original_depth]: /experiments/images/JNet_469_pretrain_2_original_depth.png
[JNet_469_pretrain_2_original_plane]: /experiments/images/JNet_469_pretrain_2_original_plane.png
[JNet_469_pretrain_2_output_depth]: /experiments/images/JNet_469_pretrain_2_output_depth.png
[JNet_469_pretrain_2_output_plane]: /experiments/images/JNet_469_pretrain_2_output_plane.png
[JNet_469_pretrain_3_label_depth]: /experiments/images/JNet_469_pretrain_3_label_depth.png
[JNet_469_pretrain_3_label_plane]: /experiments/images/JNet_469_pretrain_3_label_plane.png
[JNet_469_pretrain_3_original_depth]: /experiments/images/JNet_469_pretrain_3_original_depth.png
[JNet_469_pretrain_3_original_plane]: /experiments/images/JNet_469_pretrain_3_original_plane.png
[JNet_469_pretrain_3_output_depth]: /experiments/images/JNet_469_pretrain_3_output_depth.png
[JNet_469_pretrain_3_output_plane]: /experiments/images/JNet_469_pretrain_3_output_plane.png
[JNet_469_pretrain_4_label_depth]: /experiments/images/JNet_469_pretrain_4_label_depth.png
[JNet_469_pretrain_4_label_plane]: /experiments/images/JNet_469_pretrain_4_label_plane.png
[JNet_469_pretrain_4_original_depth]: /experiments/images/JNet_469_pretrain_4_original_depth.png
[JNet_469_pretrain_4_original_plane]: /experiments/images/JNet_469_pretrain_4_original_plane.png
[JNet_469_pretrain_4_output_depth]: /experiments/images/JNet_469_pretrain_4_output_depth.png
[JNet_469_pretrain_4_output_plane]: /experiments/images/JNet_469_pretrain_4_output_plane.png
[JNet_469_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_469_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi000_original_depth.png
[JNet_469_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi000_output_depth.png
[JNet_469_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi000_reconst_depth.png
[JNet_469_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_469_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi001_original_depth.png
[JNet_469_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi001_output_depth.png
[JNet_469_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi001_reconst_depth.png
[JNet_469_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_469_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi002_original_depth.png
[JNet_469_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi002_output_depth.png
[JNet_469_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi002_reconst_depth.png
[JNet_469_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_469_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi003_original_depth.png
[JNet_469_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi003_output_depth.png
[JNet_469_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi003_reconst_depth.png
[JNet_469_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_469_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi004_original_depth.png
[JNet_469_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi004_output_depth.png
[JNet_469_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_469_pretrain_beads_001_roi004_reconst_depth.png
[JNet_469_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_469_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi000_original_depth.png
[JNet_469_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi000_output_depth.png
[JNet_469_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi000_reconst_depth.png
[JNet_469_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_469_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi001_original_depth.png
[JNet_469_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi001_output_depth.png
[JNet_469_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi001_reconst_depth.png
[JNet_469_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_469_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi002_original_depth.png
[JNet_469_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi002_output_depth.png
[JNet_469_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_469_pretrain_beads_002_roi002_reconst_depth.png
[JNet_470_0_label_depth]: /experiments/images/JNet_470_0_label_depth.png
[JNet_470_0_label_plane]: /experiments/images/JNet_470_0_label_plane.png
[JNet_470_0_original_depth]: /experiments/images/JNet_470_0_original_depth.png
[JNet_470_0_original_plane]: /experiments/images/JNet_470_0_original_plane.png
[JNet_470_0_output_depth]: /experiments/images/JNet_470_0_output_depth.png
[JNet_470_0_output_plane]: /experiments/images/JNet_470_0_output_plane.png
[JNet_470_1_label_depth]: /experiments/images/JNet_470_1_label_depth.png
[JNet_470_1_label_plane]: /experiments/images/JNet_470_1_label_plane.png
[JNet_470_1_original_depth]: /experiments/images/JNet_470_1_original_depth.png
[JNet_470_1_original_plane]: /experiments/images/JNet_470_1_original_plane.png
[JNet_470_1_output_depth]: /experiments/images/JNet_470_1_output_depth.png
[JNet_470_1_output_plane]: /experiments/images/JNet_470_1_output_plane.png
[JNet_470_2_label_depth]: /experiments/images/JNet_470_2_label_depth.png
[JNet_470_2_label_plane]: /experiments/images/JNet_470_2_label_plane.png
[JNet_470_2_original_depth]: /experiments/images/JNet_470_2_original_depth.png
[JNet_470_2_original_plane]: /experiments/images/JNet_470_2_original_plane.png
[JNet_470_2_output_depth]: /experiments/images/JNet_470_2_output_depth.png
[JNet_470_2_output_plane]: /experiments/images/JNet_470_2_output_plane.png
[JNet_470_3_label_depth]: /experiments/images/JNet_470_3_label_depth.png
[JNet_470_3_label_plane]: /experiments/images/JNet_470_3_label_plane.png
[JNet_470_3_original_depth]: /experiments/images/JNet_470_3_original_depth.png
[JNet_470_3_original_plane]: /experiments/images/JNet_470_3_original_plane.png
[JNet_470_3_output_depth]: /experiments/images/JNet_470_3_output_depth.png
[JNet_470_3_output_plane]: /experiments/images/JNet_470_3_output_plane.png
[JNet_470_4_label_depth]: /experiments/images/JNet_470_4_label_depth.png
[JNet_470_4_label_plane]: /experiments/images/JNet_470_4_label_plane.png
[JNet_470_4_original_depth]: /experiments/images/JNet_470_4_original_depth.png
[JNet_470_4_original_plane]: /experiments/images/JNet_470_4_original_plane.png
[JNet_470_4_output_depth]: /experiments/images/JNet_470_4_output_depth.png
[JNet_470_4_output_plane]: /experiments/images/JNet_470_4_output_plane.png
[JNet_470_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_470_beads_001_roi000_heatmap_depth.png
[JNet_470_beads_001_roi000_original_depth]: /experiments/images/JNet_470_beads_001_roi000_original_depth.png
[JNet_470_beads_001_roi000_output_depth]: /experiments/images/JNet_470_beads_001_roi000_output_depth.png
[JNet_470_beads_001_roi000_reconst_depth]: /experiments/images/JNet_470_beads_001_roi000_reconst_depth.png
[JNet_470_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_470_beads_001_roi001_heatmap_depth.png
[JNet_470_beads_001_roi001_original_depth]: /experiments/images/JNet_470_beads_001_roi001_original_depth.png
[JNet_470_beads_001_roi001_output_depth]: /experiments/images/JNet_470_beads_001_roi001_output_depth.png
[JNet_470_beads_001_roi001_reconst_depth]: /experiments/images/JNet_470_beads_001_roi001_reconst_depth.png
[JNet_470_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_470_beads_001_roi002_heatmap_depth.png
[JNet_470_beads_001_roi002_original_depth]: /experiments/images/JNet_470_beads_001_roi002_original_depth.png
[JNet_470_beads_001_roi002_output_depth]: /experiments/images/JNet_470_beads_001_roi002_output_depth.png
[JNet_470_beads_001_roi002_reconst_depth]: /experiments/images/JNet_470_beads_001_roi002_reconst_depth.png
[JNet_470_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_470_beads_001_roi003_heatmap_depth.png
[JNet_470_beads_001_roi003_original_depth]: /experiments/images/JNet_470_beads_001_roi003_original_depth.png
[JNet_470_beads_001_roi003_output_depth]: /experiments/images/JNet_470_beads_001_roi003_output_depth.png
[JNet_470_beads_001_roi003_reconst_depth]: /experiments/images/JNet_470_beads_001_roi003_reconst_depth.png
[JNet_470_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_470_beads_001_roi004_heatmap_depth.png
[JNet_470_beads_001_roi004_original_depth]: /experiments/images/JNet_470_beads_001_roi004_original_depth.png
[JNet_470_beads_001_roi004_output_depth]: /experiments/images/JNet_470_beads_001_roi004_output_depth.png
[JNet_470_beads_001_roi004_reconst_depth]: /experiments/images/JNet_470_beads_001_roi004_reconst_depth.png
[JNet_470_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_470_beads_002_roi000_heatmap_depth.png
[JNet_470_beads_002_roi000_original_depth]: /experiments/images/JNet_470_beads_002_roi000_original_depth.png
[JNet_470_beads_002_roi000_output_depth]: /experiments/images/JNet_470_beads_002_roi000_output_depth.png
[JNet_470_beads_002_roi000_reconst_depth]: /experiments/images/JNet_470_beads_002_roi000_reconst_depth.png
[JNet_470_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_470_beads_002_roi001_heatmap_depth.png
[JNet_470_beads_002_roi001_original_depth]: /experiments/images/JNet_470_beads_002_roi001_original_depth.png
[JNet_470_beads_002_roi001_output_depth]: /experiments/images/JNet_470_beads_002_roi001_output_depth.png
[JNet_470_beads_002_roi001_reconst_depth]: /experiments/images/JNet_470_beads_002_roi001_reconst_depth.png
[JNet_470_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_470_beads_002_roi002_heatmap_depth.png
[JNet_470_beads_002_roi002_original_depth]: /experiments/images/JNet_470_beads_002_roi002_original_depth.png
[JNet_470_beads_002_roi002_output_depth]: /experiments/images/JNet_470_beads_002_roi002_output_depth.png
[JNet_470_beads_002_roi002_reconst_depth]: /experiments/images/JNet_470_beads_002_roi002_reconst_depth.png
[JNet_470_psf_post]: /experiments/images/JNet_470_psf_post.png
[JNet_470_psf_pre]: /experiments/images/JNet_470_psf_pre.png
