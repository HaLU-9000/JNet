



# JNet_419 Report
  
the parameters to replicate the results of JNet_419. nearest interp of PSF, logit loss = 1.0, NA = 1.0 vq loss 1  
pretrained model : JNet_417_pretrain
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
|NA|1.0||
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
|qloss_weight|1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.02492424286901951, mean BCE: 0.09433283656835556
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_0_original_plane]|![JNet_417_pretrain_0_output_plane]|![JNet_417_pretrain_0_label_plane]|
  
MSE: 0.02248373255133629, BCE: 0.07394609600305557  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_0_original_depth]|![JNet_417_pretrain_0_output_depth]|![JNet_417_pretrain_0_label_depth]|
  
MSE: 0.02248373255133629, BCE: 0.07394609600305557  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_1_original_plane]|![JNet_417_pretrain_1_output_plane]|![JNet_417_pretrain_1_label_plane]|
  
MSE: 0.025068413466215134, BCE: 0.1204979196190834  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_1_original_depth]|![JNet_417_pretrain_1_output_depth]|![JNet_417_pretrain_1_label_depth]|
  
MSE: 0.025068413466215134, BCE: 0.1204979196190834  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_2_original_plane]|![JNet_417_pretrain_2_output_plane]|![JNet_417_pretrain_2_label_plane]|
  
MSE: 0.03160369396209717, BCE: 0.10752838850021362  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_2_original_depth]|![JNet_417_pretrain_2_output_depth]|![JNet_417_pretrain_2_label_depth]|
  
MSE: 0.03160369396209717, BCE: 0.10752838850021362  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_3_original_plane]|![JNet_417_pretrain_3_output_plane]|![JNet_417_pretrain_3_label_plane]|
  
MSE: 0.02561255544424057, BCE: 0.10247237980365753  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_3_original_depth]|![JNet_417_pretrain_3_output_depth]|![JNet_417_pretrain_3_label_depth]|
  
MSE: 0.02561255544424057, BCE: 0.10247237980365753  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_4_original_plane]|![JNet_417_pretrain_4_output_plane]|![JNet_417_pretrain_4_label_plane]|
  
MSE: 0.019852815195918083, BCE: 0.06721943616867065  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_4_original_depth]|![JNet_417_pretrain_4_output_depth]|![JNet_417_pretrain_4_label_depth]|
  
MSE: 0.019852815195918083, BCE: 0.06721943616867065  
  
mean MSE: 0.028363382443785667, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_419_0_original_plane]|![JNet_419_0_output_plane]|![JNet_419_0_label_plane]|
  
MSE: 0.023195821791887283, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_419_0_original_depth]|![JNet_419_0_output_depth]|![JNet_419_0_label_depth]|
  
MSE: 0.023195821791887283, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_419_1_original_plane]|![JNet_419_1_output_plane]|![JNet_419_1_label_plane]|
  
MSE: 0.030582474544644356, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_419_1_original_depth]|![JNet_419_1_output_depth]|![JNet_419_1_label_depth]|
  
MSE: 0.030582474544644356, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_419_2_original_plane]|![JNet_419_2_output_plane]|![JNet_419_2_label_plane]|
  
MSE: 0.025730272755026817, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_419_2_original_depth]|![JNet_419_2_output_depth]|![JNet_419_2_label_depth]|
  
MSE: 0.025730272755026817, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_419_3_original_plane]|![JNet_419_3_output_plane]|![JNet_419_3_label_plane]|
  
MSE: 0.033053528517484665, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_419_3_original_depth]|![JNet_419_3_output_depth]|![JNet_419_3_label_depth]|
  
MSE: 0.033053528517484665, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_419_4_original_plane]|![JNet_419_4_output_plane]|![JNet_419_4_label_plane]|
  
MSE: 0.02925480343401432, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_419_4_original_depth]|![JNet_419_4_output_depth]|![JNet_419_4_label_depth]|
  
MSE: 0.02925480343401432, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_417_pretrain_beads_001_roi000_original_depth]|![JNet_417_pretrain_beads_001_roi000_output_depth]|![JNet_417_pretrain_beads_001_roi000_reconst_depth]|![JNet_417_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 1.1966582031250004, MSE: 0.002986322855576873, quantized loss: 0.00019565914408303797  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_417_pretrain_beads_001_roi001_original_depth]|![JNet_417_pretrain_beads_001_roi001_output_depth]|![JNet_417_pretrain_beads_001_roi001_reconst_depth]|![JNet_417_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 1.5826655273437504, MSE: 0.004666139837354422, quantized loss: 0.0003301805118098855  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_417_pretrain_beads_001_roi002_original_depth]|![JNet_417_pretrain_beads_001_roi002_output_depth]|![JNet_417_pretrain_beads_001_roi002_reconst_depth]|![JNet_417_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 1.2260853271484378, MSE: 0.0030646109953522682, quantized loss: 0.00015827635070309043  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_417_pretrain_beads_001_roi003_original_depth]|![JNet_417_pretrain_beads_001_roi003_output_depth]|![JNet_417_pretrain_beads_001_roi003_reconst_depth]|![JNet_417_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 1.7808935546875004, MSE: 0.0045853364281356335, quantized loss: 0.00038032064912840724  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_417_pretrain_beads_001_roi004_original_depth]|![JNet_417_pretrain_beads_001_roi004_output_depth]|![JNet_417_pretrain_beads_001_roi004_reconst_depth]|![JNet_417_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 1.472713989257813, MSE: 0.0026569163892418146, quantized loss: 0.00029212815570645034  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_417_pretrain_beads_002_roi000_original_depth]|![JNet_417_pretrain_beads_002_roi000_output_depth]|![JNet_417_pretrain_beads_002_roi000_reconst_depth]|![JNet_417_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 1.5913211669921878, MSE: 0.00257579842582345, quantized loss: 0.0003599865303840488  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_417_pretrain_beads_002_roi001_original_depth]|![JNet_417_pretrain_beads_002_roi001_output_depth]|![JNet_417_pretrain_beads_002_roi001_reconst_depth]|![JNet_417_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 1.4681284179687504, MSE: 0.0020705151837319136, quantized loss: 0.0002784362295642495  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_417_pretrain_beads_002_roi002_original_depth]|![JNet_417_pretrain_beads_002_roi002_output_depth]|![JNet_417_pretrain_beads_002_roi002_reconst_depth]|![JNet_417_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 1.4518350830078128, MSE: 0.002675156807526946, quantized loss: 0.00027541190502233803  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_419_beads_001_roi000_original_depth]|![JNet_419_beads_001_roi000_output_depth]|![JNet_419_beads_001_roi000_reconst_depth]|![JNet_419_beads_001_roi000_heatmap_depth]|
  
volume: 0.6880450439453126, MSE: 0.0013851335970684886, quantized loss: 5.745107773691416e-05  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_419_beads_001_roi001_original_depth]|![JNet_419_beads_001_roi001_output_depth]|![JNet_419_beads_001_roi001_reconst_depth]|![JNet_419_beads_001_roi001_heatmap_depth]|
  
volume: 1.2480209960937503, MSE: 0.0019495951710268855, quantized loss: 8.57870327308774e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_419_beads_001_roi002_original_depth]|![JNet_419_beads_001_roi002_output_depth]|![JNet_419_beads_001_roi002_reconst_depth]|![JNet_419_beads_001_roi002_heatmap_depth]|
  
volume: 0.7765664062500002, MSE: 0.0008736758027225733, quantized loss: 6.675480108242482e-05  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_419_beads_001_roi003_original_depth]|![JNet_419_beads_001_roi003_output_depth]|![JNet_419_beads_001_roi003_reconst_depth]|![JNet_419_beads_001_roi003_heatmap_depth]|
  
volume: 1.3964343261718752, MSE: 0.001470472663640976, quantized loss: 8.307646930916235e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_419_beads_001_roi004_original_depth]|![JNet_419_beads_001_roi004_output_depth]|![JNet_419_beads_001_roi004_reconst_depth]|![JNet_419_beads_001_roi004_heatmap_depth]|
  
volume: 0.873725280761719, MSE: 0.0009963986231014132, quantized loss: 4.9937058065552264e-05  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_419_beads_002_roi000_original_depth]|![JNet_419_beads_002_roi000_output_depth]|![JNet_419_beads_002_roi000_reconst_depth]|![JNet_419_beads_002_roi000_heatmap_depth]|
  
volume: 0.9783853149414065, MSE: 0.0009347745217382908, quantized loss: 4.998343501938507e-05  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_419_beads_002_roi001_original_depth]|![JNet_419_beads_002_roi001_output_depth]|![JNet_419_beads_002_roi001_reconst_depth]|![JNet_419_beads_002_roi001_heatmap_depth]|
  
volume: 0.8338186645507815, MSE: 0.0009765629656612873, quantized loss: 5.297327879816294e-05  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_419_beads_002_roi002_original_depth]|![JNet_419_beads_002_roi002_output_depth]|![JNet_419_beads_002_roi002_reconst_depth]|![JNet_419_beads_002_roi002_heatmap_depth]|
  
volume: 0.8419937133789065, MSE: 0.0012175814481452107, quantized loss: 5.486982627189718e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_419_psf_pre]|![JNet_419_psf_post]|

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
  



[JNet_417_pretrain_0_label_depth]: /experiments/images/JNet_417_pretrain_0_label_depth.png
[JNet_417_pretrain_0_label_plane]: /experiments/images/JNet_417_pretrain_0_label_plane.png
[JNet_417_pretrain_0_original_depth]: /experiments/images/JNet_417_pretrain_0_original_depth.png
[JNet_417_pretrain_0_original_plane]: /experiments/images/JNet_417_pretrain_0_original_plane.png
[JNet_417_pretrain_0_output_depth]: /experiments/images/JNet_417_pretrain_0_output_depth.png
[JNet_417_pretrain_0_output_plane]: /experiments/images/JNet_417_pretrain_0_output_plane.png
[JNet_417_pretrain_1_label_depth]: /experiments/images/JNet_417_pretrain_1_label_depth.png
[JNet_417_pretrain_1_label_plane]: /experiments/images/JNet_417_pretrain_1_label_plane.png
[JNet_417_pretrain_1_original_depth]: /experiments/images/JNet_417_pretrain_1_original_depth.png
[JNet_417_pretrain_1_original_plane]: /experiments/images/JNet_417_pretrain_1_original_plane.png
[JNet_417_pretrain_1_output_depth]: /experiments/images/JNet_417_pretrain_1_output_depth.png
[JNet_417_pretrain_1_output_plane]: /experiments/images/JNet_417_pretrain_1_output_plane.png
[JNet_417_pretrain_2_label_depth]: /experiments/images/JNet_417_pretrain_2_label_depth.png
[JNet_417_pretrain_2_label_plane]: /experiments/images/JNet_417_pretrain_2_label_plane.png
[JNet_417_pretrain_2_original_depth]: /experiments/images/JNet_417_pretrain_2_original_depth.png
[JNet_417_pretrain_2_original_plane]: /experiments/images/JNet_417_pretrain_2_original_plane.png
[JNet_417_pretrain_2_output_depth]: /experiments/images/JNet_417_pretrain_2_output_depth.png
[JNet_417_pretrain_2_output_plane]: /experiments/images/JNet_417_pretrain_2_output_plane.png
[JNet_417_pretrain_3_label_depth]: /experiments/images/JNet_417_pretrain_3_label_depth.png
[JNet_417_pretrain_3_label_plane]: /experiments/images/JNet_417_pretrain_3_label_plane.png
[JNet_417_pretrain_3_original_depth]: /experiments/images/JNet_417_pretrain_3_original_depth.png
[JNet_417_pretrain_3_original_plane]: /experiments/images/JNet_417_pretrain_3_original_plane.png
[JNet_417_pretrain_3_output_depth]: /experiments/images/JNet_417_pretrain_3_output_depth.png
[JNet_417_pretrain_3_output_plane]: /experiments/images/JNet_417_pretrain_3_output_plane.png
[JNet_417_pretrain_4_label_depth]: /experiments/images/JNet_417_pretrain_4_label_depth.png
[JNet_417_pretrain_4_label_plane]: /experiments/images/JNet_417_pretrain_4_label_plane.png
[JNet_417_pretrain_4_original_depth]: /experiments/images/JNet_417_pretrain_4_original_depth.png
[JNet_417_pretrain_4_original_plane]: /experiments/images/JNet_417_pretrain_4_original_plane.png
[JNet_417_pretrain_4_output_depth]: /experiments/images/JNet_417_pretrain_4_output_depth.png
[JNet_417_pretrain_4_output_plane]: /experiments/images/JNet_417_pretrain_4_output_plane.png
[JNet_417_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_417_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi000_original_depth.png
[JNet_417_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi000_output_depth.png
[JNet_417_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi000_reconst_depth.png
[JNet_417_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_417_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi001_original_depth.png
[JNet_417_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi001_output_depth.png
[JNet_417_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi001_reconst_depth.png
[JNet_417_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_417_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi002_original_depth.png
[JNet_417_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi002_output_depth.png
[JNet_417_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi002_reconst_depth.png
[JNet_417_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_417_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi003_original_depth.png
[JNet_417_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi003_output_depth.png
[JNet_417_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi003_reconst_depth.png
[JNet_417_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_417_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi004_original_depth.png
[JNet_417_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi004_output_depth.png
[JNet_417_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_417_pretrain_beads_001_roi004_reconst_depth.png
[JNet_417_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_417_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi000_original_depth.png
[JNet_417_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi000_output_depth.png
[JNet_417_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi000_reconst_depth.png
[JNet_417_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_417_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi001_original_depth.png
[JNet_417_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi001_output_depth.png
[JNet_417_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi001_reconst_depth.png
[JNet_417_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_417_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi002_original_depth.png
[JNet_417_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi002_output_depth.png
[JNet_417_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_417_pretrain_beads_002_roi002_reconst_depth.png
[JNet_419_0_label_depth]: /experiments/images/JNet_419_0_label_depth.png
[JNet_419_0_label_plane]: /experiments/images/JNet_419_0_label_plane.png
[JNet_419_0_original_depth]: /experiments/images/JNet_419_0_original_depth.png
[JNet_419_0_original_plane]: /experiments/images/JNet_419_0_original_plane.png
[JNet_419_0_output_depth]: /experiments/images/JNet_419_0_output_depth.png
[JNet_419_0_output_plane]: /experiments/images/JNet_419_0_output_plane.png
[JNet_419_1_label_depth]: /experiments/images/JNet_419_1_label_depth.png
[JNet_419_1_label_plane]: /experiments/images/JNet_419_1_label_plane.png
[JNet_419_1_original_depth]: /experiments/images/JNet_419_1_original_depth.png
[JNet_419_1_original_plane]: /experiments/images/JNet_419_1_original_plane.png
[JNet_419_1_output_depth]: /experiments/images/JNet_419_1_output_depth.png
[JNet_419_1_output_plane]: /experiments/images/JNet_419_1_output_plane.png
[JNet_419_2_label_depth]: /experiments/images/JNet_419_2_label_depth.png
[JNet_419_2_label_plane]: /experiments/images/JNet_419_2_label_plane.png
[JNet_419_2_original_depth]: /experiments/images/JNet_419_2_original_depth.png
[JNet_419_2_original_plane]: /experiments/images/JNet_419_2_original_plane.png
[JNet_419_2_output_depth]: /experiments/images/JNet_419_2_output_depth.png
[JNet_419_2_output_plane]: /experiments/images/JNet_419_2_output_plane.png
[JNet_419_3_label_depth]: /experiments/images/JNet_419_3_label_depth.png
[JNet_419_3_label_plane]: /experiments/images/JNet_419_3_label_plane.png
[JNet_419_3_original_depth]: /experiments/images/JNet_419_3_original_depth.png
[JNet_419_3_original_plane]: /experiments/images/JNet_419_3_original_plane.png
[JNet_419_3_output_depth]: /experiments/images/JNet_419_3_output_depth.png
[JNet_419_3_output_plane]: /experiments/images/JNet_419_3_output_plane.png
[JNet_419_4_label_depth]: /experiments/images/JNet_419_4_label_depth.png
[JNet_419_4_label_plane]: /experiments/images/JNet_419_4_label_plane.png
[JNet_419_4_original_depth]: /experiments/images/JNet_419_4_original_depth.png
[JNet_419_4_original_plane]: /experiments/images/JNet_419_4_original_plane.png
[JNet_419_4_output_depth]: /experiments/images/JNet_419_4_output_depth.png
[JNet_419_4_output_plane]: /experiments/images/JNet_419_4_output_plane.png
[JNet_419_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_419_beads_001_roi000_heatmap_depth.png
[JNet_419_beads_001_roi000_original_depth]: /experiments/images/JNet_419_beads_001_roi000_original_depth.png
[JNet_419_beads_001_roi000_output_depth]: /experiments/images/JNet_419_beads_001_roi000_output_depth.png
[JNet_419_beads_001_roi000_reconst_depth]: /experiments/images/JNet_419_beads_001_roi000_reconst_depth.png
[JNet_419_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_419_beads_001_roi001_heatmap_depth.png
[JNet_419_beads_001_roi001_original_depth]: /experiments/images/JNet_419_beads_001_roi001_original_depth.png
[JNet_419_beads_001_roi001_output_depth]: /experiments/images/JNet_419_beads_001_roi001_output_depth.png
[JNet_419_beads_001_roi001_reconst_depth]: /experiments/images/JNet_419_beads_001_roi001_reconst_depth.png
[JNet_419_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_419_beads_001_roi002_heatmap_depth.png
[JNet_419_beads_001_roi002_original_depth]: /experiments/images/JNet_419_beads_001_roi002_original_depth.png
[JNet_419_beads_001_roi002_output_depth]: /experiments/images/JNet_419_beads_001_roi002_output_depth.png
[JNet_419_beads_001_roi002_reconst_depth]: /experiments/images/JNet_419_beads_001_roi002_reconst_depth.png
[JNet_419_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_419_beads_001_roi003_heatmap_depth.png
[JNet_419_beads_001_roi003_original_depth]: /experiments/images/JNet_419_beads_001_roi003_original_depth.png
[JNet_419_beads_001_roi003_output_depth]: /experiments/images/JNet_419_beads_001_roi003_output_depth.png
[JNet_419_beads_001_roi003_reconst_depth]: /experiments/images/JNet_419_beads_001_roi003_reconst_depth.png
[JNet_419_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_419_beads_001_roi004_heatmap_depth.png
[JNet_419_beads_001_roi004_original_depth]: /experiments/images/JNet_419_beads_001_roi004_original_depth.png
[JNet_419_beads_001_roi004_output_depth]: /experiments/images/JNet_419_beads_001_roi004_output_depth.png
[JNet_419_beads_001_roi004_reconst_depth]: /experiments/images/JNet_419_beads_001_roi004_reconst_depth.png
[JNet_419_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_419_beads_002_roi000_heatmap_depth.png
[JNet_419_beads_002_roi000_original_depth]: /experiments/images/JNet_419_beads_002_roi000_original_depth.png
[JNet_419_beads_002_roi000_output_depth]: /experiments/images/JNet_419_beads_002_roi000_output_depth.png
[JNet_419_beads_002_roi000_reconst_depth]: /experiments/images/JNet_419_beads_002_roi000_reconst_depth.png
[JNet_419_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_419_beads_002_roi001_heatmap_depth.png
[JNet_419_beads_002_roi001_original_depth]: /experiments/images/JNet_419_beads_002_roi001_original_depth.png
[JNet_419_beads_002_roi001_output_depth]: /experiments/images/JNet_419_beads_002_roi001_output_depth.png
[JNet_419_beads_002_roi001_reconst_depth]: /experiments/images/JNet_419_beads_002_roi001_reconst_depth.png
[JNet_419_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_419_beads_002_roi002_heatmap_depth.png
[JNet_419_beads_002_roi002_original_depth]: /experiments/images/JNet_419_beads_002_roi002_original_depth.png
[JNet_419_beads_002_roi002_output_depth]: /experiments/images/JNet_419_beads_002_roi002_output_depth.png
[JNet_419_beads_002_roi002_reconst_depth]: /experiments/images/JNet_419_beads_002_roi002_reconst_depth.png
[JNet_419_psf_post]: /experiments/images/JNet_419_psf_post.png
[JNet_419_psf_pre]: /experiments/images/JNet_419_psf_pre.png
[finetuned]: /experiments/tmp/JNet_419_train.png
[pretrained_model]: /experiments/tmp/JNet_417_pretrain_train.png
