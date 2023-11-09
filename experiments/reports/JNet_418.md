



# JNet_418 Report
  
the parameters to replicate the results of JNet_418. nearest interp of PSF, logit loss = 1.0, NA = 0.8  
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
|qloss_weight|0.1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.023683028295636177, mean BCE: 0.08916638046503067
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_0_original_plane]|![JNet_417_pretrain_0_output_plane]|![JNet_417_pretrain_0_label_plane]|
  
MSE: 0.021945688873529434, BCE: 0.09070050716400146  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_0_original_depth]|![JNet_417_pretrain_0_output_depth]|![JNet_417_pretrain_0_label_depth]|
  
MSE: 0.021945688873529434, BCE: 0.09070050716400146  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_1_original_plane]|![JNet_417_pretrain_1_output_plane]|![JNet_417_pretrain_1_label_plane]|
  
MSE: 0.02069447562098503, BCE: 0.07488689571619034  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_1_original_depth]|![JNet_417_pretrain_1_output_depth]|![JNet_417_pretrain_1_label_depth]|
  
MSE: 0.02069447562098503, BCE: 0.07488689571619034  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_2_original_plane]|![JNet_417_pretrain_2_output_plane]|![JNet_417_pretrain_2_label_plane]|
  
MSE: 0.034977126866579056, BCE: 0.14157575368881226  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_2_original_depth]|![JNet_417_pretrain_2_output_depth]|![JNet_417_pretrain_2_label_depth]|
  
MSE: 0.034977126866579056, BCE: 0.14157575368881226  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_3_original_plane]|![JNet_417_pretrain_3_output_plane]|![JNet_417_pretrain_3_label_plane]|
  
MSE: 0.021426096558570862, BCE: 0.07430122792720795  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_3_original_depth]|![JNet_417_pretrain_3_output_depth]|![JNet_417_pretrain_3_label_depth]|
  
MSE: 0.021426096558570862, BCE: 0.07430122792720795  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_4_original_plane]|![JNet_417_pretrain_4_output_plane]|![JNet_417_pretrain_4_label_plane]|
  
MSE: 0.019371753558516502, BCE: 0.06436754018068314  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_417_pretrain_4_original_depth]|![JNet_417_pretrain_4_output_depth]|![JNet_417_pretrain_4_label_depth]|
  
MSE: 0.019371753558516502, BCE: 0.06436754018068314  
  
mean MSE: 0.02914908528327942, mean BCE: 0.31375813484191895
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_418_0_original_plane]|![JNet_418_0_output_plane]|![JNet_418_0_label_plane]|
  
MSE: 0.02263805828988552, BCE: 0.2316998988389969  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_418_0_original_depth]|![JNet_418_0_output_depth]|![JNet_418_0_label_depth]|
  
MSE: 0.02263805828988552, BCE: 0.2316998988389969  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_418_1_original_plane]|![JNet_418_1_output_plane]|![JNet_418_1_label_plane]|
  
MSE: 0.029765712097287178, BCE: 0.27528899908065796  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_418_1_original_depth]|![JNet_418_1_output_depth]|![JNet_418_1_label_depth]|
  
MSE: 0.029765712097287178, BCE: 0.27528899908065796  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_418_2_original_plane]|![JNet_418_2_output_plane]|![JNet_418_2_label_plane]|
  
MSE: 0.04742308706045151, BCE: 0.5715484023094177  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_418_2_original_depth]|![JNet_418_2_output_depth]|![JNet_418_2_label_depth]|
  
MSE: 0.04742308706045151, BCE: 0.5715484023094177  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_418_3_original_plane]|![JNet_418_3_output_plane]|![JNet_418_3_label_plane]|
  
MSE: 0.02287396974861622, BCE: 0.2822326123714447  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_418_3_original_depth]|![JNet_418_3_output_depth]|![JNet_418_3_label_depth]|
  
MSE: 0.02287396974861622, BCE: 0.2822326123714447  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_418_4_original_plane]|![JNet_418_4_output_plane]|![JNet_418_4_label_plane]|
  
MSE: 0.023044606670737267, BCE: 0.2080208659172058  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_418_4_original_depth]|![JNet_418_4_output_depth]|![JNet_418_4_label_depth]|
  
MSE: 0.023044606670737267, BCE: 0.2080208659172058  

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
|![JNet_418_beads_001_roi000_original_depth]|![JNet_418_beads_001_roi000_output_depth]|![JNet_418_beads_001_roi000_reconst_depth]|![JNet_418_beads_001_roi000_heatmap_depth]|
  
volume: 1.1115948486328127, MSE: 0.0006318566738627851, quantized loss: 0.0007632678025402129  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_418_beads_001_roi001_original_depth]|![JNet_418_beads_001_roi001_output_depth]|![JNet_418_beads_001_roi001_reconst_depth]|![JNet_418_beads_001_roi001_heatmap_depth]|
  
volume: 1.6951259765625004, MSE: 0.0015992735279724002, quantized loss: 0.0011162724113091826  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_418_beads_001_roi002_original_depth]|![JNet_418_beads_001_roi002_output_depth]|![JNet_418_beads_001_roi002_reconst_depth]|![JNet_418_beads_001_roi002_heatmap_depth]|
  
volume: 1.1335180664062503, MSE: 0.0005786232650279999, quantized loss: 0.0007827593944966793  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_418_beads_001_roi003_original_depth]|![JNet_418_beads_001_roi003_output_depth]|![JNet_418_beads_001_roi003_reconst_depth]|![JNet_418_beads_001_roi003_heatmap_depth]|
  
volume: 1.8228698730468755, MSE: 0.001264376798644662, quantized loss: 0.0011718624737113714  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_418_beads_001_roi004_original_depth]|![JNet_418_beads_001_roi004_output_depth]|![JNet_418_beads_001_roi004_reconst_depth]|![JNet_418_beads_001_roi004_heatmap_depth]|
  
volume: 1.2732279052734379, MSE: 0.0005846206331625581, quantized loss: 0.0007955585024319589  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_418_beads_002_roi000_original_depth]|![JNet_418_beads_002_roi000_output_depth]|![JNet_418_beads_002_roi000_reconst_depth]|![JNet_418_beads_002_roi000_heatmap_depth]|
  
volume: 1.3614201660156253, MSE: 0.0006177120376378298, quantized loss: 0.000803449482191354  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_418_beads_002_roi001_original_depth]|![JNet_418_beads_002_roi001_output_depth]|![JNet_418_beads_002_roi001_reconst_depth]|![JNet_418_beads_002_roi001_heatmap_depth]|
  
volume: 1.1860712890625003, MSE: 0.0005695949075743556, quantized loss: 0.0007615477661602199  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_418_beads_002_roi002_original_depth]|![JNet_418_beads_002_roi002_output_depth]|![JNet_418_beads_002_roi002_reconst_depth]|![JNet_418_beads_002_roi002_heatmap_depth]|
  
volume: 1.2741146240234378, MSE: 0.0005630037630908191, quantized loss: 0.0007952837622724473  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_418_psf_pre]|![JNet_418_psf_post]|

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
[JNet_418_0_label_depth]: /experiments/images/JNet_418_0_label_depth.png
[JNet_418_0_label_plane]: /experiments/images/JNet_418_0_label_plane.png
[JNet_418_0_original_depth]: /experiments/images/JNet_418_0_original_depth.png
[JNet_418_0_original_plane]: /experiments/images/JNet_418_0_original_plane.png
[JNet_418_0_output_depth]: /experiments/images/JNet_418_0_output_depth.png
[JNet_418_0_output_plane]: /experiments/images/JNet_418_0_output_plane.png
[JNet_418_1_label_depth]: /experiments/images/JNet_418_1_label_depth.png
[JNet_418_1_label_plane]: /experiments/images/JNet_418_1_label_plane.png
[JNet_418_1_original_depth]: /experiments/images/JNet_418_1_original_depth.png
[JNet_418_1_original_plane]: /experiments/images/JNet_418_1_original_plane.png
[JNet_418_1_output_depth]: /experiments/images/JNet_418_1_output_depth.png
[JNet_418_1_output_plane]: /experiments/images/JNet_418_1_output_plane.png
[JNet_418_2_label_depth]: /experiments/images/JNet_418_2_label_depth.png
[JNet_418_2_label_plane]: /experiments/images/JNet_418_2_label_plane.png
[JNet_418_2_original_depth]: /experiments/images/JNet_418_2_original_depth.png
[JNet_418_2_original_plane]: /experiments/images/JNet_418_2_original_plane.png
[JNet_418_2_output_depth]: /experiments/images/JNet_418_2_output_depth.png
[JNet_418_2_output_plane]: /experiments/images/JNet_418_2_output_plane.png
[JNet_418_3_label_depth]: /experiments/images/JNet_418_3_label_depth.png
[JNet_418_3_label_plane]: /experiments/images/JNet_418_3_label_plane.png
[JNet_418_3_original_depth]: /experiments/images/JNet_418_3_original_depth.png
[JNet_418_3_original_plane]: /experiments/images/JNet_418_3_original_plane.png
[JNet_418_3_output_depth]: /experiments/images/JNet_418_3_output_depth.png
[JNet_418_3_output_plane]: /experiments/images/JNet_418_3_output_plane.png
[JNet_418_4_label_depth]: /experiments/images/JNet_418_4_label_depth.png
[JNet_418_4_label_plane]: /experiments/images/JNet_418_4_label_plane.png
[JNet_418_4_original_depth]: /experiments/images/JNet_418_4_original_depth.png
[JNet_418_4_original_plane]: /experiments/images/JNet_418_4_original_plane.png
[JNet_418_4_output_depth]: /experiments/images/JNet_418_4_output_depth.png
[JNet_418_4_output_plane]: /experiments/images/JNet_418_4_output_plane.png
[JNet_418_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_418_beads_001_roi000_heatmap_depth.png
[JNet_418_beads_001_roi000_original_depth]: /experiments/images/JNet_418_beads_001_roi000_original_depth.png
[JNet_418_beads_001_roi000_output_depth]: /experiments/images/JNet_418_beads_001_roi000_output_depth.png
[JNet_418_beads_001_roi000_reconst_depth]: /experiments/images/JNet_418_beads_001_roi000_reconst_depth.png
[JNet_418_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_418_beads_001_roi001_heatmap_depth.png
[JNet_418_beads_001_roi001_original_depth]: /experiments/images/JNet_418_beads_001_roi001_original_depth.png
[JNet_418_beads_001_roi001_output_depth]: /experiments/images/JNet_418_beads_001_roi001_output_depth.png
[JNet_418_beads_001_roi001_reconst_depth]: /experiments/images/JNet_418_beads_001_roi001_reconst_depth.png
[JNet_418_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_418_beads_001_roi002_heatmap_depth.png
[JNet_418_beads_001_roi002_original_depth]: /experiments/images/JNet_418_beads_001_roi002_original_depth.png
[JNet_418_beads_001_roi002_output_depth]: /experiments/images/JNet_418_beads_001_roi002_output_depth.png
[JNet_418_beads_001_roi002_reconst_depth]: /experiments/images/JNet_418_beads_001_roi002_reconst_depth.png
[JNet_418_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_418_beads_001_roi003_heatmap_depth.png
[JNet_418_beads_001_roi003_original_depth]: /experiments/images/JNet_418_beads_001_roi003_original_depth.png
[JNet_418_beads_001_roi003_output_depth]: /experiments/images/JNet_418_beads_001_roi003_output_depth.png
[JNet_418_beads_001_roi003_reconst_depth]: /experiments/images/JNet_418_beads_001_roi003_reconst_depth.png
[JNet_418_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_418_beads_001_roi004_heatmap_depth.png
[JNet_418_beads_001_roi004_original_depth]: /experiments/images/JNet_418_beads_001_roi004_original_depth.png
[JNet_418_beads_001_roi004_output_depth]: /experiments/images/JNet_418_beads_001_roi004_output_depth.png
[JNet_418_beads_001_roi004_reconst_depth]: /experiments/images/JNet_418_beads_001_roi004_reconst_depth.png
[JNet_418_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_418_beads_002_roi000_heatmap_depth.png
[JNet_418_beads_002_roi000_original_depth]: /experiments/images/JNet_418_beads_002_roi000_original_depth.png
[JNet_418_beads_002_roi000_output_depth]: /experiments/images/JNet_418_beads_002_roi000_output_depth.png
[JNet_418_beads_002_roi000_reconst_depth]: /experiments/images/JNet_418_beads_002_roi000_reconst_depth.png
[JNet_418_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_418_beads_002_roi001_heatmap_depth.png
[JNet_418_beads_002_roi001_original_depth]: /experiments/images/JNet_418_beads_002_roi001_original_depth.png
[JNet_418_beads_002_roi001_output_depth]: /experiments/images/JNet_418_beads_002_roi001_output_depth.png
[JNet_418_beads_002_roi001_reconst_depth]: /experiments/images/JNet_418_beads_002_roi001_reconst_depth.png
[JNet_418_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_418_beads_002_roi002_heatmap_depth.png
[JNet_418_beads_002_roi002_original_depth]: /experiments/images/JNet_418_beads_002_roi002_original_depth.png
[JNet_418_beads_002_roi002_output_depth]: /experiments/images/JNet_418_beads_002_roi002_output_depth.png
[JNet_418_beads_002_roi002_reconst_depth]: /experiments/images/JNet_418_beads_002_roi002_reconst_depth.png
[JNet_418_psf_post]: /experiments/images/JNet_418_psf_post.png
[JNet_418_psf_pre]: /experiments/images/JNet_418_psf_pre.png
[finetuned]: /experiments/tmp/JNet_418_train.png
[pretrained_model]: /experiments/tmp/JNet_417_pretrain_train.png
