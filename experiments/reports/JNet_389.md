



# JNet_389 Report
  
the parameters to replicate the results of JNet_389. more large PSF (NA=0.5),   
pretrained model : JNet_387_pretrain
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
|size_z|239||
|NA|0.5||
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
|res_axial|0.05|microns|
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
|loss_fn|nn.BCELoss()|
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
|ploss_weight|0.01|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.03218569979071617, mean BCE: 0.12220815569162369
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_387_pretrain_0_original_plane]|![JNet_387_pretrain_0_output_plane]|![JNet_387_pretrain_0_label_plane]|
  
MSE: 0.036573927849531174, BCE: 0.1399082988500595  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_387_pretrain_0_original_depth]|![JNet_387_pretrain_0_output_depth]|![JNet_387_pretrain_0_label_depth]|
  
MSE: 0.036573927849531174, BCE: 0.1399082988500595  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_387_pretrain_1_original_plane]|![JNet_387_pretrain_1_output_plane]|![JNet_387_pretrain_1_label_plane]|
  
MSE: 0.03974964842200279, BCE: 0.1462126076221466  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_387_pretrain_1_original_depth]|![JNet_387_pretrain_1_output_depth]|![JNet_387_pretrain_1_label_depth]|
  
MSE: 0.03974964842200279, BCE: 0.1462126076221466  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_387_pretrain_2_original_plane]|![JNet_387_pretrain_2_output_plane]|![JNet_387_pretrain_2_label_plane]|
  
MSE: 0.02612725831568241, BCE: 0.10136667639017105  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_387_pretrain_2_original_depth]|![JNet_387_pretrain_2_output_depth]|![JNet_387_pretrain_2_label_depth]|
  
MSE: 0.02612725831568241, BCE: 0.10136667639017105  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_387_pretrain_3_original_plane]|![JNet_387_pretrain_3_output_plane]|![JNet_387_pretrain_3_label_plane]|
  
MSE: 0.02303515374660492, BCE: 0.08697357028722763  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_387_pretrain_3_original_depth]|![JNet_387_pretrain_3_output_depth]|![JNet_387_pretrain_3_label_depth]|
  
MSE: 0.02303515374660492, BCE: 0.08697357028722763  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_387_pretrain_4_original_plane]|![JNet_387_pretrain_4_output_plane]|![JNet_387_pretrain_4_label_plane]|
  
MSE: 0.03544251248240471, BCE: 0.13657964766025543  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_387_pretrain_4_original_depth]|![JNet_387_pretrain_4_output_depth]|![JNet_387_pretrain_4_label_depth]|
  
MSE: 0.03544251248240471, BCE: 0.13657964766025543  
  
mean MSE: 0.03411445394158363, mean BCE: 0.25841933488845825
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_389_0_original_plane]|![JNet_389_0_output_plane]|![JNet_389_0_label_plane]|
  
MSE: 0.031368859112262726, BCE: 0.21793924272060394  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_389_0_original_depth]|![JNet_389_0_output_depth]|![JNet_389_0_label_depth]|
  
MSE: 0.031368859112262726, BCE: 0.21793924272060394  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_389_1_original_plane]|![JNet_389_1_output_plane]|![JNet_389_1_label_plane]|
  
MSE: 0.039872538298368454, BCE: 0.32149630784988403  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_389_1_original_depth]|![JNet_389_1_output_depth]|![JNet_389_1_label_depth]|
  
MSE: 0.039872538298368454, BCE: 0.32149630784988403  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_389_2_original_plane]|![JNet_389_2_output_plane]|![JNet_389_2_label_plane]|
  
MSE: 0.03198174387216568, BCE: 0.2261401116847992  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_389_2_original_depth]|![JNet_389_2_output_depth]|![JNet_389_2_label_depth]|
  
MSE: 0.03198174387216568, BCE: 0.2261401116847992  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_389_3_original_plane]|![JNet_389_3_output_plane]|![JNet_389_3_label_plane]|
  
MSE: 0.03042569011449814, BCE: 0.24243590235710144  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_389_3_original_depth]|![JNet_389_3_output_depth]|![JNet_389_3_label_depth]|
  
MSE: 0.03042569011449814, BCE: 0.24243590235710144  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_389_4_original_plane]|![JNet_389_4_output_plane]|![JNet_389_4_label_plane]|
  
MSE: 0.03692343458533287, BCE: 0.28408515453338623  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_389_4_original_depth]|![JNet_389_4_output_depth]|![JNet_389_4_label_depth]|
  
MSE: 0.03692343458533287, BCE: 0.28408515453338623  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_387_pretrain_beads_001_roi000_original_depth]|![JNet_387_pretrain_beads_001_roi000_output_depth]|![JNet_387_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 16.659283203125003, MSE: 0.003113774349913001, quantized loss: 0.0022970056161284447  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_387_pretrain_beads_001_roi001_original_depth]|![JNet_387_pretrain_beads_001_roi001_output_depth]|![JNet_387_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 23.523742187500005, MSE: 0.005858149845153093, quantized loss: 0.0034350142814219  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_387_pretrain_beads_001_roi002_original_depth]|![JNet_387_pretrain_beads_001_roi002_output_depth]|![JNet_387_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 15.279871093750003, MSE: 0.003133531427010894, quantized loss: 0.002049094531685114  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_387_pretrain_beads_001_roi003_original_depth]|![JNet_387_pretrain_beads_001_roi003_output_depth]|![JNet_387_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 24.066677734375006, MSE: 0.00535397557541728, quantized loss: 0.0028697161469608545  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_387_pretrain_beads_001_roi004_original_depth]|![JNet_387_pretrain_beads_001_roi004_output_depth]|![JNet_387_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 16.668923828125003, MSE: 0.003780325874686241, quantized loss: 0.002142751356586814  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_387_pretrain_beads_002_roi000_original_depth]|![JNet_387_pretrain_beads_002_roi000_output_depth]|![JNet_387_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 17.838517578125003, MSE: 0.004210584331303835, quantized loss: 0.0022620991803705692  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_387_pretrain_beads_002_roi001_original_depth]|![JNet_387_pretrain_beads_002_roi001_output_depth]|![JNet_387_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 16.413060546875005, MSE: 0.003401976078748703, quantized loss: 0.0021402782294899225  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_387_pretrain_beads_002_roi002_original_depth]|![JNet_387_pretrain_beads_002_roi002_output_depth]|![JNet_387_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 17.005183593750004, MSE: 0.0038257595151662827, quantized loss: 0.002203232841566205  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_389_beads_001_roi000_original_depth]|![JNet_389_beads_001_roi000_output_depth]|![JNet_389_beads_001_roi000_reconst_depth]|
  
volume: 13.089843750000004, MSE: 0.002526982221752405, quantized loss: 0.00022844284831080586  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_389_beads_001_roi001_original_depth]|![JNet_389_beads_001_roi001_output_depth]|![JNet_389_beads_001_roi001_reconst_depth]|
  
volume: 20.275800781250005, MSE: 0.004199338145554066, quantized loss: 0.00034731789492070675  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_389_beads_001_roi002_original_depth]|![JNet_389_beads_001_roi002_output_depth]|![JNet_389_beads_001_roi002_reconst_depth]|
  
volume: 12.773665039062504, MSE: 0.0025636099744588137, quantized loss: 0.00020771825802512467  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_389_beads_001_roi003_original_depth]|![JNet_389_beads_001_roi003_output_depth]|![JNet_389_beads_001_roi003_reconst_depth]|
  
volume: 20.229919921875005, MSE: 0.004318110179156065, quantized loss: 0.000295834121061489  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_389_beads_001_roi004_original_depth]|![JNet_389_beads_001_roi004_output_depth]|![JNet_389_beads_001_roi004_reconst_depth]|
  
volume: 13.663575195312504, MSE: 0.003116331296041608, quantized loss: 0.00020535897056106478  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_389_beads_002_roi000_original_depth]|![JNet_389_beads_002_roi000_output_depth]|![JNet_389_beads_002_roi000_reconst_depth]|
  
volume: 14.529855468750004, MSE: 0.003462204011157155, quantized loss: 0.00021431531058624387  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_389_beads_002_roi001_original_depth]|![JNet_389_beads_002_roi001_output_depth]|![JNet_389_beads_002_roi001_reconst_depth]|
  
volume: 13.434575195312503, MSE: 0.0027468546759337187, quantized loss: 0.00020496871729847044  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_389_beads_002_roi002_original_depth]|![JNet_389_beads_002_roi002_output_depth]|![JNet_389_beads_002_roi002_reconst_depth]|
  
volume: 13.955289062500004, MSE: 0.0031212137546390295, quantized loss: 0.00021443312289193273  

|pre|post|
| :---: | :---: |
|![JNet_389_psf_pre]|![JNet_389_psf_post]|

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
    (conv): Conv3d(16, 2, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=same, padding_mode=replicate)  
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
  



[JNet_387_pretrain_0_label_depth]: /experiments/images/JNet_387_pretrain_0_label_depth.png
[JNet_387_pretrain_0_label_plane]: /experiments/images/JNet_387_pretrain_0_label_plane.png
[JNet_387_pretrain_0_original_depth]: /experiments/images/JNet_387_pretrain_0_original_depth.png
[JNet_387_pretrain_0_original_plane]: /experiments/images/JNet_387_pretrain_0_original_plane.png
[JNet_387_pretrain_0_output_depth]: /experiments/images/JNet_387_pretrain_0_output_depth.png
[JNet_387_pretrain_0_output_plane]: /experiments/images/JNet_387_pretrain_0_output_plane.png
[JNet_387_pretrain_1_label_depth]: /experiments/images/JNet_387_pretrain_1_label_depth.png
[JNet_387_pretrain_1_label_plane]: /experiments/images/JNet_387_pretrain_1_label_plane.png
[JNet_387_pretrain_1_original_depth]: /experiments/images/JNet_387_pretrain_1_original_depth.png
[JNet_387_pretrain_1_original_plane]: /experiments/images/JNet_387_pretrain_1_original_plane.png
[JNet_387_pretrain_1_output_depth]: /experiments/images/JNet_387_pretrain_1_output_depth.png
[JNet_387_pretrain_1_output_plane]: /experiments/images/JNet_387_pretrain_1_output_plane.png
[JNet_387_pretrain_2_label_depth]: /experiments/images/JNet_387_pretrain_2_label_depth.png
[JNet_387_pretrain_2_label_plane]: /experiments/images/JNet_387_pretrain_2_label_plane.png
[JNet_387_pretrain_2_original_depth]: /experiments/images/JNet_387_pretrain_2_original_depth.png
[JNet_387_pretrain_2_original_plane]: /experiments/images/JNet_387_pretrain_2_original_plane.png
[JNet_387_pretrain_2_output_depth]: /experiments/images/JNet_387_pretrain_2_output_depth.png
[JNet_387_pretrain_2_output_plane]: /experiments/images/JNet_387_pretrain_2_output_plane.png
[JNet_387_pretrain_3_label_depth]: /experiments/images/JNet_387_pretrain_3_label_depth.png
[JNet_387_pretrain_3_label_plane]: /experiments/images/JNet_387_pretrain_3_label_plane.png
[JNet_387_pretrain_3_original_depth]: /experiments/images/JNet_387_pretrain_3_original_depth.png
[JNet_387_pretrain_3_original_plane]: /experiments/images/JNet_387_pretrain_3_original_plane.png
[JNet_387_pretrain_3_output_depth]: /experiments/images/JNet_387_pretrain_3_output_depth.png
[JNet_387_pretrain_3_output_plane]: /experiments/images/JNet_387_pretrain_3_output_plane.png
[JNet_387_pretrain_4_label_depth]: /experiments/images/JNet_387_pretrain_4_label_depth.png
[JNet_387_pretrain_4_label_plane]: /experiments/images/JNet_387_pretrain_4_label_plane.png
[JNet_387_pretrain_4_original_depth]: /experiments/images/JNet_387_pretrain_4_original_depth.png
[JNet_387_pretrain_4_original_plane]: /experiments/images/JNet_387_pretrain_4_original_plane.png
[JNet_387_pretrain_4_output_depth]: /experiments/images/JNet_387_pretrain_4_output_depth.png
[JNet_387_pretrain_4_output_plane]: /experiments/images/JNet_387_pretrain_4_output_plane.png
[JNet_387_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi000_original_depth.png
[JNet_387_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi000_output_depth.png
[JNet_387_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi000_reconst_depth.png
[JNet_387_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi001_original_depth.png
[JNet_387_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi001_output_depth.png
[JNet_387_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi001_reconst_depth.png
[JNet_387_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi002_original_depth.png
[JNet_387_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi002_output_depth.png
[JNet_387_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi002_reconst_depth.png
[JNet_387_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi003_original_depth.png
[JNet_387_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi003_output_depth.png
[JNet_387_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi003_reconst_depth.png
[JNet_387_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi004_original_depth.png
[JNet_387_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi004_output_depth.png
[JNet_387_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_387_pretrain_beads_001_roi004_reconst_depth.png
[JNet_387_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_387_pretrain_beads_002_roi000_original_depth.png
[JNet_387_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_387_pretrain_beads_002_roi000_output_depth.png
[JNet_387_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_387_pretrain_beads_002_roi000_reconst_depth.png
[JNet_387_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_387_pretrain_beads_002_roi001_original_depth.png
[JNet_387_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_387_pretrain_beads_002_roi001_output_depth.png
[JNet_387_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_387_pretrain_beads_002_roi001_reconst_depth.png
[JNet_387_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_387_pretrain_beads_002_roi002_original_depth.png
[JNet_387_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_387_pretrain_beads_002_roi002_output_depth.png
[JNet_387_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_387_pretrain_beads_002_roi002_reconst_depth.png
[JNet_389_0_label_depth]: /experiments/images/JNet_389_0_label_depth.png
[JNet_389_0_label_plane]: /experiments/images/JNet_389_0_label_plane.png
[JNet_389_0_original_depth]: /experiments/images/JNet_389_0_original_depth.png
[JNet_389_0_original_plane]: /experiments/images/JNet_389_0_original_plane.png
[JNet_389_0_output_depth]: /experiments/images/JNet_389_0_output_depth.png
[JNet_389_0_output_plane]: /experiments/images/JNet_389_0_output_plane.png
[JNet_389_1_label_depth]: /experiments/images/JNet_389_1_label_depth.png
[JNet_389_1_label_plane]: /experiments/images/JNet_389_1_label_plane.png
[JNet_389_1_original_depth]: /experiments/images/JNet_389_1_original_depth.png
[JNet_389_1_original_plane]: /experiments/images/JNet_389_1_original_plane.png
[JNet_389_1_output_depth]: /experiments/images/JNet_389_1_output_depth.png
[JNet_389_1_output_plane]: /experiments/images/JNet_389_1_output_plane.png
[JNet_389_2_label_depth]: /experiments/images/JNet_389_2_label_depth.png
[JNet_389_2_label_plane]: /experiments/images/JNet_389_2_label_plane.png
[JNet_389_2_original_depth]: /experiments/images/JNet_389_2_original_depth.png
[JNet_389_2_original_plane]: /experiments/images/JNet_389_2_original_plane.png
[JNet_389_2_output_depth]: /experiments/images/JNet_389_2_output_depth.png
[JNet_389_2_output_plane]: /experiments/images/JNet_389_2_output_plane.png
[JNet_389_3_label_depth]: /experiments/images/JNet_389_3_label_depth.png
[JNet_389_3_label_plane]: /experiments/images/JNet_389_3_label_plane.png
[JNet_389_3_original_depth]: /experiments/images/JNet_389_3_original_depth.png
[JNet_389_3_original_plane]: /experiments/images/JNet_389_3_original_plane.png
[JNet_389_3_output_depth]: /experiments/images/JNet_389_3_output_depth.png
[JNet_389_3_output_plane]: /experiments/images/JNet_389_3_output_plane.png
[JNet_389_4_label_depth]: /experiments/images/JNet_389_4_label_depth.png
[JNet_389_4_label_plane]: /experiments/images/JNet_389_4_label_plane.png
[JNet_389_4_original_depth]: /experiments/images/JNet_389_4_original_depth.png
[JNet_389_4_original_plane]: /experiments/images/JNet_389_4_original_plane.png
[JNet_389_4_output_depth]: /experiments/images/JNet_389_4_output_depth.png
[JNet_389_4_output_plane]: /experiments/images/JNet_389_4_output_plane.png
[JNet_389_beads_001_roi000_original_depth]: /experiments/images/JNet_389_beads_001_roi000_original_depth.png
[JNet_389_beads_001_roi000_output_depth]: /experiments/images/JNet_389_beads_001_roi000_output_depth.png
[JNet_389_beads_001_roi000_reconst_depth]: /experiments/images/JNet_389_beads_001_roi000_reconst_depth.png
[JNet_389_beads_001_roi001_original_depth]: /experiments/images/JNet_389_beads_001_roi001_original_depth.png
[JNet_389_beads_001_roi001_output_depth]: /experiments/images/JNet_389_beads_001_roi001_output_depth.png
[JNet_389_beads_001_roi001_reconst_depth]: /experiments/images/JNet_389_beads_001_roi001_reconst_depth.png
[JNet_389_beads_001_roi002_original_depth]: /experiments/images/JNet_389_beads_001_roi002_original_depth.png
[JNet_389_beads_001_roi002_output_depth]: /experiments/images/JNet_389_beads_001_roi002_output_depth.png
[JNet_389_beads_001_roi002_reconst_depth]: /experiments/images/JNet_389_beads_001_roi002_reconst_depth.png
[JNet_389_beads_001_roi003_original_depth]: /experiments/images/JNet_389_beads_001_roi003_original_depth.png
[JNet_389_beads_001_roi003_output_depth]: /experiments/images/JNet_389_beads_001_roi003_output_depth.png
[JNet_389_beads_001_roi003_reconst_depth]: /experiments/images/JNet_389_beads_001_roi003_reconst_depth.png
[JNet_389_beads_001_roi004_original_depth]: /experiments/images/JNet_389_beads_001_roi004_original_depth.png
[JNet_389_beads_001_roi004_output_depth]: /experiments/images/JNet_389_beads_001_roi004_output_depth.png
[JNet_389_beads_001_roi004_reconst_depth]: /experiments/images/JNet_389_beads_001_roi004_reconst_depth.png
[JNet_389_beads_002_roi000_original_depth]: /experiments/images/JNet_389_beads_002_roi000_original_depth.png
[JNet_389_beads_002_roi000_output_depth]: /experiments/images/JNet_389_beads_002_roi000_output_depth.png
[JNet_389_beads_002_roi000_reconst_depth]: /experiments/images/JNet_389_beads_002_roi000_reconst_depth.png
[JNet_389_beads_002_roi001_original_depth]: /experiments/images/JNet_389_beads_002_roi001_original_depth.png
[JNet_389_beads_002_roi001_output_depth]: /experiments/images/JNet_389_beads_002_roi001_output_depth.png
[JNet_389_beads_002_roi001_reconst_depth]: /experiments/images/JNet_389_beads_002_roi001_reconst_depth.png
[JNet_389_beads_002_roi002_original_depth]: /experiments/images/JNet_389_beads_002_roi002_original_depth.png
[JNet_389_beads_002_roi002_output_depth]: /experiments/images/JNet_389_beads_002_roi002_output_depth.png
[JNet_389_beads_002_roi002_reconst_depth]: /experiments/images/JNet_389_beads_002_roi002_reconst_depth.png
[JNet_389_psf_post]: /experiments/images/JNet_389_psf_post.png
[JNet_389_psf_pre]: /experiments/images/JNet_389_psf_pre.png
[finetuned]: /experiments/tmp/JNet_389_train.png
[pretrained_model]: /experiments/tmp/JNet_387_pretrain_train.png
