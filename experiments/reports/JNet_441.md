



# JNet_441 Report
  
the parameters to replicate the results of JNet_441. nearest interp of PSF, NA=0.7, mu_z = 0.3, sig_z = 1.27  
pretrained model : JNet_439_pretrain
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
  
mean MSE: 0.024119237437844276, mean BCE: 0.08871608972549438
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_0_original_plane]|![JNet_439_pretrain_0_output_plane]|![JNet_439_pretrain_0_label_plane]|
  
MSE: 0.018879389390349388, BCE: 0.07098624110221863  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_0_original_depth]|![JNet_439_pretrain_0_output_depth]|![JNet_439_pretrain_0_label_depth]|
  
MSE: 0.018879389390349388, BCE: 0.07098624110221863  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_1_original_plane]|![JNet_439_pretrain_1_output_plane]|![JNet_439_pretrain_1_label_plane]|
  
MSE: 0.027540596202015877, BCE: 0.09802334010601044  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_1_original_depth]|![JNet_439_pretrain_1_output_depth]|![JNet_439_pretrain_1_label_depth]|
  
MSE: 0.027540596202015877, BCE: 0.09802334010601044  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_2_original_plane]|![JNet_439_pretrain_2_output_plane]|![JNet_439_pretrain_2_label_plane]|
  
MSE: 0.028572585433721542, BCE: 0.10451064258813858  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_2_original_depth]|![JNet_439_pretrain_2_output_depth]|![JNet_439_pretrain_2_label_depth]|
  
MSE: 0.028572585433721542, BCE: 0.10451064258813858  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_3_original_plane]|![JNet_439_pretrain_3_output_plane]|![JNet_439_pretrain_3_label_plane]|
  
MSE: 0.020793862640857697, BCE: 0.07924362272024155  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_3_original_depth]|![JNet_439_pretrain_3_output_depth]|![JNet_439_pretrain_3_label_depth]|
  
MSE: 0.020793862640857697, BCE: 0.07924362272024155  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_4_original_plane]|![JNet_439_pretrain_4_output_plane]|![JNet_439_pretrain_4_label_plane]|
  
MSE: 0.024809759110212326, BCE: 0.09081659466028214  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_4_original_depth]|![JNet_439_pretrain_4_output_depth]|![JNet_439_pretrain_4_label_depth]|
  
MSE: 0.024809759110212326, BCE: 0.09081659466028214  
  
mean MSE: 0.03459161892533302, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_441_0_original_plane]|![JNet_441_0_output_plane]|![JNet_441_0_label_plane]|
  
MSE: 0.04408739134669304, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_441_0_original_depth]|![JNet_441_0_output_depth]|![JNet_441_0_label_depth]|
  
MSE: 0.04408739134669304, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_441_1_original_plane]|![JNet_441_1_output_plane]|![JNet_441_1_label_plane]|
  
MSE: 0.015700524672865868, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_441_1_original_depth]|![JNet_441_1_output_depth]|![JNet_441_1_label_depth]|
  
MSE: 0.015700524672865868, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_441_2_original_plane]|![JNet_441_2_output_plane]|![JNet_441_2_label_plane]|
  
MSE: 0.0425429530441761, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_441_2_original_depth]|![JNet_441_2_output_depth]|![JNet_441_2_label_depth]|
  
MSE: 0.0425429530441761, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_441_3_original_plane]|![JNet_441_3_output_plane]|![JNet_441_3_label_plane]|
  
MSE: 0.04906449466943741, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_441_3_original_depth]|![JNet_441_3_output_depth]|![JNet_441_3_label_depth]|
  
MSE: 0.04906449466943741, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_441_4_original_plane]|![JNet_441_4_output_plane]|![JNet_441_4_label_plane]|
  
MSE: 0.021562734618782997, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_441_4_original_depth]|![JNet_441_4_output_depth]|![JNet_441_4_label_depth]|
  
MSE: 0.021562734618782997, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_439_pretrain_beads_001_roi000_original_depth]|![JNet_439_pretrain_beads_001_roi000_output_depth]|![JNet_439_pretrain_beads_001_roi000_reconst_depth]|![JNet_439_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 20.852056640625005, MSE: 0.0021086863707751036, quantized loss: 0.002919599646702409  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_439_pretrain_beads_001_roi001_original_depth]|![JNet_439_pretrain_beads_001_roi001_output_depth]|![JNet_439_pretrain_beads_001_roi001_reconst_depth]|![JNet_439_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 30.75900781250001, MSE: 0.0037715614307671785, quantized loss: 0.004034939222037792  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_439_pretrain_beads_001_roi002_original_depth]|![JNet_439_pretrain_beads_001_roi002_output_depth]|![JNet_439_pretrain_beads_001_roi002_reconst_depth]|![JNet_439_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 19.288605468750003, MSE: 0.0021511579398065805, quantized loss: 0.0023915329948067665  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_439_pretrain_beads_001_roi003_original_depth]|![JNet_439_pretrain_beads_001_roi003_output_depth]|![JNet_439_pretrain_beads_001_roi003_reconst_depth]|![JNet_439_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 31.64290234375001, MSE: 0.0036672065034508705, quantized loss: 0.003974692430347204  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_439_pretrain_beads_001_roi004_original_depth]|![JNet_439_pretrain_beads_001_roi004_output_depth]|![JNet_439_pretrain_beads_001_roi004_reconst_depth]|![JNet_439_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 21.512791015625005, MSE: 0.002746155485510826, quantized loss: 0.0027180127799510956  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_439_pretrain_beads_002_roi000_original_depth]|![JNet_439_pretrain_beads_002_roi000_output_depth]|![JNet_439_pretrain_beads_002_roi000_reconst_depth]|![JNet_439_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 23.536382812500005, MSE: 0.0031677011866122484, quantized loss: 0.0030188874807208776  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_439_pretrain_beads_002_roi001_original_depth]|![JNet_439_pretrain_beads_002_roi001_output_depth]|![JNet_439_pretrain_beads_002_roi001_reconst_depth]|![JNet_439_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 21.124314453125006, MSE: 0.0024076534900814295, quantized loss: 0.0026687614154070616  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_439_pretrain_beads_002_roi002_original_depth]|![JNet_439_pretrain_beads_002_roi002_output_depth]|![JNet_439_pretrain_beads_002_roi002_reconst_depth]|![JNet_439_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 21.693822265625005, MSE: 0.0027437107637524605, quantized loss: 0.002681557321920991  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_441_beads_001_roi000_original_depth]|![JNet_441_beads_001_roi000_output_depth]|![JNet_441_beads_001_roi000_reconst_depth]|![JNet_441_beads_001_roi000_heatmap_depth]|
  
volume: 13.468257812500003, MSE: 0.000217472726944834, quantized loss: 1.1856672244903166e-05  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_441_beads_001_roi001_original_depth]|![JNet_441_beads_001_roi001_output_depth]|![JNet_441_beads_001_roi001_reconst_depth]|![JNet_441_beads_001_roi001_heatmap_depth]|
  
volume: 21.202248046875006, MSE: 0.0007340277661569417, quantized loss: 1.7802596630644985e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_441_beads_001_roi002_original_depth]|![JNet_441_beads_001_roi002_output_depth]|![JNet_441_beads_001_roi002_reconst_depth]|![JNet_441_beads_001_roi002_heatmap_depth]|
  
volume: 13.224513671875004, MSE: 0.00015368193271569908, quantized loss: 1.0547028978180606e-05  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_441_beads_001_roi003_original_depth]|![JNet_441_beads_001_roi003_output_depth]|![JNet_441_beads_001_roi003_reconst_depth]|![JNet_441_beads_001_roi003_heatmap_depth]|
  
volume: 21.682679687500006, MSE: 0.0005649434751830995, quantized loss: 1.704510759736877e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_441_beads_001_roi004_original_depth]|![JNet_441_beads_001_roi004_output_depth]|![JNet_441_beads_001_roi004_reconst_depth]|![JNet_441_beads_001_roi004_heatmap_depth]|
  
volume: 14.286613281250004, MSE: 0.00026109785540029407, quantized loss: 1.1395245564926881e-05  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_441_beads_002_roi000_original_depth]|![JNet_441_beads_002_roi000_output_depth]|![JNet_441_beads_002_roi000_reconst_depth]|![JNet_441_beads_002_roi000_heatmap_depth]|
  
volume: 15.224276367187503, MSE: 0.00033114958205260336, quantized loss: 1.1780627573898528e-05  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_441_beads_002_roi001_original_depth]|![JNet_441_beads_002_roi001_output_depth]|![JNet_441_beads_002_roi001_reconst_depth]|![JNet_441_beads_002_roi001_heatmap_depth]|
  
volume: 14.120273437500003, MSE: 0.00021928001660853624, quantized loss: 1.0992317584168632e-05  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_441_beads_002_roi002_original_depth]|![JNet_441_beads_002_roi002_output_depth]|![JNet_441_beads_002_roi002_reconst_depth]|![JNet_441_beads_002_roi002_heatmap_depth]|
  
volume: 14.567299804687503, MSE: 0.000259572610957548, quantized loss: 1.228267137776129e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_441_psf_pre]|![JNet_441_psf_post]|

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
  



[JNet_439_pretrain_0_label_depth]: /experiments/images/JNet_439_pretrain_0_label_depth.png
[JNet_439_pretrain_0_label_plane]: /experiments/images/JNet_439_pretrain_0_label_plane.png
[JNet_439_pretrain_0_original_depth]: /experiments/images/JNet_439_pretrain_0_original_depth.png
[JNet_439_pretrain_0_original_plane]: /experiments/images/JNet_439_pretrain_0_original_plane.png
[JNet_439_pretrain_0_output_depth]: /experiments/images/JNet_439_pretrain_0_output_depth.png
[JNet_439_pretrain_0_output_plane]: /experiments/images/JNet_439_pretrain_0_output_plane.png
[JNet_439_pretrain_1_label_depth]: /experiments/images/JNet_439_pretrain_1_label_depth.png
[JNet_439_pretrain_1_label_plane]: /experiments/images/JNet_439_pretrain_1_label_plane.png
[JNet_439_pretrain_1_original_depth]: /experiments/images/JNet_439_pretrain_1_original_depth.png
[JNet_439_pretrain_1_original_plane]: /experiments/images/JNet_439_pretrain_1_original_plane.png
[JNet_439_pretrain_1_output_depth]: /experiments/images/JNet_439_pretrain_1_output_depth.png
[JNet_439_pretrain_1_output_plane]: /experiments/images/JNet_439_pretrain_1_output_plane.png
[JNet_439_pretrain_2_label_depth]: /experiments/images/JNet_439_pretrain_2_label_depth.png
[JNet_439_pretrain_2_label_plane]: /experiments/images/JNet_439_pretrain_2_label_plane.png
[JNet_439_pretrain_2_original_depth]: /experiments/images/JNet_439_pretrain_2_original_depth.png
[JNet_439_pretrain_2_original_plane]: /experiments/images/JNet_439_pretrain_2_original_plane.png
[JNet_439_pretrain_2_output_depth]: /experiments/images/JNet_439_pretrain_2_output_depth.png
[JNet_439_pretrain_2_output_plane]: /experiments/images/JNet_439_pretrain_2_output_plane.png
[JNet_439_pretrain_3_label_depth]: /experiments/images/JNet_439_pretrain_3_label_depth.png
[JNet_439_pretrain_3_label_plane]: /experiments/images/JNet_439_pretrain_3_label_plane.png
[JNet_439_pretrain_3_original_depth]: /experiments/images/JNet_439_pretrain_3_original_depth.png
[JNet_439_pretrain_3_original_plane]: /experiments/images/JNet_439_pretrain_3_original_plane.png
[JNet_439_pretrain_3_output_depth]: /experiments/images/JNet_439_pretrain_3_output_depth.png
[JNet_439_pretrain_3_output_plane]: /experiments/images/JNet_439_pretrain_3_output_plane.png
[JNet_439_pretrain_4_label_depth]: /experiments/images/JNet_439_pretrain_4_label_depth.png
[JNet_439_pretrain_4_label_plane]: /experiments/images/JNet_439_pretrain_4_label_plane.png
[JNet_439_pretrain_4_original_depth]: /experiments/images/JNet_439_pretrain_4_original_depth.png
[JNet_439_pretrain_4_original_plane]: /experiments/images/JNet_439_pretrain_4_original_plane.png
[JNet_439_pretrain_4_output_depth]: /experiments/images/JNet_439_pretrain_4_output_depth.png
[JNet_439_pretrain_4_output_plane]: /experiments/images/JNet_439_pretrain_4_output_plane.png
[JNet_439_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_439_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi000_original_depth.png
[JNet_439_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi000_output_depth.png
[JNet_439_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi000_reconst_depth.png
[JNet_439_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_439_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi001_original_depth.png
[JNet_439_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi001_output_depth.png
[JNet_439_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi001_reconst_depth.png
[JNet_439_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_439_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi002_original_depth.png
[JNet_439_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi002_output_depth.png
[JNet_439_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi002_reconst_depth.png
[JNet_439_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_439_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi003_original_depth.png
[JNet_439_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi003_output_depth.png
[JNet_439_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi003_reconst_depth.png
[JNet_439_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_439_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi004_original_depth.png
[JNet_439_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi004_output_depth.png
[JNet_439_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_439_pretrain_beads_001_roi004_reconst_depth.png
[JNet_439_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_439_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi000_original_depth.png
[JNet_439_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi000_output_depth.png
[JNet_439_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi000_reconst_depth.png
[JNet_439_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_439_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi001_original_depth.png
[JNet_439_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi001_output_depth.png
[JNet_439_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi001_reconst_depth.png
[JNet_439_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_439_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi002_original_depth.png
[JNet_439_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi002_output_depth.png
[JNet_439_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_439_pretrain_beads_002_roi002_reconst_depth.png
[JNet_441_0_label_depth]: /experiments/images/JNet_441_0_label_depth.png
[JNet_441_0_label_plane]: /experiments/images/JNet_441_0_label_plane.png
[JNet_441_0_original_depth]: /experiments/images/JNet_441_0_original_depth.png
[JNet_441_0_original_plane]: /experiments/images/JNet_441_0_original_plane.png
[JNet_441_0_output_depth]: /experiments/images/JNet_441_0_output_depth.png
[JNet_441_0_output_plane]: /experiments/images/JNet_441_0_output_plane.png
[JNet_441_1_label_depth]: /experiments/images/JNet_441_1_label_depth.png
[JNet_441_1_label_plane]: /experiments/images/JNet_441_1_label_plane.png
[JNet_441_1_original_depth]: /experiments/images/JNet_441_1_original_depth.png
[JNet_441_1_original_plane]: /experiments/images/JNet_441_1_original_plane.png
[JNet_441_1_output_depth]: /experiments/images/JNet_441_1_output_depth.png
[JNet_441_1_output_plane]: /experiments/images/JNet_441_1_output_plane.png
[JNet_441_2_label_depth]: /experiments/images/JNet_441_2_label_depth.png
[JNet_441_2_label_plane]: /experiments/images/JNet_441_2_label_plane.png
[JNet_441_2_original_depth]: /experiments/images/JNet_441_2_original_depth.png
[JNet_441_2_original_plane]: /experiments/images/JNet_441_2_original_plane.png
[JNet_441_2_output_depth]: /experiments/images/JNet_441_2_output_depth.png
[JNet_441_2_output_plane]: /experiments/images/JNet_441_2_output_plane.png
[JNet_441_3_label_depth]: /experiments/images/JNet_441_3_label_depth.png
[JNet_441_3_label_plane]: /experiments/images/JNet_441_3_label_plane.png
[JNet_441_3_original_depth]: /experiments/images/JNet_441_3_original_depth.png
[JNet_441_3_original_plane]: /experiments/images/JNet_441_3_original_plane.png
[JNet_441_3_output_depth]: /experiments/images/JNet_441_3_output_depth.png
[JNet_441_3_output_plane]: /experiments/images/JNet_441_3_output_plane.png
[JNet_441_4_label_depth]: /experiments/images/JNet_441_4_label_depth.png
[JNet_441_4_label_plane]: /experiments/images/JNet_441_4_label_plane.png
[JNet_441_4_original_depth]: /experiments/images/JNet_441_4_original_depth.png
[JNet_441_4_original_plane]: /experiments/images/JNet_441_4_original_plane.png
[JNet_441_4_output_depth]: /experiments/images/JNet_441_4_output_depth.png
[JNet_441_4_output_plane]: /experiments/images/JNet_441_4_output_plane.png
[JNet_441_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_441_beads_001_roi000_heatmap_depth.png
[JNet_441_beads_001_roi000_original_depth]: /experiments/images/JNet_441_beads_001_roi000_original_depth.png
[JNet_441_beads_001_roi000_output_depth]: /experiments/images/JNet_441_beads_001_roi000_output_depth.png
[JNet_441_beads_001_roi000_reconst_depth]: /experiments/images/JNet_441_beads_001_roi000_reconst_depth.png
[JNet_441_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_441_beads_001_roi001_heatmap_depth.png
[JNet_441_beads_001_roi001_original_depth]: /experiments/images/JNet_441_beads_001_roi001_original_depth.png
[JNet_441_beads_001_roi001_output_depth]: /experiments/images/JNet_441_beads_001_roi001_output_depth.png
[JNet_441_beads_001_roi001_reconst_depth]: /experiments/images/JNet_441_beads_001_roi001_reconst_depth.png
[JNet_441_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_441_beads_001_roi002_heatmap_depth.png
[JNet_441_beads_001_roi002_original_depth]: /experiments/images/JNet_441_beads_001_roi002_original_depth.png
[JNet_441_beads_001_roi002_output_depth]: /experiments/images/JNet_441_beads_001_roi002_output_depth.png
[JNet_441_beads_001_roi002_reconst_depth]: /experiments/images/JNet_441_beads_001_roi002_reconst_depth.png
[JNet_441_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_441_beads_001_roi003_heatmap_depth.png
[JNet_441_beads_001_roi003_original_depth]: /experiments/images/JNet_441_beads_001_roi003_original_depth.png
[JNet_441_beads_001_roi003_output_depth]: /experiments/images/JNet_441_beads_001_roi003_output_depth.png
[JNet_441_beads_001_roi003_reconst_depth]: /experiments/images/JNet_441_beads_001_roi003_reconst_depth.png
[JNet_441_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_441_beads_001_roi004_heatmap_depth.png
[JNet_441_beads_001_roi004_original_depth]: /experiments/images/JNet_441_beads_001_roi004_original_depth.png
[JNet_441_beads_001_roi004_output_depth]: /experiments/images/JNet_441_beads_001_roi004_output_depth.png
[JNet_441_beads_001_roi004_reconst_depth]: /experiments/images/JNet_441_beads_001_roi004_reconst_depth.png
[JNet_441_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_441_beads_002_roi000_heatmap_depth.png
[JNet_441_beads_002_roi000_original_depth]: /experiments/images/JNet_441_beads_002_roi000_original_depth.png
[JNet_441_beads_002_roi000_output_depth]: /experiments/images/JNet_441_beads_002_roi000_output_depth.png
[JNet_441_beads_002_roi000_reconst_depth]: /experiments/images/JNet_441_beads_002_roi000_reconst_depth.png
[JNet_441_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_441_beads_002_roi001_heatmap_depth.png
[JNet_441_beads_002_roi001_original_depth]: /experiments/images/JNet_441_beads_002_roi001_original_depth.png
[JNet_441_beads_002_roi001_output_depth]: /experiments/images/JNet_441_beads_002_roi001_output_depth.png
[JNet_441_beads_002_roi001_reconst_depth]: /experiments/images/JNet_441_beads_002_roi001_reconst_depth.png
[JNet_441_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_441_beads_002_roi002_heatmap_depth.png
[JNet_441_beads_002_roi002_original_depth]: /experiments/images/JNet_441_beads_002_roi002_original_depth.png
[JNet_441_beads_002_roi002_output_depth]: /experiments/images/JNet_441_beads_002_roi002_output_depth.png
[JNet_441_beads_002_roi002_reconst_depth]: /experiments/images/JNet_441_beads_002_roi002_reconst_depth.png
[JNet_441_psf_post]: /experiments/images/JNet_441_psf_post.png
[JNet_441_psf_pre]: /experiments/images/JNet_441_psf_pre.png
[finetuned]: /experiments/tmp/JNet_441_train.png
[pretrained_model]: /experiments/tmp/JNet_439_pretrain_train.png
