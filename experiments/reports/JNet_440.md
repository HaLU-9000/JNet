



# JNet_440 Report
  
the parameters to replicate the results of JNet_440. nearest interp of PSF, NA=0.7, mu_z = 0.3, sig_z = 1.27  
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
  
mean MSE: 0.021374497562646866, mean BCE: 0.07804447412490845
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_0_original_plane]|![JNet_439_pretrain_0_output_plane]|![JNet_439_pretrain_0_label_plane]|
  
MSE: 0.017170364037156105, BCE: 0.0630248486995697  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_0_original_depth]|![JNet_439_pretrain_0_output_depth]|![JNet_439_pretrain_0_label_depth]|
  
MSE: 0.017170364037156105, BCE: 0.0630248486995697  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_1_original_plane]|![JNet_439_pretrain_1_output_plane]|![JNet_439_pretrain_1_label_plane]|
  
MSE: 0.019116532057523727, BCE: 0.06756970286369324  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_1_original_depth]|![JNet_439_pretrain_1_output_depth]|![JNet_439_pretrain_1_label_depth]|
  
MSE: 0.019116532057523727, BCE: 0.06756970286369324  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_2_original_plane]|![JNet_439_pretrain_2_output_plane]|![JNet_439_pretrain_2_label_plane]|
  
MSE: 0.017944572493433952, BCE: 0.06605803221464157  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_2_original_depth]|![JNet_439_pretrain_2_output_depth]|![JNet_439_pretrain_2_label_depth]|
  
MSE: 0.017944572493433952, BCE: 0.06605803221464157  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_3_original_plane]|![JNet_439_pretrain_3_output_plane]|![JNet_439_pretrain_3_label_plane]|
  
MSE: 0.017758838832378387, BCE: 0.06395862996578217  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_3_original_depth]|![JNet_439_pretrain_3_output_depth]|![JNet_439_pretrain_3_label_depth]|
  
MSE: 0.017758838832378387, BCE: 0.06395862996578217  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_4_original_plane]|![JNet_439_pretrain_4_output_plane]|![JNet_439_pretrain_4_label_plane]|
  
MSE: 0.03488217666745186, BCE: 0.12961111962795258  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_439_pretrain_4_original_depth]|![JNet_439_pretrain_4_output_depth]|![JNet_439_pretrain_4_label_depth]|
  
MSE: 0.03488217666745186, BCE: 0.12961111962795258  
  
mean MSE: 0.03431936353445053, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_440_0_original_plane]|![JNet_440_0_output_plane]|![JNet_440_0_label_plane]|
  
MSE: 0.04408399760723114, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_440_0_original_depth]|![JNet_440_0_output_depth]|![JNet_440_0_label_depth]|
  
MSE: 0.04408399760723114, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_440_1_original_plane]|![JNet_440_1_output_plane]|![JNet_440_1_label_plane]|
  
MSE: 0.03679049760103226, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_440_1_original_depth]|![JNet_440_1_output_depth]|![JNet_440_1_label_depth]|
  
MSE: 0.03679049760103226, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_440_2_original_plane]|![JNet_440_2_output_plane]|![JNet_440_2_label_plane]|
  
MSE: 0.03644803538918495, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_440_2_original_depth]|![JNet_440_2_output_depth]|![JNet_440_2_label_depth]|
  
MSE: 0.03644803538918495, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_440_3_original_plane]|![JNet_440_3_output_plane]|![JNet_440_3_label_plane]|
  
MSE: 0.0326593741774559, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_440_3_original_depth]|![JNet_440_3_output_depth]|![JNet_440_3_label_depth]|
  
MSE: 0.0326593741774559, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_440_4_original_plane]|![JNet_440_4_output_plane]|![JNet_440_4_label_plane]|
  
MSE: 0.021614914759993553, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_440_4_original_depth]|![JNet_440_4_output_depth]|![JNet_440_4_label_depth]|
  
MSE: 0.021614914759993553, BCE: nan  

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
|![JNet_440_beads_001_roi000_original_depth]|![JNet_440_beads_001_roi000_output_depth]|![JNet_440_beads_001_roi000_reconst_depth]|![JNet_440_beads_001_roi000_heatmap_depth]|
  
volume: 7.565687988281252, MSE: 0.00012743992556352168, quantized loss: 6.0260513237153646e-06  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_440_beads_001_roi001_original_depth]|![JNet_440_beads_001_roi001_output_depth]|![JNet_440_beads_001_roi001_reconst_depth]|![JNet_440_beads_001_roi001_heatmap_depth]|
  
volume: 12.041467773437503, MSE: 0.0005629445076920092, quantized loss: 8.691951734363101e-06  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_440_beads_001_roi002_original_depth]|![JNet_440_beads_001_roi002_output_depth]|![JNet_440_beads_001_roi002_reconst_depth]|![JNet_440_beads_001_roi002_heatmap_depth]|
  
volume: 7.509671875000002, MSE: 8.983881707536057e-05, quantized loss: 6.1523714975919574e-06  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_440_beads_001_roi003_original_depth]|![JNet_440_beads_001_roi003_output_depth]|![JNet_440_beads_001_roi003_reconst_depth]|![JNet_440_beads_001_roi003_heatmap_depth]|
  
volume: 12.648093750000003, MSE: 0.00033747198176570237, quantized loss: 9.622136531106662e-06  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_440_beads_001_roi004_original_depth]|![JNet_440_beads_001_roi004_output_depth]|![JNet_440_beads_001_roi004_reconst_depth]|![JNet_440_beads_001_roi004_heatmap_depth]|
  
volume: 8.139292968750002, MSE: 0.00011818311031674966, quantized loss: 6.4813643803063314e-06  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_440_beads_002_roi000_original_depth]|![JNet_440_beads_002_roi000_output_depth]|![JNet_440_beads_002_roi000_reconst_depth]|![JNet_440_beads_002_roi000_heatmap_depth]|
  
volume: 8.617209960937503, MSE: 0.00012601821799762547, quantized loss: 6.609957836190006e-06  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_440_beads_002_roi001_original_depth]|![JNet_440_beads_002_roi001_output_depth]|![JNet_440_beads_002_roi001_reconst_depth]|![JNet_440_beads_002_roi001_heatmap_depth]|
  
volume: 8.027253906250001, MSE: 8.830626757116988e-05, quantized loss: 6.330762062134454e-06  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_440_beads_002_roi002_original_depth]|![JNet_440_beads_002_roi002_output_depth]|![JNet_440_beads_002_roi002_reconst_depth]|![JNet_440_beads_002_roi002_heatmap_depth]|
  
volume: 8.275458007812501, MSE: 0.0001048069098033011, quantized loss: 5.984782546875067e-06  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_440_psf_pre]|![JNet_440_psf_post]|

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
[JNet_440_0_label_depth]: /experiments/images/JNet_440_0_label_depth.png
[JNet_440_0_label_plane]: /experiments/images/JNet_440_0_label_plane.png
[JNet_440_0_original_depth]: /experiments/images/JNet_440_0_original_depth.png
[JNet_440_0_original_plane]: /experiments/images/JNet_440_0_original_plane.png
[JNet_440_0_output_depth]: /experiments/images/JNet_440_0_output_depth.png
[JNet_440_0_output_plane]: /experiments/images/JNet_440_0_output_plane.png
[JNet_440_1_label_depth]: /experiments/images/JNet_440_1_label_depth.png
[JNet_440_1_label_plane]: /experiments/images/JNet_440_1_label_plane.png
[JNet_440_1_original_depth]: /experiments/images/JNet_440_1_original_depth.png
[JNet_440_1_original_plane]: /experiments/images/JNet_440_1_original_plane.png
[JNet_440_1_output_depth]: /experiments/images/JNet_440_1_output_depth.png
[JNet_440_1_output_plane]: /experiments/images/JNet_440_1_output_plane.png
[JNet_440_2_label_depth]: /experiments/images/JNet_440_2_label_depth.png
[JNet_440_2_label_plane]: /experiments/images/JNet_440_2_label_plane.png
[JNet_440_2_original_depth]: /experiments/images/JNet_440_2_original_depth.png
[JNet_440_2_original_plane]: /experiments/images/JNet_440_2_original_plane.png
[JNet_440_2_output_depth]: /experiments/images/JNet_440_2_output_depth.png
[JNet_440_2_output_plane]: /experiments/images/JNet_440_2_output_plane.png
[JNet_440_3_label_depth]: /experiments/images/JNet_440_3_label_depth.png
[JNet_440_3_label_plane]: /experiments/images/JNet_440_3_label_plane.png
[JNet_440_3_original_depth]: /experiments/images/JNet_440_3_original_depth.png
[JNet_440_3_original_plane]: /experiments/images/JNet_440_3_original_plane.png
[JNet_440_3_output_depth]: /experiments/images/JNet_440_3_output_depth.png
[JNet_440_3_output_plane]: /experiments/images/JNet_440_3_output_plane.png
[JNet_440_4_label_depth]: /experiments/images/JNet_440_4_label_depth.png
[JNet_440_4_label_plane]: /experiments/images/JNet_440_4_label_plane.png
[JNet_440_4_original_depth]: /experiments/images/JNet_440_4_original_depth.png
[JNet_440_4_original_plane]: /experiments/images/JNet_440_4_original_plane.png
[JNet_440_4_output_depth]: /experiments/images/JNet_440_4_output_depth.png
[JNet_440_4_output_plane]: /experiments/images/JNet_440_4_output_plane.png
[JNet_440_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_440_beads_001_roi000_heatmap_depth.png
[JNet_440_beads_001_roi000_original_depth]: /experiments/images/JNet_440_beads_001_roi000_original_depth.png
[JNet_440_beads_001_roi000_output_depth]: /experiments/images/JNet_440_beads_001_roi000_output_depth.png
[JNet_440_beads_001_roi000_reconst_depth]: /experiments/images/JNet_440_beads_001_roi000_reconst_depth.png
[JNet_440_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_440_beads_001_roi001_heatmap_depth.png
[JNet_440_beads_001_roi001_original_depth]: /experiments/images/JNet_440_beads_001_roi001_original_depth.png
[JNet_440_beads_001_roi001_output_depth]: /experiments/images/JNet_440_beads_001_roi001_output_depth.png
[JNet_440_beads_001_roi001_reconst_depth]: /experiments/images/JNet_440_beads_001_roi001_reconst_depth.png
[JNet_440_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_440_beads_001_roi002_heatmap_depth.png
[JNet_440_beads_001_roi002_original_depth]: /experiments/images/JNet_440_beads_001_roi002_original_depth.png
[JNet_440_beads_001_roi002_output_depth]: /experiments/images/JNet_440_beads_001_roi002_output_depth.png
[JNet_440_beads_001_roi002_reconst_depth]: /experiments/images/JNet_440_beads_001_roi002_reconst_depth.png
[JNet_440_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_440_beads_001_roi003_heatmap_depth.png
[JNet_440_beads_001_roi003_original_depth]: /experiments/images/JNet_440_beads_001_roi003_original_depth.png
[JNet_440_beads_001_roi003_output_depth]: /experiments/images/JNet_440_beads_001_roi003_output_depth.png
[JNet_440_beads_001_roi003_reconst_depth]: /experiments/images/JNet_440_beads_001_roi003_reconst_depth.png
[JNet_440_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_440_beads_001_roi004_heatmap_depth.png
[JNet_440_beads_001_roi004_original_depth]: /experiments/images/JNet_440_beads_001_roi004_original_depth.png
[JNet_440_beads_001_roi004_output_depth]: /experiments/images/JNet_440_beads_001_roi004_output_depth.png
[JNet_440_beads_001_roi004_reconst_depth]: /experiments/images/JNet_440_beads_001_roi004_reconst_depth.png
[JNet_440_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_440_beads_002_roi000_heatmap_depth.png
[JNet_440_beads_002_roi000_original_depth]: /experiments/images/JNet_440_beads_002_roi000_original_depth.png
[JNet_440_beads_002_roi000_output_depth]: /experiments/images/JNet_440_beads_002_roi000_output_depth.png
[JNet_440_beads_002_roi000_reconst_depth]: /experiments/images/JNet_440_beads_002_roi000_reconst_depth.png
[JNet_440_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_440_beads_002_roi001_heatmap_depth.png
[JNet_440_beads_002_roi001_original_depth]: /experiments/images/JNet_440_beads_002_roi001_original_depth.png
[JNet_440_beads_002_roi001_output_depth]: /experiments/images/JNet_440_beads_002_roi001_output_depth.png
[JNet_440_beads_002_roi001_reconst_depth]: /experiments/images/JNet_440_beads_002_roi001_reconst_depth.png
[JNet_440_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_440_beads_002_roi002_heatmap_depth.png
[JNet_440_beads_002_roi002_original_depth]: /experiments/images/JNet_440_beads_002_roi002_original_depth.png
[JNet_440_beads_002_roi002_output_depth]: /experiments/images/JNet_440_beads_002_roi002_output_depth.png
[JNet_440_beads_002_roi002_reconst_depth]: /experiments/images/JNet_440_beads_002_roi002_reconst_depth.png
[JNet_440_psf_post]: /experiments/images/JNet_440_psf_post.png
[JNet_440_psf_pre]: /experiments/images/JNet_440_psf_pre.png
[finetuned]: /experiments/tmp/JNet_440_train.png
[pretrained_model]: /experiments/tmp/JNet_439_pretrain_train.png
