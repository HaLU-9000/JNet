



# JNet_460 Report
  
the parameters to replicate the results of JNet_460. vibrate in fine tuning, , mu_z = 1.2, sig_z = 1.27  
pretrained model : JNet_459_pretrain
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
|NA|1.0||
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
  
mean MSE: 0.025148844346404076, mean BCE: 0.09901462495326996
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_0_original_plane]|![JNet_459_pretrain_0_output_plane]|![JNet_459_pretrain_0_label_plane]|
  
MSE: 0.03577422723174095, BCE: 0.13951830565929413  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_0_original_depth]|![JNet_459_pretrain_0_output_depth]|![JNet_459_pretrain_0_label_depth]|
  
MSE: 0.03577422723174095, BCE: 0.13951830565929413  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_1_original_plane]|![JNet_459_pretrain_1_output_plane]|![JNet_459_pretrain_1_label_plane]|
  
MSE: 0.026515021920204163, BCE: 0.09852592647075653  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_1_original_depth]|![JNet_459_pretrain_1_output_depth]|![JNet_459_pretrain_1_label_depth]|
  
MSE: 0.026515021920204163, BCE: 0.09852592647075653  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_2_original_plane]|![JNet_459_pretrain_2_output_plane]|![JNet_459_pretrain_2_label_plane]|
  
MSE: 0.023136641830205917, BCE: 0.09406254440546036  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_2_original_depth]|![JNet_459_pretrain_2_output_depth]|![JNet_459_pretrain_2_label_depth]|
  
MSE: 0.023136641830205917, BCE: 0.09406254440546036  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_3_original_plane]|![JNet_459_pretrain_3_output_plane]|![JNet_459_pretrain_3_label_plane]|
  
MSE: 0.02682955004274845, BCE: 0.1032748818397522  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_3_original_depth]|![JNet_459_pretrain_3_output_depth]|![JNet_459_pretrain_3_label_depth]|
  
MSE: 0.02682955004274845, BCE: 0.1032748818397522  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_4_original_plane]|![JNet_459_pretrain_4_output_plane]|![JNet_459_pretrain_4_label_plane]|
  
MSE: 0.013488785363733768, BCE: 0.05969148874282837  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_4_original_depth]|![JNet_459_pretrain_4_output_depth]|![JNet_459_pretrain_4_label_depth]|
  
MSE: 0.013488785363733768, BCE: 0.05969148874282837  
  
mean MSE: 0.03472191467881203, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_460_0_original_plane]|![JNet_460_0_output_plane]|![JNet_460_0_label_plane]|
  
MSE: 0.027852624654769897, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_460_0_original_depth]|![JNet_460_0_output_depth]|![JNet_460_0_label_depth]|
  
MSE: 0.027852624654769897, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_460_1_original_plane]|![JNet_460_1_output_plane]|![JNet_460_1_label_plane]|
  
MSE: 0.04266282543540001, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_460_1_original_depth]|![JNet_460_1_output_depth]|![JNet_460_1_label_depth]|
  
MSE: 0.04266282543540001, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_460_2_original_plane]|![JNet_460_2_output_plane]|![JNet_460_2_label_plane]|
  
MSE: 0.02708100900053978, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_460_2_original_depth]|![JNet_460_2_output_depth]|![JNet_460_2_label_depth]|
  
MSE: 0.02708100900053978, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_460_3_original_plane]|![JNet_460_3_output_plane]|![JNet_460_3_label_plane]|
  
MSE: 0.0350257083773613, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_460_3_original_depth]|![JNet_460_3_output_depth]|![JNet_460_3_label_depth]|
  
MSE: 0.0350257083773613, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_460_4_original_plane]|![JNet_460_4_output_plane]|![JNet_460_4_label_plane]|
  
MSE: 0.04098739102482796, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_460_4_original_depth]|![JNet_460_4_output_depth]|![JNet_460_4_label_depth]|
  
MSE: 0.04098739102482796, BCE: nan  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_459_pretrain_beads_001_roi000_original_depth]|![JNet_459_pretrain_beads_001_roi000_output_depth]|![JNet_459_pretrain_beads_001_roi000_reconst_depth]|![JNet_459_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 17.169593750000004, MSE: 0.0019207323202863336, quantized loss: 0.002217733534052968  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_459_pretrain_beads_001_roi001_original_depth]|![JNet_459_pretrain_beads_001_roi001_output_depth]|![JNet_459_pretrain_beads_001_roi001_reconst_depth]|![JNet_459_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 22.692263671875004, MSE: 0.002105971099808812, quantized loss: 0.0023864051327109337  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_459_pretrain_beads_001_roi002_original_depth]|![JNet_459_pretrain_beads_001_roi002_output_depth]|![JNet_459_pretrain_beads_001_roi002_reconst_depth]|![JNet_459_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 19.397166015625004, MSE: 0.0014782886719331145, quantized loss: 0.0021182303316891193  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_459_pretrain_beads_001_roi003_original_depth]|![JNet_459_pretrain_beads_001_roi003_output_depth]|![JNet_459_pretrain_beads_001_roi003_reconst_depth]|![JNet_459_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 28.835136718750007, MSE: 0.0027383111882954836, quantized loss: 0.0032621813006699085  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_459_pretrain_beads_001_roi004_original_depth]|![JNet_459_pretrain_beads_001_roi004_output_depth]|![JNet_459_pretrain_beads_001_roi004_reconst_depth]|![JNet_459_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 20.346513671875005, MSE: 0.0017446147976443172, quantized loss: 0.002174000022932887  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_459_pretrain_beads_002_roi000_original_depth]|![JNet_459_pretrain_beads_002_roi000_output_depth]|![JNet_459_pretrain_beads_002_roi000_reconst_depth]|![JNet_459_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 21.781513671875004, MSE: 0.002179734641686082, quantized loss: 0.002291907323524356  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_459_pretrain_beads_002_roi001_original_depth]|![JNet_459_pretrain_beads_002_roi001_output_depth]|![JNet_459_pretrain_beads_002_roi001_reconst_depth]|![JNet_459_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 19.927615234375004, MSE: 0.0018557821167632937, quantized loss: 0.0021698966156691313  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_459_pretrain_beads_002_roi002_original_depth]|![JNet_459_pretrain_beads_002_roi002_output_depth]|![JNet_459_pretrain_beads_002_roi002_reconst_depth]|![JNet_459_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 20.751730468750004, MSE: 0.0019171764142811298, quantized loss: 0.002215395448729396  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_460_beads_001_roi000_original_depth]|![JNet_460_beads_001_roi000_output_depth]|![JNet_460_beads_001_roi000_reconst_depth]|![JNet_460_beads_001_roi000_heatmap_depth]|
  
volume: 6.221910644531252, MSE: 0.00019342562882229686, quantized loss: 1.0924088201136328e-05  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_460_beads_001_roi001_original_depth]|![JNet_460_beads_001_roi001_output_depth]|![JNet_460_beads_001_roi001_reconst_depth]|![JNet_460_beads_001_roi001_heatmap_depth]|
  
volume: 9.707064453125001, MSE: 0.0007079380447976291, quantized loss: 1.5819532563909888e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_460_beads_001_roi002_original_depth]|![JNet_460_beads_001_roi002_output_depth]|![JNet_460_beads_001_roi002_reconst_depth]|![JNet_460_beads_001_roi002_heatmap_depth]|
  
volume: 6.268752929687501, MSE: 0.00013932929141446948, quantized loss: 1.1847054338431917e-05  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_460_beads_001_roi003_original_depth]|![JNet_460_beads_001_roi003_output_depth]|![JNet_460_beads_001_roi003_reconst_depth]|![JNet_460_beads_001_roi003_heatmap_depth]|
  
volume: 10.187797851562502, MSE: 0.0004723460879176855, quantized loss: 1.6087280528154224e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_460_beads_001_roi004_original_depth]|![JNet_460_beads_001_roi004_output_depth]|![JNet_460_beads_001_roi004_reconst_depth]|![JNet_460_beads_001_roi004_heatmap_depth]|
  
volume: 6.683825195312502, MSE: 0.00017983435827773064, quantized loss: 1.2994693861401174e-05  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_460_beads_002_roi000_original_depth]|![JNet_460_beads_002_roi000_output_depth]|![JNet_460_beads_002_roi000_reconst_depth]|![JNet_460_beads_002_roi000_heatmap_depth]|
  
volume: 7.043940917968752, MSE: 0.0001917435583891347, quantized loss: 1.0732403097790666e-05  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_460_beads_002_roi001_original_depth]|![JNet_460_beads_002_roi001_output_depth]|![JNet_460_beads_002_roi001_reconst_depth]|![JNet_460_beads_002_roi001_heatmap_depth]|
  
volume: 6.564736816406252, MSE: 0.00016544012760277838, quantized loss: 1.1628738320723642e-05  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_460_beads_002_roi002_original_depth]|![JNet_460_beads_002_roi002_output_depth]|![JNet_460_beads_002_roi002_reconst_depth]|![JNet_460_beads_002_roi002_heatmap_depth]|
  
volume: 6.772329101562502, MSE: 0.0001692990626906976, quantized loss: 1.1031378562620375e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_460_psf_pre]|![JNet_460_psf_post]|

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
  



[JNet_459_pretrain_0_label_depth]: /experiments/images/JNet_459_pretrain_0_label_depth.png
[JNet_459_pretrain_0_label_plane]: /experiments/images/JNet_459_pretrain_0_label_plane.png
[JNet_459_pretrain_0_original_depth]: /experiments/images/JNet_459_pretrain_0_original_depth.png
[JNet_459_pretrain_0_original_plane]: /experiments/images/JNet_459_pretrain_0_original_plane.png
[JNet_459_pretrain_0_output_depth]: /experiments/images/JNet_459_pretrain_0_output_depth.png
[JNet_459_pretrain_0_output_plane]: /experiments/images/JNet_459_pretrain_0_output_plane.png
[JNet_459_pretrain_1_label_depth]: /experiments/images/JNet_459_pretrain_1_label_depth.png
[JNet_459_pretrain_1_label_plane]: /experiments/images/JNet_459_pretrain_1_label_plane.png
[JNet_459_pretrain_1_original_depth]: /experiments/images/JNet_459_pretrain_1_original_depth.png
[JNet_459_pretrain_1_original_plane]: /experiments/images/JNet_459_pretrain_1_original_plane.png
[JNet_459_pretrain_1_output_depth]: /experiments/images/JNet_459_pretrain_1_output_depth.png
[JNet_459_pretrain_1_output_plane]: /experiments/images/JNet_459_pretrain_1_output_plane.png
[JNet_459_pretrain_2_label_depth]: /experiments/images/JNet_459_pretrain_2_label_depth.png
[JNet_459_pretrain_2_label_plane]: /experiments/images/JNet_459_pretrain_2_label_plane.png
[JNet_459_pretrain_2_original_depth]: /experiments/images/JNet_459_pretrain_2_original_depth.png
[JNet_459_pretrain_2_original_plane]: /experiments/images/JNet_459_pretrain_2_original_plane.png
[JNet_459_pretrain_2_output_depth]: /experiments/images/JNet_459_pretrain_2_output_depth.png
[JNet_459_pretrain_2_output_plane]: /experiments/images/JNet_459_pretrain_2_output_plane.png
[JNet_459_pretrain_3_label_depth]: /experiments/images/JNet_459_pretrain_3_label_depth.png
[JNet_459_pretrain_3_label_plane]: /experiments/images/JNet_459_pretrain_3_label_plane.png
[JNet_459_pretrain_3_original_depth]: /experiments/images/JNet_459_pretrain_3_original_depth.png
[JNet_459_pretrain_3_original_plane]: /experiments/images/JNet_459_pretrain_3_original_plane.png
[JNet_459_pretrain_3_output_depth]: /experiments/images/JNet_459_pretrain_3_output_depth.png
[JNet_459_pretrain_3_output_plane]: /experiments/images/JNet_459_pretrain_3_output_plane.png
[JNet_459_pretrain_4_label_depth]: /experiments/images/JNet_459_pretrain_4_label_depth.png
[JNet_459_pretrain_4_label_plane]: /experiments/images/JNet_459_pretrain_4_label_plane.png
[JNet_459_pretrain_4_original_depth]: /experiments/images/JNet_459_pretrain_4_original_depth.png
[JNet_459_pretrain_4_original_plane]: /experiments/images/JNet_459_pretrain_4_original_plane.png
[JNet_459_pretrain_4_output_depth]: /experiments/images/JNet_459_pretrain_4_output_depth.png
[JNet_459_pretrain_4_output_plane]: /experiments/images/JNet_459_pretrain_4_output_plane.png
[JNet_459_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_459_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi000_original_depth.png
[JNet_459_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi000_output_depth.png
[JNet_459_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi000_reconst_depth.png
[JNet_459_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_459_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi001_original_depth.png
[JNet_459_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi001_output_depth.png
[JNet_459_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi001_reconst_depth.png
[JNet_459_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_459_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi002_original_depth.png
[JNet_459_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi002_output_depth.png
[JNet_459_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi002_reconst_depth.png
[JNet_459_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_459_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi003_original_depth.png
[JNet_459_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi003_output_depth.png
[JNet_459_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi003_reconst_depth.png
[JNet_459_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_459_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi004_original_depth.png
[JNet_459_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi004_output_depth.png
[JNet_459_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_459_pretrain_beads_001_roi004_reconst_depth.png
[JNet_459_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_459_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi000_original_depth.png
[JNet_459_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi000_output_depth.png
[JNet_459_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi000_reconst_depth.png
[JNet_459_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_459_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi001_original_depth.png
[JNet_459_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi001_output_depth.png
[JNet_459_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi001_reconst_depth.png
[JNet_459_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_459_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi002_original_depth.png
[JNet_459_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi002_output_depth.png
[JNet_459_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_459_pretrain_beads_002_roi002_reconst_depth.png
[JNet_460_0_label_depth]: /experiments/images/JNet_460_0_label_depth.png
[JNet_460_0_label_plane]: /experiments/images/JNet_460_0_label_plane.png
[JNet_460_0_original_depth]: /experiments/images/JNet_460_0_original_depth.png
[JNet_460_0_original_plane]: /experiments/images/JNet_460_0_original_plane.png
[JNet_460_0_output_depth]: /experiments/images/JNet_460_0_output_depth.png
[JNet_460_0_output_plane]: /experiments/images/JNet_460_0_output_plane.png
[JNet_460_1_label_depth]: /experiments/images/JNet_460_1_label_depth.png
[JNet_460_1_label_plane]: /experiments/images/JNet_460_1_label_plane.png
[JNet_460_1_original_depth]: /experiments/images/JNet_460_1_original_depth.png
[JNet_460_1_original_plane]: /experiments/images/JNet_460_1_original_plane.png
[JNet_460_1_output_depth]: /experiments/images/JNet_460_1_output_depth.png
[JNet_460_1_output_plane]: /experiments/images/JNet_460_1_output_plane.png
[JNet_460_2_label_depth]: /experiments/images/JNet_460_2_label_depth.png
[JNet_460_2_label_plane]: /experiments/images/JNet_460_2_label_plane.png
[JNet_460_2_original_depth]: /experiments/images/JNet_460_2_original_depth.png
[JNet_460_2_original_plane]: /experiments/images/JNet_460_2_original_plane.png
[JNet_460_2_output_depth]: /experiments/images/JNet_460_2_output_depth.png
[JNet_460_2_output_plane]: /experiments/images/JNet_460_2_output_plane.png
[JNet_460_3_label_depth]: /experiments/images/JNet_460_3_label_depth.png
[JNet_460_3_label_plane]: /experiments/images/JNet_460_3_label_plane.png
[JNet_460_3_original_depth]: /experiments/images/JNet_460_3_original_depth.png
[JNet_460_3_original_plane]: /experiments/images/JNet_460_3_original_plane.png
[JNet_460_3_output_depth]: /experiments/images/JNet_460_3_output_depth.png
[JNet_460_3_output_plane]: /experiments/images/JNet_460_3_output_plane.png
[JNet_460_4_label_depth]: /experiments/images/JNet_460_4_label_depth.png
[JNet_460_4_label_plane]: /experiments/images/JNet_460_4_label_plane.png
[JNet_460_4_original_depth]: /experiments/images/JNet_460_4_original_depth.png
[JNet_460_4_original_plane]: /experiments/images/JNet_460_4_original_plane.png
[JNet_460_4_output_depth]: /experiments/images/JNet_460_4_output_depth.png
[JNet_460_4_output_plane]: /experiments/images/JNet_460_4_output_plane.png
[JNet_460_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_460_beads_001_roi000_heatmap_depth.png
[JNet_460_beads_001_roi000_original_depth]: /experiments/images/JNet_460_beads_001_roi000_original_depth.png
[JNet_460_beads_001_roi000_output_depth]: /experiments/images/JNet_460_beads_001_roi000_output_depth.png
[JNet_460_beads_001_roi000_reconst_depth]: /experiments/images/JNet_460_beads_001_roi000_reconst_depth.png
[JNet_460_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_460_beads_001_roi001_heatmap_depth.png
[JNet_460_beads_001_roi001_original_depth]: /experiments/images/JNet_460_beads_001_roi001_original_depth.png
[JNet_460_beads_001_roi001_output_depth]: /experiments/images/JNet_460_beads_001_roi001_output_depth.png
[JNet_460_beads_001_roi001_reconst_depth]: /experiments/images/JNet_460_beads_001_roi001_reconst_depth.png
[JNet_460_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_460_beads_001_roi002_heatmap_depth.png
[JNet_460_beads_001_roi002_original_depth]: /experiments/images/JNet_460_beads_001_roi002_original_depth.png
[JNet_460_beads_001_roi002_output_depth]: /experiments/images/JNet_460_beads_001_roi002_output_depth.png
[JNet_460_beads_001_roi002_reconst_depth]: /experiments/images/JNet_460_beads_001_roi002_reconst_depth.png
[JNet_460_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_460_beads_001_roi003_heatmap_depth.png
[JNet_460_beads_001_roi003_original_depth]: /experiments/images/JNet_460_beads_001_roi003_original_depth.png
[JNet_460_beads_001_roi003_output_depth]: /experiments/images/JNet_460_beads_001_roi003_output_depth.png
[JNet_460_beads_001_roi003_reconst_depth]: /experiments/images/JNet_460_beads_001_roi003_reconst_depth.png
[JNet_460_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_460_beads_001_roi004_heatmap_depth.png
[JNet_460_beads_001_roi004_original_depth]: /experiments/images/JNet_460_beads_001_roi004_original_depth.png
[JNet_460_beads_001_roi004_output_depth]: /experiments/images/JNet_460_beads_001_roi004_output_depth.png
[JNet_460_beads_001_roi004_reconst_depth]: /experiments/images/JNet_460_beads_001_roi004_reconst_depth.png
[JNet_460_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_460_beads_002_roi000_heatmap_depth.png
[JNet_460_beads_002_roi000_original_depth]: /experiments/images/JNet_460_beads_002_roi000_original_depth.png
[JNet_460_beads_002_roi000_output_depth]: /experiments/images/JNet_460_beads_002_roi000_output_depth.png
[JNet_460_beads_002_roi000_reconst_depth]: /experiments/images/JNet_460_beads_002_roi000_reconst_depth.png
[JNet_460_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_460_beads_002_roi001_heatmap_depth.png
[JNet_460_beads_002_roi001_original_depth]: /experiments/images/JNet_460_beads_002_roi001_original_depth.png
[JNet_460_beads_002_roi001_output_depth]: /experiments/images/JNet_460_beads_002_roi001_output_depth.png
[JNet_460_beads_002_roi001_reconst_depth]: /experiments/images/JNet_460_beads_002_roi001_reconst_depth.png
[JNet_460_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_460_beads_002_roi002_heatmap_depth.png
[JNet_460_beads_002_roi002_original_depth]: /experiments/images/JNet_460_beads_002_roi002_original_depth.png
[JNet_460_beads_002_roi002_output_depth]: /experiments/images/JNet_460_beads_002_roi002_output_depth.png
[JNet_460_beads_002_roi002_reconst_depth]: /experiments/images/JNet_460_beads_002_roi002_reconst_depth.png
[JNet_460_psf_post]: /experiments/images/JNet_460_psf_post.png
[JNet_460_psf_pre]: /experiments/images/JNet_460_psf_pre.png
[finetuned]: /experiments/tmp/JNet_460_train.png
[pretrained_model]: /experiments/tmp/JNet_459_pretrain_train.png
