



# JNet_461 Report
  
the parameters to replicate the results of JNet_460. ewc and vibrate in fine tuning, , mu_z = 1.2, sig_z = 1.27  
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
|partial|None|
|ewc|ewc|
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
  
mean MSE: 0.022610744461417198, mean BCE: 0.090041883289814
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_0_original_plane]|![JNet_459_pretrain_0_output_plane]|![JNet_459_pretrain_0_label_plane]|
  
MSE: 0.028607603162527084, BCE: 0.11940757930278778  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_0_original_depth]|![JNet_459_pretrain_0_output_depth]|![JNet_459_pretrain_0_label_depth]|
  
MSE: 0.028607603162527084, BCE: 0.11940757930278778  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_1_original_plane]|![JNet_459_pretrain_1_output_plane]|![JNet_459_pretrain_1_label_plane]|
  
MSE: 0.027734005823731422, BCE: 0.09770502895116806  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_1_original_depth]|![JNet_459_pretrain_1_output_depth]|![JNet_459_pretrain_1_label_depth]|
  
MSE: 0.027734005823731422, BCE: 0.09770502895116806  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_2_original_plane]|![JNet_459_pretrain_2_output_plane]|![JNet_459_pretrain_2_label_plane]|
  
MSE: 0.01458511222153902, BCE: 0.06390125304460526  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_2_original_depth]|![JNet_459_pretrain_2_output_depth]|![JNet_459_pretrain_2_label_depth]|
  
MSE: 0.01458511222153902, BCE: 0.06390125304460526  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_3_original_plane]|![JNet_459_pretrain_3_output_plane]|![JNet_459_pretrain_3_label_plane]|
  
MSE: 0.015996938571333885, BCE: 0.07352433353662491  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_3_original_depth]|![JNet_459_pretrain_3_output_depth]|![JNet_459_pretrain_3_label_depth]|
  
MSE: 0.015996938571333885, BCE: 0.07352433353662491  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_4_original_plane]|![JNet_459_pretrain_4_output_plane]|![JNet_459_pretrain_4_label_plane]|
  
MSE: 0.026130056008696556, BCE: 0.09567124396562576  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_459_pretrain_4_original_depth]|![JNet_459_pretrain_4_output_depth]|![JNet_459_pretrain_4_label_depth]|
  
MSE: 0.026130056008696556, BCE: 0.09567124396562576  
  
mean MSE: 0.03945676237344742, mean BCE: 0.6840388774871826
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_461_0_original_plane]|![JNet_461_0_output_plane]|![JNet_461_0_label_plane]|
  
MSE: 0.03117813915014267, BCE: 0.3939947187900543  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_461_0_original_depth]|![JNet_461_0_output_depth]|![JNet_461_0_label_depth]|
  
MSE: 0.03117813915014267, BCE: 0.3939947187900543  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_461_1_original_plane]|![JNet_461_1_output_plane]|![JNet_461_1_label_plane]|
  
MSE: 0.04006427153944969, BCE: 0.5111342072486877  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_461_1_original_depth]|![JNet_461_1_output_depth]|![JNet_461_1_label_depth]|
  
MSE: 0.04006427153944969, BCE: 0.5111342072486877  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_461_2_original_plane]|![JNet_461_2_output_plane]|![JNet_461_2_label_plane]|
  
MSE: 0.04153478890657425, BCE: 0.5910204648971558  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_461_2_original_depth]|![JNet_461_2_output_depth]|![JNet_461_2_label_depth]|
  
MSE: 0.04153478890657425, BCE: 0.5910204648971558  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_461_3_original_plane]|![JNet_461_3_output_plane]|![JNet_461_3_label_plane]|
  
MSE: 0.03151712566614151, BCE: 0.593936026096344  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_461_3_original_depth]|![JNet_461_3_output_depth]|![JNet_461_3_label_depth]|
  
MSE: 0.03151712566614151, BCE: 0.593936026096344  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_461_4_original_plane]|![JNet_461_4_output_plane]|![JNet_461_4_label_plane]|
  
MSE: 0.052989497780799866, BCE: 1.330108880996704  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_461_4_original_depth]|![JNet_461_4_output_depth]|![JNet_461_4_label_depth]|
  
MSE: 0.052989497780799866, BCE: 1.330108880996704  

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
|![JNet_461_beads_001_roi000_original_depth]|![JNet_461_beads_001_roi000_output_depth]|![JNet_461_beads_001_roi000_reconst_depth]|![JNet_461_beads_001_roi000_heatmap_depth]|
  
volume: 4.365792968750001, MSE: 0.00020077689259778708, quantized loss: 5.521262210095301e-05  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_461_beads_001_roi001_original_depth]|![JNet_461_beads_001_roi001_output_depth]|![JNet_461_beads_001_roi001_reconst_depth]|![JNet_461_beads_001_roi001_heatmap_depth]|
  
volume: 7.077781738281252, MSE: 0.0006488024373538792, quantized loss: 8.035884820856154e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_461_beads_001_roi002_original_depth]|![JNet_461_beads_001_roi002_output_depth]|![JNet_461_beads_001_roi002_reconst_depth]|![JNet_461_beads_001_roi002_heatmap_depth]|
  
volume: 4.564823730468751, MSE: 0.00013280939310789108, quantized loss: 5.8363901189295575e-05  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_461_beads_001_roi003_original_depth]|![JNet_461_beads_001_roi003_output_depth]|![JNet_461_beads_001_roi003_reconst_depth]|![JNet_461_beads_001_roi003_heatmap_depth]|
  
volume: 6.9453066406250015, MSE: 0.00045354850590229034, quantized loss: 8.225481724366546e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_461_beads_001_roi004_original_depth]|![JNet_461_beads_001_roi004_output_depth]|![JNet_461_beads_001_roi004_reconst_depth]|![JNet_461_beads_001_roi004_heatmap_depth]|
  
volume: 4.468585449218751, MSE: 0.00014601687144022435, quantized loss: 4.748973879031837e-05  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_461_beads_002_roi000_original_depth]|![JNet_461_beads_002_roi000_output_depth]|![JNet_461_beads_002_roi000_reconst_depth]|![JNet_461_beads_002_roi000_heatmap_depth]|
  
volume: 4.666521484375001, MSE: 0.00015802898269612342, quantized loss: 5.1264189096400514e-05  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_461_beads_002_roi001_original_depth]|![JNet_461_beads_002_roi001_output_depth]|![JNet_461_beads_002_roi001_reconst_depth]|![JNet_461_beads_002_roi001_heatmap_depth]|
  
volume: 4.637620605468751, MSE: 0.00015219235501717776, quantized loss: 5.611370215774514e-05  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_461_beads_002_roi002_original_depth]|![JNet_461_beads_002_roi002_output_depth]|![JNet_461_beads_002_roi002_reconst_depth]|![JNet_461_beads_002_roi002_heatmap_depth]|
  
volume: 4.7147812500000015, MSE: 0.00014196474512573332, quantized loss: 5.0665468734223396e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_461_psf_pre]|![JNet_461_psf_post]|

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
[JNet_461_0_label_depth]: /experiments/images/JNet_461_0_label_depth.png
[JNet_461_0_label_plane]: /experiments/images/JNet_461_0_label_plane.png
[JNet_461_0_original_depth]: /experiments/images/JNet_461_0_original_depth.png
[JNet_461_0_original_plane]: /experiments/images/JNet_461_0_original_plane.png
[JNet_461_0_output_depth]: /experiments/images/JNet_461_0_output_depth.png
[JNet_461_0_output_plane]: /experiments/images/JNet_461_0_output_plane.png
[JNet_461_1_label_depth]: /experiments/images/JNet_461_1_label_depth.png
[JNet_461_1_label_plane]: /experiments/images/JNet_461_1_label_plane.png
[JNet_461_1_original_depth]: /experiments/images/JNet_461_1_original_depth.png
[JNet_461_1_original_plane]: /experiments/images/JNet_461_1_original_plane.png
[JNet_461_1_output_depth]: /experiments/images/JNet_461_1_output_depth.png
[JNet_461_1_output_plane]: /experiments/images/JNet_461_1_output_plane.png
[JNet_461_2_label_depth]: /experiments/images/JNet_461_2_label_depth.png
[JNet_461_2_label_plane]: /experiments/images/JNet_461_2_label_plane.png
[JNet_461_2_original_depth]: /experiments/images/JNet_461_2_original_depth.png
[JNet_461_2_original_plane]: /experiments/images/JNet_461_2_original_plane.png
[JNet_461_2_output_depth]: /experiments/images/JNet_461_2_output_depth.png
[JNet_461_2_output_plane]: /experiments/images/JNet_461_2_output_plane.png
[JNet_461_3_label_depth]: /experiments/images/JNet_461_3_label_depth.png
[JNet_461_3_label_plane]: /experiments/images/JNet_461_3_label_plane.png
[JNet_461_3_original_depth]: /experiments/images/JNet_461_3_original_depth.png
[JNet_461_3_original_plane]: /experiments/images/JNet_461_3_original_plane.png
[JNet_461_3_output_depth]: /experiments/images/JNet_461_3_output_depth.png
[JNet_461_3_output_plane]: /experiments/images/JNet_461_3_output_plane.png
[JNet_461_4_label_depth]: /experiments/images/JNet_461_4_label_depth.png
[JNet_461_4_label_plane]: /experiments/images/JNet_461_4_label_plane.png
[JNet_461_4_original_depth]: /experiments/images/JNet_461_4_original_depth.png
[JNet_461_4_original_plane]: /experiments/images/JNet_461_4_original_plane.png
[JNet_461_4_output_depth]: /experiments/images/JNet_461_4_output_depth.png
[JNet_461_4_output_plane]: /experiments/images/JNet_461_4_output_plane.png
[JNet_461_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_461_beads_001_roi000_heatmap_depth.png
[JNet_461_beads_001_roi000_original_depth]: /experiments/images/JNet_461_beads_001_roi000_original_depth.png
[JNet_461_beads_001_roi000_output_depth]: /experiments/images/JNet_461_beads_001_roi000_output_depth.png
[JNet_461_beads_001_roi000_reconst_depth]: /experiments/images/JNet_461_beads_001_roi000_reconst_depth.png
[JNet_461_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_461_beads_001_roi001_heatmap_depth.png
[JNet_461_beads_001_roi001_original_depth]: /experiments/images/JNet_461_beads_001_roi001_original_depth.png
[JNet_461_beads_001_roi001_output_depth]: /experiments/images/JNet_461_beads_001_roi001_output_depth.png
[JNet_461_beads_001_roi001_reconst_depth]: /experiments/images/JNet_461_beads_001_roi001_reconst_depth.png
[JNet_461_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_461_beads_001_roi002_heatmap_depth.png
[JNet_461_beads_001_roi002_original_depth]: /experiments/images/JNet_461_beads_001_roi002_original_depth.png
[JNet_461_beads_001_roi002_output_depth]: /experiments/images/JNet_461_beads_001_roi002_output_depth.png
[JNet_461_beads_001_roi002_reconst_depth]: /experiments/images/JNet_461_beads_001_roi002_reconst_depth.png
[JNet_461_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_461_beads_001_roi003_heatmap_depth.png
[JNet_461_beads_001_roi003_original_depth]: /experiments/images/JNet_461_beads_001_roi003_original_depth.png
[JNet_461_beads_001_roi003_output_depth]: /experiments/images/JNet_461_beads_001_roi003_output_depth.png
[JNet_461_beads_001_roi003_reconst_depth]: /experiments/images/JNet_461_beads_001_roi003_reconst_depth.png
[JNet_461_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_461_beads_001_roi004_heatmap_depth.png
[JNet_461_beads_001_roi004_original_depth]: /experiments/images/JNet_461_beads_001_roi004_original_depth.png
[JNet_461_beads_001_roi004_output_depth]: /experiments/images/JNet_461_beads_001_roi004_output_depth.png
[JNet_461_beads_001_roi004_reconst_depth]: /experiments/images/JNet_461_beads_001_roi004_reconst_depth.png
[JNet_461_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_461_beads_002_roi000_heatmap_depth.png
[JNet_461_beads_002_roi000_original_depth]: /experiments/images/JNet_461_beads_002_roi000_original_depth.png
[JNet_461_beads_002_roi000_output_depth]: /experiments/images/JNet_461_beads_002_roi000_output_depth.png
[JNet_461_beads_002_roi000_reconst_depth]: /experiments/images/JNet_461_beads_002_roi000_reconst_depth.png
[JNet_461_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_461_beads_002_roi001_heatmap_depth.png
[JNet_461_beads_002_roi001_original_depth]: /experiments/images/JNet_461_beads_002_roi001_original_depth.png
[JNet_461_beads_002_roi001_output_depth]: /experiments/images/JNet_461_beads_002_roi001_output_depth.png
[JNet_461_beads_002_roi001_reconst_depth]: /experiments/images/JNet_461_beads_002_roi001_reconst_depth.png
[JNet_461_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_461_beads_002_roi002_heatmap_depth.png
[JNet_461_beads_002_roi002_original_depth]: /experiments/images/JNet_461_beads_002_roi002_original_depth.png
[JNet_461_beads_002_roi002_output_depth]: /experiments/images/JNet_461_beads_002_roi002_output_depth.png
[JNet_461_beads_002_roi002_reconst_depth]: /experiments/images/JNet_461_beads_002_roi002_reconst_depth.png
[JNet_461_psf_post]: /experiments/images/JNet_461_psf_post.png
[JNet_461_psf_pre]: /experiments/images/JNet_461_psf_pre.png
[finetuned]: /experiments/tmp/JNet_461_train.png
[pretrained_model]: /experiments/tmp/JNet_459_pretrain_train.png
