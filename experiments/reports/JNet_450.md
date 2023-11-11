



# JNet_450 Report
  
the parameters to replicate the results of JNet_450. no vibrate, bright NA=0.7, mu_z = 1.2, sig_z = 1.27  
pretrained model : JNet_448_pretrain
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
|mu_z|2.0||
|sig_z|0.3||
|blur_mode|gaussian|`gaussian` or `gibsonlanni`|
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
|is_vibrate|False|
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
|is_vibrate|False|
|loss_weight|1|
|qloss_weight|1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.014508751221001148, mean BCE: 0.05155162140727043
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_448_pretrain_0_original_plane]|![JNet_448_pretrain_0_output_plane]|![JNet_448_pretrain_0_label_plane]|
  
MSE: 0.013092170469462872, BCE: 0.046894777566194534  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_448_pretrain_0_original_depth]|![JNet_448_pretrain_0_output_depth]|![JNet_448_pretrain_0_label_depth]|
  
MSE: 0.013092170469462872, BCE: 0.046894777566194534  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_448_pretrain_1_original_plane]|![JNet_448_pretrain_1_output_plane]|![JNet_448_pretrain_1_label_plane]|
  
MSE: 0.018089227378368378, BCE: 0.06425729393959045  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_448_pretrain_1_original_depth]|![JNet_448_pretrain_1_output_depth]|![JNet_448_pretrain_1_label_depth]|
  
MSE: 0.018089227378368378, BCE: 0.06425729393959045  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_448_pretrain_2_original_plane]|![JNet_448_pretrain_2_output_plane]|![JNet_448_pretrain_2_label_plane]|
  
MSE: 0.01663256250321865, BCE: 0.056784022599458694  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_448_pretrain_2_original_depth]|![JNet_448_pretrain_2_output_depth]|![JNet_448_pretrain_2_label_depth]|
  
MSE: 0.01663256250321865, BCE: 0.056784022599458694  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_448_pretrain_3_original_plane]|![JNet_448_pretrain_3_output_plane]|![JNet_448_pretrain_3_label_plane]|
  
MSE: 0.015248122625052929, BCE: 0.054917704313993454  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_448_pretrain_3_original_depth]|![JNet_448_pretrain_3_output_depth]|![JNet_448_pretrain_3_label_depth]|
  
MSE: 0.015248122625052929, BCE: 0.054917704313993454  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_448_pretrain_4_original_plane]|![JNet_448_pretrain_4_output_plane]|![JNet_448_pretrain_4_label_plane]|
  
MSE: 0.009481672197580338, BCE: 0.03490430489182472  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_448_pretrain_4_original_depth]|![JNet_448_pretrain_4_output_depth]|![JNet_448_pretrain_4_label_depth]|
  
MSE: 0.009481672197580338, BCE: 0.03490430489182472  
  
mean MSE: 0.025852208957076073, mean BCE: 0.1545598804950714
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_450_0_original_plane]|![JNet_450_0_output_plane]|![JNet_450_0_label_plane]|
  
MSE: 0.024630557745695114, BCE: 0.13575361669063568  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_450_0_original_depth]|![JNet_450_0_output_depth]|![JNet_450_0_label_depth]|
  
MSE: 0.024630557745695114, BCE: 0.13575361669063568  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_450_1_original_plane]|![JNet_450_1_output_plane]|![JNet_450_1_label_plane]|
  
MSE: 0.025157703086733818, BCE: 0.14308731257915497  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_450_1_original_depth]|![JNet_450_1_output_depth]|![JNet_450_1_label_depth]|
  
MSE: 0.025157703086733818, BCE: 0.14308731257915497  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_450_2_original_plane]|![JNet_450_2_output_plane]|![JNet_450_2_label_plane]|
  
MSE: 0.023576876148581505, BCE: 0.12079066038131714  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_450_2_original_depth]|![JNet_450_2_output_depth]|![JNet_450_2_label_depth]|
  
MSE: 0.023576876148581505, BCE: 0.12079066038131714  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_450_3_original_plane]|![JNet_450_3_output_plane]|![JNet_450_3_label_plane]|
  
MSE: 0.024815436452627182, BCE: 0.13008013367652893  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_450_3_original_depth]|![JNet_450_3_output_depth]|![JNet_450_3_label_depth]|
  
MSE: 0.024815436452627182, BCE: 0.13008013367652893  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_450_4_original_plane]|![JNet_450_4_output_plane]|![JNet_450_4_label_plane]|
  
MSE: 0.031080465763807297, BCE: 0.24308772385120392  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_450_4_original_depth]|![JNet_450_4_output_depth]|![JNet_450_4_label_depth]|
  
MSE: 0.031080465763807297, BCE: 0.24308772385120392  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_448_pretrain_beads_001_roi000_original_depth]|![JNet_448_pretrain_beads_001_roi000_output_depth]|![JNet_448_pretrain_beads_001_roi000_reconst_depth]|![JNet_448_pretrain_beads_001_roi000_heatmap_depth]|
  
volume: 11.289762695312502, MSE: 0.0004491706204134971, quantized loss: 0.0010117326164618134  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_448_pretrain_beads_001_roi001_original_depth]|![JNet_448_pretrain_beads_001_roi001_output_depth]|![JNet_448_pretrain_beads_001_roi001_reconst_depth]|![JNet_448_pretrain_beads_001_roi001_heatmap_depth]|
  
volume: 14.348054687500003, MSE: 0.0011819741921499372, quantized loss: 0.0011784917442128062  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_448_pretrain_beads_001_roi002_original_depth]|![JNet_448_pretrain_beads_001_roi002_output_depth]|![JNet_448_pretrain_beads_001_roi002_reconst_depth]|![JNet_448_pretrain_beads_001_roi002_heatmap_depth]|
  
volume: 10.400447265625003, MSE: 0.0003540804609656334, quantized loss: 0.0005686294753104448  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_448_pretrain_beads_001_roi003_original_depth]|![JNet_448_pretrain_beads_001_roi003_output_depth]|![JNet_448_pretrain_beads_001_roi003_reconst_depth]|![JNet_448_pretrain_beads_001_roi003_heatmap_depth]|
  
volume: 14.156715820312503, MSE: 0.0008147477055899799, quantized loss: 0.0012399029219523072  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_448_pretrain_beads_001_roi004_original_depth]|![JNet_448_pretrain_beads_001_roi004_output_depth]|![JNet_448_pretrain_beads_001_roi004_reconst_depth]|![JNet_448_pretrain_beads_001_roi004_heatmap_depth]|
  
volume: 10.888596679687502, MSE: 0.0004539088113233447, quantized loss: 0.0006475560949184  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_448_pretrain_beads_002_roi000_original_depth]|![JNet_448_pretrain_beads_002_roi000_output_depth]|![JNet_448_pretrain_beads_002_roi000_reconst_depth]|![JNet_448_pretrain_beads_002_roi000_heatmap_depth]|
  
volume: 11.076193359375003, MSE: 0.0005177909624762833, quantized loss: 0.0007315680850297213  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_448_pretrain_beads_002_roi001_original_depth]|![JNet_448_pretrain_beads_002_roi001_output_depth]|![JNet_448_pretrain_beads_002_roi001_reconst_depth]|![JNet_448_pretrain_beads_002_roi001_heatmap_depth]|
  
volume: 10.715389648437503, MSE: 0.0003868350468110293, quantized loss: 0.0006850848440080881  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_448_pretrain_beads_002_roi002_original_depth]|![JNet_448_pretrain_beads_002_roi002_output_depth]|![JNet_448_pretrain_beads_002_roi002_reconst_depth]|![JNet_448_pretrain_beads_002_roi002_heatmap_depth]|
  
volume: 10.608755859375002, MSE: 0.00045705074444413185, quantized loss: 0.000660719582810998  

### beads_001_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_450_beads_001_roi000_original_depth]|![JNet_450_beads_001_roi000_output_depth]|![JNet_450_beads_001_roi000_reconst_depth]|![JNet_450_beads_001_roi000_heatmap_depth]|
  
volume: 2.7463845214843756, MSE: 0.00019054707081522793, quantized loss: 6.008845230098814e-05  

### beads_001_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_450_beads_001_roi001_original_depth]|![JNet_450_beads_001_roi001_output_depth]|![JNet_450_beads_001_roi001_reconst_depth]|![JNet_450_beads_001_roi001_heatmap_depth]|
  
volume: 4.192929687500001, MSE: 0.0007327498751692474, quantized loss: 7.880779594415799e-05  

### beads_001_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_450_beads_001_roi002_original_depth]|![JNet_450_beads_001_roi002_output_depth]|![JNet_450_beads_001_roi002_reconst_depth]|![JNet_450_beads_001_roi002_heatmap_depth]|
  
volume: 2.7322656250000006, MSE: 0.0001364332711091265, quantized loss: 4.747408092953265e-05  

### beads_001_roi003

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_450_beads_001_roi003_original_depth]|![JNet_450_beads_001_roi003_output_depth]|![JNet_450_beads_001_roi003_reconst_depth]|![JNet_450_beads_001_roi003_heatmap_depth]|
  
volume: 4.143095214843751, MSE: 0.000325904373312369, quantized loss: 7.922879740362987e-05  

### beads_001_roi004

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_450_beads_001_roi004_original_depth]|![JNet_450_beads_001_roi004_output_depth]|![JNet_450_beads_001_roi004_reconst_depth]|![JNet_450_beads_001_roi004_heatmap_depth]|
  
volume: 2.7443413085937505, MSE: 0.00015277808415703475, quantized loss: 4.8811707529239357e-05  

### beads_002_roi000

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_450_beads_002_roi000_original_depth]|![JNet_450_beads_002_roi000_output_depth]|![JNet_450_beads_002_roi000_reconst_depth]|![JNet_450_beads_002_roi000_heatmap_depth]|
  
volume: 2.8160864257812506, MSE: 0.00015352216723840684, quantized loss: 4.902011642116122e-05  

### beads_002_roi001

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_450_beads_002_roi001_original_depth]|![JNet_450_beads_002_roi001_output_depth]|![JNet_450_beads_002_roi001_reconst_depth]|![JNet_450_beads_002_roi001_heatmap_depth]|
  
volume: 2.7383498535156257, MSE: 0.00010805784404510632, quantized loss: 5.187895294511691e-05  

### beads_002_roi002

|original|output|reconst|heatmap|
| :---: | :---: | :---: | :---: |
|![JNet_450_beads_002_roi002_original_depth]|![JNet_450_beads_002_roi002_output_depth]|![JNet_450_beads_002_roi002_reconst_depth]|![JNet_450_beads_002_roi002_heatmap_depth]|
  
volume: 2.7834768066406257, MSE: 0.00013917482283432037, quantized loss: 4.83700969198253e-05  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_450_psf_pre]|![JNet_450_psf_post]|

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
  



[JNet_448_pretrain_0_label_depth]: /experiments/images/JNet_448_pretrain_0_label_depth.png
[JNet_448_pretrain_0_label_plane]: /experiments/images/JNet_448_pretrain_0_label_plane.png
[JNet_448_pretrain_0_original_depth]: /experiments/images/JNet_448_pretrain_0_original_depth.png
[JNet_448_pretrain_0_original_plane]: /experiments/images/JNet_448_pretrain_0_original_plane.png
[JNet_448_pretrain_0_output_depth]: /experiments/images/JNet_448_pretrain_0_output_depth.png
[JNet_448_pretrain_0_output_plane]: /experiments/images/JNet_448_pretrain_0_output_plane.png
[JNet_448_pretrain_1_label_depth]: /experiments/images/JNet_448_pretrain_1_label_depth.png
[JNet_448_pretrain_1_label_plane]: /experiments/images/JNet_448_pretrain_1_label_plane.png
[JNet_448_pretrain_1_original_depth]: /experiments/images/JNet_448_pretrain_1_original_depth.png
[JNet_448_pretrain_1_original_plane]: /experiments/images/JNet_448_pretrain_1_original_plane.png
[JNet_448_pretrain_1_output_depth]: /experiments/images/JNet_448_pretrain_1_output_depth.png
[JNet_448_pretrain_1_output_plane]: /experiments/images/JNet_448_pretrain_1_output_plane.png
[JNet_448_pretrain_2_label_depth]: /experiments/images/JNet_448_pretrain_2_label_depth.png
[JNet_448_pretrain_2_label_plane]: /experiments/images/JNet_448_pretrain_2_label_plane.png
[JNet_448_pretrain_2_original_depth]: /experiments/images/JNet_448_pretrain_2_original_depth.png
[JNet_448_pretrain_2_original_plane]: /experiments/images/JNet_448_pretrain_2_original_plane.png
[JNet_448_pretrain_2_output_depth]: /experiments/images/JNet_448_pretrain_2_output_depth.png
[JNet_448_pretrain_2_output_plane]: /experiments/images/JNet_448_pretrain_2_output_plane.png
[JNet_448_pretrain_3_label_depth]: /experiments/images/JNet_448_pretrain_3_label_depth.png
[JNet_448_pretrain_3_label_plane]: /experiments/images/JNet_448_pretrain_3_label_plane.png
[JNet_448_pretrain_3_original_depth]: /experiments/images/JNet_448_pretrain_3_original_depth.png
[JNet_448_pretrain_3_original_plane]: /experiments/images/JNet_448_pretrain_3_original_plane.png
[JNet_448_pretrain_3_output_depth]: /experiments/images/JNet_448_pretrain_3_output_depth.png
[JNet_448_pretrain_3_output_plane]: /experiments/images/JNet_448_pretrain_3_output_plane.png
[JNet_448_pretrain_4_label_depth]: /experiments/images/JNet_448_pretrain_4_label_depth.png
[JNet_448_pretrain_4_label_plane]: /experiments/images/JNet_448_pretrain_4_label_plane.png
[JNet_448_pretrain_4_original_depth]: /experiments/images/JNet_448_pretrain_4_original_depth.png
[JNet_448_pretrain_4_original_plane]: /experiments/images/JNet_448_pretrain_4_original_plane.png
[JNet_448_pretrain_4_output_depth]: /experiments/images/JNet_448_pretrain_4_output_depth.png
[JNet_448_pretrain_4_output_plane]: /experiments/images/JNet_448_pretrain_4_output_plane.png
[JNet_448_pretrain_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi000_heatmap_depth.png
[JNet_448_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi000_original_depth.png
[JNet_448_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi000_output_depth.png
[JNet_448_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi000_reconst_depth.png
[JNet_448_pretrain_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi001_heatmap_depth.png
[JNet_448_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi001_original_depth.png
[JNet_448_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi001_output_depth.png
[JNet_448_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi001_reconst_depth.png
[JNet_448_pretrain_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi002_heatmap_depth.png
[JNet_448_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi002_original_depth.png
[JNet_448_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi002_output_depth.png
[JNet_448_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi002_reconst_depth.png
[JNet_448_pretrain_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi003_heatmap_depth.png
[JNet_448_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi003_original_depth.png
[JNet_448_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi003_output_depth.png
[JNet_448_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi003_reconst_depth.png
[JNet_448_pretrain_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi004_heatmap_depth.png
[JNet_448_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi004_original_depth.png
[JNet_448_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi004_output_depth.png
[JNet_448_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_448_pretrain_beads_001_roi004_reconst_depth.png
[JNet_448_pretrain_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi000_heatmap_depth.png
[JNet_448_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi000_original_depth.png
[JNet_448_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi000_output_depth.png
[JNet_448_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi000_reconst_depth.png
[JNet_448_pretrain_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi001_heatmap_depth.png
[JNet_448_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi001_original_depth.png
[JNet_448_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi001_output_depth.png
[JNet_448_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi001_reconst_depth.png
[JNet_448_pretrain_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi002_heatmap_depth.png
[JNet_448_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi002_original_depth.png
[JNet_448_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi002_output_depth.png
[JNet_448_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_448_pretrain_beads_002_roi002_reconst_depth.png
[JNet_450_0_label_depth]: /experiments/images/JNet_450_0_label_depth.png
[JNet_450_0_label_plane]: /experiments/images/JNet_450_0_label_plane.png
[JNet_450_0_original_depth]: /experiments/images/JNet_450_0_original_depth.png
[JNet_450_0_original_plane]: /experiments/images/JNet_450_0_original_plane.png
[JNet_450_0_output_depth]: /experiments/images/JNet_450_0_output_depth.png
[JNet_450_0_output_plane]: /experiments/images/JNet_450_0_output_plane.png
[JNet_450_1_label_depth]: /experiments/images/JNet_450_1_label_depth.png
[JNet_450_1_label_plane]: /experiments/images/JNet_450_1_label_plane.png
[JNet_450_1_original_depth]: /experiments/images/JNet_450_1_original_depth.png
[JNet_450_1_original_plane]: /experiments/images/JNet_450_1_original_plane.png
[JNet_450_1_output_depth]: /experiments/images/JNet_450_1_output_depth.png
[JNet_450_1_output_plane]: /experiments/images/JNet_450_1_output_plane.png
[JNet_450_2_label_depth]: /experiments/images/JNet_450_2_label_depth.png
[JNet_450_2_label_plane]: /experiments/images/JNet_450_2_label_plane.png
[JNet_450_2_original_depth]: /experiments/images/JNet_450_2_original_depth.png
[JNet_450_2_original_plane]: /experiments/images/JNet_450_2_original_plane.png
[JNet_450_2_output_depth]: /experiments/images/JNet_450_2_output_depth.png
[JNet_450_2_output_plane]: /experiments/images/JNet_450_2_output_plane.png
[JNet_450_3_label_depth]: /experiments/images/JNet_450_3_label_depth.png
[JNet_450_3_label_plane]: /experiments/images/JNet_450_3_label_plane.png
[JNet_450_3_original_depth]: /experiments/images/JNet_450_3_original_depth.png
[JNet_450_3_original_plane]: /experiments/images/JNet_450_3_original_plane.png
[JNet_450_3_output_depth]: /experiments/images/JNet_450_3_output_depth.png
[JNet_450_3_output_plane]: /experiments/images/JNet_450_3_output_plane.png
[JNet_450_4_label_depth]: /experiments/images/JNet_450_4_label_depth.png
[JNet_450_4_label_plane]: /experiments/images/JNet_450_4_label_plane.png
[JNet_450_4_original_depth]: /experiments/images/JNet_450_4_original_depth.png
[JNet_450_4_original_plane]: /experiments/images/JNet_450_4_original_plane.png
[JNet_450_4_output_depth]: /experiments/images/JNet_450_4_output_depth.png
[JNet_450_4_output_plane]: /experiments/images/JNet_450_4_output_plane.png
[JNet_450_beads_001_roi000_heatmap_depth]: /experiments/images/JNet_450_beads_001_roi000_heatmap_depth.png
[JNet_450_beads_001_roi000_original_depth]: /experiments/images/JNet_450_beads_001_roi000_original_depth.png
[JNet_450_beads_001_roi000_output_depth]: /experiments/images/JNet_450_beads_001_roi000_output_depth.png
[JNet_450_beads_001_roi000_reconst_depth]: /experiments/images/JNet_450_beads_001_roi000_reconst_depth.png
[JNet_450_beads_001_roi001_heatmap_depth]: /experiments/images/JNet_450_beads_001_roi001_heatmap_depth.png
[JNet_450_beads_001_roi001_original_depth]: /experiments/images/JNet_450_beads_001_roi001_original_depth.png
[JNet_450_beads_001_roi001_output_depth]: /experiments/images/JNet_450_beads_001_roi001_output_depth.png
[JNet_450_beads_001_roi001_reconst_depth]: /experiments/images/JNet_450_beads_001_roi001_reconst_depth.png
[JNet_450_beads_001_roi002_heatmap_depth]: /experiments/images/JNet_450_beads_001_roi002_heatmap_depth.png
[JNet_450_beads_001_roi002_original_depth]: /experiments/images/JNet_450_beads_001_roi002_original_depth.png
[JNet_450_beads_001_roi002_output_depth]: /experiments/images/JNet_450_beads_001_roi002_output_depth.png
[JNet_450_beads_001_roi002_reconst_depth]: /experiments/images/JNet_450_beads_001_roi002_reconst_depth.png
[JNet_450_beads_001_roi003_heatmap_depth]: /experiments/images/JNet_450_beads_001_roi003_heatmap_depth.png
[JNet_450_beads_001_roi003_original_depth]: /experiments/images/JNet_450_beads_001_roi003_original_depth.png
[JNet_450_beads_001_roi003_output_depth]: /experiments/images/JNet_450_beads_001_roi003_output_depth.png
[JNet_450_beads_001_roi003_reconst_depth]: /experiments/images/JNet_450_beads_001_roi003_reconst_depth.png
[JNet_450_beads_001_roi004_heatmap_depth]: /experiments/images/JNet_450_beads_001_roi004_heatmap_depth.png
[JNet_450_beads_001_roi004_original_depth]: /experiments/images/JNet_450_beads_001_roi004_original_depth.png
[JNet_450_beads_001_roi004_output_depth]: /experiments/images/JNet_450_beads_001_roi004_output_depth.png
[JNet_450_beads_001_roi004_reconst_depth]: /experiments/images/JNet_450_beads_001_roi004_reconst_depth.png
[JNet_450_beads_002_roi000_heatmap_depth]: /experiments/images/JNet_450_beads_002_roi000_heatmap_depth.png
[JNet_450_beads_002_roi000_original_depth]: /experiments/images/JNet_450_beads_002_roi000_original_depth.png
[JNet_450_beads_002_roi000_output_depth]: /experiments/images/JNet_450_beads_002_roi000_output_depth.png
[JNet_450_beads_002_roi000_reconst_depth]: /experiments/images/JNet_450_beads_002_roi000_reconst_depth.png
[JNet_450_beads_002_roi001_heatmap_depth]: /experiments/images/JNet_450_beads_002_roi001_heatmap_depth.png
[JNet_450_beads_002_roi001_original_depth]: /experiments/images/JNet_450_beads_002_roi001_original_depth.png
[JNet_450_beads_002_roi001_output_depth]: /experiments/images/JNet_450_beads_002_roi001_output_depth.png
[JNet_450_beads_002_roi001_reconst_depth]: /experiments/images/JNet_450_beads_002_roi001_reconst_depth.png
[JNet_450_beads_002_roi002_heatmap_depth]: /experiments/images/JNet_450_beads_002_roi002_heatmap_depth.png
[JNet_450_beads_002_roi002_original_depth]: /experiments/images/JNet_450_beads_002_roi002_original_depth.png
[JNet_450_beads_002_roi002_output_depth]: /experiments/images/JNet_450_beads_002_roi002_output_depth.png
[JNet_450_beads_002_roi002_reconst_depth]: /experiments/images/JNet_450_beads_002_roi002_reconst_depth.png
[JNet_450_psf_post]: /experiments/images/JNet_450_psf_post.png
[JNet_450_psf_pre]: /experiments/images/JNet_450_psf_pre.png
[finetuned]: /experiments/tmp/JNet_450_train.png
[pretrained_model]: /experiments/tmp/JNet_448_pretrain_train.png
