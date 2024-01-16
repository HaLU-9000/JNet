



# JNet_476 Report
  
new data generation with more objects. axon deconv  
pretrained model : JNet_475_pretrain
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
|bet_z|30.0||
|bet_xy|3.0||
|sig_eps|0.01||
|background|0.01||
|scale|10||
|mid|20|num of NeurIPSF middle channel|
|loss_fn|nn.MSELoss()|loss func for NeurIPSF|
|lr|0.01|lr for pre-training NeurIPSF|
|num_iter_psf_pretrain|1000|epoch for pre-training of NeurIPSF|
|device|cuda||

## Datasets and other training details

### simulation_data_generation

|Parameter|Value|
| :--- | :--- |
|dataset_name|_var_num_realisticdata1|
|train_num|16|
|valid_num|4|
|image_size|[1200, 500, 500]|
|train_object_num_min|40000|
|train_object_num_max|100000|
|valid_object_num_min|62500|
|valid_object_num_max|77500|

### pretrain_dataset

|Parameter|Value|
| :--- | :--- |
|folderpath|_var_num_realisticdata1|
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
|folderpath|_var_num_realisticdata1|
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
|loss_fnx|nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=params['device']))|
|loss_fnz|nn.MSELoss()|
|path|model|
|savefig_path|train|
|partial|params['partial']|
|ewc|None|
|es_patience|10|
|is_vibrate|True|
|weight_x|1|
|weight_z|1|

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
|adjust_luminance|True|
|loss_weight|1|
|ewc_weight|1000000|
|qloss_weight|1|
|ploss_weight|0.0|

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
Segmentation: mean MSE: 0.26449304819107056, mean BCE: 0.7221738696098328  
Luminance Estimation: mean MSE: 0.07974787056446075, mean BCE: 0.3352468013763428
### 0

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_475_pretrain_0_original_plane]|![JNet_475_pretrain_0_outputx_plane]|![JNet_475_pretrain_0_labelx_plane]|![JNet_475_pretrain_0_outputz_plane]|![JNet_475_pretrain_0_labelz_plane]|
  
MSEx: 0.2693798542022705, BCEx: 0.7320020198822021  
MSEz: 0.0790831446647644, BCEz: 0.33657053112983704  

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_475_pretrain_0_original_depth]|![JNet_475_pretrain_0_outputx_depth]|![JNet_475_pretrain_0_labelx_depth]|![JNet_475_pretrain_0_outputz_depth]|![JNet_475_pretrain_0_labelz_depth]|
  
MSEx: 0.2693798542022705, BCEx: 0.7320020198822021  
MSEz: 0.0790831446647644, BCEz: 0.33657053112983704  

### 1

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_475_pretrain_1_original_plane]|![JNet_475_pretrain_1_outputx_plane]|![JNet_475_pretrain_1_labelx_plane]|![JNet_475_pretrain_1_outputz_plane]|![JNet_475_pretrain_1_labelz_plane]|
  
MSEx: 0.26266881823539734, BCEx: 0.7184998989105225  
MSEz: 0.0799720287322998, BCEz: 0.33480241894721985  

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_475_pretrain_1_original_depth]|![JNet_475_pretrain_1_outputx_depth]|![JNet_475_pretrain_1_labelx_depth]|![JNet_475_pretrain_1_outputz_depth]|![JNet_475_pretrain_1_labelz_depth]|
  
MSEx: 0.26266881823539734, BCEx: 0.7184998989105225  
MSEz: 0.0799720287322998, BCEz: 0.33480241894721985  

### 2

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_475_pretrain_2_original_plane]|![JNet_475_pretrain_2_outputx_plane]|![JNet_475_pretrain_2_labelx_plane]|![JNet_475_pretrain_2_outputz_plane]|![JNet_475_pretrain_2_labelz_plane]|
  
MSEx: 0.26674649119377136, BCEx: 0.7267048954963684  
MSEz: 0.07956562936306, BCEz: 0.3355673551559448  

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_475_pretrain_2_original_depth]|![JNet_475_pretrain_2_outputx_depth]|![JNet_475_pretrain_2_labelx_depth]|![JNet_475_pretrain_2_outputz_depth]|![JNet_475_pretrain_2_labelz_depth]|
  
MSEx: 0.26674649119377136, BCEx: 0.7267048954963684  
MSEz: 0.07956562936306, BCEz: 0.3355673551559448  

### 3

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_475_pretrain_3_original_plane]|![JNet_475_pretrain_3_outputx_plane]|![JNet_475_pretrain_3_labelx_plane]|![JNet_475_pretrain_3_outputz_plane]|![JNet_475_pretrain_3_labelz_plane]|
  
MSEx: 0.2625885009765625, BCEx: 0.7183371186256409  
MSEz: 0.07997925579547882, BCEz: 0.33479732275009155  

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_475_pretrain_3_original_depth]|![JNet_475_pretrain_3_outputx_depth]|![JNet_475_pretrain_3_labelx_depth]|![JNet_475_pretrain_3_outputz_depth]|![JNet_475_pretrain_3_labelz_depth]|
  
MSEx: 0.2625885009765625, BCEx: 0.7183371186256409  
MSEz: 0.07997925579547882, BCEz: 0.33479732275009155  

### 4

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_475_pretrain_4_original_plane]|![JNet_475_pretrain_4_outputx_plane]|![JNet_475_pretrain_4_labelx_plane]|![JNet_475_pretrain_4_outputz_plane]|![JNet_475_pretrain_4_labelz_plane]|
  
MSEx: 0.26108160614967346, BCEx: 0.7153255343437195  
MSEz: 0.08013929426670074, BCEz: 0.33449622988700867  

|original|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: |
|![JNet_475_pretrain_4_original_depth]|![JNet_475_pretrain_4_outputx_depth]|![JNet_475_pretrain_4_labelx_depth]|![JNet_475_pretrain_4_outputz_depth]|![JNet_475_pretrain_4_labelz_depth]|
  
MSEx: 0.26108160614967346, BCEx: 0.7153255343437195  
MSEz: 0.08013929426670074, BCEz: 0.33449622988700867  

### image 0

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_476_0_original_depth]|![JNet_476_0_reconst_depth]|![JNet_476_0_heatmap_depth]|![JNet_476_0_outputx_depth]|![JNet_476_0_labelx_depth]|![JNet_476_0_outputz_depth]|![JNet_476_0_labelz_depth]|
  
MSEz: 0.07963735610246658, quantized loss: 1.839501173606095e-08  

### image 1

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_476_1_original_depth]|![JNet_476_1_reconst_depth]|![JNet_476_1_heatmap_depth]|![JNet_476_1_outputx_depth]|![JNet_476_1_labelx_depth]|![JNet_476_1_outputz_depth]|![JNet_476_1_labelz_depth]|
  
MSEz: 0.07993865013122559, quantized loss: 2.296467904727706e-09  

### image 2

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_476_2_original_depth]|![JNet_476_2_reconst_depth]|![JNet_476_2_heatmap_depth]|![JNet_476_2_outputx_depth]|![JNet_476_2_labelx_depth]|![JNet_476_2_outputz_depth]|![JNet_476_2_labelz_depth]|
  
MSEz: 0.07913623005151749, quantized loss: 2.2868499627293204e-07  

### image 3

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_476_3_original_depth]|![JNet_476_3_reconst_depth]|![JNet_476_3_heatmap_depth]|![JNet_476_3_outputx_depth]|![JNet_476_3_labelx_depth]|![JNet_476_3_outputz_depth]|![JNet_476_3_labelz_depth]|
  
MSEz: 0.07980892807245255, quantized loss: 7.126992329631321e-10  

### image 4

|original|reconst|heatmap|outputx|labelx|outputz|labelz|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|![JNet_476_4_original_depth]|![JNet_476_4_reconst_depth]|![JNet_476_4_heatmap_depth]|![JNet_476_4_outputx_depth]|![JNet_476_4_labelx_depth]|![JNet_476_4_outputz_depth]|![JNet_476_4_labelz_depth]|
  
MSEz: 0.07912808656692505, quantized loss: 4.1085752400249476e-07  
  
If the pixels are red, the reconstructed image is brighter than the original. If they are blue, the reconstructed image is darker.
|pre|post|
| :---: | :---: |
|![JNet_476_psf_pre]|![JNet_476_psf_post]|

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
  (post1): JNetBlockN(  
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
    (upsample): Upsample(scale_factor=(10.0, 1.0, 1.0), mode='trilinear')  
  )  
  (vq): VectorQuantizer()  
)  
```  
  



[JNet_475_pretrain_0_labelx_depth]: /experiments/images/JNet_475_pretrain_0_labelx_depth.png
[JNet_475_pretrain_0_labelx_plane]: /experiments/images/JNet_475_pretrain_0_labelx_plane.png
[JNet_475_pretrain_0_labelz_depth]: /experiments/images/JNet_475_pretrain_0_labelz_depth.png
[JNet_475_pretrain_0_labelz_plane]: /experiments/images/JNet_475_pretrain_0_labelz_plane.png
[JNet_475_pretrain_0_original_depth]: /experiments/images/JNet_475_pretrain_0_original_depth.png
[JNet_475_pretrain_0_original_plane]: /experiments/images/JNet_475_pretrain_0_original_plane.png
[JNet_475_pretrain_0_outputx_depth]: /experiments/images/JNet_475_pretrain_0_outputx_depth.png
[JNet_475_pretrain_0_outputx_plane]: /experiments/images/JNet_475_pretrain_0_outputx_plane.png
[JNet_475_pretrain_0_outputz_depth]: /experiments/images/JNet_475_pretrain_0_outputz_depth.png
[JNet_475_pretrain_0_outputz_plane]: /experiments/images/JNet_475_pretrain_0_outputz_plane.png
[JNet_475_pretrain_1_labelx_depth]: /experiments/images/JNet_475_pretrain_1_labelx_depth.png
[JNet_475_pretrain_1_labelx_plane]: /experiments/images/JNet_475_pretrain_1_labelx_plane.png
[JNet_475_pretrain_1_labelz_depth]: /experiments/images/JNet_475_pretrain_1_labelz_depth.png
[JNet_475_pretrain_1_labelz_plane]: /experiments/images/JNet_475_pretrain_1_labelz_plane.png
[JNet_475_pretrain_1_original_depth]: /experiments/images/JNet_475_pretrain_1_original_depth.png
[JNet_475_pretrain_1_original_plane]: /experiments/images/JNet_475_pretrain_1_original_plane.png
[JNet_475_pretrain_1_outputx_depth]: /experiments/images/JNet_475_pretrain_1_outputx_depth.png
[JNet_475_pretrain_1_outputx_plane]: /experiments/images/JNet_475_pretrain_1_outputx_plane.png
[JNet_475_pretrain_1_outputz_depth]: /experiments/images/JNet_475_pretrain_1_outputz_depth.png
[JNet_475_pretrain_1_outputz_plane]: /experiments/images/JNet_475_pretrain_1_outputz_plane.png
[JNet_475_pretrain_2_labelx_depth]: /experiments/images/JNet_475_pretrain_2_labelx_depth.png
[JNet_475_pretrain_2_labelx_plane]: /experiments/images/JNet_475_pretrain_2_labelx_plane.png
[JNet_475_pretrain_2_labelz_depth]: /experiments/images/JNet_475_pretrain_2_labelz_depth.png
[JNet_475_pretrain_2_labelz_plane]: /experiments/images/JNet_475_pretrain_2_labelz_plane.png
[JNet_475_pretrain_2_original_depth]: /experiments/images/JNet_475_pretrain_2_original_depth.png
[JNet_475_pretrain_2_original_plane]: /experiments/images/JNet_475_pretrain_2_original_plane.png
[JNet_475_pretrain_2_outputx_depth]: /experiments/images/JNet_475_pretrain_2_outputx_depth.png
[JNet_475_pretrain_2_outputx_plane]: /experiments/images/JNet_475_pretrain_2_outputx_plane.png
[JNet_475_pretrain_2_outputz_depth]: /experiments/images/JNet_475_pretrain_2_outputz_depth.png
[JNet_475_pretrain_2_outputz_plane]: /experiments/images/JNet_475_pretrain_2_outputz_plane.png
[JNet_475_pretrain_3_labelx_depth]: /experiments/images/JNet_475_pretrain_3_labelx_depth.png
[JNet_475_pretrain_3_labelx_plane]: /experiments/images/JNet_475_pretrain_3_labelx_plane.png
[JNet_475_pretrain_3_labelz_depth]: /experiments/images/JNet_475_pretrain_3_labelz_depth.png
[JNet_475_pretrain_3_labelz_plane]: /experiments/images/JNet_475_pretrain_3_labelz_plane.png
[JNet_475_pretrain_3_original_depth]: /experiments/images/JNet_475_pretrain_3_original_depth.png
[JNet_475_pretrain_3_original_plane]: /experiments/images/JNet_475_pretrain_3_original_plane.png
[JNet_475_pretrain_3_outputx_depth]: /experiments/images/JNet_475_pretrain_3_outputx_depth.png
[JNet_475_pretrain_3_outputx_plane]: /experiments/images/JNet_475_pretrain_3_outputx_plane.png
[JNet_475_pretrain_3_outputz_depth]: /experiments/images/JNet_475_pretrain_3_outputz_depth.png
[JNet_475_pretrain_3_outputz_plane]: /experiments/images/JNet_475_pretrain_3_outputz_plane.png
[JNet_475_pretrain_4_labelx_depth]: /experiments/images/JNet_475_pretrain_4_labelx_depth.png
[JNet_475_pretrain_4_labelx_plane]: /experiments/images/JNet_475_pretrain_4_labelx_plane.png
[JNet_475_pretrain_4_labelz_depth]: /experiments/images/JNet_475_pretrain_4_labelz_depth.png
[JNet_475_pretrain_4_labelz_plane]: /experiments/images/JNet_475_pretrain_4_labelz_plane.png
[JNet_475_pretrain_4_original_depth]: /experiments/images/JNet_475_pretrain_4_original_depth.png
[JNet_475_pretrain_4_original_plane]: /experiments/images/JNet_475_pretrain_4_original_plane.png
[JNet_475_pretrain_4_outputx_depth]: /experiments/images/JNet_475_pretrain_4_outputx_depth.png
[JNet_475_pretrain_4_outputx_plane]: /experiments/images/JNet_475_pretrain_4_outputx_plane.png
[JNet_475_pretrain_4_outputz_depth]: /experiments/images/JNet_475_pretrain_4_outputz_depth.png
[JNet_475_pretrain_4_outputz_plane]: /experiments/images/JNet_475_pretrain_4_outputz_plane.png
[JNet_476_0_heatmap_depth]: /experiments/images/JNet_476_0_heatmap_depth.png
[JNet_476_0_labelx_depth]: /experiments/images/JNet_476_0_labelx_depth.png
[JNet_476_0_labelz_depth]: /experiments/images/JNet_476_0_labelz_depth.png
[JNet_476_0_original_depth]: /experiments/images/JNet_476_0_original_depth.png
[JNet_476_0_outputx_depth]: /experiments/images/JNet_476_0_outputx_depth.png
[JNet_476_0_outputz_depth]: /experiments/images/JNet_476_0_outputz_depth.png
[JNet_476_0_reconst_depth]: /experiments/images/JNet_476_0_reconst_depth.png
[JNet_476_1_heatmap_depth]: /experiments/images/JNet_476_1_heatmap_depth.png
[JNet_476_1_labelx_depth]: /experiments/images/JNet_476_1_labelx_depth.png
[JNet_476_1_labelz_depth]: /experiments/images/JNet_476_1_labelz_depth.png
[JNet_476_1_original_depth]: /experiments/images/JNet_476_1_original_depth.png
[JNet_476_1_outputx_depth]: /experiments/images/JNet_476_1_outputx_depth.png
[JNet_476_1_outputz_depth]: /experiments/images/JNet_476_1_outputz_depth.png
[JNet_476_1_reconst_depth]: /experiments/images/JNet_476_1_reconst_depth.png
[JNet_476_2_heatmap_depth]: /experiments/images/JNet_476_2_heatmap_depth.png
[JNet_476_2_labelx_depth]: /experiments/images/JNet_476_2_labelx_depth.png
[JNet_476_2_labelz_depth]: /experiments/images/JNet_476_2_labelz_depth.png
[JNet_476_2_original_depth]: /experiments/images/JNet_476_2_original_depth.png
[JNet_476_2_outputx_depth]: /experiments/images/JNet_476_2_outputx_depth.png
[JNet_476_2_outputz_depth]: /experiments/images/JNet_476_2_outputz_depth.png
[JNet_476_2_reconst_depth]: /experiments/images/JNet_476_2_reconst_depth.png
[JNet_476_3_heatmap_depth]: /experiments/images/JNet_476_3_heatmap_depth.png
[JNet_476_3_labelx_depth]: /experiments/images/JNet_476_3_labelx_depth.png
[JNet_476_3_labelz_depth]: /experiments/images/JNet_476_3_labelz_depth.png
[JNet_476_3_original_depth]: /experiments/images/JNet_476_3_original_depth.png
[JNet_476_3_outputx_depth]: /experiments/images/JNet_476_3_outputx_depth.png
[JNet_476_3_outputz_depth]: /experiments/images/JNet_476_3_outputz_depth.png
[JNet_476_3_reconst_depth]: /experiments/images/JNet_476_3_reconst_depth.png
[JNet_476_4_heatmap_depth]: /experiments/images/JNet_476_4_heatmap_depth.png
[JNet_476_4_labelx_depth]: /experiments/images/JNet_476_4_labelx_depth.png
[JNet_476_4_labelz_depth]: /experiments/images/JNet_476_4_labelz_depth.png
[JNet_476_4_original_depth]: /experiments/images/JNet_476_4_original_depth.png
[JNet_476_4_outputx_depth]: /experiments/images/JNet_476_4_outputx_depth.png
[JNet_476_4_outputz_depth]: /experiments/images/JNet_476_4_outputz_depth.png
[JNet_476_4_reconst_depth]: /experiments/images/JNet_476_4_reconst_depth.png
[JNet_476_psf_post]: /experiments/images/JNet_476_psf_post.png
[JNet_476_psf_pre]: /experiments/images/JNet_476_psf_pre.png
[finetuned]: /experiments/tmp/JNet_476_train.png
[pretrained_model]: /experiments/tmp/JNet_475_pretrain_train.png
