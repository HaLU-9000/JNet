



# JNet_362 Report
  
the parameters to replicate the results of JNet_362. attention added.  
pretrained model : JNet_361_pretrain
## Model Parameters
  

|Parameter|Value|Comment|
| :--- | :--- | :--- |
|hidden_channels_list|[16, 32, 64, 128, 256]||
|attn_list|[False, False, False, False, True]||
|nblocks|2||
|activation|nn.ReLU(inplace=True)||
|dropout|0.5||
|superres|True||
|partial|None||
|reconstruct|False||
|apply_vq|False||
|use_x_quantized|False||
|use_fftconv|True||
|mu_z|0.1||
|sig_z|0.1||
|blur_mode|gibsonlanni|`gaussian` or `gibsonlanni`|
|size_x|101||
|size_y|101||
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
|res_axial|0.05|microns|
|pZ|0|microns, particle distance from coverslip|
|bet_z|30.0||
|bet_xy|3.0||
|sig_eps|0.02||
|scale|10||
|device|cuda||

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
          (attn): CrossAttentionBlock(  
            (attn): CrossAttention(  
              (to_q): Linear(in_features=256, out_features=256, bias=False)  
              (to_k): Linear(in_features=256, out_features=256, bias=False)  
              (to_v): Linear(in_features=256, out_features=256, bias=False)  
              (to_out): Sequential(  
                (0): Linear(in_features=256, out_features=256, bias=True)  
              )  
            )  
            (norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)  
          )  
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

## Results
  
mean MSE: 0.2820992171764374, mean BCE: 0.7574471235275269
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_361_pretrain_0_original_plane]|![JNet_361_pretrain_0_output_plane]|![JNet_361_pretrain_0_label_plane]|
  
MSE: 0.28204047679901123, BCE: 0.757329523563385  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_361_pretrain_0_original_depth]|![JNet_361_pretrain_0_output_depth]|![JNet_361_pretrain_0_label_depth]|
  
MSE: 0.28204047679901123, BCE: 0.757329523563385  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_361_pretrain_1_original_plane]|![JNet_361_pretrain_1_output_plane]|![JNet_361_pretrain_1_label_plane]|
  
MSE: 0.28436845541000366, BCE: 0.7619986534118652  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_361_pretrain_1_original_depth]|![JNet_361_pretrain_1_output_depth]|![JNet_361_pretrain_1_label_depth]|
  
MSE: 0.28436845541000366, BCE: 0.7619986534118652  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_361_pretrain_2_original_plane]|![JNet_361_pretrain_2_output_plane]|![JNet_361_pretrain_2_label_plane]|
  
MSE: 0.2828574776649475, BCE: 0.7589715719223022  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_361_pretrain_2_original_depth]|![JNet_361_pretrain_2_output_depth]|![JNet_361_pretrain_2_label_depth]|
  
MSE: 0.2828574776649475, BCE: 0.7589715719223022  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_361_pretrain_3_original_plane]|![JNet_361_pretrain_3_output_plane]|![JNet_361_pretrain_3_label_plane]|
  
MSE: 0.28101596236228943, BCE: 0.7552724480628967  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_361_pretrain_3_original_depth]|![JNet_361_pretrain_3_output_depth]|![JNet_361_pretrain_3_label_depth]|
  
MSE: 0.28101596236228943, BCE: 0.7552724480628967  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_361_pretrain_4_original_plane]|![JNet_361_pretrain_4_output_plane]|![JNet_361_pretrain_4_label_plane]|
  
MSE: 0.28021371364593506, BCE: 0.7536635994911194  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_361_pretrain_4_original_depth]|![JNet_361_pretrain_4_output_depth]|![JNet_361_pretrain_4_label_depth]|
  
MSE: 0.28021371364593506, BCE: 0.7536635994911194  
  
mean MSE: 0.26444298028945923, mean BCE: 0.7220429182052612
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_362_0_original_plane]|![JNet_362_0_output_plane]|![JNet_362_0_label_plane]|
  
MSE: 0.26446542143821716, BCE: 0.7220878601074219  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_362_0_original_depth]|![JNet_362_0_output_depth]|![JNet_362_0_label_depth]|
  
MSE: 0.26446542143821716, BCE: 0.7220878601074219  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_362_1_original_plane]|![JNet_362_1_output_plane]|![JNet_362_1_label_plane]|
  
MSE: 0.2646002173423767, BCE: 0.7223570346832275  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_362_1_original_depth]|![JNet_362_1_output_depth]|![JNet_362_1_label_depth]|
  
MSE: 0.2646002173423767, BCE: 0.7223570346832275  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_362_2_original_plane]|![JNet_362_2_output_plane]|![JNet_362_2_label_plane]|
  
MSE: 0.2637953758239746, BCE: 0.720747709274292  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_362_2_original_depth]|![JNet_362_2_output_depth]|![JNet_362_2_label_depth]|
  
MSE: 0.2637953758239746, BCE: 0.720747709274292  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_362_3_original_plane]|![JNet_362_3_output_plane]|![JNet_362_3_label_plane]|
  
MSE: 0.2649853825569153, BCE: 0.7231284379959106  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_362_3_original_depth]|![JNet_362_3_output_depth]|![JNet_362_3_label_depth]|
  
MSE: 0.2649853825569153, BCE: 0.7231284379959106  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_362_4_original_plane]|![JNet_362_4_output_plane]|![JNet_362_4_label_plane]|
  
MSE: 0.26436847448349, BCE: 0.7218936681747437  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_362_4_original_depth]|![JNet_362_4_output_depth]|![JNet_362_4_label_depth]|
  
MSE: 0.26436847448349, BCE: 0.7218936681747437  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_361_pretrain_beads_001_roi000_original_depth]|![JNet_361_pretrain_beads_001_roi000_output_depth]|![JNet_361_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.532200276851654, quantized loss: 0.24088191986083984  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_361_pretrain_beads_001_roi001_original_depth]|![JNet_361_pretrain_beads_001_roi001_output_depth]|![JNet_361_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5208145380020142, quantized loss: 0.23988686501979828  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_361_pretrain_beads_001_roi002_original_depth]|![JNet_361_pretrain_beads_001_roi002_output_depth]|![JNet_361_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5362014770507812, quantized loss: 0.24106796085834503  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_361_pretrain_beads_001_roi003_original_depth]|![JNet_361_pretrain_beads_001_roi003_output_depth]|![JNet_361_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5194132328033447, quantized loss: 0.23993229866027832  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_361_pretrain_beads_001_roi004_original_depth]|![JNet_361_pretrain_beads_001_roi004_output_depth]|![JNet_361_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5328969955444336, quantized loss: 0.240861177444458  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_361_pretrain_beads_002_roi000_original_depth]|![JNet_361_pretrain_beads_002_roi000_output_depth]|![JNet_361_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5308762192726135, quantized loss: 0.24072612822055817  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_361_pretrain_beads_002_roi001_original_depth]|![JNet_361_pretrain_beads_002_roi001_output_depth]|![JNet_361_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5341352820396423, quantized loss: 0.24092625081539154  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_361_pretrain_beads_002_roi002_original_depth]|![JNet_361_pretrain_beads_002_roi002_output_depth]|![JNet_361_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 491.5200000000001, MSE: 0.5327744483947754, quantized loss: 0.24085097014904022  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_362_beads_001_roi000_original_depth]|![JNet_362_beads_001_roi000_output_depth]|![JNet_362_beads_001_roi000_reconst_depth]|
  
volume: 0.7535000000000002, MSE: 0.009032287634909153, quantized loss: 0.23789648711681366  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_362_beads_001_roi001_original_depth]|![JNet_362_beads_001_roi001_output_depth]|![JNet_362_beads_001_roi001_reconst_depth]|
  
volume: 1.3823750000000004, MSE: 0.01433609426021576, quantized loss: 0.23698118329048157  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_362_beads_001_roi002_original_depth]|![JNet_362_beads_001_roi002_output_depth]|![JNet_362_beads_001_roi002_reconst_depth]|
  
volume: 0.6433750000000001, MSE: 0.008825917728245258, quantized loss: 0.23818811774253845  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_362_beads_001_roi003_original_depth]|![JNet_362_beads_001_roi003_output_depth]|![JNet_362_beads_001_roi003_reconst_depth]|
  
volume: 0.8637500000000002, MSE: 0.015565279871225357, quantized loss: 0.23611852526664734  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_362_beads_001_roi004_original_depth]|![JNet_362_beads_001_roi004_output_depth]|![JNet_362_beads_001_roi004_reconst_depth]|
  
volume: 0.8618750000000002, MSE: 0.010591001249849796, quantized loss: 0.23789721727371216  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_362_beads_002_roi000_original_depth]|![JNet_362_beads_002_roi000_output_depth]|![JNet_362_beads_002_roi000_reconst_depth]|
  
volume: 1.0087500000000003, MSE: 0.01184847205877304, quantized loss: 0.23772020637989044  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_362_beads_002_roi001_original_depth]|![JNet_362_beads_002_roi001_output_depth]|![JNet_362_beads_002_roi001_reconst_depth]|
  
volume: 0.8417500000000002, MSE: 0.009971690364181995, quantized loss: 0.23800252377986908  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_362_beads_002_roi002_original_depth]|![JNet_362_beads_002_roi002_output_depth]|![JNet_362_beads_002_roi002_reconst_depth]|
  
volume: 0.8737500000000002, MSE: 0.010725263506174088, quantized loss: 0.2378915250301361  

|pre|post|
| :---: | :---: |
|![JNet_362_psf_pre]|![JNet_362_psf_post]|
  



[JNet_361_pretrain_0_label_depth]: /experiments/images/JNet_361_pretrain_0_label_depth.png
[JNet_361_pretrain_0_label_plane]: /experiments/images/JNet_361_pretrain_0_label_plane.png
[JNet_361_pretrain_0_original_depth]: /experiments/images/JNet_361_pretrain_0_original_depth.png
[JNet_361_pretrain_0_original_plane]: /experiments/images/JNet_361_pretrain_0_original_plane.png
[JNet_361_pretrain_0_output_depth]: /experiments/images/JNet_361_pretrain_0_output_depth.png
[JNet_361_pretrain_0_output_plane]: /experiments/images/JNet_361_pretrain_0_output_plane.png
[JNet_361_pretrain_1_label_depth]: /experiments/images/JNet_361_pretrain_1_label_depth.png
[JNet_361_pretrain_1_label_plane]: /experiments/images/JNet_361_pretrain_1_label_plane.png
[JNet_361_pretrain_1_original_depth]: /experiments/images/JNet_361_pretrain_1_original_depth.png
[JNet_361_pretrain_1_original_plane]: /experiments/images/JNet_361_pretrain_1_original_plane.png
[JNet_361_pretrain_1_output_depth]: /experiments/images/JNet_361_pretrain_1_output_depth.png
[JNet_361_pretrain_1_output_plane]: /experiments/images/JNet_361_pretrain_1_output_plane.png
[JNet_361_pretrain_2_label_depth]: /experiments/images/JNet_361_pretrain_2_label_depth.png
[JNet_361_pretrain_2_label_plane]: /experiments/images/JNet_361_pretrain_2_label_plane.png
[JNet_361_pretrain_2_original_depth]: /experiments/images/JNet_361_pretrain_2_original_depth.png
[JNet_361_pretrain_2_original_plane]: /experiments/images/JNet_361_pretrain_2_original_plane.png
[JNet_361_pretrain_2_output_depth]: /experiments/images/JNet_361_pretrain_2_output_depth.png
[JNet_361_pretrain_2_output_plane]: /experiments/images/JNet_361_pretrain_2_output_plane.png
[JNet_361_pretrain_3_label_depth]: /experiments/images/JNet_361_pretrain_3_label_depth.png
[JNet_361_pretrain_3_label_plane]: /experiments/images/JNet_361_pretrain_3_label_plane.png
[JNet_361_pretrain_3_original_depth]: /experiments/images/JNet_361_pretrain_3_original_depth.png
[JNet_361_pretrain_3_original_plane]: /experiments/images/JNet_361_pretrain_3_original_plane.png
[JNet_361_pretrain_3_output_depth]: /experiments/images/JNet_361_pretrain_3_output_depth.png
[JNet_361_pretrain_3_output_plane]: /experiments/images/JNet_361_pretrain_3_output_plane.png
[JNet_361_pretrain_4_label_depth]: /experiments/images/JNet_361_pretrain_4_label_depth.png
[JNet_361_pretrain_4_label_plane]: /experiments/images/JNet_361_pretrain_4_label_plane.png
[JNet_361_pretrain_4_original_depth]: /experiments/images/JNet_361_pretrain_4_original_depth.png
[JNet_361_pretrain_4_original_plane]: /experiments/images/JNet_361_pretrain_4_original_plane.png
[JNet_361_pretrain_4_output_depth]: /experiments/images/JNet_361_pretrain_4_output_depth.png
[JNet_361_pretrain_4_output_plane]: /experiments/images/JNet_361_pretrain_4_output_plane.png
[JNet_361_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi000_original_depth.png
[JNet_361_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi000_output_depth.png
[JNet_361_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi000_reconst_depth.png
[JNet_361_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi001_original_depth.png
[JNet_361_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi001_output_depth.png
[JNet_361_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi001_reconst_depth.png
[JNet_361_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi002_original_depth.png
[JNet_361_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi002_output_depth.png
[JNet_361_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi002_reconst_depth.png
[JNet_361_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi003_original_depth.png
[JNet_361_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi003_output_depth.png
[JNet_361_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi003_reconst_depth.png
[JNet_361_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi004_original_depth.png
[JNet_361_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi004_output_depth.png
[JNet_361_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_361_pretrain_beads_001_roi004_reconst_depth.png
[JNet_361_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_361_pretrain_beads_002_roi000_original_depth.png
[JNet_361_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_361_pretrain_beads_002_roi000_output_depth.png
[JNet_361_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_361_pretrain_beads_002_roi000_reconst_depth.png
[JNet_361_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_361_pretrain_beads_002_roi001_original_depth.png
[JNet_361_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_361_pretrain_beads_002_roi001_output_depth.png
[JNet_361_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_361_pretrain_beads_002_roi001_reconst_depth.png
[JNet_361_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_361_pretrain_beads_002_roi002_original_depth.png
[JNet_361_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_361_pretrain_beads_002_roi002_output_depth.png
[JNet_361_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_361_pretrain_beads_002_roi002_reconst_depth.png
[JNet_362_0_label_depth]: /experiments/images/JNet_362_0_label_depth.png
[JNet_362_0_label_plane]: /experiments/images/JNet_362_0_label_plane.png
[JNet_362_0_original_depth]: /experiments/images/JNet_362_0_original_depth.png
[JNet_362_0_original_plane]: /experiments/images/JNet_362_0_original_plane.png
[JNet_362_0_output_depth]: /experiments/images/JNet_362_0_output_depth.png
[JNet_362_0_output_plane]: /experiments/images/JNet_362_0_output_plane.png
[JNet_362_1_label_depth]: /experiments/images/JNet_362_1_label_depth.png
[JNet_362_1_label_plane]: /experiments/images/JNet_362_1_label_plane.png
[JNet_362_1_original_depth]: /experiments/images/JNet_362_1_original_depth.png
[JNet_362_1_original_plane]: /experiments/images/JNet_362_1_original_plane.png
[JNet_362_1_output_depth]: /experiments/images/JNet_362_1_output_depth.png
[JNet_362_1_output_plane]: /experiments/images/JNet_362_1_output_plane.png
[JNet_362_2_label_depth]: /experiments/images/JNet_362_2_label_depth.png
[JNet_362_2_label_plane]: /experiments/images/JNet_362_2_label_plane.png
[JNet_362_2_original_depth]: /experiments/images/JNet_362_2_original_depth.png
[JNet_362_2_original_plane]: /experiments/images/JNet_362_2_original_plane.png
[JNet_362_2_output_depth]: /experiments/images/JNet_362_2_output_depth.png
[JNet_362_2_output_plane]: /experiments/images/JNet_362_2_output_plane.png
[JNet_362_3_label_depth]: /experiments/images/JNet_362_3_label_depth.png
[JNet_362_3_label_plane]: /experiments/images/JNet_362_3_label_plane.png
[JNet_362_3_original_depth]: /experiments/images/JNet_362_3_original_depth.png
[JNet_362_3_original_plane]: /experiments/images/JNet_362_3_original_plane.png
[JNet_362_3_output_depth]: /experiments/images/JNet_362_3_output_depth.png
[JNet_362_3_output_plane]: /experiments/images/JNet_362_3_output_plane.png
[JNet_362_4_label_depth]: /experiments/images/JNet_362_4_label_depth.png
[JNet_362_4_label_plane]: /experiments/images/JNet_362_4_label_plane.png
[JNet_362_4_original_depth]: /experiments/images/JNet_362_4_original_depth.png
[JNet_362_4_original_plane]: /experiments/images/JNet_362_4_original_plane.png
[JNet_362_4_output_depth]: /experiments/images/JNet_362_4_output_depth.png
[JNet_362_4_output_plane]: /experiments/images/JNet_362_4_output_plane.png
[JNet_362_beads_001_roi000_original_depth]: /experiments/images/JNet_362_beads_001_roi000_original_depth.png
[JNet_362_beads_001_roi000_output_depth]: /experiments/images/JNet_362_beads_001_roi000_output_depth.png
[JNet_362_beads_001_roi000_reconst_depth]: /experiments/images/JNet_362_beads_001_roi000_reconst_depth.png
[JNet_362_beads_001_roi001_original_depth]: /experiments/images/JNet_362_beads_001_roi001_original_depth.png
[JNet_362_beads_001_roi001_output_depth]: /experiments/images/JNet_362_beads_001_roi001_output_depth.png
[JNet_362_beads_001_roi001_reconst_depth]: /experiments/images/JNet_362_beads_001_roi001_reconst_depth.png
[JNet_362_beads_001_roi002_original_depth]: /experiments/images/JNet_362_beads_001_roi002_original_depth.png
[JNet_362_beads_001_roi002_output_depth]: /experiments/images/JNet_362_beads_001_roi002_output_depth.png
[JNet_362_beads_001_roi002_reconst_depth]: /experiments/images/JNet_362_beads_001_roi002_reconst_depth.png
[JNet_362_beads_001_roi003_original_depth]: /experiments/images/JNet_362_beads_001_roi003_original_depth.png
[JNet_362_beads_001_roi003_output_depth]: /experiments/images/JNet_362_beads_001_roi003_output_depth.png
[JNet_362_beads_001_roi003_reconst_depth]: /experiments/images/JNet_362_beads_001_roi003_reconst_depth.png
[JNet_362_beads_001_roi004_original_depth]: /experiments/images/JNet_362_beads_001_roi004_original_depth.png
[JNet_362_beads_001_roi004_output_depth]: /experiments/images/JNet_362_beads_001_roi004_output_depth.png
[JNet_362_beads_001_roi004_reconst_depth]: /experiments/images/JNet_362_beads_001_roi004_reconst_depth.png
[JNet_362_beads_002_roi000_original_depth]: /experiments/images/JNet_362_beads_002_roi000_original_depth.png
[JNet_362_beads_002_roi000_output_depth]: /experiments/images/JNet_362_beads_002_roi000_output_depth.png
[JNet_362_beads_002_roi000_reconst_depth]: /experiments/images/JNet_362_beads_002_roi000_reconst_depth.png
[JNet_362_beads_002_roi001_original_depth]: /experiments/images/JNet_362_beads_002_roi001_original_depth.png
[JNet_362_beads_002_roi001_output_depth]: /experiments/images/JNet_362_beads_002_roi001_output_depth.png
[JNet_362_beads_002_roi001_reconst_depth]: /experiments/images/JNet_362_beads_002_roi001_reconst_depth.png
[JNet_362_beads_002_roi002_original_depth]: /experiments/images/JNet_362_beads_002_roi002_original_depth.png
[JNet_362_beads_002_roi002_output_depth]: /experiments/images/JNet_362_beads_002_roi002_output_depth.png
[JNet_362_beads_002_roi002_reconst_depth]: /experiments/images/JNet_362_beads_002_roi002_reconst_depth.png
[JNet_362_psf_post]: /experiments/images/JNet_362_psf_post.png
[JNet_362_psf_pre]: /experiments/images/JNet_362_psf_pre.png
