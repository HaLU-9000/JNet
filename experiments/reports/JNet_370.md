



# JNet_370 Report
  
the parameters to replicate the results of JNet_370. background=0.01  
pretrained model : JNet_369_pretrain
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
|use_fftconv|True||
|mu_z|0.1||
|sig_z|0.1||
|blur_mode|gibsonlanni|`gaussian` or `gibsonlanni`|
|size_x|51||
|size_y|51||
|size_z|161||
|NA|0.8||
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
|background|0.001||
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
|mask|False|
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

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.019879257306456566, mean BCE: 0.08214499056339264
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_369_pretrain_0_original_plane]|![JNet_369_pretrain_0_output_plane]|![JNet_369_pretrain_0_label_plane]|
  
MSE: 0.03120328113436699, BCE: 0.14587382972240448  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_369_pretrain_0_original_depth]|![JNet_369_pretrain_0_output_depth]|![JNet_369_pretrain_0_label_depth]|
  
MSE: 0.03120328113436699, BCE: 0.14587382972240448  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_369_pretrain_1_original_plane]|![JNet_369_pretrain_1_output_plane]|![JNet_369_pretrain_1_label_plane]|
  
MSE: 0.01565755158662796, BCE: 0.05942375212907791  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_369_pretrain_1_original_depth]|![JNet_369_pretrain_1_output_depth]|![JNet_369_pretrain_1_label_depth]|
  
MSE: 0.01565755158662796, BCE: 0.05942375212907791  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_369_pretrain_2_original_plane]|![JNet_369_pretrain_2_output_plane]|![JNet_369_pretrain_2_label_plane]|
  
MSE: 0.02137651853263378, BCE: 0.08620844036340714  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_369_pretrain_2_original_depth]|![JNet_369_pretrain_2_output_depth]|![JNet_369_pretrain_2_label_depth]|
  
MSE: 0.02137651853263378, BCE: 0.08620844036340714  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_369_pretrain_3_original_plane]|![JNet_369_pretrain_3_output_plane]|![JNet_369_pretrain_3_label_plane]|
  
MSE: 0.017720064148306847, BCE: 0.062553271651268  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_369_pretrain_3_original_depth]|![JNet_369_pretrain_3_output_depth]|![JNet_369_pretrain_3_label_depth]|
  
MSE: 0.017720064148306847, BCE: 0.062553271651268  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_369_pretrain_4_original_plane]|![JNet_369_pretrain_4_output_plane]|![JNet_369_pretrain_4_label_plane]|
  
MSE: 0.013438873924314976, BCE: 0.05666566640138626  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_369_pretrain_4_original_depth]|![JNet_369_pretrain_4_output_depth]|![JNet_369_pretrain_4_label_depth]|
  
MSE: 0.013438873924314976, BCE: 0.05666566640138626  
  
mean MSE: 0.03122904524207115, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_370_0_original_plane]|![JNet_370_0_output_plane]|![JNet_370_0_label_plane]|
  
MSE: 0.027867572382092476, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_370_0_original_depth]|![JNet_370_0_output_depth]|![JNet_370_0_label_depth]|
  
MSE: 0.027867572382092476, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_370_1_original_plane]|![JNet_370_1_output_plane]|![JNet_370_1_label_plane]|
  
MSE: 0.0368536114692688, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_370_1_original_depth]|![JNet_370_1_output_depth]|![JNet_370_1_label_depth]|
  
MSE: 0.0368536114692688, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_370_2_original_plane]|![JNet_370_2_output_plane]|![JNet_370_2_label_plane]|
  
MSE: 0.03466992452740669, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_370_2_original_depth]|![JNet_370_2_output_depth]|![JNet_370_2_label_depth]|
  
MSE: 0.03466992452740669, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_370_3_original_plane]|![JNet_370_3_output_plane]|![JNet_370_3_label_plane]|
  
MSE: 0.02735789492726326, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_370_3_original_depth]|![JNet_370_3_output_depth]|![JNet_370_3_label_depth]|
  
MSE: 0.02735789492726326, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_370_4_original_plane]|![JNet_370_4_output_plane]|![JNet_370_4_label_plane]|
  
MSE: 0.02939623035490513, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_370_4_original_depth]|![JNet_370_4_output_depth]|![JNet_370_4_label_depth]|
  
MSE: 0.02939623035490513, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_369_pretrain_beads_001_roi000_original_depth]|![JNet_369_pretrain_beads_001_roi000_output_depth]|![JNet_369_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 12.995750000000003, MSE: 0.0012509251246228814, quantized loss: 0.0014264661585912108  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_369_pretrain_beads_001_roi001_original_depth]|![JNet_369_pretrain_beads_001_roi001_output_depth]|![JNet_369_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 19.831750000000003, MSE: 0.0021369822788983583, quantized loss: 0.0018113723490387201  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_369_pretrain_beads_001_roi002_original_depth]|![JNet_369_pretrain_beads_001_roi002_output_depth]|![JNet_369_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 12.700250000000002, MSE: 0.0011792111909016967, quantized loss: 0.0013350477674975991  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_369_pretrain_beads_001_roi003_original_depth]|![JNet_369_pretrain_beads_001_roi003_output_depth]|![JNet_369_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 20.460000000000004, MSE: 0.002060588914901018, quantized loss: 0.0018820672994479537  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_369_pretrain_beads_001_roi004_original_depth]|![JNet_369_pretrain_beads_001_roi004_output_depth]|![JNet_369_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 13.996500000000003, MSE: 0.0015093852998688817, quantized loss: 0.001299420022405684  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_369_pretrain_beads_002_roi000_original_depth]|![JNet_369_pretrain_beads_002_roi000_output_depth]|![JNet_369_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 14.844875000000004, MSE: 0.0016992706805467606, quantized loss: 0.0013135416666045785  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_369_pretrain_beads_002_roi001_original_depth]|![JNet_369_pretrain_beads_002_roi001_output_depth]|![JNet_369_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 13.839625000000003, MSE: 0.0012611792190000415, quantized loss: 0.001386154443025589  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_369_pretrain_beads_002_roi002_original_depth]|![JNet_369_pretrain_beads_002_roi002_output_depth]|![JNet_369_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 14.110000000000003, MSE: 0.001471961964853108, quantized loss: 0.0013026886153966188  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_370_beads_001_roi000_original_depth]|![JNet_370_beads_001_roi000_output_depth]|![JNet_370_beads_001_roi000_reconst_depth]|
  
volume: 11.363250000000003, MSE: 0.0005304909427650273, quantized loss: 3.688725882966537e-06  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_370_beads_001_roi001_original_depth]|![JNet_370_beads_001_roi001_output_depth]|![JNet_370_beads_001_roi001_reconst_depth]|
  
volume: 17.778000000000006, MSE: 0.0009197432082146406, quantized loss: 5.191192940401379e-06  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_370_beads_001_roi002_original_depth]|![JNet_370_beads_001_roi002_output_depth]|![JNet_370_beads_001_roi002_reconst_depth]|
  
volume: 11.419625000000003, MSE: 0.0005100183770991862, quantized loss: 3.9270762499654666e-06  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_370_beads_001_roi003_original_depth]|![JNet_370_beads_001_roi003_output_depth]|![JNet_370_beads_001_roi003_reconst_depth]|
  
volume: 18.682375000000004, MSE: 0.0007733766105957329, quantized loss: 6.413363280444173e-06  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_370_beads_001_roi004_original_depth]|![JNet_370_beads_001_roi004_output_depth]|![JNet_370_beads_001_roi004_reconst_depth]|
  
volume: 12.330875000000002, MSE: 0.000458145426819101, quantized loss: 4.232700121065136e-06  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_370_beads_002_roi000_original_depth]|![JNet_370_beads_002_roi000_output_depth]|![JNet_370_beads_002_roi000_reconst_depth]|
  
volume: 13.138000000000003, MSE: 0.00047114284825511277, quantized loss: 4.040351996081881e-06  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_370_beads_002_roi001_original_depth]|![JNet_370_beads_002_roi001_output_depth]|![JNet_370_beads_002_roi001_reconst_depth]|
  
volume: 12.060000000000002, MSE: 0.0004934866447001696, quantized loss: 4.3286695472488645e-06  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_370_beads_002_roi002_original_depth]|![JNet_370_beads_002_roi002_output_depth]|![JNet_370_beads_002_roi002_reconst_depth]|
  
volume: 12.534750000000003, MSE: 0.0004584674024954438, quantized loss: 3.8082250739535084e-06  

|pre|post|
| :---: | :---: |
|![JNet_370_psf_pre]|![JNet_370_psf_post]|
  



[JNet_369_pretrain_0_label_depth]: /experiments/images/JNet_369_pretrain_0_label_depth.png
[JNet_369_pretrain_0_label_plane]: /experiments/images/JNet_369_pretrain_0_label_plane.png
[JNet_369_pretrain_0_original_depth]: /experiments/images/JNet_369_pretrain_0_original_depth.png
[JNet_369_pretrain_0_original_plane]: /experiments/images/JNet_369_pretrain_0_original_plane.png
[JNet_369_pretrain_0_output_depth]: /experiments/images/JNet_369_pretrain_0_output_depth.png
[JNet_369_pretrain_0_output_plane]: /experiments/images/JNet_369_pretrain_0_output_plane.png
[JNet_369_pretrain_1_label_depth]: /experiments/images/JNet_369_pretrain_1_label_depth.png
[JNet_369_pretrain_1_label_plane]: /experiments/images/JNet_369_pretrain_1_label_plane.png
[JNet_369_pretrain_1_original_depth]: /experiments/images/JNet_369_pretrain_1_original_depth.png
[JNet_369_pretrain_1_original_plane]: /experiments/images/JNet_369_pretrain_1_original_plane.png
[JNet_369_pretrain_1_output_depth]: /experiments/images/JNet_369_pretrain_1_output_depth.png
[JNet_369_pretrain_1_output_plane]: /experiments/images/JNet_369_pretrain_1_output_plane.png
[JNet_369_pretrain_2_label_depth]: /experiments/images/JNet_369_pretrain_2_label_depth.png
[JNet_369_pretrain_2_label_plane]: /experiments/images/JNet_369_pretrain_2_label_plane.png
[JNet_369_pretrain_2_original_depth]: /experiments/images/JNet_369_pretrain_2_original_depth.png
[JNet_369_pretrain_2_original_plane]: /experiments/images/JNet_369_pretrain_2_original_plane.png
[JNet_369_pretrain_2_output_depth]: /experiments/images/JNet_369_pretrain_2_output_depth.png
[JNet_369_pretrain_2_output_plane]: /experiments/images/JNet_369_pretrain_2_output_plane.png
[JNet_369_pretrain_3_label_depth]: /experiments/images/JNet_369_pretrain_3_label_depth.png
[JNet_369_pretrain_3_label_plane]: /experiments/images/JNet_369_pretrain_3_label_plane.png
[JNet_369_pretrain_3_original_depth]: /experiments/images/JNet_369_pretrain_3_original_depth.png
[JNet_369_pretrain_3_original_plane]: /experiments/images/JNet_369_pretrain_3_original_plane.png
[JNet_369_pretrain_3_output_depth]: /experiments/images/JNet_369_pretrain_3_output_depth.png
[JNet_369_pretrain_3_output_plane]: /experiments/images/JNet_369_pretrain_3_output_plane.png
[JNet_369_pretrain_4_label_depth]: /experiments/images/JNet_369_pretrain_4_label_depth.png
[JNet_369_pretrain_4_label_plane]: /experiments/images/JNet_369_pretrain_4_label_plane.png
[JNet_369_pretrain_4_original_depth]: /experiments/images/JNet_369_pretrain_4_original_depth.png
[JNet_369_pretrain_4_original_plane]: /experiments/images/JNet_369_pretrain_4_original_plane.png
[JNet_369_pretrain_4_output_depth]: /experiments/images/JNet_369_pretrain_4_output_depth.png
[JNet_369_pretrain_4_output_plane]: /experiments/images/JNet_369_pretrain_4_output_plane.png
[JNet_369_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi000_original_depth.png
[JNet_369_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi000_output_depth.png
[JNet_369_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi000_reconst_depth.png
[JNet_369_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi001_original_depth.png
[JNet_369_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi001_output_depth.png
[JNet_369_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi001_reconst_depth.png
[JNet_369_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi002_original_depth.png
[JNet_369_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi002_output_depth.png
[JNet_369_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi002_reconst_depth.png
[JNet_369_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi003_original_depth.png
[JNet_369_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi003_output_depth.png
[JNet_369_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi003_reconst_depth.png
[JNet_369_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi004_original_depth.png
[JNet_369_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi004_output_depth.png
[JNet_369_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_369_pretrain_beads_001_roi004_reconst_depth.png
[JNet_369_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_369_pretrain_beads_002_roi000_original_depth.png
[JNet_369_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_369_pretrain_beads_002_roi000_output_depth.png
[JNet_369_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_369_pretrain_beads_002_roi000_reconst_depth.png
[JNet_369_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_369_pretrain_beads_002_roi001_original_depth.png
[JNet_369_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_369_pretrain_beads_002_roi001_output_depth.png
[JNet_369_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_369_pretrain_beads_002_roi001_reconst_depth.png
[JNet_369_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_369_pretrain_beads_002_roi002_original_depth.png
[JNet_369_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_369_pretrain_beads_002_roi002_output_depth.png
[JNet_369_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_369_pretrain_beads_002_roi002_reconst_depth.png
[JNet_370_0_label_depth]: /experiments/images/JNet_370_0_label_depth.png
[JNet_370_0_label_plane]: /experiments/images/JNet_370_0_label_plane.png
[JNet_370_0_original_depth]: /experiments/images/JNet_370_0_original_depth.png
[JNet_370_0_original_plane]: /experiments/images/JNet_370_0_original_plane.png
[JNet_370_0_output_depth]: /experiments/images/JNet_370_0_output_depth.png
[JNet_370_0_output_plane]: /experiments/images/JNet_370_0_output_plane.png
[JNet_370_1_label_depth]: /experiments/images/JNet_370_1_label_depth.png
[JNet_370_1_label_plane]: /experiments/images/JNet_370_1_label_plane.png
[JNet_370_1_original_depth]: /experiments/images/JNet_370_1_original_depth.png
[JNet_370_1_original_plane]: /experiments/images/JNet_370_1_original_plane.png
[JNet_370_1_output_depth]: /experiments/images/JNet_370_1_output_depth.png
[JNet_370_1_output_plane]: /experiments/images/JNet_370_1_output_plane.png
[JNet_370_2_label_depth]: /experiments/images/JNet_370_2_label_depth.png
[JNet_370_2_label_plane]: /experiments/images/JNet_370_2_label_plane.png
[JNet_370_2_original_depth]: /experiments/images/JNet_370_2_original_depth.png
[JNet_370_2_original_plane]: /experiments/images/JNet_370_2_original_plane.png
[JNet_370_2_output_depth]: /experiments/images/JNet_370_2_output_depth.png
[JNet_370_2_output_plane]: /experiments/images/JNet_370_2_output_plane.png
[JNet_370_3_label_depth]: /experiments/images/JNet_370_3_label_depth.png
[JNet_370_3_label_plane]: /experiments/images/JNet_370_3_label_plane.png
[JNet_370_3_original_depth]: /experiments/images/JNet_370_3_original_depth.png
[JNet_370_3_original_plane]: /experiments/images/JNet_370_3_original_plane.png
[JNet_370_3_output_depth]: /experiments/images/JNet_370_3_output_depth.png
[JNet_370_3_output_plane]: /experiments/images/JNet_370_3_output_plane.png
[JNet_370_4_label_depth]: /experiments/images/JNet_370_4_label_depth.png
[JNet_370_4_label_plane]: /experiments/images/JNet_370_4_label_plane.png
[JNet_370_4_original_depth]: /experiments/images/JNet_370_4_original_depth.png
[JNet_370_4_original_plane]: /experiments/images/JNet_370_4_original_plane.png
[JNet_370_4_output_depth]: /experiments/images/JNet_370_4_output_depth.png
[JNet_370_4_output_plane]: /experiments/images/JNet_370_4_output_plane.png
[JNet_370_beads_001_roi000_original_depth]: /experiments/images/JNet_370_beads_001_roi000_original_depth.png
[JNet_370_beads_001_roi000_output_depth]: /experiments/images/JNet_370_beads_001_roi000_output_depth.png
[JNet_370_beads_001_roi000_reconst_depth]: /experiments/images/JNet_370_beads_001_roi000_reconst_depth.png
[JNet_370_beads_001_roi001_original_depth]: /experiments/images/JNet_370_beads_001_roi001_original_depth.png
[JNet_370_beads_001_roi001_output_depth]: /experiments/images/JNet_370_beads_001_roi001_output_depth.png
[JNet_370_beads_001_roi001_reconst_depth]: /experiments/images/JNet_370_beads_001_roi001_reconst_depth.png
[JNet_370_beads_001_roi002_original_depth]: /experiments/images/JNet_370_beads_001_roi002_original_depth.png
[JNet_370_beads_001_roi002_output_depth]: /experiments/images/JNet_370_beads_001_roi002_output_depth.png
[JNet_370_beads_001_roi002_reconst_depth]: /experiments/images/JNet_370_beads_001_roi002_reconst_depth.png
[JNet_370_beads_001_roi003_original_depth]: /experiments/images/JNet_370_beads_001_roi003_original_depth.png
[JNet_370_beads_001_roi003_output_depth]: /experiments/images/JNet_370_beads_001_roi003_output_depth.png
[JNet_370_beads_001_roi003_reconst_depth]: /experiments/images/JNet_370_beads_001_roi003_reconst_depth.png
[JNet_370_beads_001_roi004_original_depth]: /experiments/images/JNet_370_beads_001_roi004_original_depth.png
[JNet_370_beads_001_roi004_output_depth]: /experiments/images/JNet_370_beads_001_roi004_output_depth.png
[JNet_370_beads_001_roi004_reconst_depth]: /experiments/images/JNet_370_beads_001_roi004_reconst_depth.png
[JNet_370_beads_002_roi000_original_depth]: /experiments/images/JNet_370_beads_002_roi000_original_depth.png
[JNet_370_beads_002_roi000_output_depth]: /experiments/images/JNet_370_beads_002_roi000_output_depth.png
[JNet_370_beads_002_roi000_reconst_depth]: /experiments/images/JNet_370_beads_002_roi000_reconst_depth.png
[JNet_370_beads_002_roi001_original_depth]: /experiments/images/JNet_370_beads_002_roi001_original_depth.png
[JNet_370_beads_002_roi001_output_depth]: /experiments/images/JNet_370_beads_002_roi001_output_depth.png
[JNet_370_beads_002_roi001_reconst_depth]: /experiments/images/JNet_370_beads_002_roi001_reconst_depth.png
[JNet_370_beads_002_roi002_original_depth]: /experiments/images/JNet_370_beads_002_roi002_original_depth.png
[JNet_370_beads_002_roi002_output_depth]: /experiments/images/JNet_370_beads_002_roi002_output_depth.png
[JNet_370_beads_002_roi002_reconst_depth]: /experiments/images/JNet_370_beads_002_roi002_reconst_depth.png
[JNet_370_psf_post]: /experiments/images/JNet_370_psf_post.png
[JNet_370_psf_pre]: /experiments/images/JNet_370_psf_pre.png
[finetuned]: /experiments/tmp/JNet_370_train.png
[pretrained_model]: /experiments/tmp/JNet_369_pretrain_train.png
