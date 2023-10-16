



# JNet_356 Report
  
the parameters to replicate the results of JNet_356. large psf and noise  
pretrained model : JNet_355_pretrain
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

## Training Curves
  

### Pretraining
  
![pretrained_model]
### Finetuning
  
![finetuned]
## Results
  
mean MSE: 0.02839030884206295, mean BCE: 0.10404862463474274
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_355_pretrain_0_original_plane]|![JNet_355_pretrain_0_output_plane]|![JNet_355_pretrain_0_label_plane]|
  
MSE: 0.028929945081472397, BCE: 0.10837559401988983  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_355_pretrain_0_original_depth]|![JNet_355_pretrain_0_output_depth]|![JNet_355_pretrain_0_label_depth]|
  
MSE: 0.028929945081472397, BCE: 0.10837559401988983  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_355_pretrain_1_original_plane]|![JNet_355_pretrain_1_output_plane]|![JNet_355_pretrain_1_label_plane]|
  
MSE: 0.02605924755334854, BCE: 0.09378647804260254  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_355_pretrain_1_original_depth]|![JNet_355_pretrain_1_output_depth]|![JNet_355_pretrain_1_label_depth]|
  
MSE: 0.02605924755334854, BCE: 0.09378647804260254  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_355_pretrain_2_original_plane]|![JNet_355_pretrain_2_output_plane]|![JNet_355_pretrain_2_label_plane]|
  
MSE: 0.02897154912352562, BCE: 0.10571563988924026  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_355_pretrain_2_original_depth]|![JNet_355_pretrain_2_output_depth]|![JNet_355_pretrain_2_label_depth]|
  
MSE: 0.02897154912352562, BCE: 0.10571563988924026  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_355_pretrain_3_original_plane]|![JNet_355_pretrain_3_output_plane]|![JNet_355_pretrain_3_label_plane]|
  
MSE: 0.02613002061843872, BCE: 0.09319940954446793  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_355_pretrain_3_original_depth]|![JNet_355_pretrain_3_output_depth]|![JNet_355_pretrain_3_label_depth]|
  
MSE: 0.02613002061843872, BCE: 0.09319940954446793  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_355_pretrain_4_original_plane]|![JNet_355_pretrain_4_output_plane]|![JNet_355_pretrain_4_label_plane]|
  
MSE: 0.03186078369617462, BCE: 0.11916596442461014  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_355_pretrain_4_original_depth]|![JNet_355_pretrain_4_output_depth]|![JNet_355_pretrain_4_label_depth]|
  
MSE: 0.03186078369617462, BCE: 0.11916596442461014  
  
mean MSE: 0.03295479714870453, mean BCE: nan
### 0

|original|output|label|
| :---: | :---: | :---: |
|![JNet_356_0_original_plane]|![JNet_356_0_output_plane]|![JNet_356_0_label_plane]|
  
MSE: 0.023659054189920425, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_356_0_original_depth]|![JNet_356_0_output_depth]|![JNet_356_0_label_depth]|
  
MSE: 0.023659054189920425, BCE: nan  

### 1

|original|output|label|
| :---: | :---: | :---: |
|![JNet_356_1_original_plane]|![JNet_356_1_output_plane]|![JNet_356_1_label_plane]|
  
MSE: 0.037024009972810745, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_356_1_original_depth]|![JNet_356_1_output_depth]|![JNet_356_1_label_depth]|
  
MSE: 0.037024009972810745, BCE: nan  

### 2

|original|output|label|
| :---: | :---: | :---: |
|![JNet_356_2_original_plane]|![JNet_356_2_output_plane]|![JNet_356_2_label_plane]|
  
MSE: 0.036435745656490326, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_356_2_original_depth]|![JNet_356_2_output_depth]|![JNet_356_2_label_depth]|
  
MSE: 0.036435745656490326, BCE: nan  

### 3

|original|output|label|
| :---: | :---: | :---: |
|![JNet_356_3_original_plane]|![JNet_356_3_output_plane]|![JNet_356_3_label_plane]|
  
MSE: 0.03367944806814194, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_356_3_original_depth]|![JNet_356_3_output_depth]|![JNet_356_3_label_depth]|
  
MSE: 0.03367944806814194, BCE: nan  

### 4

|original|output|label|
| :---: | :---: | :---: |
|![JNet_356_4_original_plane]|![JNet_356_4_output_plane]|![JNet_356_4_label_plane]|
  
MSE: 0.033975739032030106, BCE: nan  

|original|output|label|
| :---: | :---: | :---: |
|![JNet_356_4_original_depth]|![JNet_356_4_output_depth]|![JNet_356_4_label_depth]|
  
MSE: 0.033975739032030106, BCE: nan  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_355_pretrain_beads_001_roi000_original_depth]|![JNet_355_pretrain_beads_001_roi000_output_depth]|![JNet_355_pretrain_beads_001_roi000_reconst_depth]|
  
volume: 13.798125000000002, MSE: 0.0024329752195626497, quantized loss: 0.0019008326344192028  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_355_pretrain_beads_001_roi001_original_depth]|![JNet_355_pretrain_beads_001_roi001_output_depth]|![JNet_355_pretrain_beads_001_roi001_reconst_depth]|
  
volume: 21.828750000000007, MSE: 0.004136601462960243, quantized loss: 0.0027099582366645336  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_355_pretrain_beads_001_roi002_original_depth]|![JNet_355_pretrain_beads_001_roi002_output_depth]|![JNet_355_pretrain_beads_001_roi002_reconst_depth]|
  
volume: 13.632750000000003, MSE: 0.002469595754519105, quantized loss: 0.0017313974676653743  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_355_pretrain_beads_001_roi003_original_depth]|![JNet_355_pretrain_beads_001_roi003_output_depth]|![JNet_355_pretrain_beads_001_roi003_reconst_depth]|
  
volume: 22.856875000000006, MSE: 0.004199383780360222, quantized loss: 0.002928594360128045  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_355_pretrain_beads_001_roi004_original_depth]|![JNet_355_pretrain_beads_001_roi004_output_depth]|![JNet_355_pretrain_beads_001_roi004_reconst_depth]|
  
volume: 14.784125000000003, MSE: 0.0030559541191905737, quantized loss: 0.0018603857606649399  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_355_pretrain_beads_002_roi000_original_depth]|![JNet_355_pretrain_beads_002_roi000_output_depth]|![JNet_355_pretrain_beads_002_roi000_reconst_depth]|
  
volume: 15.775750000000004, MSE: 0.0034366182517260313, quantized loss: 0.0019799235742539167  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_355_pretrain_beads_002_roi001_original_depth]|![JNet_355_pretrain_beads_002_roi001_output_depth]|![JNet_355_pretrain_beads_002_roi001_reconst_depth]|
  
volume: 14.723375000000004, MSE: 0.0026592968497425318, quantized loss: 0.0019344587344676256  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_355_pretrain_beads_002_roi002_original_depth]|![JNet_355_pretrain_beads_002_roi002_output_depth]|![JNet_355_pretrain_beads_002_roi002_reconst_depth]|
  
volume: 15.060125000000003, MSE: 0.00303634419105947, quantized loss: 0.0019371983362361789  

### beads_001_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_356_beads_001_roi000_original_depth]|![JNet_356_beads_001_roi000_output_depth]|![JNet_356_beads_001_roi000_reconst_depth]|
  
volume: 12.343250000000003, MSE: 0.0005973908700980246, quantized loss: 4.691481990448665e-06  

### beads_001_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_356_beads_001_roi001_original_depth]|![JNet_356_beads_001_roi001_output_depth]|![JNet_356_beads_001_roi001_reconst_depth]|
  
volume: 19.551500000000004, MSE: 0.0009121598559431732, quantized loss: 6.176833267090842e-06  

### beads_001_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_356_beads_001_roi002_original_depth]|![JNet_356_beads_001_roi002_output_depth]|![JNet_356_beads_001_roi002_reconst_depth]|
  
volume: 12.255625000000004, MSE: 0.0005562982405535877, quantized loss: 4.886659553449135e-06  

### beads_001_roi003

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_356_beads_001_roi003_original_depth]|![JNet_356_beads_001_roi003_output_depth]|![JNet_356_beads_001_roi003_reconst_depth]|
  
volume: 20.133625000000006, MSE: 0.0008550831116735935, quantized loss: 6.249853868212085e-06  

### beads_001_roi004

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_356_beads_001_roi004_original_depth]|![JNet_356_beads_001_roi004_output_depth]|![JNet_356_beads_001_roi004_reconst_depth]|
  
volume: 13.438375000000002, MSE: 0.0006186806713230908, quantized loss: 4.539599103736691e-06  

### beads_002_roi000

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_356_beads_002_roi000_original_depth]|![JNet_356_beads_002_roi000_output_depth]|![JNet_356_beads_002_roi000_reconst_depth]|
  
volume: 14.426750000000004, MSE: 0.000701680313795805, quantized loss: 4.770101895701373e-06  

### beads_002_roi001

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_356_beads_002_roi001_original_depth]|![JNet_356_beads_002_roi001_output_depth]|![JNet_356_beads_002_roi001_reconst_depth]|
  
volume: 13.131375000000004, MSE: 0.0006339598330669105, quantized loss: 4.576045284920838e-06  

### beads_002_roi002

|original|output|reconst|
| :---: | :---: | :---: |
|![JNet_356_beads_002_roi002_original_depth]|![JNet_356_beads_002_roi002_output_depth]|![JNet_356_beads_002_roi002_reconst_depth]|
  
volume: 13.683250000000003, MSE: 0.0006434224196709692, quantized loss: 5.217364559939597e-06  

|pre|post|
| :---: | :---: |
|![JNet_356_psf_pre]|![JNet_356_psf_post]|
  



[JNet_355_pretrain_0_label_depth]: /experiments/images/JNet_355_pretrain_0_label_depth.png
[JNet_355_pretrain_0_label_plane]: /experiments/images/JNet_355_pretrain_0_label_plane.png
[JNet_355_pretrain_0_original_depth]: /experiments/images/JNet_355_pretrain_0_original_depth.png
[JNet_355_pretrain_0_original_plane]: /experiments/images/JNet_355_pretrain_0_original_plane.png
[JNet_355_pretrain_0_output_depth]: /experiments/images/JNet_355_pretrain_0_output_depth.png
[JNet_355_pretrain_0_output_plane]: /experiments/images/JNet_355_pretrain_0_output_plane.png
[JNet_355_pretrain_1_label_depth]: /experiments/images/JNet_355_pretrain_1_label_depth.png
[JNet_355_pretrain_1_label_plane]: /experiments/images/JNet_355_pretrain_1_label_plane.png
[JNet_355_pretrain_1_original_depth]: /experiments/images/JNet_355_pretrain_1_original_depth.png
[JNet_355_pretrain_1_original_plane]: /experiments/images/JNet_355_pretrain_1_original_plane.png
[JNet_355_pretrain_1_output_depth]: /experiments/images/JNet_355_pretrain_1_output_depth.png
[JNet_355_pretrain_1_output_plane]: /experiments/images/JNet_355_pretrain_1_output_plane.png
[JNet_355_pretrain_2_label_depth]: /experiments/images/JNet_355_pretrain_2_label_depth.png
[JNet_355_pretrain_2_label_plane]: /experiments/images/JNet_355_pretrain_2_label_plane.png
[JNet_355_pretrain_2_original_depth]: /experiments/images/JNet_355_pretrain_2_original_depth.png
[JNet_355_pretrain_2_original_plane]: /experiments/images/JNet_355_pretrain_2_original_plane.png
[JNet_355_pretrain_2_output_depth]: /experiments/images/JNet_355_pretrain_2_output_depth.png
[JNet_355_pretrain_2_output_plane]: /experiments/images/JNet_355_pretrain_2_output_plane.png
[JNet_355_pretrain_3_label_depth]: /experiments/images/JNet_355_pretrain_3_label_depth.png
[JNet_355_pretrain_3_label_plane]: /experiments/images/JNet_355_pretrain_3_label_plane.png
[JNet_355_pretrain_3_original_depth]: /experiments/images/JNet_355_pretrain_3_original_depth.png
[JNet_355_pretrain_3_original_plane]: /experiments/images/JNet_355_pretrain_3_original_plane.png
[JNet_355_pretrain_3_output_depth]: /experiments/images/JNet_355_pretrain_3_output_depth.png
[JNet_355_pretrain_3_output_plane]: /experiments/images/JNet_355_pretrain_3_output_plane.png
[JNet_355_pretrain_4_label_depth]: /experiments/images/JNet_355_pretrain_4_label_depth.png
[JNet_355_pretrain_4_label_plane]: /experiments/images/JNet_355_pretrain_4_label_plane.png
[JNet_355_pretrain_4_original_depth]: /experiments/images/JNet_355_pretrain_4_original_depth.png
[JNet_355_pretrain_4_original_plane]: /experiments/images/JNet_355_pretrain_4_original_plane.png
[JNet_355_pretrain_4_output_depth]: /experiments/images/JNet_355_pretrain_4_output_depth.png
[JNet_355_pretrain_4_output_plane]: /experiments/images/JNet_355_pretrain_4_output_plane.png
[JNet_355_pretrain_beads_001_roi000_original_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi000_original_depth.png
[JNet_355_pretrain_beads_001_roi000_output_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi000_output_depth.png
[JNet_355_pretrain_beads_001_roi000_reconst_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi000_reconst_depth.png
[JNet_355_pretrain_beads_001_roi001_original_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi001_original_depth.png
[JNet_355_pretrain_beads_001_roi001_output_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi001_output_depth.png
[JNet_355_pretrain_beads_001_roi001_reconst_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi001_reconst_depth.png
[JNet_355_pretrain_beads_001_roi002_original_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi002_original_depth.png
[JNet_355_pretrain_beads_001_roi002_output_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi002_output_depth.png
[JNet_355_pretrain_beads_001_roi002_reconst_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi002_reconst_depth.png
[JNet_355_pretrain_beads_001_roi003_original_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi003_original_depth.png
[JNet_355_pretrain_beads_001_roi003_output_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi003_output_depth.png
[JNet_355_pretrain_beads_001_roi003_reconst_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi003_reconst_depth.png
[JNet_355_pretrain_beads_001_roi004_original_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi004_original_depth.png
[JNet_355_pretrain_beads_001_roi004_output_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi004_output_depth.png
[JNet_355_pretrain_beads_001_roi004_reconst_depth]: /experiments/images/JNet_355_pretrain_beads_001_roi004_reconst_depth.png
[JNet_355_pretrain_beads_002_roi000_original_depth]: /experiments/images/JNet_355_pretrain_beads_002_roi000_original_depth.png
[JNet_355_pretrain_beads_002_roi000_output_depth]: /experiments/images/JNet_355_pretrain_beads_002_roi000_output_depth.png
[JNet_355_pretrain_beads_002_roi000_reconst_depth]: /experiments/images/JNet_355_pretrain_beads_002_roi000_reconst_depth.png
[JNet_355_pretrain_beads_002_roi001_original_depth]: /experiments/images/JNet_355_pretrain_beads_002_roi001_original_depth.png
[JNet_355_pretrain_beads_002_roi001_output_depth]: /experiments/images/JNet_355_pretrain_beads_002_roi001_output_depth.png
[JNet_355_pretrain_beads_002_roi001_reconst_depth]: /experiments/images/JNet_355_pretrain_beads_002_roi001_reconst_depth.png
[JNet_355_pretrain_beads_002_roi002_original_depth]: /experiments/images/JNet_355_pretrain_beads_002_roi002_original_depth.png
[JNet_355_pretrain_beads_002_roi002_output_depth]: /experiments/images/JNet_355_pretrain_beads_002_roi002_output_depth.png
[JNet_355_pretrain_beads_002_roi002_reconst_depth]: /experiments/images/JNet_355_pretrain_beads_002_roi002_reconst_depth.png
[JNet_356_0_label_depth]: /experiments/images/JNet_356_0_label_depth.png
[JNet_356_0_label_plane]: /experiments/images/JNet_356_0_label_plane.png
[JNet_356_0_original_depth]: /experiments/images/JNet_356_0_original_depth.png
[JNet_356_0_original_plane]: /experiments/images/JNet_356_0_original_plane.png
[JNet_356_0_output_depth]: /experiments/images/JNet_356_0_output_depth.png
[JNet_356_0_output_plane]: /experiments/images/JNet_356_0_output_plane.png
[JNet_356_1_label_depth]: /experiments/images/JNet_356_1_label_depth.png
[JNet_356_1_label_plane]: /experiments/images/JNet_356_1_label_plane.png
[JNet_356_1_original_depth]: /experiments/images/JNet_356_1_original_depth.png
[JNet_356_1_original_plane]: /experiments/images/JNet_356_1_original_plane.png
[JNet_356_1_output_depth]: /experiments/images/JNet_356_1_output_depth.png
[JNet_356_1_output_plane]: /experiments/images/JNet_356_1_output_plane.png
[JNet_356_2_label_depth]: /experiments/images/JNet_356_2_label_depth.png
[JNet_356_2_label_plane]: /experiments/images/JNet_356_2_label_plane.png
[JNet_356_2_original_depth]: /experiments/images/JNet_356_2_original_depth.png
[JNet_356_2_original_plane]: /experiments/images/JNet_356_2_original_plane.png
[JNet_356_2_output_depth]: /experiments/images/JNet_356_2_output_depth.png
[JNet_356_2_output_plane]: /experiments/images/JNet_356_2_output_plane.png
[JNet_356_3_label_depth]: /experiments/images/JNet_356_3_label_depth.png
[JNet_356_3_label_plane]: /experiments/images/JNet_356_3_label_plane.png
[JNet_356_3_original_depth]: /experiments/images/JNet_356_3_original_depth.png
[JNet_356_3_original_plane]: /experiments/images/JNet_356_3_original_plane.png
[JNet_356_3_output_depth]: /experiments/images/JNet_356_3_output_depth.png
[JNet_356_3_output_plane]: /experiments/images/JNet_356_3_output_plane.png
[JNet_356_4_label_depth]: /experiments/images/JNet_356_4_label_depth.png
[JNet_356_4_label_plane]: /experiments/images/JNet_356_4_label_plane.png
[JNet_356_4_original_depth]: /experiments/images/JNet_356_4_original_depth.png
[JNet_356_4_original_plane]: /experiments/images/JNet_356_4_original_plane.png
[JNet_356_4_output_depth]: /experiments/images/JNet_356_4_output_depth.png
[JNet_356_4_output_plane]: /experiments/images/JNet_356_4_output_plane.png
[JNet_356_beads_001_roi000_original_depth]: /experiments/images/JNet_356_beads_001_roi000_original_depth.png
[JNet_356_beads_001_roi000_output_depth]: /experiments/images/JNet_356_beads_001_roi000_output_depth.png
[JNet_356_beads_001_roi000_reconst_depth]: /experiments/images/JNet_356_beads_001_roi000_reconst_depth.png
[JNet_356_beads_001_roi001_original_depth]: /experiments/images/JNet_356_beads_001_roi001_original_depth.png
[JNet_356_beads_001_roi001_output_depth]: /experiments/images/JNet_356_beads_001_roi001_output_depth.png
[JNet_356_beads_001_roi001_reconst_depth]: /experiments/images/JNet_356_beads_001_roi001_reconst_depth.png
[JNet_356_beads_001_roi002_original_depth]: /experiments/images/JNet_356_beads_001_roi002_original_depth.png
[JNet_356_beads_001_roi002_output_depth]: /experiments/images/JNet_356_beads_001_roi002_output_depth.png
[JNet_356_beads_001_roi002_reconst_depth]: /experiments/images/JNet_356_beads_001_roi002_reconst_depth.png
[JNet_356_beads_001_roi003_original_depth]: /experiments/images/JNet_356_beads_001_roi003_original_depth.png
[JNet_356_beads_001_roi003_output_depth]: /experiments/images/JNet_356_beads_001_roi003_output_depth.png
[JNet_356_beads_001_roi003_reconst_depth]: /experiments/images/JNet_356_beads_001_roi003_reconst_depth.png
[JNet_356_beads_001_roi004_original_depth]: /experiments/images/JNet_356_beads_001_roi004_original_depth.png
[JNet_356_beads_001_roi004_output_depth]: /experiments/images/JNet_356_beads_001_roi004_output_depth.png
[JNet_356_beads_001_roi004_reconst_depth]: /experiments/images/JNet_356_beads_001_roi004_reconst_depth.png
[JNet_356_beads_002_roi000_original_depth]: /experiments/images/JNet_356_beads_002_roi000_original_depth.png
[JNet_356_beads_002_roi000_output_depth]: /experiments/images/JNet_356_beads_002_roi000_output_depth.png
[JNet_356_beads_002_roi000_reconst_depth]: /experiments/images/JNet_356_beads_002_roi000_reconst_depth.png
[JNet_356_beads_002_roi001_original_depth]: /experiments/images/JNet_356_beads_002_roi001_original_depth.png
[JNet_356_beads_002_roi001_output_depth]: /experiments/images/JNet_356_beads_002_roi001_output_depth.png
[JNet_356_beads_002_roi001_reconst_depth]: /experiments/images/JNet_356_beads_002_roi001_reconst_depth.png
[JNet_356_beads_002_roi002_original_depth]: /experiments/images/JNet_356_beads_002_roi002_original_depth.png
[JNet_356_beads_002_roi002_output_depth]: /experiments/images/JNet_356_beads_002_roi002_output_depth.png
[JNet_356_beads_002_roi002_reconst_depth]: /experiments/images/JNet_356_beads_002_roi002_reconst_depth.png
[JNet_356_psf_post]: /experiments/images/JNet_356_psf_post.png
[JNet_356_psf_pre]: /experiments/images/JNet_356_psf_pre.png
[finetuned]: /experiments/tmp/JNet_356_train.png
[pretrained_model]: /experiments/tmp/JNet_355_pretrain_train.png
