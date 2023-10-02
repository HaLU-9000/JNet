



# JNet_347_json_test
  
the parameters to replicate the results of JNet_347  
pretrained model : JNet_348_pretrain_json_test
## parameters
  
hidden_channels_list				[4, 8, 16, 32, 64]  
attn_list				[False, False, False, False, False]  
nblocks				2  
activation				nn.ReLU(inplace=True)  
dropout				0.5  
superres				True  
partial				None  
reconstruct				False  
apply_vq				False  
use_x_quantized				False  
use_fftconv				True  
mu_z				0.1  
sig_z				0.1  
blur_mode				gaussian  
$blur_mode_comments				gaussian or gibsonlanni  
size_x				51  
size_y				51  
size_z				161  
NA				0.8  
wavelength				0.91  
$wavelength comments				microns  
M				25  
&M  comments				magnification  
ns				1.4  
$ns comments				specimen refractive index (RI)  
ng0				1.5  
$ng0 comments				coverslip RI design value  
ng				1.5  
$ng comments				coverslip RI experimental value  
ni0				1.5  
$ni0 comments				immersion medium RI design value  
ni				1.5  
$ni comments				immersion medium RI experimental value  
ti0				150  
$ti0 comments				microns, working distance (immersion medium thickness) design value  
tg0				170  
$tg0 comments				microns, coverslip thickness design value  
tg				170  
$tg comments				microns, coverslip thickness experimental value  
res_lateral				0.05  
$res_lateral comments				microns  
res_axial				0.05  
$res_axial comments				microns  
pZ				0  
$pZ comments				microns, particle distance from coverslip  
bet_z				30.0  
bet_xy				3.0  
sig_eps				0.01  
scale				10  
device				cuda