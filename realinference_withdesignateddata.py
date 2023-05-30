import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from dataset import RealDensityDataset
import model

vis_mseloss = False

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Inference on device {device}.")

scale    = 6
surround = False
surround_size = [32, 4, 4]
image_path    = 'images_psj/'
image_name    = 'cropped_normalized_175-11w-D3-xyz6-020'
image__name   = 'cropped_raw_175-11w-D3-xyz6-020'
images        = torch.load('./' + image_path + image_name  + '.pt').to(device)
images_       = torch.load('./' + image_path + image__name + '.pt').to(device)
images =  [images[:, :, :112, :112].clone(),
           images[:, :, :112, 112:].clone(),
           images[:, :, 112:, :112].clone(),
           images[:, :, 112:, 112:].clone(),]
images_ = [images_[:, :, :112, :112].clone(),
           images_[:, :, :112, 112:].clone(),
           images_[:, :, 112:, :112].clone(),
           images_[:, :, 112:, 112:].clone(),]

model_name           = 'JNet_183_x6_start01'
hidden_channels_list = [16, 32, 64, 128, 256]
scale_factor         = (scale, 1, 1)
nblocks              = 2
s_nblocks            = 2
activation           = nn.ReLU(inplace=True)
dropout              = 0.5
superres             = True if scale > 1 else False
reconstruct          = True
JNet = model.JNet(hidden_channels_list  = hidden_channels_list ,
                  nblocks               = nblocks              ,
                  activation            = activation           ,
                  dropout               = dropout              ,
                  scale_factor          = scale_factor         ,
                  mu_z                  = 0.2                  ,
                  sig_z                 = 0.2                  , 
                  bet_xy                = 4.43864              ,
                  bet_z                 = 27.7052              ,
                  alpha                 = 74.9664              ,
                  superres              = superres             ,
                  reconstruct           = reconstruct          ,
                  )
JNet = JNet.to(device = device)
JNet.set_tau(0.1)
j   = 12
j_s = j // scale
i = 56

JNet.load_state_dict(torch.load(f'model/{model_name}.pt'), strict=False)
JNet.eval()

for k, image in enumerate(zip(images, images_)):
    image, image_ = image
    output, reconst= JNet(image.to("cuda").unsqueeze(0))
    output  = output.detach().cpu().numpy()
    reconst = reconst.squeeze(0).detach().cpu().numpy()
    torch.save(output, f'./result_psj/{image_name}_result{k}.pt')
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_axis_off()
    ax2.set_axis_off()

    ax1.set_title('original image')
    ax2.set_title(f'reconstruct image\n{model_name}')
    plt.subplots_adjust(hspace=-0.0)
    ax1.imshow(image_[0, :, i, :].to(device='cpu'),
            cmap='gray', vmin=0.0, vmax=1.0, aspect=scale)
    ax2.imshow(output[0, 0, :, i, :],
            cmap='gray', vmin=0.0, vmax=1.0, aspect=1)
    plt.savefig(f'result_psj/{model_name}_{image_name}_{k}.png', format='png', dpi=250) 
else:
    pass