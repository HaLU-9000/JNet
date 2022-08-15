from curses import tparm
import torch

def jiffs(image1, image2, smooth=1e-8):
    """
    Jaccard index (or IoU) for 3d image fuzzy segmentation.
    input  : 4d tensors (x2)
    output : jaccard index

    refs:
    Taha, A.A., Hanbury, A.
    Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool.
    BMC Med Imaging 15, 29 (2015). https://doi.org/10.1186/s12880-015-0068-x

    # ToDo
    - 5d version that outputs mIoU over the samples in minibatch
    """
    assert len(image1.shape) == 4       ,\
        'image1 must be 4d tensor'
    assert len(image2.shape) == 4       ,\
        'image2 must be 4d tensor'
    assert image1.shape == image2.shape ,\
        '\nimage1 and image2 should have the same shape:\n'+\
        f'\timage1 shape : {image1.shape}\n'+\
        f'\timage2 shape : {image2.shape}'
    tp = torch.sum(torch.minimum(image1, image2))
    fp = torch.sum(torch.maximum((image1 - image2), torch.zeros_like(image1)))
    fn = torch.sum(torch.maximum((1.0 - image1) - (1.0 - image2), torch.zeros_like(image1)))
    #tn = torch.sum(torch.minimum(1.0 - image1, 1.0 - image2))
    ji = (tp + smooth) / (tp + fp + fn + smooth)
    return ji

def kagglejiffs(image1, image2, smooth=1e-8):
    '''
    ref:
    https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    '''
    image1 = image1.view(-1)
    image2 = image2.view(-1)
    intersection = (image1 * image2).sum()
    total = torch.sum(image1 + image2)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

if __name__ == '__main__':
    image1 = torch.sigmoid(torch.randn(1, 100, 100, 100))
    image2 = torch.sigmoid(torch.randn(1, 100, 100, 100))
    print(jiffs(image1, image2))
