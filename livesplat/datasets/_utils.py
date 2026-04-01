import torch
import kornia

def create_img_and_mask(img, seg_img, seg_labels, threshold=0.005, kernel=torch.ones((5, 5))):
                
    mask = torch.zeros((1, img.shape[1], img.shape[2]))
        
    for cur_label in seg_labels:
        cur_label_norm = cur_label[..., None, None] / 255
        binary_mask_head = (torch.abs(seg_img - cur_label_norm) < threshold).all(axis=0)
        binary_mask_head_dilated = kornia.morphology.dilation(
            binary_mask_head[None, None, ...], 
            kernel,border_type='constant', border_value=0.0)
        mask = torch.logical_or(binary_mask_head_dilated[0], mask)
        
    img_masked = img * mask.repeat(3, 1, 1)

    return img_masked, mask