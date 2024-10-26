import torch
import numpy as np

def image_level_loss(preds, smooth = 1e-10):
    """Image level cross-entropy loss. Defined in https://arxiv.org/pdf/1506.02106 (page 5).
    
    The presence_loss encourages the target class to have a high probability on at least one pixel in the image.
    The absence_loss corresponds to the fact the background class should have a low probability on at least one pixel in the image.

    Args:
        pred: [B, 1, H, W] tensor, model output

    Returns:
        loss: scalar tensor, image level cross-entropy loss
    """
    S = torch.sigmoid(preds)
    S = S.view(S.shape[0], -1)
    
    S_tc_true = S.max(dim=1).values
    S_tc_false = (1 - S).max(dim=1).values

    presence_loss = -torch.log(S_tc_true + smooth)
    absence_loss = -torch.log(1 - S_tc_false + smooth)

    loss = presence_loss + absence_loss
    return loss # shape [batch_size]

def point_level_loss(preds, point_supervision, smooth = 1e-10):
    """Point level cross-entropy loss. Equal to the sum of the cross-entropy losses of the foreground and background points + the image level loss.
    Assumes that a_i (importance) is uniform for every point i.

    Args:
        preds: [B, 1, H, W] tensor, model output
        point_supervision: [B, 1, H, W] tensor, point level supervision (0: background, 1: foreground, -1: ignore)

    Returns:
        loss: scalar tensor, point level cross-entropy loss
    """
    S = torch.sigmoid(preds)
    loss = image_level_loss(preds, smooth)

    for i, mask in enumerate(point_supervision):
        foreground_points = np.argwhere(mask == 1).T
        background_points = np.argwhere(mask == 0).T
        for (x, y) in foreground_points:
            SiGi = S[i, :, x, y]
            loss[i] += -torch.log(SiGi + smooth)[0] # presence loss (foreground)
        
        for (x, y) in background_points:
            SiGi= S[i, :, x, y]
            loss[i] += -torch.log(1 - SiGi + smooth)[0] # absence loss (background)

    return loss # shape [batch_size]

if __name__ == '__main__':
    preds = torch.rand(16, 1, 512, 512) # simulate predictions from model: [batch_size, 1, H, W]
    point_supervision = torch.full((16, 512, 512), -1, dtype=torch.float32) # simulate point supervision: [batch_size, H, W]
    foreground_points = np.random.randint(0, 512, size=(16, 2, 10))
    background_points = np.random.randint(0, 512, size=(16, 2, 10))
    
    for i, (x, y) in enumerate(foreground_points):
        point_supervision[i, x, y] = 1
    
    for i, (x, y) in enumerate(background_points):
        point_supervision[i, x, y] = 0
    
    print("Prediction shape:", preds.shape)
    print("Point supervision shape:", point_supervision.shape)
    pll = point_level_loss(preds, point_supervision)
    print("Point level loss shape:", pll.shape)
    print("Point level loss:", pll)