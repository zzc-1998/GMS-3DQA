import torch
import torch.nn as nn


from .swin_transformer import SwinTransformer as swint_tiny
from .head import IQAHead


class GMS_3DQA(nn.Module):
    def __init__(self,pretrained = True, checkpoint = 'path_to_checkpoint'):
        super().__init__()
        self.backbone = swint_tiny()
        self.iqa_head = IQAHead()
        if pretrained:          
            self.load(self.backbone,checkpoint)
        #print(self.backbone.state_dict())
            
    def load(self,model,checkpoint):
        state_dict = torch.load(checkpoint)   
        state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=False)


    def forward(self, image):
        image_size = image.shape
        image = image.view(-1, image_size[2], image_size[3], image_size[4])
        feat = self.backbone(image)
        feat = self.iqa_head(feat)
        avg_feat = torch.mean(feat,dim=1)
        avg_feat = avg_feat.view(image_size[0],image_size[1])
        score = torch.mean(avg_feat,dim=1)
        return score
