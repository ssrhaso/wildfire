import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ViT_B_16_Weights

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.1):
        super(HybridCNNViT, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        embed_dim = 768  # fixed by ViT-B/16

        # Pretrained ResNet50 backbone (conv1 through layer3, outputs 1024-ch feature maps)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])

        # Project CNN feature channels to transformer embed_dim
        self.conv_proj = nn.Conv2d(in_channels=1024, out_channels=embed_dim, kernel_size=1)

        # Pretrained ViT-B/16 transformer (includes pos_embedding, 12 blocks, layer norm)
        vit = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        self.cls_token = vit.class_token
        self.transformer = vit.encoder

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)  # (B, 196, embed_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, 197, embed_dim)
        x = self.transformer(x)  # pos_embed + transformer blocks + layer norm
        cls_output = x[:, 0, :]
        logits = self.classifier(self.dropout(cls_output))
        return logits

if __name__ == "__main__":
    model = HybridCNNViT(num_classes=2, dropout_rate=0.1)
    dummy_image = torch.randn(8, 3, 224, 224)
    output = model(dummy_image)
    print("Output shape:", output.shape)  # Should be [8, 2]
