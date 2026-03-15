import torch
import torch.nn as nn
import torchvision.models as models

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes, embed_dim, num_heads, depth):
        super(HybridCNNViT, self).__init__()
        # Initialize the parameters for the model
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        #CNN model
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)# Load ResNet550
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])# reomove last two layers (avgpool and fc) to get feature maps instead of classification output
        self.conv_proj = nn.Conv2d(in_channels=2048, out_channels=embed_dim, kernel_size=1)#reform the output of resnet to fit the input of transformer
        #ViT model
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)#initialize the transformer encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))# create a learnable class token for the transformer
        self.classifier = nn.Linear(embed_dim, num_classes) # final linear layer to classify into Fire/Non-Fire

#forward pass
    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2) # Reshape the feature maps to (Batch, Sequence Length, Embedding Dimension)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: (Batch, 50, 768)
        x = self.transformer(x)
        cls_output = x[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits

if __name__ == "__main__":
    model = HybridCNNViT(num_classes=3)
    dummy_image = torch.randn(8, 2, 224, 224) # Batch of 8, 2 color channels, 224x224
    output = model(dummy_image)
    print("Output shape:", output.shape) # Should be [8, 3]