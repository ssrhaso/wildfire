import torch
import torch.nn as nn
import torchvision.models as models

class HybridCNNViT(nn.Module):
    def __init__(self, num_classes, embed_dim, num_heads, depth, dropout_rate):
        super(HybridCNNViT, self).__init__()
        # Initialize the parameters for the model
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.depth = depth
        self.dropout_rate = dropout_rate
        #CNN model
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)# Load ResNet50
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])# reomove last three layers (avgpool, fc, layer4) to get feature maps instead of classification output
        self.conv_proj = nn.Conv2d(in_channels=1024, out_channels=embed_dim, kernel_size=1)#reform the output of resnet to fit the input of transformer
        #ViT model
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))# create a learnable class token for the transformer
        self.pos_embed = nn.Parameter(torch.randn(1, 197, embed_dim))# create positional encoding for 196 patches + 1 class token
        self.pos_drop = nn.Dropout(p=dropout_rate)# dropout for positional encoding
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True, dropout=dropout_rate)#initialize the transformer encoder
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(embed_dim, num_classes) # final linear layer to classify into Fire/Non-Fire

#forward pass
    def forward(self, x):
        batch_size = x.size(0)
        x = self.backbone(x)
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2) # Reshape the feature maps to (Batch, Sequence Length, Embedding Dimension)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Shape: (Batch, 197, 768)
        x = x + self.pos_embed # add positional encoding to the tokens so it understands spatial reasoning
        x = self.pos_drop(x) # apply dropout to the embeddings
        x = self.transformer(x)
        cls_output = x[:, 0, :]
        logits = self.classifier(self.dropout(cls_output))
        return logits

if __name__ == "__main__":
    model = HybridCNNViT(num_classes=2, embed_dim=768, num_heads=12, depth=12, dropout_rate=0.1)
    dummy_image = torch.randn(8, 3, 224, 224) # Batch of 8, 3 color channels, 224x224
    output = model(dummy_image)
    print("Output shape:", output.shape) # Should be [8, 2]