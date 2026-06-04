import torch
import torch.nn as nn
from models import GNN

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class DomainClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, num_domains=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x, alpha=1.0):
        # Reverse the gradient flow during backpropagation
        reversed_x = GradientReversal.apply(x, alpha)
        return self.net(reversed_x)

class GNN_CrossAttention_DANN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_textual_features, num_structural_features,
                 activation_fn=torch.nn.ReLU(), dropout_p=0.2, gnn_type='gcn'):
        super().__init__()
        self.gnn = GNN(num_node_features=num_node_features * 2,
                       hidden_dim=hidden_dim * 2, num_classes=num_classes,
                       dropout_p=dropout_p, gnn_type=gnn_type)
        self.cross_attention_to_text = torch.nn.Linear(num_structural_features, hidden_dim)
        self.cross_attention_to_struct = torch.nn.Linear(num_textual_features, hidden_dim)
        self.struct_projector = torch.nn.Sequential(torch.nn.Linear(num_structural_features, hidden_dim),
                                                    torch.nn.ReLU())
        self.text_projector = torch.nn.Sequential(torch.nn.Linear(num_textual_features, hidden_dim),
                                                  torch.nn.ReLU())
        self.joint_projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.ReLU(),
        )

    def forward(self, text_node_features, struct_node_features, edge_index, return_features=False):
        struct_projection = self.struct_projector(struct_node_features) * self.cross_attention_to_struct(text_node_features)
        text_projection = self.text_projector(text_node_features) * self.cross_attention_to_text(struct_node_features)
        multimodal_node_features = self.joint_projector(torch.concat([struct_projection, text_projection], dim=-1))
        
        preds = self.gnn(multimodal_node_features, edge_index)
        if return_features:
            return preds, multimodal_node_features
        return preds
