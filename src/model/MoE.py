from .base import Base
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_channels, interm_channels, out_channels, hidden_layers, layerNorm):
        super().__init__()
        layers = []

        # creates each of the hidden layers 
        for i in range(hidden_layers):
            if(i == 0):
                layers.append(nn.Linear(in_channels, interm_channels))
            else:
                layers.append(nn.Linear(interm_channels, interm_channels))
            layers.append(nn.GELU())

        # adds in last linear layer converting from interm to out channels
        layers.append(nn.Linear(interm_channels,out_channels))

        # adding layer normalization
        if(layerNorm):
            layers.append(nn.LayerNorm(out_channels, elementwise_affine=False))

        # combining it all into one big model
        self.full_MLP = nn.Sequential(*layers)


    def forward(self, x):
        return self.full_MLP(x)

class ResNet(nn.Module):
    def __init__(self, in_channels, hidden_layers, layerNorm):
        super().__init__()

        # one Resnet layer = one MLP
        self.one_Restnet = MLP(in_channels, in_channels, in_channels, hidden_layers, layerNorm)

    def forward(self, x):
        # keeping copy of old values in x
        identity = x
        # moving forward and changing x value
        x = self.one_Restnet(x)
        # the resnet part where old values added back in
        if(identity.size() != x.size()):
            print("Skip Connection Failed")
        x = x + identity
        # x = self.relu(x)
        return x

# Expert class
class Expert(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.latent_dim =   cfg.latent_dim
        self.hidden_layer = cfg.hidden_layer
        self.input_dim = cfg.input_dim
        self.target_dim = cfg.target_dim
        self.num_resnet = cfg.num_resnet

        # creating encoder to go from input_dim to latent_dim
        self.encoder = MLP(self.input_dim, self.latent_dim, self.latent_dim, self.hidden_layer, layerNorm=True)
        # Instantiate ResNet and MLP
        self.resnet_blocks = nn.ModuleList([
            ResNet(self.latent_dim, self.hidden_layer, layerNorm=True)
            for i in range(self.num_resnet)
        ])
        # creating decoder to go from latent_dim to output_dim
        self.decoder = MLP(self.latent_dim, self.latent_dim, self.target_dim, self.hidden_layer, layerNorm=False)

    def forward(self, input, **kwargs):
        x = input
        x = self.encoder(x)
        # Pass through ResNet blocks
        for resnet in self.resnet_blocks:
            x = resnet(x)
        # Final linear layer
        return self.decoder(x)  
    
# MoE class
class MoE(Base):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.num_experts = cfg.num_experts
        self.input_dim = cfg.input_dim
        self.target_dim = cfg.target_dim
        
        # Create expert networks
        self.experts = nn.ModuleList([
            Expert(cfg) for _ in range(self.num_experts)
        ])
        
        # Gating network
        self.gate = MLP(
            in_channels=self.input_dim,
            interm_channels=cfg.gate_hidden_dim,
            out_channels=self.num_experts,
            hidden_layers=cfg.gate_hidden_layers,
            layerNorm=True
        )

    def _forward(self, input, **kwargs):

        batch_size = input.shape[0]
        
        # Get router logits and probabilities
        router_logits = self.gate(input)
        router_probs = torch.softmax(router_logits, dim=-1)
        
        # getting the expert to use
        expert_idx = torch.argmax(router_probs, dim=-1)

        # Prepare output tensor
        outputs = torch.zeros(batch_size, self.target_dim, device=input.device)
        
        # Process each expert's batch
        for expert_id in range(self.num_experts):
            # Find indices of samples routed to this expert
            indices = torch.where(expert_idx == expert_id)[0]
            
            if indices.size(0) > 0:
                # Get inputs for this expert and compute outputs
                expert_inputs = input[indices]
                expert_outputs = self.experts[expert_id].forward(expert_inputs)
                
                # Place outputs back in the correct positions
                outputs[indices] = expert_outputs
        
        # Return output and gating logits
        return outputs, router_logits