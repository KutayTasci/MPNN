import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from typing import Dict

from torch.autograd import Function
from torch.utils.cpp_extension import load_inline


# Merge bindings + dispatcher in single .cu file
with open("custom_kernels/reverse_scatter_bindings.cu", "r") as f:
    cuda_src = f.read()

load_inline(
    name="reverse_scatter",
    cpp_sources="#include <torch/extension.h>\nTORCH_LIBRARY_FRAGMENT(reverse_scatter, m) {}",  # force linkage
    cuda_sources=cuda_src,
    extra_cuda_cflags=[],
    extra_cflags=[],
    functions=[],
    verbose=True
)

@torch._dynamo.disable
def safe_reverse_scatter(input, mapping, output):
    return torch.ops.reverse_scatter.forward(input, mapping, output)


class GatedMLP(nn.Module):
    """
    Gated MLP with fixed batch normalization (always applied) before activation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        activation: str = "silu",
        bias: bool = True,
    ):
        super().__init__()
        # Core and gate MLPs
        #self.mlp_core = nn.Linear(input_dim, output_dim, bias=bias) # This will be moved to outside the class
        #self.mlp_gate = nn.Linear(input_dim, output_dim, bias=bias)
        self.activation = nn.ReLU() if activation == "relu" else nn.SiLU() if activation == "silu" else nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # Fixed batch normalization layers
        self.bn_core = nn.BatchNorm1d(output_dim)
        self.bn_gate = nn.BatchNorm1d(output_dim)

    def forward(self, core, gate: torch.Tensor) -> torch.Tensor:
        # core = self.mlp_core(x)
        #gate = self.mlp_gate(x)
        # Apply batch normalization (fixed)
        core = self.bn_core(core)
        gate = self.bn_gate(gate)
        core = self.activation(core)
        gate = self.sigmoid(gate)
        return core * gate

# --- Placeholder convolution modules ---
class AtomConvSum(nn.Module):
    def __init__(
            self, 
            in_dim, 
            bond_dim, 
            hidden_dim,
            dropout: float = 0.0,
            activation: str = "silu",
            use_mlp_out: bool = True,
            mlp_out_bias: bool = False,
            resnet: bool = True):
        super().__init__()
        self.resnet = resnet
        self.use_mlp_out = use_mlp_out
        self.activation = nn.ReLU() if activation == "relu" else nn.SiLU() if activation == "silu" else nn.GELU()

        # Input dimension
        self.in_dim = in_dim
        self.bond_dim = bond_dim

        self.core_src_mlp = nn.Linear(in_dim, in_dim, bias=False)  # Core MLP for atom features
        self.core_dst_mlp = nn.Linear(in_dim, in_dim, bias=False)
        self.core_bond_mlp = nn.Linear(bond_dim, in_dim, bias=False)

        self.src_mlp_gate = nn.Linear(in_dim, in_dim, bias=False)
        self.dst_mlp_gate = nn.Linear(in_dim, in_dim, bias=False)
        self.bond_mlp_gate = nn.Linear(bond_dim, in_dim, bias=False)
        # Gated MLP accepts concatenated center, bond & neighbor
        self.twoBody = GatedMLP(
            input_dim = in_dim,
            output_dim = in_dim,
            dropout = dropout,
            activation = activation,
        )

        if self.use_mlp_out:
            self.mlp_out = nn.Linear(in_dim, in_dim, bias=mlp_out_bias)


    def forward(self, vertex_feat, edge_feat, edge_index):
    
        src, dst = edge_index[0,:], edge_index[1,:]

        """"
        center = vertex_feat[src]            # [E, in_dim]
        neighbor = vertex_feat[dst]          # [E, in_dim]
        bonds = edge_feat  # [E, bond_dim]

        # Build messages using concatenation
        msg = torch.cat([center, bonds, neighbor], dim=-1)  # [E, 2*in_dim + bond_dim]
        core = self.core_mlp(msg)  # [E, in_dim]
        gate = self.mlp_gate(msg)  # [E, in_dim]
        """
        center = self.core_src_mlp(vertex_feat)            # [E, in_dim]
        neighbor = self.core_dst_mlp(vertex_feat)          # [E, in_dim]
        bonds = self.core_bond_mlp(edge_feat)  # [E, in_dim]

        tmp = safe_reverse_scatter(center, src, bonds)
        core = safe_reverse_scatter(neighbor, dst, tmp)  # [E, in_dim]

        center = self.src_mlp_gate(vertex_feat)            # [E, in_dim]
        neighbor = self.dst_mlp_gate(vertex_feat)          # [E, in_dim]
        bonds = self.bond_mlp_gate(edge_feat)  # [E, in_dim]

        tmp = safe_reverse_scatter(center, src, bonds)
        gate = safe_reverse_scatter(neighbor, dst, tmp)  # [E, in_dim]

        msg = self.twoBody(core, gate)


        # Aggregate messages per center atom
        new = scatter(msg, src, dim=0, dim_size=vertex_feat.size(0), reduce='add')

        if self.use_mlp_out:
            new = self.mlp_out(new)

        if self.resnet:
            new = new + vertex_feat


        return new

class AtomConvCat(nn.Module):
    def __init__(
            self, 
            in_dim, 
            bond_dim, 
            hidden_dim,
            dropout: float = 0.0,
            activation: str = "silu",
            use_mlp_out: bool = True,
            mlp_out_bias: bool = False,
            resnet: bool = True):
        super().__init__()
        self.resnet = resnet
        self.use_mlp_out = use_mlp_out
        self.activation = nn.ReLU() if activation == "relu" else nn.SiLU() if activation == "silu" else nn.GELU()

        # Input dimension
        self.in_dim = in_dim
        self.bond_dim = bond_dim
        self.core_mlp = nn.Linear(2 * in_dim + bond_dim, in_dim, bias=False)  # Core MLP for atom features
        self.mlp_gate = nn.Linear(2 * in_dim + bond_dim, in_dim, bias=False)
        # Gated MLP accepts concatenated center, bond & neighbor
        self.twoBody = GatedMLP(
            input_dim = in_dim,
            output_dim = in_dim,
            dropout = dropout,
            activation = activation,
        )

        if self.use_mlp_out:
            self.mlp_out = nn.Linear(in_dim, in_dim, bias=mlp_out_bias)


    def forward(self, vertex_feat, edge_feat, edge_index):
    
        src, dst = edge_index[0,:], edge_index[1,:]

        center = vertex_feat[src]            # [E, in_dim]
        neighbor = vertex_feat[dst]          # [E, in_dim]
        bonds = edge_feat  # [E, bond_dim]

        # Build messages using concatenation
        msg = torch.cat([center, bonds, neighbor], dim=-1)  # [E, 2*in_dim + bond_dim]
        core = self.core_mlp(msg)  # [E, in_dim]
        gate = self.mlp_gate(msg)  # [E, in_dim]
        msg = self.twoBody(core, gate)


        # Aggregate messages per center atom
        new = scatter(msg, src, dim=0, dim_size=vertex_feat.size(0), reduce='add')

        if self.use_mlp_out:
            new = self.mlp_out(new)

        if self.resnet:
            new = new + vertex_feat


        return new

class BondConvSum(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        angle_dim: int,
        dropout: float = 0.0,
        activation: str = "silu",
        use_mlp_out: bool = True,
        mlp_out_bias: bool = False,
        resnet: bool = True,
    ):
        super().__init__()
        self.resnet = resnet
        self.use_mlp_out = use_mlp_out
        self.activation = nn.ReLU() if activation == "relu" else nn.SiLU() if activation == "silu" else nn.GELU()

        # Gated MLP input dim = center atom + two bond ends + angle features
        #in_dim = atom_dim + 2 * bond_dim + angle_dim
        #self.core_mlp = nn.Linear(in_dim, bond_dim, bias=False)  # Core MLP for bond features
        self.core_src_mlp = nn.Linear(atom_dim, bond_dim, bias=False)  # Core MLP for bond features
        self.core_dst_mlp = nn.Linear(atom_dim, bond_dim, bias=False)
        self.core_bond_mlp = nn.Linear(bond_dim, bond_dim, bias=False)
        self.core_angle_mlp = nn.Linear(angle_dim, bond_dim, bias=False)


        #self.mlp_gate = nn.Linear(in_dim, bond_dim, bias=False)  # Gate MLP for bond features
        self.src_mlp_gate = nn.Linear(atom_dim, bond_dim, bias=False)
        self.dst_mlp_gate = nn.Linear(atom_dim, bond_dim, bias=False)
        self.bond_mlp_gate = nn.Linear(bond_dim, bond_dim, bias=False)
        self.angle_mlp_gate = nn.Linear(angle_dim, bond_dim, bias=False)
        self.twoBody = GatedMLP(
            input_dim=bond_dim,
            output_dim=bond_dim,
            dropout=dropout,
            activation=activation,
        )
        if use_mlp_out:
            self.mlp_out = nn.Linear(bond_dim, bond_dim, bias=mlp_out_bias)
        

    def forward(
        self,
        vertex_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        angle_feat: torch.Tensor,
        edge_index: torch.Tensor,
        triplet_info: torch.Tensor,
    ) -> torch.Tensor:
        # bond_graph: [num_angles, 3] => columns [center_atom_idx, bond_i_idx, bond_j_idx]
        b_i, a, b_j = triplet_info['k_idx'], triplet_info['j_idx'], triplet_info['i_idx']
        """
        center = vertex_feat[a]         # [T, atom_dim]
        bond_i = edge_feat[b_i]         # [T, bond_dim]
        bond_j = edge_feat[b_j]         # [T, bond_dim]
        angles = angle_feat            # [T, angle_dim]

        # Concatenate features
        total = torch.cat([center, bond_i, bond_j, angles], dim=-1)  # [T, *]
        core = self.core_mlp(total)  # [T, bond_dim]
        gate = self.mlp_gate(total)  # [T, bond_dim]
        """

        center = self.core_src_mlp(vertex_feat)         # [V, bond_dim]
        bond_i = self.core_bond_mlp(edge_feat)         # [E, bond_dim]
        bond_j = self.core_dst_mlp(vertex_feat)         # [E, bond_dim]
        angles = self.core_angle_mlp(angle_feat)            # [T, bond_dim]
        # Reverse scatter to core in the size of angles
        tmp = safe_reverse_scatter(center, a, angles)  # [T, bond_dim]
        tmp = safe_reverse_scatter(bond_i, b_i, tmp)  # [T, bond_dim]
        core = safe_reverse_scatter(bond_j, b_j, tmp)  # [T, bond_dim]

        center = self.src_mlp_gate(vertex_feat)         # [V, bond_dim]
        bond_i = self.bond_mlp_gate(edge_feat)         # [E, bond_dim]
        bond_j = self.dst_mlp_gate(vertex_feat)         # [E, bond_dim]
        angles = self.angle_mlp_gate(angle_feat)            # [T, bond_dim]
        # Reverse scatter to core in the size of angles
        tmp = safe_reverse_scatter(center, a, angles)  # [T, bond_dim]
        tmp = safe_reverse_scatter(bond_i, b_i, tmp)  # [T, bond_dim]
        gate = safe_reverse_scatter(bond_j, b_j, tmp)  # [T, bond_dim]

        update = self.twoBody(core, gate)  # [T, bond_dim]

        # Aggregate messages to each bond_i index
        new_bond = scatter(update, b_i, dim=0, dim_size=edge_feat.size(0), reduce="add")

        if self.use_mlp_out:
            new_bond = self.mlp_out(new_bond)
        if self.resnet:
            new_bond = new_bond + edge_feat

        return new_bond

class BondConvCat(nn.Module):
    """BondConv layer with concatenation-based update."""

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        angle_dim: int,
        dropout: float = 0.0,
        activation: str = "silu",
        use_mlp_out: bool = True,
        mlp_out_bias: bool = False,
        resnet: bool = True,
    ):
        super().__init__()
        self.resnet = resnet
        self.use_mlp_out = use_mlp_out
        self.activation = nn.ReLU() if activation == "relu" else nn.SiLU() if activation == "silu" else nn.GELU()

        # Gated MLP input dim = center atom + two bond ends + angle features
        in_dim = atom_dim + 2 * bond_dim + angle_dim
        self.core_mlp = nn.Linear(in_dim, bond_dim, bias=False)  # Core MLP for bond features
        self.mlp_gate = nn.Linear(in_dim, bond_dim, bias=False)  # Gate MLP for bond features
        self.twoBody = GatedMLP(
            input_dim=bond_dim,
            output_dim=bond_dim,
            dropout=dropout,
            activation=activation,
        )
        if use_mlp_out:
            self.mlp_out = nn.Linear(bond_dim, bond_dim, bias=mlp_out_bias)
        

    def forward(
        self,
        vertex_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        angle_feat: torch.Tensor,
        edge_index: torch.Tensor,
        triplet_info: torch.Tensor,
    ) -> torch.Tensor:
        # bond_graph: [num_angles, 3] => columns [center_atom_idx, bond_i_idx, bond_j_idx]
        b_i, a, b_j = triplet_info['k_idx'], triplet_info['j_idx'], triplet_info['i_idx']
        
        center = vertex_feat[a]         # [T, atom_dim]
        bond_i = edge_feat[b_i]         # [T, bond_dim]
        bond_j = edge_feat[b_j]         # [T, bond_dim]
        angles = angle_feat            # [T, angle_dim]

        # Concatenate features
        total = torch.cat([center, bond_i, bond_j, angles], dim=-1)  # [T, *]
        core = self.core_mlp(total)  # [T, bond_dim]
        gate = self.mlp_gate(total)  # [T, bond_dim]
        update = self.twoBody(core, gate)  # [T, bond_dim]

        # Aggregate messages to each bond_i index
        new_bond = scatter(update, b_i, dim=0, dim_size=edge_feat.size(0), reduce="add")

        if self.use_mlp_out:
            new_bond = self.mlp_out(new_bond)
        if self.resnet:
            new_bond = new_bond + edge_feat

        return new_bond

class AngleConvSum(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        angle_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        activation: str = "silu",
        resnet: bool = True,
    ):
        super().__init__()
        self.resnet = resnet
        self.activation = nn.ReLU() if activation == "relu" else nn.SiLU() if activation == "silu" else nn.GELU()

        # Core linear layer to reduce concatenated size to angle_dim
        #in_dim = atom_dim + 2 * bond_dim + angle_dim
        self.core_src_mlp = nn.Linear(atom_dim, bond_dim, bias=False)  # Core MLP for bond features
        self.core_dst_mlp = nn.Linear(atom_dim, bond_dim, bias=False)
        self.core_bond_mlp = nn.Linear(bond_dim, bond_dim, bias=False)
        self.core_angle_mlp = nn.Linear(angle_dim, bond_dim, bias=False)


        #self.mlp_gate = nn.Linear(in_dim, bond_dim, bias=False)  # Gate MLP for bond features
        self.src_mlp_gate = nn.Linear(atom_dim, bond_dim, bias=False)
        self.dst_mlp_gate = nn.Linear(atom_dim, bond_dim, bias=False)
        self.bond_mlp_gate = nn.Linear(bond_dim, bond_dim, bias=False)
        self.angle_mlp_gate = nn.Linear(angle_dim, bond_dim, bias=False)

        # GatedMLP for refined mixing
        self.gated = GatedMLP(
            input_dim=angle_dim,
            output_dim=angle_dim,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        vertex_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        angle_feat: torch.Tensor,
        edge_index: torch.Tensor,  # not used, but kept for signature
        triplet_info: dict,
    ) -> torch.Tensor:
        # Extract indices from triplet_info: k→j→i corresponds to bonds
        b_i = triplet_info['k_idx']  # bond index i (sending)
        atom_j = triplet_info['j_idx']  # center atom
        b_j = triplet_info['i_idx']  # bond index j (receiving)

        """
        center = vertex_feat[a]         # [T, atom_dim]
        bond_i = edge_feat[b_i]         # [T, bond_dim]
        bond_j = edge_feat[b_j]         # [T, bond_dim]
        angles = angle_feat            # [T, angle_dim]

        # Concatenate features
        total = torch.cat([center, bond_i, bond_j, angles], dim=-1)  # [T, *]
        core = self.core_mlp(total)  # [T, bond_dim]
        gate = self.mlp_gate(total)  # [T, bond_dim]
        """

        center = self.core_src_mlp(vertex_feat)         # [V, bond_dim]
        bond_i = self.core_bond_mlp(edge_feat)         # [E, bond_dim]
        bond_j = self.core_dst_mlp(vertex_feat)         # [E, bond_dim]
        angles = self.core_angle_mlp(angle_feat)            # [T, bond_dim]
        # Reverse scatter to core in the size of angles
        tmp = safe_reverse_scatter(center, atom_j, angles)  # [T, bond_dim]
        tmp = safe_reverse_scatter(bond_i, b_i, tmp)  # [T, bond_dim]
        core = safe_reverse_scatter(bond_j, b_j, tmp)  # [T, bond_dim]

        center = self.src_mlp_gate(vertex_feat)         # [V, bond_dim]
        bond_i = self.bond_mlp_gate(edge_feat)         # [E, bond_dim]
        bond_j = self.dst_mlp_gate(vertex_feat)         # [E, bond_dim]
        angles = self.angle_mlp_gate(angle_feat)            # [T, bond_dim]
        # Reverse scatter to core in the size of angles
        tmp = safe_reverse_scatter(center, atom_j, angles)  # [T, bond_dim]
        tmp = safe_reverse_scatter(bond_i, b_i, tmp)  # [T, bond_dim]
        gate = safe_reverse_scatter(bond_j, b_j, tmp)  # [T, bond_dim]
        x = self.gated(core, gate)              # [T, angle_dim]

        # Residual connection
        new_angle = x + angle_feat if self.resnet else x


        return new_angle
    
class AngleConvCat(nn.Module):
    """AngleUpdate layer with concatenation-based update (cat mode)."""

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        angle_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        activation: str = "silu",
        resnet: bool = True,
    ):
        super().__init__()
        self.resnet = resnet
        self.activation = nn.ReLU() if activation == "relu" else nn.SiLU() if activation == "silu" else nn.GELU()

        # Core linear layer to reduce concatenated size to angle_dim
        in_dim = atom_dim + 2 * bond_dim + angle_dim
        self.core_mlp = nn.Linear(in_dim, angle_dim, bias=False)
        self.mlp_gate = nn.Linear(in_dim, angle_dim, bias=False)  # Gate MLP for angle features

        # GatedMLP for refined mixing
        self.gated = GatedMLP(
            input_dim=angle_dim,
            output_dim=angle_dim,
            dropout=dropout,
            activation=activation,
        )

    def forward(
        self,
        vertex_feat: torch.Tensor,
        edge_feat: torch.Tensor,
        angle_feat: torch.Tensor,
        edge_index: torch.Tensor,  # not used, but kept for signature
        triplet_info: dict,
    ) -> torch.Tensor:
        # Extract indices from triplet_info: k→j→i corresponds to bonds
        b_i = triplet_info['k_idx']  # bond index i (sending)
        atom_j = triplet_info['j_idx']  # center atom
        b_j = triplet_info['i_idx']  # bond index j (receiving)

        # Retrieve features
        center = vertex_feat[atom_j]        # [T, atom_dim]
        bond_i = edge_feat[b_i]            # [T, bond_dim]
        bond_j = edge_feat[b_j]            # [T, bond_dim]
        legs = angle_feat                  # [T, angle_dim]

        # Concatenate
        total = torch.cat([center, bond_i, bond_j, legs], dim=-1)  # [T, *]
        core = self.core_mlp(total)       # [T, angle_dim]
        gate = self.mlp_gate(total)       # [T, angle_dim]
        x = self.gated(core, gate)              # [T, angle_dim]

        # Residual connection
        new_angle = x + angle_feat if self.resnet else x


        return new_angle

# --- CHGNetSimple class ---
class CHGNetSimple(nn.Module):
    def __init__(
        self,
        in_channels=64,
        hidden_channels=64,
        out_channels=64,
        bond_fea_dim=6,
        angle_fea_dim=13,
        num_layers=4,
        mode="cat",  # 'cat' or 'sum'
        task="regression",  # 'regression' or 'classification'
    ):
        super().__init__()
        self.mode = mode
        self.num_layers = num_layers

        # Embeddings
        self.atom_emb = nn.Linear(in_channels, hidden_channels) 
        self.bond_emb = nn.Linear(bond_fea_dim, hidden_channels, bias=False)
        self.angle_emb = nn.Linear(angle_fea_dim, hidden_channels, bias=False)

        # Interaction blocks
        self.atom_convs = nn.ModuleList()
        self.bond_convs = nn.ModuleList()
        self.angle_convs = nn.ModuleList()

        for i in range(num_layers):
            # Atom conv
            AtomConv = AtomConvCat if mode == "cat" else AtomConvSum
            self.atom_convs.append(AtomConv(hidden_channels, hidden_channels, hidden_channels))

            if i < num_layers - 1:
                BondConv = BondConvCat if mode == "cat" else BondConvSum
                self.bond_convs.append(BondConv(hidden_channels, hidden_channels, hidden_channels))

                AngleConv = AngleConvCat if mode == "cat" else AngleConvSum
                self.angle_convs.append(AngleConv(hidden_channels, hidden_channels, hidden_channels))

        # Read‑out MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data):
        """
        data should have:
          - atomic_numbers [N]
          - bond_feats [E, num_layers]
          - angle_feats [T, num_layers]
          - batch [N]
          - graph connectivity info passed as kwargs
        """


        x, edge_index = data.x, data.edge_index
        
        rbf = data.rbf
        triplet_info = data.triplet_info
        cbf_feat = triplet_info['cbf']
        batch = data.batch

        x = self.atom_emb(x)
        b = self.bond_emb(rbf)
        a = self.angle_emb(cbf_feat) 
        for i in range(self.num_layers):
            x = self.atom_convs[i](vertex_feat=x, edge_feat=b,
                                   edge_index=edge_index)

            if i < self.num_layers - 1:
                b = self.bond_convs[i](vertex_feat=x, edge_feat=b,
                                       angle_feat=a, edge_index=edge_index, triplet_info=triplet_info)
                if a is not None:
                    a = self.angle_convs[i](vertex_feat=x, edge_feat=b,
                                       angle_feat=a, edge_index=edge_index, triplet_info=triplet_info)

        # global pooling
        if batch is not None:
            # Manually compute dim_size to avoid data-dependent guard errors
            dim_size = int(batch.max().item()) + 1
            x_out = scatter(x, batch, dim=0, dim_size=dim_size, reduce='mean')

        # output
        return self.readout(x_out)