"""Heterogeneous message-passing GNN on (flow, link) graphs, RouteNet-style.

One iteration k = two half-steps (h = hidden state, e = edge attributes of the hop):

  1. links listen to the flows they carry
       h_l <- LN( h_l + Dropout(ReLU( CONV_f->l( {h_f, e_fl : f crosses l}, h_l ) )) )
  2. flows listen to the links on their path (already updated in step 1)
       h_f <- LN( h_f + Dropout(ReLU( CONV_l->f( {h_l, e_fl : l on path of f}, h_f ) )) )

After K iterations: delay_hat_f = MLP(h_f)   (log10 seconds).

CONV is one of
  * "sage"  : EdgeSAGE (below)  -> W_root h_dst + mean_j ReLU(W_msg [h_j || e_ij])
  * "gatv2" : GATv2Conv with edge_dim -> sum_j alpha_ij W h_j, alpha learned from (h_dst, h_j, e_ij)
K = 0 gives an MLP on the flow's own features: the "no message passing" ablation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv, MessagePassing

F2L = ("flow", "traverses", "link")
L2F = ("link", "carries", "flow")


class EdgeSAGE(MessagePassing):
    """GraphSAGE (mean aggregation) whose message also sees the edge attributes.

    Plain SAGEConv ignores edge features; here the WFQ share of the hop must reach
    the link/flow, so the message is ReLU(W_msg [h_src || e]).
    """

    def __init__(self, hidden: int, edge_dim: int):
        super().__init__(aggr="mean")
        self.msg = nn.Linear(hidden + edge_dim, hidden)
        self.root = nn.Linear(hidden, hidden)

    def forward(self, x_src, x_dst, edge_index, edge_attr):
        agg = self.propagate(edge_index, x=(x_src, x_dst), edge_attr=edge_attr,
                             size=(x_src.size(0), x_dst.size(0)))
        return self.root(x_dst) + agg

    def message(self, x_j, edge_attr):           # x_j = state of the sending node
        return torch.relu(self.msg(torch.cat([x_j, edge_attr], dim=-1)))


def make_conv(kind: str, hidden: int, edge_dim: int, heads: int):
    if kind == "sage":
        return EdgeSAGE(hidden, edge_dim)
    if kind == "gatv2":
        return GATv2Conv((hidden, hidden), hidden, heads=heads, concat=False,
                         edge_dim=edge_dim, add_self_loops=False)
    raise ValueError(kind)


def mlp(d_in: int, hidden: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, d_out))


class FlowLinkGNN(nn.Module):
    def __init__(self, flow_dim: int, link_dim: int, edge_dim: int, conv: str = "gatv2",
                 hidden: int = 128, layers: int = 3, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv_kind, self.layers = conv, layers
        self.enc_flow = mlp(flow_dim, hidden, hidden, dropout)
        self.enc_link = mlp(link_dim, hidden, hidden, dropout)
        self.f2l = nn.ModuleList(make_conv(conv, hidden, edge_dim, heads) for _ in range(layers))
        self.l2f = nn.ModuleList(make_conv(conv, hidden, edge_dim, heads) for _ in range(layers))
        self.norm_l = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.norm_f = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.readout = mlp(hidden, hidden, 1, dropout)

    def forward(self, data, return_attention: bool = False):
        h_f = self.enc_flow(data["flow"].x)
        h_l = self.enc_link(data["link"].x)
        ei_fl, ei_lf = data[F2L].edge_index, data[L2F].edge_index
        e = data[F2L].edge_attr                     # same hop attributes in both directions
        attn = []
        for k in range(self.layers):
            # 1. each link aggregates the flows it carries
            m_l = self._conv(self.f2l[k], h_f, h_l, ei_fl, e, attn if return_attention else None)
            h_l = self.norm_l[k](h_l + self.drop(torch.relu(m_l)))
            # 2. each flow aggregates the links of its path (already updated in step 1)
            m_f = self._conv(self.l2f[k], h_l, h_f, ei_lf, e, None)
            h_f = self.norm_f[k](h_f + self.drop(torch.relu(m_f)))
        out = self.readout(h_f).squeeze(-1)         # predicted log10(delay) per flow
        return (out, attn) if return_attention else out

    def _conv(self, conv, x_src, x_dst, edge_index, e, attn_store):
        """Call either conv type with the same arguments; optionally keep GATv2 attention."""
        if self.conv_kind == "sage":
            return conv(x_src, x_dst, edge_index, e)
        if attn_store is not None:
            out, (ei, alpha) = conv((x_src, x_dst), edge_index, edge_attr=e, return_attention_weights=True)
            attn_store.append(alpha.detach())        # one weight per (flow -> link) edge and head
            return out
        return conv((x_src, x_dst), edge_index, edge_attr=e)
