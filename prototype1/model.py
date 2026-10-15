"""Edge-aware message-passing GNN for AC power-flow state prediction."""
import torch
from torch import nn
from common import edge_features, node_features, pf_mask


class MessageLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(2 * hidden + 6, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden), nn.SiLU())
        self.update = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, ei, ef):
        src, dst = ei
        msg = self.message(torch.cat([h[src], h[dst], ef], dim=-1))
        agg = h.new_zeros(h.shape).index_add(0, dst, msg)
        deg = h.new_zeros((h.shape[0], 1)).index_add(
            0, dst, h.new_ones((dst.shape[0], 1))).clamp_min_(1.0)
        return self.norm(h + self.update(torch.cat([h, agg / deg], dim=-1)))


class PowerFlowGNN(nn.Module):
    """Predict normalized bus state; observed values are preserved by reconstruction."""
    def __init__(self, hidden=128, layers=5):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(18, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([MessageLayer(hidden) for _ in range(layers)])
        self.decoder = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 6))

    def forward(self, g):
        mask = pf_mask(g)
        x = self.encoder(node_features(g, mask))
        # Physics stores each physical branch once; message passing uses both directions.
        ei = torch.cat([g.ei, g.ei.flip(0)], dim=1)
        ef = edge_features(g)
        for layer in self.layers:
            x = layer(x, ei, ef)
        return self.decoder(x)
