from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SimpleGATLayer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int = 2):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.projection = nn.Linear(in_dim, hidden_dim)
        self.attention = nn.Parameter(torch.empty(num_heads, 2 * self.head_dim))
        self.residual_projection = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.ELU()
        self.leaky_relu = nn.LeakyReLU(0.2)
        nn.init.xavier_uniform_(self.attention)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = node_features.shape
        projected = self.projection(node_features).view(
            batch_size,
            num_nodes,
            self.num_heads,
            self.head_dim,
        )

        source = projected.unsqueeze(2).expand(-1, -1, num_nodes, -1, -1)
        target = projected.unsqueeze(1).expand(-1, num_nodes, -1, -1, -1)
        attention_input = torch.cat([source, target], dim=-1)
        scores = (attention_input * self.attention.view(1, 1, 1, self.num_heads, -1)).sum(dim=-1)
        scores = self.leaky_relu(scores).permute(0, 3, 1, 2)

        adjacency_mask = adjacency.unsqueeze(1) > 0.0
        scores = scores.masked_fill(~adjacency_mask, -1e9)
        attention_weights = torch.softmax(scores, dim=-1)
        aggregated = torch.einsum("bhij,bjhd->bihd", attention_weights, projected)
        aggregated = aggregated.reshape(batch_size, num_nodes, self.hidden_dim)

        residual = self.residual_projection(node_features)
        return self.layer_norm(self.activation(aggregated) + residual)


class TaskGraphFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        hidden_dim: int = 64,
        num_gat_layers: int = 2,
        num_heads: int = 2,
    ):
        spaces_dict = observation_space.spaces
        max_ready_tasks = spaces_dict["ready_task_node_ids"].shape[0]
        task_feature_dim = spaces_dict["task_features"].shape[1]
        num_resources = spaces_dict["resource_features"].shape[0]
        resource_feature_dim = spaces_dict["resource_features"].shape[1]
        features_dim = max_ready_tasks * hidden_dim + num_resources * hidden_dim + hidden_dim
        super().__init__(observation_space, features_dim)

        layers: list[SimpleGATLayer] = []
        current_dim = task_feature_dim
        for _ in range(num_gat_layers):
            layers.append(SimpleGATLayer(current_dim, hidden_dim, num_heads=num_heads))
            current_dim = hidden_dim
        self.gat_layers = nn.ModuleList(layers)
        self.resource_encoder = nn.Sequential(
            nn.Linear(resource_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        task_features = observations["task_features"].float()
        task_adjacency = observations["task_adjacency"].float()
        task_valid_mask = observations["task_valid_mask"].float()
        ready_task_node_ids = observations["ready_task_node_ids"].long()
        resource_features = observations["resource_features"].float()

        node_embeddings = task_features
        valid_mask_expanded = task_valid_mask.unsqueeze(-1)
        for layer in self.gat_layers:
            node_embeddings = layer(node_embeddings, task_adjacency)
            node_embeddings = node_embeddings * valid_mask_expanded

        valid_counts = task_valid_mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        graph_embedding = (node_embeddings * valid_mask_expanded).sum(dim=1) / valid_counts

        max_nodes = node_embeddings.shape[1]
        ready_ids = ready_task_node_ids.clamp(min=0, max=max_nodes - 1)
        batch_indices = torch.arange(node_embeddings.shape[0], device=node_embeddings.device).unsqueeze(1)
        ready_embeddings = node_embeddings[batch_indices, ready_ids]

        resource_embeddings = self.resource_encoder(resource_features)
        return torch.cat(
            [
                ready_embeddings.flatten(start_dim=1),
                resource_embeddings.flatten(start_dim=1),
                graph_embedding,
            ],
            dim=1,
        )

