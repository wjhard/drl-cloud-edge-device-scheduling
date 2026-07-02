from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Resource:
    id: str
    tier: str
    compute_power: float
    bandwidth: float
    available_time: float = 0.0


class ResourceConfig:
    def __init__(self, resources: list[Resource]):
        if not resources:
            raise ValueError("at least one resource is required")
        self.resources = resources
        self._by_id = {resource.id: resource for resource in resources}
        if len(self._by_id) != len(resources):
            raise ValueError("resource ids must be unique")

    def get_resource(self, resource: Resource | str | int) -> Resource:
        if isinstance(resource, Resource):
            return resource
        if isinstance(resource, int):
            return self.resources[resource]
        return self._by_id[resource]

    def get_execution_time(self, task: dict[str, Any] | float, resource: Resource | str | int) -> float:
        resource_obj = self.get_resource(resource)
        if resource_obj.compute_power <= 0:
            raise ValueError(f"resource {resource_obj.id} has non-positive compute_power")

        if isinstance(task, dict):
            computation_cost = float(task["computation_cost"])
        else:
            computation_cost = float(task)
        return computation_cost / resource_obj.compute_power

    def get_communication_time(
        self,
        data_size: float,
        resource_a: Resource | str | int,
        resource_b: Resource | str | int,
    ) -> float:
        first = self.get_resource(resource_a)
        second = self.get_resource(resource_b)
        if first.id == second.id:
            return 0.0
        bottleneck_bandwidth = min(first.bandwidth, second.bandwidth)
        if bottleneck_bandwidth <= 0:
            raise ValueError("bandwidth must be positive for cross-resource communication")
        return float(data_size) / bottleneck_bandwidth

    def reset(self) -> None:
        for resource in self.resources:
            resource.available_time = 0.0


def load_resource_config(filepath: str | Path) -> ResourceConfig:
    data = yaml.safe_load(Path(filepath).read_text(encoding="utf-8"))
    resources = [
        Resource(
            id=str(item["id"]),
            tier=str(item["tier"]),
            compute_power=float(item["compute_power"]),
            bandwidth=float(item["bandwidth"]),
        )
        for item in data.get("resources", [])
    ]
    return ResourceConfig(resources)


def default_resource_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "resource_default.yaml"

