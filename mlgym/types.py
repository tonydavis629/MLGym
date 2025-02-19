"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Adapted from SWE-agent/sweagent/types.py
This file has types/dataclass definitions that are used in MLGym for exchanging information between modules/functions/classes.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, TypedDict


class TrajectoryStep(TypedDict):
    action: str
    observation: str
    response: str
    state: str | None
    thought: str
    execution_time: float


class _HistoryItem(TypedDict):
    role: str


class HistoryItem(_HistoryItem, total=False):
    content: str | None
    agent: str
    is_demo: bool
    thought: str
    action: str | None


History = list[HistoryItem]
Trajectory = list[TrajectoryStep]


class AgentInfo(defaultdict):
    def __init__(self):
        super().__init__(lambda: None)
        self.model_stats: dict[str, float] = {}
        self.exit_status: str = ""
        self.submission: str | None = None
        self.score: list[dict[str, float]] = []
        self.summarizer: dict = {}
    
    def __getattr__(self, name: str) -> Any:
        return self[name]
    
    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value
        
    def update(self, other: dict[str, Any]) -> None:
        for key, value in other.items():
            if key == 'score' and isinstance(value, list):
                self.score.extend(value)
            else:
                self[key] = value
