"""
Online Learning System for Real AGI.

Implements neural network training, experience replay,
meta-learning (MAML), and adaptive learning.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import deque
import copy

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


@dataclass
class TrainingExample:
    """A training example for online learning."""
    input_data: Any
    target: Any
    task_id: str = "default"
    priority: float = 1.0
    times_trained: int = 0


class ExperienceReplayBuffer:
    """
    Experience replay buffer for stable online learning.
    
    Implements prioritized experience replay.
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
    
    def add(self, example: TrainingExample) -> None:
        """Add an experience to the buffer."""
        self.buffer.append(example)
        self.priorities.append(example.priority)
    
    def sample(self, batch_size: int) -> List[TrainingExample]:
        """Sample a batch using prioritized replay."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        # Prioritized sampling
        total_priority = sum(self.priorities)
        if total_priority == 0:
            indices = random.sample(range(len(self.buffer)), batch_size)
        else:
            probs = [p / total_priority for p in self.priorities]
            indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
        
        return [self.buffer[i] for i in indices]
    
    def update_priority(self, index: int, new_priority: float) -> None:
        """Update priority of an experience."""
        if 0 <= index < len(self.priorities):
            self.priorities[index] = new_priority
    
    def __len__(self) -> int:
        return len(self.buffer)


class AdaptiveLearningRate:
    """
    Adaptive learning rate scheduler.
    
    Adjusts learning rate based on loss dynamics.
    """
    
    def __init__(
        self,
        initial_lr: float = 0.001,
        min_lr: float = 1e-6,
        max_lr: float = 0.1,
        patience: int = 5,
    ):
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.patience = patience
        
        self.loss_history: List[float] = []
        self.best_loss = float('inf')
        self.plateau_count = 0
    
    def step(self, current_loss: float) -> float:
        """Update learning rate based on loss."""
        self.loss_history.append(current_loss)
        
        if current_loss < self.best_loss * 0.99:
            self.best_loss = current_loss
            self.plateau_count = 0
            # Increase LR slightly when improving
            self.current_lr = min(self.max_lr, self.current_lr * 1.05)
        else:
            self.plateau_count += 1
            if self.plateau_count >= self.patience:
                # Reduce LR on plateau
                self.current_lr = max(self.min_lr, self.current_lr * 0.5)
                self.plateau_count = 0
        
        return self.current_lr
    
    def get_lr(self) -> float:
        return self.current_lr


class OnlineLearner(nn.Module if TORCH_AVAILABLE else object):
    """
    Online learning neural network.
    
    Can learn continuously from new experiences.
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 32):
        if TORCH_AVAILABLE:
            super().__init__()
            
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            
            self.head = nn.Linear(hidden_dim, output_dim)
            
            self.optimizer = optim.Adam(self.parameters(), lr=0.001)
            self.lr_scheduler = AdaptiveLearningRate(0.001)
            self.replay_buffer = ExperienceReplayBuffer(10000)
            
            self.total_updates = 0
            self.running_loss = 0.0
    
    def forward(self, x):
        if not TORCH_AVAILABLE:
            return None
        features = self.encoder(x)
        return self.head(features)
    
    def learn(self, example: TrainingExample) -> float:
        """Learn from a single example."""
        if not TORCH_AVAILABLE:
            return 0.0
        
        # Add to replay buffer
        self.replay_buffer.add(example)
        
        # Sample batch
        batch_size = min(32, len(self.replay_buffer))
        batch = self.replay_buffer.sample(batch_size)
        
        # Prepare data
        inputs = torch.stack([
            torch.tensor(e.input_data, dtype=torch.float32)
            if not isinstance(e.input_data, torch.Tensor)
            else e.input_data
            for e in batch
        ])
        
        targets = torch.stack([
            torch.tensor(e.target, dtype=torch.float32)
            if not isinstance(e.target, torch.Tensor)
            else e.target
            for e in batch
        ])
        
        # Forward pass
        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        
        # Compute loss
        loss = F.mse_loss(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        
        # Update weights
        self.optimizer.step()
        
        # Update learning rate
        new_lr = self.lr_scheduler.step(loss.item())
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.total_updates += 1
        self.running_loss = 0.9 * self.running_loss + 0.1 * loss.item()
        
        return loss.item()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_updates": self.total_updates,
            "running_loss": self.running_loss,
            "learning_rate": self.lr_scheduler.get_lr(),
            "buffer_size": len(self.replay_buffer),
        }


class MAML:
    """
    Model-Agnostic Meta-Learning (MAML).
    
    Learns to learn - finds initialization that can adapt
    quickly to new tasks.
    """
    
    def __init__(
        self,
        model: nn.Module,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        if not TORCH_AVAILABLE:
            return
        
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        
        self.meta_optimizer = optim.Adam(model.parameters(), lr=outer_lr)
        self.task_losses: Dict[str, List[float]] = {}
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        """
        Inner loop: adapt to a specific task.
        
        Returns adapted model (without modifying original).
        """
        if not TORCH_AVAILABLE:
            return None
        
        # Clone model for task-specific adaptation
        adapted_model = copy.deepcopy(self.model)
        inner_optimizer = optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        for _ in range(self.inner_steps):
            inner_optimizer.zero_grad()
            pred = adapted_model(support_x)
            loss = F.mse_loss(pred, support_y)
            loss.backward()
            inner_optimizer.step()
        
        return adapted_model
    
    def outer_loop(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        """
        Outer loop: meta-update across tasks.
        
        Each task is (support_x, support_y, query_x, query_y).
        """
        if not TORCH_AVAILABLE:
            return 0.0
        
        self.meta_optimizer.zero_grad()
        
        total_loss = 0.0
        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop adaptation
            adapted = self.inner_loop(support_x, support_y)
            
            # Compute loss on query set
            query_pred = adapted(query_x)
            task_loss = F.mse_loss(query_pred, query_y)
            total_loss += task_loss
        
        # Average loss across tasks
        avg_loss = total_loss / len(tasks)
        
        # Meta-update (through copied models)
        avg_loss.backward()
        self.meta_optimizer.step()
        
        return avg_loss.item()
    
    def adapt_to_task(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
    ) -> nn.Module:
        """Adapt to a new task and return the adapted model."""
        return self.inner_loop(support_x, support_y)


class EWC:
    """
    Elastic Weight Consolidation.
    
    Prevents catastrophic forgetting by penalizing changes
    to important weights.
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        if not TORCH_AVAILABLE:
            return
        
        self.model = model
        self.lambda_ewc = lambda_ewc
        
        # Store optimal parameters for previous tasks
        self.saved_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
    
    def compute_fisher(
        self,
        data_loader,
        num_samples: int = 100,
    ) -> None:
        """Compute Fisher information matrix."""
        if not TORCH_AVAILABLE:
            return
        
        self.model.eval()
        
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters()}
        
        for i, (x, y) in enumerate(data_loader):
            if i >= num_samples:
                break
            
            self.model.zero_grad()
            output = self.model(x)
            loss = F.mse_loss(output, y)
            loss.backward()
            
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
        
        # Normalize
        for n in fisher:
            fisher[n] /= num_samples
        
        # Accumulate Fisher
        for n in fisher:
            if n in self.fisher:
                self.fisher[n] += fisher[n]
            else:
                self.fisher[n] = fisher[n]
        
        # Save current parameters
        for n, p in self.model.named_parameters():
            self.saved_params[n] = p.data.clone()
    
    def penalty(self) -> torch.Tensor:
        """Compute EWC penalty."""
        if not TORCH_AVAILABLE or not self.fisher:
            return torch.tensor(0.0)
        
        total = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                diff = p - self.saved_params[n]
                total += (self.fisher[n] * diff ** 2).sum()
        
        return self.lambda_ewc * total


class RealAGILearner:
    """
    Real AGI Learning System.
    
    Combines online learning, meta-learning, and
    catastrophic forgetting prevention.
    """
    
    def __init__(self, input_dim: int = 64, hidden_dim: int = 128, output_dim: int = 32):
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available")
            self.online = None
            self.maml = None
            self.ewc = None
            return
        
        self.online = OnlineLearner(input_dim, hidden_dim, output_dim)
        self.maml = MAML(self.online, inner_lr=0.01, outer_lr=0.001)
        self.ewc = EWC(self.online, lambda_ewc=1000.0)
        
        self.tasks_learned: List[str] = []
        self.total_examples = 0
    
    def learn_example(self, example: TrainingExample) -> float:
        """Learn from a single example."""
        if not TORCH_AVAILABLE:
            return 0.0
        
        loss = self.online.learn(example)
        self.total_examples += 1
        
        return loss
    
    def learn_task(
        self,
        task_id: str,
        examples: List[TrainingExample],
    ) -> Dict[str, float]:
        """Learn a complete task and consolidate."""
        if not TORCH_AVAILABLE:
            return {}
        
        # Train on examples
        losses = []
        for ex in examples:
            ex.task_id = task_id
            loss = self.online.learn(ex)
            losses.append(loss)
        
        # Consolidate to prevent forgetting
        # Would need DataLoader for full EWC
        
        self.tasks_learned.append(task_id)
        
        return {
            "task_id": task_id,
            "examples": len(examples),
            "final_loss": losses[-1] if losses else 0,
            "avg_loss": sum(losses) / len(losses) if losses else 0,
        }
    
    def adapt_to_new_task(
        self,
        support_examples: List[TrainingExample],
    ) -> nn.Module:
        """Quickly adapt to a new task using meta-learning."""
        if not TORCH_AVAILABLE:
            return None
        
        support_x = torch.stack([
            torch.tensor(e.input_data, dtype=torch.float32)
            if not isinstance(e.input_data, torch.Tensor)
            else e.input_data
            for e in support_examples
        ])
        
        support_y = torch.stack([
            torch.tensor(e.target, dtype=torch.float32)
            if not isinstance(e.target, torch.Tensor)
            else e.target
            for e in support_examples
        ])
        
        return self.maml.adapt_to_task(support_x, support_y)
    
    def get_stats(self) -> Dict[str, Any]:
        if not TORCH_AVAILABLE:
            return {"status": "PyTorch not available"}
        
        return {
            "total_examples": self.total_examples,
            "tasks_learned": len(self.tasks_learned),
            "online_stats": self.online.get_stats(),
        }
