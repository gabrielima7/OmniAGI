"""Transfer Learning module."""

try:
    from omniagi.transfer.domain_adapter import (
        TransferLearner,
        DomainMapping,
        TransferResult,
        TransferType,
    )
except ImportError:
    TransferLearner = None
    DomainMapping = None
    TransferResult = None
    TransferType = None

try:
    from omniagi.transfer.zero_shot import (
        ZeroShotTransferSystem,
        Task,
        Concept,
    )
except ImportError:
    ZeroShotTransferSystem = None
    Task = None
    Concept = None

__all__ = [
    "TransferLearner",
    "DomainMapping",
    "TransferResult",
    "TransferType",
    "ZeroShotTransferSystem",
    "Task",
    "Concept",
]

