"""Shared library for the reproduction pipeline.

Stage scripts import submodules directly (lib.campaign, lib.experiment,
lib.encoders.gram, ...); this package init only installs the runtime
guardrails.
"""

# Auto-guardrails: RAM watchdog + SIGTERM flush for every lib consumer,
# even scripts that forget init_runtime(). Opt out: EXPERIMENT_AUTO_INIT=0.
from .runtime import soft_init as _soft_init
_soft_init()
del _soft_init
