"""Shim: campaign result persistence moved to lib/campaign.py.

Kept so the historical `from campaign_io import save_result` import in
the campaign scripts keeps working; the implementation (flat [src:]
file + provenance-stamped registry copy) lives in lib.campaign.

The caller-detection contract is preserved: save_result here forwards
the caller's script path explicitly, so provenance records the
producing script, not this shim.
"""
import inspect
import os
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))

from lib.campaign import save_result as _save_result


def save_result(name, results, **kwargs):
    if "script" not in kwargs:
        caller = inspect.stack()[1].filename
        try:
            kwargs["script"] = os.path.relpath(caller, WORK)
        except ValueError:
            kwargs["script"] = caller
    return _save_result(name, results, **kwargs)
