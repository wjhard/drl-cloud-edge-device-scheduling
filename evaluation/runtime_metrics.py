from __future__ import annotations

import ctypes
import os
from pathlib import Path


MIB = 1024.0 * 1024.0


def process_memory_mib() -> dict[str, float | None]:
    """Return current and peak resident memory without third-party packages."""

    if os.name == "nt":
        try:
            from ctypes import wintypes

            class ProcessMemoryCounters(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = ProcessMemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            psapi = ctypes.WinDLL("psapi", use_last_error=True)
            kernel32.GetCurrentProcess.argtypes = []
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi.GetProcessMemoryInfo.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(ProcessMemoryCounters),
                wintypes.DWORD,
            ]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            handle = kernel32.GetCurrentProcess()
            ok = psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            )
            if not ok:
                raise ctypes.WinError(ctypes.get_last_error())
            return {
                "rss_mib": counters.WorkingSetSize / MIB,
                "peak_rss_mib": counters.PeakWorkingSetSize / MIB,
            }
        except (AttributeError, OSError, ValueError):
            return {"rss_mib": None, "peak_rss_mib": None}

    status_path = Path("/proc/self/status")
    if status_path.exists():
        values: dict[str, float] = {}
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, raw_value, *_ = line.split()
                values[key.rstrip(":")] = float(raw_value) / 1024.0
        return {
            "rss_mib": values.get("VmRSS"),
            "peak_rss_mib": values.get("VmHWM"),
        }

    return {"rss_mib": None, "peak_rss_mib": None}


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
