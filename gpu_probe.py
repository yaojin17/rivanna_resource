"""Read one job's GPUs through NVML and print a JSON object per card.

This runs as a job step inside the job being measured (see ``_probe_gpu`` in
app.py), so the cgroup already limits NVML to the cards that job was given --
``nvmlDeviceGetCount`` returns the job's GPUs, not the node's.

Utilisation and power are taken from the ring buffer the driver fills on its
own, which is why a multi-second average costs nothing to read: the samples are
already there and the process exits in well under a second.  Everything else
(memory, temperature, the power cap) is a point reading, because it is a level
rather than a rate and does not bounce between refreshes.

Prints warnings to stderr and leaves the field null rather than dying, so one
unsupported reading on one card does not cost the whole panel.
"""

import ctypes
import json
import sys
import time

# How far back to average over, in seconds, overridable as the one argument so
# app.py stays the single place the window is chosen.  The driver keeps roughly
# 14s of utilisation samples but only ~2.4s of power samples, so the window
# actually achieved is reported alongside each average rather than assumed.
DEFAULT_WINDOW_SECONDS = 5.0

NVML_TOTAL_POWER_SAMPLES = 0
NVML_GPU_UTILIZATION_SAMPLES = 1
NVML_TEMPERATURE_GPU = 0
NVML_SUCCESS = 0
NVML_ERROR_NOT_SUPPORTED = 3
NVML_ERROR_NOT_FOUND = 6
# nvmlValueType_t
VALUE_FIELDS = {0: "dVal", 1: "uiVal", 2: "ulVal", 3: "ullVal", 4: "sllVal"}


class NvmlValue(ctypes.Union):
    _fields_ = [
        ("dVal", ctypes.c_double),
        ("uiVal", ctypes.c_uint),
        ("ulVal", ctypes.c_ulong),
        ("ullVal", ctypes.c_ulonglong),
        ("sllVal", ctypes.c_longlong),
    ]


class NvmlSample(ctypes.Structure):
    _fields_ = [("timeStamp", ctypes.c_ulonglong), ("sampleValue", NvmlValue)]


class NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def load_nvml():
    """Open libnvidia-ml and declare every prototype this script calls."""
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    device = ctypes.c_void_p
    signatures = {
        "nvmlInit_v2": [],
        "nvmlShutdown": [],
        "nvmlErrorString": [ctypes.c_int],
        "nvmlDeviceGetCount_v2": [ctypes.POINTER(ctypes.c_uint)],
        "nvmlDeviceGetHandleByIndex_v2": [ctypes.c_uint, ctypes.POINTER(device)],
        "nvmlDeviceGetName": [device, ctypes.c_char_p, ctypes.c_uint],
        "nvmlDeviceGetMemoryInfo": [device, ctypes.POINTER(NvmlMemory)],
        "nvmlDeviceGetTemperature": [device, ctypes.c_int, ctypes.POINTER(ctypes.c_uint)],
        "nvmlDeviceGetEnforcedPowerLimit": [device, ctypes.POINTER(ctypes.c_uint)],
        "nvmlDeviceGetSamples": [
            device,
            ctypes.c_int,
            ctypes.c_ulonglong,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(NvmlSample),
        ],
    }
    for name, argtypes in signatures.items():
        func = getattr(nvml, name)
        func.argtypes = argtypes
        func.restype = ctypes.c_char_p if name == "nvmlErrorString" else ctypes.c_int
    return nvml


def fail(nvml, call, code):
    """Warn about a failed NVML call; the caller nulls the field and carries on."""
    reason = (nvml.nvmlErrorString(code) or b"").decode("utf-8", "replace")
    print(f"Warning: {call} failed: {reason or code}", file=sys.stderr)


def sampled_average(nvml, dev, kind, label, since_us):
    """Mean of the buffered samples newer than ``since_us``.

    Returns ``(average, count, span_seconds)``, with the average None when the
    driver has nothing buffered.  An idle card legitimately has no samples, and
    a MIG slice does not support the call at all; neither is an error.
    """
    value_type = ctypes.c_int()
    count = ctypes.c_uint()
    code = nvml.nvmlDeviceGetSamples(
        dev, kind, since_us, ctypes.byref(value_type), ctypes.byref(count), None
    )
    if code in (NVML_ERROR_NOT_FOUND, NVML_ERROR_NOT_SUPPORTED):
        return None, 0, 0.0
    if code != NVML_SUCCESS:
        fail(nvml, f"nvmlDeviceGetSamples({label}) sizing", code)
        return None, 0, 0.0

    buffer = (NvmlSample * count.value)()
    code = nvml.nvmlDeviceGetSamples(
        dev, kind, since_us, ctypes.byref(value_type), ctypes.byref(count), buffer
    )
    if code in (NVML_ERROR_NOT_FOUND, NVML_ERROR_NOT_SUPPORTED):
        return None, 0, 0.0
    if code != NVML_SUCCESS:
        fail(nvml, f"nvmlDeviceGetSamples({label})", code)
        return None, 0, 0.0

    field = VALUE_FIELDS.get(value_type.value)
    if field is None:
        print(
            f"Warning: nvmlDeviceGetSamples({label}) returned unknown value type "
            f"{value_type.value}, ignoring the samples",
            file=sys.stderr,
        )
        return None, 0, 0.0

    samples = buffer[: count.value]
    if not samples:
        return None, 0, 0.0
    values = [float(getattr(s.sampleValue, field)) for s in samples]
    span = (samples[-1].timeStamp - samples[0].timeStamp) / 1e6
    return sum(values) / len(values), len(values), span


def scalar(nvml, call, dev, *extra):
    """One unsigned-int NVML reading, or None when the card will not give it."""
    out = ctypes.c_uint()
    code = getattr(nvml, call)(dev, *extra, ctypes.byref(out))
    if code != NVML_SUCCESS:
        fail(nvml, call, code)
        return None
    return out.value


def describe(nvml, index, window):
    """Every reading for one card, as the dict app.py turns into a table row."""
    dev = ctypes.c_void_p()
    code = nvml.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(dev))
    if code != NVML_SUCCESS:
        fail(nvml, f"nvmlDeviceGetHandleByIndex_v2({index})", code)
        return None

    since_us = ctypes.c_ulonglong(int((time.time() - window) * 1e6))
    util, util_n, util_span = sampled_average(
        nvml, dev, NVML_GPU_UTILIZATION_SAMPLES, "utilisation", since_us
    )
    power, power_n, power_span = sampled_average(
        nvml, dev, NVML_TOTAL_POWER_SAMPLES, "power", since_us
    )

    name_buffer = ctypes.create_string_buffer(96)
    code = nvml.nvmlDeviceGetName(dev, name_buffer, len(name_buffer))
    if code != NVML_SUCCESS:
        fail(nvml, "nvmlDeviceGetName", code)
    name = name_buffer.value.decode("utf-8", "replace") if code == NVML_SUCCESS else None

    memory = NvmlMemory()
    code = nvml.nvmlDeviceGetMemoryInfo(dev, ctypes.byref(memory))
    if code != NVML_SUCCESS:
        fail(nvml, "nvmlDeviceGetMemoryInfo", code)
        mem_used_mb = mem_total_mb = None
    else:
        mem_used_mb = memory.used / 1024 / 1024
        mem_total_mb = memory.total / 1024 / 1024

    power_limit_mw = scalar(nvml, "nvmlDeviceGetEnforcedPowerLimit", dev)
    return {
        "index": index,
        "name": name,
        # Samples are in whole percent and milliwatts.
        "util_avg": util,
        "util_n": util_n,
        "util_window_s": util_span,
        "power_avg_w": power / 1000 if power is not None else None,
        "power_n": power_n,
        "power_window_s": power_span,
        "mem_used_mb": mem_used_mb,
        "mem_total_mb": mem_total_mb,
        "power_limit_w": power_limit_mw / 1000 if power_limit_mw is not None else None,
        "temp_c": scalar(nvml, "nvmlDeviceGetTemperature", dev, NVML_TEMPERATURE_GPU),
    }


def main(argv):
    window = DEFAULT_WINDOW_SECONDS
    if argv:
        try:
            window = float(argv[0])
        except ValueError:
            print(
                f"Warning: unparsable window {argv[0]!r}, "
                f"averaging over {window}s instead",
                file=sys.stderr,
            )

    try:
        nvml = load_nvml()
    except OSError as exc:
        print(f"Warning: could not load libnvidia-ml.so.1: {exc}", file=sys.stderr)
        return 1

    code = nvml.nvmlInit_v2()
    if code != NVML_SUCCESS:
        fail(nvml, "nvmlInit_v2", code)
        return 1
    try:
        count = ctypes.c_uint()
        code = nvml.nvmlDeviceGetCount_v2(ctypes.byref(count))
        if code != NVML_SUCCESS:
            fail(nvml, "nvmlDeviceGetCount_v2", code)
            return 1
        for index in range(count.value):
            card = describe(nvml, index, window)
            if card is not None:
                print(json.dumps(card))
    finally:
        nvml.nvmlShutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
