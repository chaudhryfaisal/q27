import sys
import pynvml as nv

nv.nvmlInit()
h = nv.nvmlDeviceGetHandleByIndex(0)
off = int(sys.argv[1])
try:
    nv.nvmlDeviceSetMemClkVfOffset(h, off)
    print("legacy SetMemClkVfOffset OK ->", nv.nvmlDeviceGetMemClkVfOffset(h))
except nv.NVMLError as e:
    print("legacy setter:", e)
# also check min/max via the dedicated range getter if present
for fn in ("nvmlDeviceGetMinMaxClockOfPState", "nvmlDeviceGetSupportedPerformanceStates"):
    if hasattr(nv, fn):
        try:
            print(fn, getattr(nv, fn)(h))
        except Exception as e:
            print(fn, "->", e)
