# -*- coding: utf-8 -*-
"""Launch permutation test immediately; wait for memory then launch random control."""
import os, subprocess, time, ctypes
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "3"
env["MKL_NUM_THREADS"] = "3"
env["PATH_AGNN_BATCH_SIZE"] = "64"

def free_gb():
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / (1024**3)

def spawn(script, tag):
    out = open(os.path.join(root, "work", tag + ".log"), "w", encoding="utf-8")
    err = open(os.path.join(root, "work", tag + ".err"), "w", encoding="utf-8")
    p = subprocess.Popen([py, os.path.join(root, "work", script)], cwd=root, env=env, stdout=out, stderr=err)
    print("started", tag, p.pid, flush=True)
    return p, out, err

p1, o1, e1 = spawn("permutation_test.py", "perm1")
# wait for memory headroom before the heavier random control
while free_gb() < 6.0 and p1.poll() is None:
    time.sleep(20)
print("memory ok: %.1f GB" % free_gb(), flush=True)
p2, o2, e2 = spawn("random_control.py", "randctrl1")
for tag, p, o, e in [("perm1", p1, o1, e1), ("randctrl1", p2, o2, e2)]:
    rc = p.wait()
    o.close(); e.close()
    print("done", tag, "exit", rc, flush=True)
print("ALL_STRENGTHEN_DONE", flush=True)
