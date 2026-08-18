# -*- coding: utf-8 -*-
"""Single rewiring LUAD test: OMP=3, batch=64, memory monitor."""
import os, subprocess, time, ctypes
root = r"D:\TT paper\Path-AGNN-Cox Pathway-Constrained Adaptive Graph Neural Network for Interpretable Survival Analysis"
py = r"C:\Users\Administrator\AppData\Local\Programs\Python\Python310\python.exe"
env = os.environ.copy()
env["OMP_NUM_THREADS"] = "3"
env["MKL_NUM_THREADS"] = "3"
env["PATH_AGNN_BATCH_SIZE"] = "64"
cmd = [py, "-m", "benchmark.rewiring_analysis", "--dataset", "LUAD",
       "--train-csv", os.path.join(root, "data", "processed", "LUAD", "train.csv"),
       "--gmt", os.path.join(root, "data", "pathways", "kegg_cancer_core.gmt"),
       "--clinical-csv", os.path.join(root, "data", "processed", "rewiring", "clinical_LUAD.csv"),
       "--out", os.path.join(root, "results", "rewiring", "LUAD"),
       "--known-pathways-file", os.path.join(root, "data", "pathways", "luad_known_pathways.txt")]
out = open(os.path.join(root, "work", "rw4_LUAD.log"), "w", encoding="utf-8")
err = open(os.path.join(root, "work", "rw4_LUAD.err"), "w", encoding="utf-8")
p = subprocess.Popen(cmd, cwd=root, env=env, stdout=out, stderr=err)
print("started LUAD pid", p.pid, flush=True)
mem = open(os.path.join(root, "work", "rw4_mem.log"), "w", encoding="utf-8")
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
for i in range(60):
    time.sleep(5)
    mem.write(f"t={i*5}s availGB={free_gb():.2f}\n"); mem.flush()
    if p.poll() is not None:
        mem.write(f"EXIT rc={p.returncode}\n"); mem.flush()
        break
mem.close()
rc = p.wait()
out.close(); err.close()
print("done LUAD exit", rc, flush=True)
print("SINGLE_TEST_DONE", flush=True)
