import sys
print("step1", flush=True)
import ncnn
print("step2 import ok", flush=True)
import numpy as np
print("step3 numpy ok", flush=True)
net = ncnn.Net()
print("step4 net ok", flush=True)
net.load_param("runs/jump_pose-2/weights/best_ncnn_model/model.ncnn.param")
print("step5 param ok", flush=True)
net.load_model("runs/jump_pose-2/weights/best_ncnn_model/model.ncnn.bin")
print("step6 bin ok", flush=True)
x = np.zeros((640, 640, 3), np.float32)
try:
    m = ncnn.Mat(x)
    print("step7 Mat(HWC) ok dims", m.dims, m.w, m.h, m.c, flush=True)
except Exception as e:
    print("step7 ERR", e, flush=True)
try:
    m2 = ncnn.Mat(np.zeros((3, 640, 640), np.float32))
    print("step8 Mat(CHW) ok dims", m2.dims, m2.w, m2.h, m2.c, flush=True)
except Exception as e:
    print("step8 ERR", e, flush=True)
