from rknn.api import RKNN

ONNX = '/home/seaway/sdb/ljl/Spm_llama/transmodel/vision_448_op13.onnx'
OUT  = './vision_448.rknn'

rknn = RKNN(verbose=True)
rknn.config(
    target_platform='rk3588',
    float_dtype='float16',
)
assert rknn.load_onnx(model=ONNX) == 0
assert rknn.build(do_quantization=False) == 0
assert rknn.export_rknn(OUT) == 0
rknn.release()
print('exported', OUT)
