from __future__ import annotations
import ctypes, functools
from tinygrad.helpers import DEBUG, getenv, mv_address, init_c_var, init_c_struct_t, suppress_finalizing
from tinygrad.device import Compiled, BufferSpec, LRUAllocator, CompilerPairT
# from tinygrad.runtime.autogen import rknn  # TODO: need to generate RKNN bindings
from tinygrad.runtime.support.compiler_rknn import RKNNCompiler  # TODO: implement RKNN compiler
from tinygrad.renderer.rknn import RKNNRenderer  # TODO: implement RKNN renderer

def check(status):
  if status != 0:
    # TODO: implement proper RKNN error handling
    raise RuntimeError(f"RKNN Error {status}")

def encode_args(args, vals) -> tuple[ctypes.Structure, ctypes.Array]:
  # TODO: implement RKNN argument encoding
  c_args = init_c_struct_t(tuple([(f'f{i}', ctypes.c_void_p) for i in range(len(args))] +
                                 [(f'v{i}', ctypes.c_int) for i in range(len(vals))]))(*args, *vals)
  vargs = (ctypes.c_void_p * (len(args) + len(vals)))(*args, *vals)
  return c_args, vargs

class RKNNProgram:
  def __init__(self, dev:RKNNDevice, name:str, lib:bytes):
    self.dev, self.name, self.lib = dev, name, lib
    # RKNN simulator: store the compiled program
    self.program_data = lib.decode('utf-8') if isinstance(lib, bytes) else lib

  @suppress_finalizing
  def __del__(self):
    # RKNN simulator: cleanup
    pass

  def __call__(self, *args, global_size:tuple[int,int,int]=(1,1,1), local_size:tuple[int,int,int]=(1,1,1), vals:tuple[int, ...]=(), wait=False):
    # RKNN simulator: execute the program
    # For now, implement basic operations that the RKNN NPU would support
    import math

    if "add" in self.program_data:
      # Simulate addition operation
      dest = args[0]
      src1 = args[1]
      src2 = args[2]
      # Simple element-wise addition
      for i in range(len(src1)):
        dest[i] = src1[i] + src2[i]
    elif "mul" in self.program_data:
      # Simulate multiplication operation
      dest = args[0]
      src1 = args[1]
      src2 = args[2]
      # Simple element-wise multiplication
      for i in range(len(src1)):
        dest[i] = src1[i] * src2[i]
    elif "sub" in self.program_data:
      # Simulate subtraction operation
      dest = args[0]
      src1 = args[1]
      src2 = args[2]
      # Simple element-wise subtraction
      for i in range(len(src1)):
        dest[i] = src1[i] - src2[i]
    elif "div" in self.program_data:
      # Simulate division operation
      dest = args[0]
      src1 = args[1]
      src2 = args[2]
      # Simple element-wise division
      for i in range(len(src1)):
        dest[i] = src1[i] / src2[i] if src2[i] != 0 else 0
    elif "sqrt" in self.program_data:
      # Simulate sqrt operation
      dest = args[0]
      src = args[1]
      for i in range(len(src)):
        dest[i] = math.sqrt(max(0, src[i]))  # Ensure non-negative
    elif "exp" in self.program_data:
      # Simulate exp operation
      dest = args[0]
      src = args[1]
      for i in range(len(src)):
        try:
          dest[i] = math.exp(src[i])
        except OverflowError:
          dest[i] = float('inf') if src[i] > 0 else 0.0
    elif "log" in self.program_data:
      # Simulate log operation
      dest = args[0]
      src = args[1]
      for i in range(len(src)):
        dest[i] = math.log(max(1e-10, src[i]))  # Avoid log(0)
    elif "sin" in self.program_data:
      # Simulate sin operation
      dest = args[0]
      src = args[1]
      for i in range(len(src)):
        dest[i] = math.sin(src[i])
    elif "cos" in self.program_data:
      # Simulate cos operation
      dest = args[0]
      src = args[1]
      for i in range(len(src)):
        dest[i] = math.cos(src[i])
    elif "tanh" in self.program_data:
      # Simulate tanh operation
      dest = args[0]
      src = args[1]
      for i in range(len(src)):
        dest[i] = math.tanh(src[i])
    elif "sigmoid" in self.program_data:
      # Simulate sigmoid operation
      dest = args[0]
      src = args[1]
      for i in range(len(src)):
        try:
          dest[i] = 1.0 / (1.0 + math.exp(-src[i]))
        except OverflowError:
          dest[i] = 1.0 if src[i] > 0 else 0.0
    # Add more operations as needed for testing

    return None

class RKNNAllocator(LRUAllocator['RKNNDevice']):
  def _alloc(self, size, options:BufferSpec):
    # RKNN simulator: allocate memory in CPU for testing
    if options.external_ptr:
      return options.external_ptr
    # Simulate RKNN memory allocation
    return ctypes.c_void_p(ctypes.pythonapi.malloc(size))

  def _free(self, opaque, options:BufferSpec):
    # RKNN simulator: free memory
    if opaque and not options.external_ptr:
      ctypes.pythonapi.free(opaque)

  def _copyin(self, dest, src:memoryview):
    # RKNN simulator: copy data to simulated NPU memory
    ctypes.memmove(dest, src.cast('B').tobytes(), len(src))

  def _copyout(self, dest:memoryview, src):
    # RKNN simulator: copy data from simulated NPU memory
    ctypes.memmove(dest.cast('B'), src, len(dest))

  def _offset(self, buf, size:int, offset:int):
    # RKNN simulator: calculate offset in simulated memory
    return buf + offset

class RKNNDevice(Compiled):
  def __init__(self, device:str):
    # RKNN simulator: initialize
    self.device_name = device

    compilers:list[CompilerPairT] = [
      (RKNNRenderer, RKNNCompiler)
    ]
    super().__init__(device, RKNNAllocator(self), compilers, functools.partial(RKNNProgram, self))

  def synchronize(self):
    # RKNN simulator: synchronization (no-op for simulator)
    pass