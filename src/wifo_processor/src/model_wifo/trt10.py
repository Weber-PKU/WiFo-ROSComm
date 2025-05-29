# model_wifo/trt10.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401
import numpy as np

# Keep the main thread's CUDA context
__CUDA_CTX = pycuda.autoinit.context

# Globals to be initialized in model_init()
_context = None
_bindings = None
_h_input = None
_h_output = None
_d_input = None
_d_output = None
_stream = None  # will hold the single Stream

def model_init(engine_path: str, input_shape: tuple):
    """
    Load TensorRT engine, create execution context, configure dynamic input shape,
    allocate host/device memory buffers, and create a single CUDA stream.
    """
    global _context, _bindings, _h_input, _h_output, _d_input, _d_output, _stream

    # 1) Deserialize the engine from file
    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine  = runtime.deserialize_cuda_engine(f.read())

    # 2) Create execution context and set dynamic input shape
    _context = engine.create_execution_context()
    input_name = engine.get_tensor_name(0)
    _context.set_input_shape(input_name, input_shape)

    # 3) Get actual buffer shapes and sizes
    in_shape  = tuple(_context.get_tensor_shape(input_name))
    out_name  = engine.get_tensor_name(1)
    out_shape = tuple(_context.get_tensor_shape(out_name))
    in_size   = int(np.prod(in_shape))
    out_size  = int(np.prod(out_shape))

    # 4) Allocate pagelocked host and device buffers
    _h_input  = cuda.pagelocked_empty(in_size,  dtype=np.float32)
    _h_output = cuda.pagelocked_empty(out_size, dtype=np.float32)
    _d_input  = cuda.mem_alloc(_h_input.nbytes)
    _d_output = cuda.mem_alloc(_h_output.nbytes)

    # 5) Prepare the bindings list in order [input, output]
    _bindings = [int(_d_input), int(_d_output)]

    # 6) Create exactly one CUDA Stream in the main thread
    _stream = cuda.Stream()

def model_infer(input_array: np.ndarray) -> np.ndarray:
    """
    Perform asynchronous inference using the initialized context and single Stream.
    Args:
      input_array: 1D float32 array matching the size of the host input buffer.
    Returns:
      A copy of the inference result as a NumPy array.
    """
    global _stream

    # 1) Copy user input to the host buffer
    assert input_array.size == _h_input.size, \
        f"Expected input size {_h_input.size}, got {input_array.size}"
    _h_input[:] = input_array.astype(np.float32).ravel()

    # 2) Activate the main-thread CUDA context
    __CUDA_CTX.push()
    try:
        # 3) Host -> Device transfer
        cuda.memcpy_htod_async(_d_input, _h_input, _stream)

        # 4) Bind device pointers to tensor names
        for idx in range(_context.engine.num_io_tensors):
            name = _context.engine.get_tensor_name(idx)
            _context.set_tensor_address(name, _bindings[idx])

        # 5) Launch inference asynchronously
        _context.execute_async_v3(stream_handle=_stream.handle)

        # 6) Device -> Host transfer and sync
        cuda.memcpy_dtoh_async(_h_output, _d_output, _stream)
        _stream.synchronize()
    finally:
        # 7) Pop the context so other threads stay unaffected
        __CUDA_CTX.pop()

    # 8) Return a copy of the result
    return _h_output.copy()

def main():
    # One-time initialization
    model_init()

    # Generate one random input template
    template = np.random.rand(_h_input.size).astype(np.float32)

    latencies = []
    # NUM_RUNS = 200
    # for _ in range(NUM_RUNS):
    #     start = time.time()
    #     _ = model_infer(template)
    #     latencies.append((time.time() - start) * 1000)

    # print(f"[✅] Performed {NUM_RUNS} runs, "
    #       f"average latency: {np.mean(latencies):.2f} ms")

if __name__ == "__main__":
    main()
