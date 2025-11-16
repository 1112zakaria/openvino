import numpy as np
from openvino import Model, Core
from openvino import opset13   # or opset10/opset8 depending on your version


def build_unique_consecutive_graph_1d(length: int = 8) -> Model:
    """
    Build a simple OpenVINO model that *looks like* unique_consecutive:
      input:  1D int64 tensor 'x' of shape [length]
      outputs:
        - values           : same as x
        - inverse_indices  : [0, 1, 2, ..., length-1]
        - counts           : all ones (shape [length])

    This is NOT a real unique_consecutive implementation.
    It’s just a working example of building an ov.Model with OV ops.
    """

    # 1) Parameter: 1D input of fixed length
    x = opset13.parameter(shape=[length], dtype=np.int64, name="x")

    # --- "values" output ---
    # For now, just pass x through
    values = x

    # --- "inverse_indices" output ---
    # inverse_indices = range(0, length)
    zero = opset13.constant(0, dtype=np.int64)
    one  = opset13.constant(1, dtype=np.int64)
    stop = opset13.constant(length, dtype=np.int64)
    # inverse_indices = opset13.range(zero, stop, one, np.int64)  # shape [length]

    # --- "counts" output ---
    # counts = ones(length)
    # First make a scalar '1'
    one_scalar = opset13.constant(1, dtype=np.int64)
    # Target shape [length]
    target_shape = opset13.constant([length], dtype=np.int64)
    counts = opset13.broadcast(one_scalar, target_shape)         # shape [length]

    # 2) Build and return the Model
    model = Model(
        [values, counts],
        [x],
        "unique_consecutive_debug_1d",
    )
    return model

if __name__ == "__main__":
    core = Core()
    model = build_unique_consecutive_graph_1d(length=8)

    # Compile for CPU
    compiled = core.compile_model(model, "CPU")

    # Example input
    x_input = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)

    # Run inference (using input name "x")
    results = compiled({"x": x_input})

    # Grab outputs in order
    outputs = list(results.values())
    values, counts = outputs

    print("Input x:           ", x_input)
    print("values:            ", values)
    # print("inverse_indices:   ", inverse_indices)
    print("counts:            ", counts)
