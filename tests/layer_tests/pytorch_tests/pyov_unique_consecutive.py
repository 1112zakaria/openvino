import numpy as np
from openvino import Model, Core
from openvino import opset13, opset8   # or opset10/opset8 depending on your version



def build_unique_consecutive_model(
    shape,                     # e.g. [ -1 ] for 1-D or [B, N] for 2-D (rank must be known)
    dim=None,                  # None, or int (can be negative)
    return_inverse=False,
    return_counts=False,
    dtype=np.int64,
    model_name="unique_consecutive_py"
) -> Model:
    """
    Builds an OpenVINO graph that performs torch.unique_consecutive along axis `dim`.
    If dim=None, the input is flattened and axis=0 is used.
    Returns an opset8.Model with outputs: values[, inverse][, counts].
    """

    # ---- Parameter (rank must be known; -1 is OK for dynamic sizes) ----
    x = opset8.parameter(shape=shape, dtype=dtype, name="x")
    rank = len(shape)  # fixed rank at build-time (needed for python int axes)

    # ---- Axis normalization (python int) ----
    if dim is None:
        axis_index = 0
        # Flatten to 1-D
        prepared = opset8.reshape(x, opset8.constant(np.array([-1], np.int64)), False)
        flatten_back_shape = opset8.shape_of(x)  # used to reshape inverse back
        flattened = True
    else:
        axis_index = dim if dim >= 0 else (dim + rank)
        assert 0 <= axis_index < rank, f"dim={dim} not valid for rank={rank}"
        prepared = x
        flatten_back_shape = None
        flattened = False

    # Tensor axis (for ops that expect tensor scalars)
    axis_const = opset8.constant(np.int64(axis_index))

    # ---- Shapes / small constants ----
    prep_shape   = opset8.shape_of(prepared)                     # i64[rank]
    axis_idx_vec = opset8.unsqueeze(axis_const, opset8.constant(0))  # i64[1] = {axis}
    zero_1d      = opset8.constant(np.array([0], np.int64))
    one_1d       = opset8.constant(np.array([1], np.int64))
    zero_s       = opset8.constant(np.int64(0))
    one_s        = opset8.constant(np.int64(1))
    print(one_s)

    # ---- Axis length (scalar) and 1D forms for Slice ----
    axis_len     = opset8.gather(prep_shape, axis_idx_vec, 0)    # scalar
    print(axis_len.get_output_shape(0))
    axis_len_m1  = opset8.subtract(axis_len, one_s)              # scalar
    print(axis_len_m1.get_output_shape(0))
    axis_len_1d  = opset8.unsqueeze(axis_len,     opset8.constant(0))  # {len}
    axis_m1_1d   = opset8.unsqueeze(axis_len_m1,  opset8.constant(0))  # {len-1}

    print(axis_len_1d.get_output_shape(0))
    print(axis_m1_1d.get_output_shape(0))

    # ---- Neighbor slices along axis: head = [0:len-1], tail = [1:len] ----
    head = opset8.slice(prepared, zero_1d, axis_m1_1d, one_1d, axis_idx_vec)
    tail = opset8.slice(prepared, one_1d,  axis_len_1d, one_1d, axis_idx_vec)

    # ---- Run boundaries: change = NOT (head == tail), marks starts at i+1 ----
    eq     = opset8.equal(head, tail)
    change = opset8.logical_not(eq)

    # ---- Build prefix of True with length 1 along axis (same rank as change) ----
    # first_slice = prepared[0:1] along axis; true_prefix = (first_slice == first_slice)
    first_slice  = opset8.slice(prepared, zero_1d, one_1d, one_1d, axis_idx_vec)
    true_prefix  = opset8.equal(first_slice, first_slice)  # boolean of same shape as first_slice

    # keep = Concat([true_prefix, change], axis=axis_index)  (axis must be python int)
    keep = opset8.concat([true_prefix, change], axis_index)

    # ---- Start indices and values ----
    nz       = opset8.non_zero(keep)                        # i64[rank, N]
    nz_axis  = opset8.gather(nz, opset8.constant(np.int64(axis_index)), 0)  # i64[N], row for our axis
    values   = opset8.gather(prepared, nz_axis, axis_index) # unique run values along axis

    outputs = [values]

    # ---- Counts (optional) ----
    if return_counts or return_inverse:
        starts_plus = opset8.concat([nz_axis, axis_len_1d], 0)   # i64[N+1]
        L           = opset8.gather(opset8.shape_of(starts_plus), opset8.constant(0), 0)  # scalar (N+1)
        Lm1         = opset8.subtract(L, one_s)                  # scalar
        L_1d        = opset8.unsqueeze(L,   opset8.constant(0))      # {N+1}
        Lm1_1d      = opset8.unsqueeze(Lm1, opset8.constant(0))      # {N}

        head_idx = opset8.slice(starts_plus, zero_1d, Lm1_1d, one_1d)  # i64[N]
        tail_idx = opset8.slice(starts_plus, one_1d,  L_1d,   one_1d)  # i64[N]
        counts   = opset8.subtract(tail_idx, head_idx)                 # i64[N]

        if return_counts:
            outputs.append(counts)

    # ---- Inverse indices (optional) ----
    if return_inverse:
        keep_i64   = opset8.convert(keep, np.int64)                                # i64
        cumsum     = opset8.cum_sum(keep_i64, axis_const, False, False)            # labels 1..K
        inverse_pre= opset8.subtract(cumsum, one_s)                                # 0..K-1

        if flattened:
            inverse = opset8.reshape(inverse_pre, flatten_back_shape, False)
        else:
            inverse = inverse_pre

        # If counts already appended, order in PyTorch is (values, inverse, counts).
        # If not, just append inverse now.
        if return_counts:
            outputs.insert(1, inverse)
        else:
            outputs.append(inverse)

    return Model(outputs, [x], model_name)


# --- quick sanity run ---
if __name__ == "__main__":
    core = Core()

    # 1-D example: dim=None (flatten)
    m1 = build_unique_consecutive_model(shape=[-1], dim=None, return_inverse=True, return_counts=True)
    exec1 = core.compile_model(m1, "CPU")
    x1 = np.array([1,1,2,2,2,1], np.int64)
    outs = exec1([x1])
    print("\n[1-D, dim=None]")
    for i, o in enumerate(outs):
        print(f"out[{i}] -> shape={o.shape}, dtype={o.dtype}, data={o}")

    # 2-D example: along axis=1 (dim=1)
    m2 = build_unique_consecutive_model(shape=[-1, -1], dim=1, return_inverse=True, return_counts=True)
    exec2 = core.compile_model(m2, "CPU")
    x2 = np.array([[1,1,2,2,2,1],
                   [5,5,5,9,9,9]], np.int64)
    outs2 = exec2([x2])
    print("\n[2-D, dim=1]")
    for i, o in enumerate(outs2):
        print(f"out[{i}] -> shape={o.shape}, dtype={o.dtype}, data={o}")