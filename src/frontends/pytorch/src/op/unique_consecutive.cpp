// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// #include "openvino/op/unique_consecutive.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

// Add missing op headers
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/logical_not.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
// #include "openvino/op/shapeof.hpp" // sometimes duplicate; harmless if present
#include "openvino/op/reduce_prod.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;

// ...existing code...
OutputVector translate_unique_consecutive(const NodeContext& context) {
    // aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int dim=None) ->
    // (Tensor output, Tensor inverse_indices, Tensor counts)
    num_inputs_check(context, 1, 4);

    auto input = context.get_input(0);

    bool return_inverse = false;
    if (!context.input_is_none(1)) {
        return_inverse = context.const_input<bool>(1);
    }

    bool return_counts = false;
    if (!context.input_is_none(2)) {
        return_counts = context.const_input<bool>(2);
    }

    int64_t dim = -1;
    bool dim_is_none = context.input_is_none(3);
    if (!dim_is_none) {
        dim = context.const_input<int64_t>(3);
    }

    OutputVector outputs;
    Output<Node> prepared_input;
    Output<Node> axis_const;

    // Step 1: Choose the axis and prepare input
    if (dim_is_none) {
        // If dim is None, flatten the input tensor first
        auto flatten_shape = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
        prepared_input = context.mark_node(std::make_shared<v1::Reshape>(input, flatten_shape, false));
        // Use axis 0 for flattened tensor
        axis_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    } else {
        // Use input as-is with specified dimension
        prepared_input = input;
        // Normalize negative dim if desired — here keep as constant (negative dims are supported elsewhere)
        axis_const = context.mark_node(v0::Constant::create(element::i64, Shape{}, {dim}));
    }

    // Step 2: Compare neighbors along the chosen axis
    auto one_scalar = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));

    // Get shape of prepared_input -> [rank]
    auto prepared_shape = context.mark_node(std::make_shared<v0::ShapeOf>(prepared_input));  // i64[rank]

    // axis_index we use as compile-time int where available
    int64_t axis_index = dim_is_none ? 0 : dim;

    // 1D constant to use for Unsqueeze and for gather indices
    auto axis0 = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    auto axis_index_vec = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {axis_index}));

    // axis_len_scalar = prepared_shape[axis_index]  (scalar)
    auto axis_len_scalar = context.mark_node(std::make_shared<v8::Gather>(
        prepared_shape,
        axis_index_vec,                                                      // indices
        context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}))  // axis=0
        ));

    auto axis_len_minus_one = context.mark_node(std::make_shared<v1::Subtract>(axis_len_scalar, one_scalar));

    // Build 1D start/stop/step vectors for Slice along axis_index
    auto start_head = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
    auto start_tail = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
    auto stop_head = context.mark_node(std::make_shared<v0::Unsqueeze>(axis_len_minus_one, axis0));  // [len-1]
    auto stop_tail = context.mark_node(std::make_shared<v0::Unsqueeze>(axis_len_scalar, axis0));     // [len]
    auto step_vec = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
    auto axes_vec = axis_index_vec;

    auto head = context.mark_node(std::make_shared<v8::Slice>(prepared_input, start_head, stop_head, step_vec, axes_vec));
    auto tail = context.mark_node(std::make_shared<v8::Slice>(prepared_input, start_tail, stop_tail, step_vec, axes_vec));

    // elementwise equality of neighbor slices (may be multi-dim slices if axis != last)
    auto equal = context.mark_node(std::make_shared<v1::Equal>(head, tail));

    // Step 3 - Build a keep mask of run starts
    // change = not(equal) -> True where element i != i+1
    auto change = context.mark_node(std::make_shared<v1::LogicalNot>(equal));

    // Prepend True for the first element. Need a prefix shaped like `change` with size 1 on axis_index.
    // Build a true-prefix by gathering the first element along axis and comparing it to itself to get correct shape.
    auto idx0 = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto first_elem = context.mark_node(std::make_shared<v8::Gather>(prepared_input, idx0, axis_const)); // may need unsqueeze depending on Gather semantics
    // Make an all-true tensor with same shape as first_elem by equal(first_elem, first_elem)
    auto true_prefix = context.mark_node(std::make_shared<v1::Equal>(first_elem, first_elem));

    // Concat true_prefix and change along the axis_index to get keep
    auto keep = context.mark_node(std::make_shared<v0::Concat>(OutputVector{true_prefix, change}, static_cast<int64_t>(axis_index)));

    // Step 4 - Get run start indices and the values output
    auto nonzero = context.mark_node(std::make_shared<v3::NonZero>(keep));
    auto axis_row_idx = context.mark_node(v0::Constant::create(element::i64, Shape{}, {axis_index}));
    auto nonzero_axis = context.mark_node(std::make_shared<v8::Gather>(nonzero, axis_row_idx, context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}))));

    // Gather values along axis_const at positions nonzero_axis
    auto values = context.mark_node(std::make_shared<v8::Gather>(prepared_input, nonzero_axis, axis_const));
    outputs.push_back(values);

    // Step 5 - Compute counts (optional)
    if (return_counts || return_inverse) {
        // concat axis constant for Unsqueeze (1D)
        auto concat_axis0 = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));

        // make axis_len 1-D so we can concat with nonzero_axis
        auto axis_len_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(axis_len_scalar, concat_axis0)); // shape{1}

        // concat starts + sentinel (1-D)
        auto starts_with_sentinel = context.mark_node(std::make_shared<v0::Concat>(OutputVector{nonzero_axis, axis_len_1d}, 0)); // 1-D

        // compute size L = len(starts_with_sentinel)
        auto starts_shape = context.mark_node(std::make_shared<v0::ShapeOf>(starts_with_sentinel)); // [1]
        auto size_scalar = context.mark_node(std::make_shared<v8::Gather>(
            starts_shape,
            context.mark_node(v0::Constant::create(element::i64, Shape{}, {0})),
            context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}))
            )); // scalar

        // slice indices for head = starts_with_sentinel[0 : L-1], tail = starts_with_sentinel[1 : L]
        auto zero_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {0}));
        auto one_1d = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {1}));
        auto size_minus_one = context.mark_node(std::make_shared<v1::Subtract>(size_scalar, one_scalar)); // scalar

        auto head_stop_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(size_minus_one, concat_axis0)); // {L-1}
        auto tail_stop_1d = context.mark_node(std::make_shared<v0::Unsqueeze>(size_scalar, concat_axis0));       // {L}

        auto head_indices = context.mark_node(std::make_shared<v8::Slice>(starts_with_sentinel, zero_1d, head_stop_1d, one_1d));
        auto tail_indices = context.mark_node(std::make_shared<v8::Slice>(starts_with_sentinel, one_1d, tail_stop_1d, one_1d));

        auto counts = context.mark_node(std::make_shared<v1::Subtract>(tail_indices, head_indices));

        if (return_counts) {
            outputs.push_back(counts);
        }

        // Step 6 - Compute inverse indices (optional)
        if (return_inverse) {
            // convert keep (bool) -> integer so we can CumSum
            auto keep_int = context.mark_node(std::make_shared<v0::Convert>(keep, element::i64));

            // CumSum along the chosen axis (inclusive). Produces per-position run labels
            auto cumsum = context.mark_node(std::make_shared<v0::CumSum>(keep_int, axis_const, /*exclusive*/ false, /*reverse*/ false));

            // Subtract 1 to make run ids 0-based
            auto inverse_pre = context.mark_node(std::make_shared<v1::Subtract>(cumsum, one_scalar));

            // If we flattened the input (dim_is_none), reshape inverse back to original input shape
            Output<Node> inverse;
            if (dim_is_none) {
                auto orig_shape = context.mark_node(std::make_shared<v0::ShapeOf>(input)); // original input shape
                inverse = context.mark_node(std::make_shared<v1::Reshape>(inverse_pre, orig_shape, false));
            } else {
                // inverse_pre already matches prepared_input shape
                inverse = inverse_pre;
            }

            outputs.push_back(inverse);
        }
    }

    return outputs;
}
// ...existing code...
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov