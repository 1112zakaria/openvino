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
    // aten::unique_consecutive(
    //   Tensor self,
    //   bool return_inverse=False,
    //   bool return_counts=False,
    //   int? dim=None
    // ) -> (Tensor, Tensor, Tensor)
    num_inputs_check(context, 1, 4);

    auto input = context.get_input(0);

    // ---- flags ----
    bool return_inverse = false;
    if (!context.input_is_none(1)) {
        return_inverse = context.const_input<bool>(1);
    }

    bool return_counts = false;
    if (!context.input_is_none(2)) {
        return_counts = context.const_input<bool>(2);
    }

    bool dim_is_none = context.input_is_none(3);
    int64_t dim_attr = 0;
    if (!dim_is_none) {
        dim_attr = context.const_input<int64_t>(3);
    }

    // ---- Step 1: choose axis & prepare input ----
    Output<Node> prepared_input;
    int64_t axis_index = 0;

    if (dim_is_none) {
        // Flatten and use axis 0
        auto flatten_shape = context.mark_node(
            v0::Constant::create(element::i64, Shape{1}, {-1}));
        prepared_input = context.mark_node(
            std::make_shared<v1::Reshape>(input, flatten_shape, false));
        axis_index = 0;
    } else {
        prepared_input = input;
        axis_index = dim_attr;

        // Normalize negative dim if we know rank statically
        if (dim_attr < 0) {
            auto pshape = input.get_partial_shape();
            if (pshape.rank().is_static()) {
                auto rank = pshape.rank().get_length();
                axis_index = dim_attr + rank;
            }
        }
    }

    auto axis_const = context.mark_node(
        v0::Constant::create(element::i64, Shape{}, {axis_index}));

    // Common scalars
    auto one_scalar = context.mark_node(
        v0::Constant::create(element::i64, Shape{}, {1}));
    auto zero_scalar = context.mark_node(
        v0::Constant::create(element::i64, Shape{}, {0}));

    // ---- Step 2: compare neighbors along axis ----
    auto prepared_shape = context.mark_node(
        std::make_shared<v0::ShapeOf>(prepared_input));  // i64[rank]

    auto axis0_1d = context.mark_node(
        v0::Constant::create(element::i64, Shape{1}, {0}));

    auto axis_index_scalar = context.mark_node(
        v0::Constant::create(element::i64, Shape{}, {axis_index}));
    auto axis_index_vec = context.mark_node(
        v0::Constant::create(element::i64, Shape{1}, {axis_index}));

    // axis_len_scalar = shape[axis_index] -> scalar
    auto axis_len_scalar = context.mark_node(std::make_shared<v8::Gather>(
        prepared_shape,
        axis_index_scalar,
        zero_scalar /* axis = 0 */));

    auto axis_len_minus_one = context.mark_node(
        std::make_shared<v1::Subtract>(axis_len_scalar, one_scalar));

    // 1D start/stop/step/axes vectors for Slice
    auto start_head = context.mark_node(
        v0::Constant::create(element::i64, Shape{1}, {0}));
    auto start_tail = context.mark_node(
        v0::Constant::create(element::i64, Shape{1}, {1}));

    auto stop_head = context.mark_node(
        std::make_shared<v0::Unsqueeze>(axis_len_minus_one, axis0_1d));  // [len-1]
    auto stop_tail = context.mark_node(
        std::make_shared<v0::Unsqueeze>(axis_len_scalar, axis0_1d));     // [len]

    auto step_vec = context.mark_node(
        v0::Constant::create(element::i64, Shape{1}, {1}));

    auto axes_vec = axis_index_vec;  // [axis_index]

    // head: slice [0 : len-1] along axis_index
    auto head = context.mark_node(std::make_shared<v8::Slice>(
        prepared_input, start_head, stop_head, step_vec, axes_vec));
    // tail: slice [1 : len] along axis_index
    auto tail = context.mark_node(std::make_shared<v8::Slice>(
        prepared_input, start_tail, stop_tail, step_vec, axes_vec));

    // elementwise equality of neighbor slices
    auto equal = context.mark_node(
        std::make_shared<v1::Equal>(head, tail));

    // ---- Step 3: keep mask of run starts ----
    // change = True where neighbor values differ
    auto change = context.mark_node(
        std::make_shared<v1::LogicalNot>(equal));

    // First element always starts a run.
    // Build a true-prefix with the same shape as prepared_input sliced at index 0.
    auto idx0_scalar = context.mark_node(
        v0::Constant::create(element::i64, Shape{}, {0}));

    auto first_elem = context.mark_node(std::make_shared<v8::Gather>(
        prepared_input, idx0_scalar, axis_const));

    auto true_prefix = context.mark_node(
        std::make_shared<v1::Equal>(first_elem, first_elem));  // all True, same shape as first_elem

    // Concat prefix and change along axis_index → mask of run starts
    auto keep = context.mark_node(std::make_shared<v0::Concat>(
        OutputVector{true_prefix, change}, axis_index));

    // ---- Step 4: run start indices and values ----
    // NonZero(keep) gives coordinates of True entries.
    auto nonzero = context.mark_node(
        std::make_shared<v3::NonZero>(keep));  // shape [rank, N]

    // Extract row corresponding to axis_index -> 1D tensor of start indices
    auto axis_row_idx = context.mark_node(
        v0::Constant::create(element::i64, Shape{}, {axis_index}));

    auto nonzero_axis = context.mark_node(std::make_shared<v8::Gather>(
        nonzero, axis_row_idx, zero_scalar));  // 1D [N]

    // Gather values at those positions along axis_const
    auto values = context.mark_node(std::make_shared<v8::Gather>(
        prepared_input, nonzero_axis, axis_const));

    Output<Node> counts_node;
    bool have_counts = false;
    Output<Node> inverse_node;
    bool have_inverse = false;

    // ---- Step 5: counts (optional) ----
    if (return_counts) {
        // Append sentinel = axis length, then diff successive elements.
        auto concat_axis0 = axis0_1d;  // [0]

        auto axis_len_1d = context.mark_node(
            std::make_shared<v0::Unsqueeze>(axis_len_scalar, concat_axis0));  // [1]

        auto starts_with_sentinel = context.mark_node(
            std::make_shared<v0::Concat>(
                OutputVector{nonzero_axis, axis_len_1d}, 0));  // 1D [num_runs + 1]

        auto starts_shape = context.mark_node(
            std::make_shared<v0::ShapeOf>(starts_with_sentinel));  // [1]

        auto size_scalar = context.mark_node(std::make_shared<v8::Gather>(
            starts_shape, zero_scalar, zero_scalar));  // scalar L

        auto zero_1d = context.mark_node(
            v0::Constant::create(element::i64, Shape{1}, {0}));
        auto one_1d = context.mark_node(
            v0::Constant::create(element::i64, Shape{1}, {1}));

        auto size_minus_one = context.mark_node(
            std::make_shared<v1::Subtract>(size_scalar, one_scalar));  // L-1

        auto head_stop_1d = context.mark_node(
            std::make_shared<v0::Unsqueeze>(size_minus_one, concat_axis0));  // [L-1]
        auto tail_stop_1d = context.mark_node(
            std::make_shared<v0::Unsqueeze>(size_scalar, concat_axis0));     // [L]

        auto head_indices = context.mark_node(std::make_shared<v8::Slice>(
            starts_with_sentinel, zero_1d, head_stop_1d, one_1d));
        auto tail_indices = context.mark_node(std::make_shared<v8::Slice>(
            starts_with_sentinel, one_1d, tail_stop_1d, one_1d));

        counts_node = context.mark_node(
            std::make_shared<v1::Subtract>(tail_indices, head_indices));  // 1D [num_runs]
        have_counts = true;
    }

    // ---- Step 6: inverse indices (optional) ----
    if (return_inverse) {
        // Convert keep (bool) → int64 and CumSum along axis.
        auto keep_int = context.mark_node(
            std::make_shared<v0::Convert>(keep, element::i64));

        auto cumsum = context.mark_node(
            std::make_shared<v0::CumSum>(keep_int, axis_const, false, false));

        // Make run ids 0-based: [1,1,2,2,...] → [0,0,1,1,...]
        auto inverse_pre = context.mark_node(
            std::make_shared<v1::Subtract>(cumsum, one_scalar));

        if (dim_is_none) {
            // If we flattened, reshape back to original input shape
            auto orig_shape = context.mark_node(
                std::make_shared<v0::ShapeOf>(input));
            inverse_node = context.mark_node(
                std::make_shared<v1::Reshape>(inverse_pre, orig_shape, false));
        } else {
            inverse_node = inverse_pre;
        }
        have_inverse = true;
    }

    // ---- Assemble outputs in PyTorch order ----
    // (values, [inverse], [counts])
    OutputVector outputs;
    outputs.push_back(values);
    if (have_inverse) {
        outputs.push_back(inverse_node);
    }
    if (have_counts) {
        outputs.push_back(counts_node);
    }

    return outputs;
}

// ...existing code...
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov