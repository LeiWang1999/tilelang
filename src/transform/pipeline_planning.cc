/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file pipeline_planning.cc
 * \brief Plan the software pipeline
 */

#include <tvm/arith/analyzer.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

/*!
 * \brief Check whether two regions have intersections.
 * \param region1 The first region.
 * \param region2 The second region.
 * \return Whether region1 and region2 have intersections.
 */
bool MayConflict(Region region1, Region region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); i++) {
    Range dim1 = region1[i];
    Range dim2 = region2[i];
    auto int_set1 = arith::IntSet::FromRange(dim1);
    auto int_set2 = arith::IntSet::FromRange(dim2);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

/*!
 * \brief Detect if a statement follows the global memory copy pattern:
 *        1. Contains exactly one buffer store operation
 *        2. Source buffer must be in global memory scope
 *        3. Destination buffer must be in local or shared memory scope
 */
class GlobalCopyPatternDetector : public StmtExprVisitor {
public:
  static bool Detect(const Stmt &stmt) {
    GlobalCopyPatternDetector detector;
    detector.VisitStmt(stmt);
    return detector.is_global_copy_pattern_;
  }

private:
  void VisitStmt_(const BufferStoreNode *op) final {
    Buffer store_buffer = op->buffer;
    is_global_read_ = false;
    this->VisitExpr(op->value);
    if (is_global_read_ && (store_buffer.scope() == "shared" ||
                            store_buffer.scope() == "shared.dyn" ||
                            store_buffer.scope() == "local")) {
      is_global_copy_pattern_ = true;
    }
    is_global_read_ = false;
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (op->buffer.scope() == "global") {
      is_global_read_ = true;
    }
  }

  void VisitExpr_(const CallNode *op) final {
    auto args = op->args;
    if (op->op.same_as(tir::builtin::if_then_else())) {
      // Simplify nested if_then_else
      // if (cond) { if (inner_cond) { inner_then_expr } else { inner_else_expr
      // } } else { else_expr }
      // => if (cond && inner_cond) { inner_then_expr } else { else_expr }
      const PrimExpr &cond = op->args[0];
      const PrimExpr &then_expr = op->args[1];
      const PrimExpr &else_expr = op->args[2];
      this->VisitExpr(then_expr);
      this->VisitExpr(else_expr);
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    // Skip condition
    this->VisitStmt(op->then_case);
    if (op->else_case.defined()) {
      this->VisitStmt(op->else_case.value());
    }
  }

private:
  bool is_global_read_ = false;
  bool under_buffer_store_ = false;
  bool is_global_copy_pattern_ = false;
};

class PipelinePlanner : public StmtExprMutator {
public:
  static Stmt Substitute(const PrimFunc &f) {
    PipelinePlanner substituter;
    for (const auto &[_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "Pipeline_Planning: Require the target attribute";
    substituter.target_ = target.value();
    return substituter.VisitStmt(f->body);
  }

private:
  PipelinePlanner() = default;

  struct PipelineStageInfo {
    Array<BufferRegion> reads, writes;
    int original_order;
    int order = -1, stage = -1;
    bool copy_stage = false;
    int last_use_stage = -1;
  };

  PipelineStageInfo MakePipelineStageInfo(Stmt stmt, int idx) {
    Block block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{}, /*name_hint=*/"",
                /*body*/ stmt);
    Array<Array<BufferRegion>> access =
        GetBlockReadWriteRegion(block, buffer_data_to_buffer_);

    PipelineStageInfo pinfo;
    pinfo.reads = std::move(access[0]);
    pinfo.writes = std::move(access[1]);
    pinfo.original_order = idx;
    pinfo.copy_stage = GlobalCopyPatternDetector::Detect(stmt);

    return std::move(pinfo);
  }

  Stmt VisitStmt_(const ForNode *loop) final {
    auto num_stages_anno = loop->annotations.Get("num_stages");
    if (!num_stages_anno.defined())
      return StmtExprMutator::VisitStmt_(loop);
    int num_stages = num_stages_anno.as<IntImmNode>()->value;
    Stmt pipeline_body{nullptr};
    if (const auto *realize = loop->body.as<BlockRealizeNode>()) {
      const auto &block = realize->block;
      for (const auto &buffer : block->alloc_buffers) {
        ICHECK(buffer->IsInstance<BufferNode>());
        buffer_data_to_buffer_.Set(buffer->data, buffer);
      }
      if (const auto *seq_stmt = block->body.as<SeqStmtNode>()) {
        pipeline_body = block->body;
      } else if (const auto *if_then_else = block->body.as<IfThenElseNode>()) {
        // should assert else case is nullptr
        ICHECK(!if_then_else->else_case.defined())
            << "Pipeline_Planning: Can't handle the body of the loop because "
               "it is not a SeqStmt";
        pipeline_body = if_then_else->then_case;
      } else {
        LOG(FATAL) << "Pipeline_Planning: Can't handle the body of the loop "
                      "because it is not a SeqStmt or IfThenElse";
      }
    } else {
      pipeline_body = loop->body;
    }
    const SeqStmtNode *pipeline_body_seq = pipeline_body.as<SeqStmtNode>();
    CHECK(pipeline_body_seq)
        << "ValueError: The body of the software pipeline "
           "should be SeqStmt, got "
        << pipeline_body->GetTypeKey() << " " << pipeline_body;
    CHECK(num_stages >= 1);
    CHECK(loop->kind == ForKind::kSerial);

    std::vector<PipelineStageInfo> pipeline_stage_infos;
    for (size_t i = 0; i < pipeline_body_seq->size(); i++) {
      auto pinfo = MakePipelineStageInfo(pipeline_body_seq->seq[i], i);
      pipeline_stage_infos.push_back(std::move(pinfo));
    }

    // analysis use-def chain
    for (auto &pinfo : pipeline_stage_infos) {
      for (int i = pinfo.original_order + 1;
           i < static_cast<int>(pipeline_body_seq->size()); i++) {
        if (!pinfo.copy_stage)
          continue;
        for (const BufferRegion &read : pipeline_stage_infos[i].reads) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const BufferRegion &r) {
                             return r->buffer == read->buffer &&
                                    MayConflict(r->region, read->region);
                           }) != pinfo.writes.end()) {
            pinfo.last_use_stage = std::max(pinfo.last_use_stage, i);
          }
        }
        for (const BufferRegion &write : pipeline_stage_infos[i].writes) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const BufferRegion &r) {
                             return r->buffer == write->buffer &&
                                    MayConflict(r->region, write->region);
                           }) != pinfo.writes.end()) {
            LOG(FATAL) << "Pipeline planning error: Multiple writes to "
                          "overlapping buffer regions detected. "
                       << "Stage " << pinfo.original_order << " and stage " << i
                       << " are both writing to buffer '" << write->buffer->name
                       << "' with overlapping regions. This is not supported "
                          "in pipeline planning.";
          }
        }
      }
    }

    // Making stages and orders
    int order_idx = 0;
    for (auto &pinfo : pipeline_stage_infos) {
      if (pinfo.copy_stage && pinfo.last_use_stage != -1)
        continue;
      pinfo.order = order_idx++;
      pinfo.stage = num_stages;
      for (auto &pinfo_1 : pipeline_stage_infos) {
        if (pinfo_1.copy_stage &&
            pinfo_1.last_use_stage == pinfo.original_order) {
          pinfo_1.order = order_idx++;
          pinfo_1.stage = 0;
        }
      }
    }
    ICHECK(size_t(order_idx) == pipeline_stage_infos.size())
        << "The number of stages should be equal to the number of pipeline "
           "stages. "
        << "Got " << order_idx << " stages and " << pipeline_stage_infos.size()
        << " pipeline stages.";

    // if all the copy is at the end of the order, we can move these copy to the
    // beginning of the order and shrink the stage offset by 1.
    int copy_stage_at_end = [&]() {
      int copy_stage_cnt = 0;
      int copy_order_min = pipeline_stage_infos.size();
      int non_copy_order_max = 0;
      for (auto &pinfo : pipeline_stage_infos) {
        if (pinfo.copy_stage) {
          copy_stage_cnt++;
          copy_order_min = std::min(copy_order_min, pinfo.order);
        } else {
          non_copy_order_max = std::max(non_copy_order_max, pinfo.order);
        }
      }
      if (copy_order_min > non_copy_order_max)
        return copy_stage_cnt;
      return -1;
    }();
    if (copy_stage_at_end > 0 && num_stages >= 2) {
      for (auto &pinfo : pipeline_stage_infos) { // move copy to the beginning
        pinfo.order =
            (pinfo.order + copy_stage_at_end) % pipeline_stage_infos.size();
        if (!pinfo.copy_stage)
          pinfo.stage--;
      }
    }

    // Finally, make the pipeline annotation
    Map<String, ObjectRef> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages") {
        annotations.Set(key, value);
      }
    }

    std::vector<Integer> orders, stages;
    orders.reserve(pipeline_stage_infos.size());
    stages.reserve(pipeline_stage_infos.size());
    for (auto &pinfo : pipeline_stage_infos) {
      orders.push_back(pinfo.order);
      stages.push_back(pinfo.stage);
    }

    annotations.Set(tir::attr::software_pipeline_stage, Array<Integer>(stages));
    annotations.Set(tir::attr::software_pipeline_order, Array<Integer>(orders));
    if (TargetHasAsyncCopy(target_))
      annotations.Set(tir::attr::software_pipeline_async_stages,
                      Array<Integer>{0});

    return For(loop->loop_var, loop->min, loop->extent, loop->kind, loop->body,
               loop->thread_binding, annotations);
  }

  Stmt VisitStmt_(const BlockNode *op) final {
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    Block block = Downcast<Block>(StmtExprMutator::VisitStmt_(op));
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return std::move(block);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
};

tvm::transform::Pass PipelinePlanning() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = PipelinePlanner::Substitute(f);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PipelinePlanning", {});
}

TVM_REGISTER_GLOBAL("tl.transform.PipelinePlanning")
    .set_body_typed(PipelinePlanning);

} // namespace tl
} // namespace tvm
