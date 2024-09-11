# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import paddle.distributed as dist
from paddle.distributed.auto_parallel.static.process_group import (
    new_process_group,
)
import paddle
from paddle.distributed.fleet.meta_optimizers.common import (
    OpRole,
)

from ..auto_parallel.static.utils import (
    get_logger,
)
from .pass_base import PassBase, register_pass

logger = get_logger(logging.INFO)


def check_a_b(opera, name_a, name_b):
    if opera.name() != name_a:
        return False
    if len(opera.results()) == 0:
        return False
    res = opera.result(0)
    if len(res.all_used_ops()) >= 1 and res.all_used_ops()[0].name() == name_b:
        return True
    return False


@register_pass("auto_parallel_c_embedding_pass")
class AutoParallelCEmbeddingPass(PassBase):
    def __init__(self):
        super().__init__()

    def _check_self(self):
        return True

    def _check_conflict(self, other_pass):
        return True

    def _apply_single_impl(self, main_program, startup_program, context):
        print("### lzx ### _apply_single_impl")
        block = main_program.global_block()
        startup_block = startup_program.global_block()
        ops = block.ops

        for i, op in enumerate(ops):

            if op.name() == 'builtin.parameter':
                res = op.results()[0]
                print("### lzx ### dir(op)",dir(op))
                for t in res.all_used_ops():
                    if t.name()=='pd_op.embedding':
                        print("&&& lzx &&& get builtin.parameter")
                        print(dir(op.results()[0]))
                        placements = op.results()[0].placements
                        print(placements)
                        dim_map, partial_status = dist.auto_parallel.placement_type.to_dim_map(
                            placements, len(placements)
                        )
                        dim_map = [1,-1] # 先固定修改
                        print("dim_map:",dim_map,"partial_status",partial_status)
                        dist_attr = (
                            paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                                op.results()[0].process_mesh, dim_map, partial_status
                            )
                        )
                        # # print("dist_attr:",dist_attr)
                        dist_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                            op.results()[0].type(), dist_attr
                        )
                        op.results()[0].set_type(dist_type)
                        op.dist_attr = (
                            paddle.base.libpaddle.pir.create_op_dist_attribute(
                                op.results()[0].process_mesh, [], [dist_attr]
                            )
                        )
                        print("over dist_attr")

            elif op.name() == 'pd_op.data':
                res = op.results()[0]
                for t in res.all_used_ops():
                    if t.name()=='pd_op.embedding':
                        print("&&& lzx &&& get builtin.parameter")
                        print(dir(op.results()[0]))
                        placements = op.results()[0].placements
                        print(placements)
                        dim_map, partial_status = dist.auto_parallel.placement_type.to_dim_map(
                            placements, len(placements)
                        )
                        dim_map = [-1,1]
                        print("dim_map:",dim_map,"partial_status",partial_status)
                        dist_attr = (
                            paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                                op.results()[0].process_mesh, dim_map, partial_status
                            )
                        )
                        # # print("dist_attr:",dist_attr)
                        dist_type = paddle.base.libpaddle.pir.cvt_to_dist_type(
                            op.results()[0].type(), dist_attr
                        )
                        op.results()[0].set_type(dist_type)
                        op.dist_attr = (
                            paddle.base.libpaddle.pir.create_op_dist_attribute(
                                op.results()[0].process_mesh, [], [dist_attr]
                            )
                        )
                        print("over dist_attr")
            elif op.name() == 'pd_op.embedding':
                # placements2 = op.results()[0].placements
                # print("res placements test:",placements2)
                # dim_map2, partial_status_good = dist.auto_parallel.placement_type.to_dim_map(
                #     placements2, len(placements2)
                # )
                # dim_map2 = [-1,-1,-1] # 修改输入的切分
                # print("dim_map test :",dim_map2,"partial_status_good",partial_status_good)


                paddle.pir.set_insertion_point(op)
                # results = op.dist_attr.results()

                t_op = paddle._C_ops.c_embedding(
                    op.operand(1).source(), op.operand(0).source(), 0, -1
                )
           
                t_op.get_defining_op().op_role = int(OpRole.Optimize)
                new_op = t_op.get_defining_op()
                # 删除op
                op.result(0).replace_all_uses_with(t_op)
                op.erase()
    

                # aalreduce

                # paddle.pir.set_insertion_point(op)
                # src_mesh = new_op.results()[0].process_mesh
                # group = new_process_group(sorted(src_mesh.process_ids))
                # comm_op_t = paddle._C_ops.c_allreduce_sum(
                #     new_op.results()[0], group.id, True, False
                # )
                # comm_op_t.get_defining_op().op_role = int(OpRole.Optimize)
                # comm_op = comm_op_t.get_defining_op()
                # op.result(0).replace_all_uses_with(comm_op_t)
                # op.erase()

                # placements3 = comm_op.results()[0].placements
                # print("res placements3:",placements3)
                # dim_map3, partial_status3 = dist.auto_parallel.placement_type.to_dim_map(
                #     placements3, len(placements3)
                # )
                # dim_map3 = [-1,-1,-1] # 修改输入的切分
                # # partial_on_dims = comm_op.dist_attr.operand(0).as_tensor_dist_attr().dims_mappings()
                # print("dim_map3:",dim_map3,"partial_status3",partial_status3)
                # # print("partial_on_dims:",partial_on_dims)
                # dist_attr3 = (
                #     paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                #         # comm_op.results()[0].process_mesh, dim_map2, [partial_on_dims]
                #         comm_op.results()[0].process_mesh, dim_map3, partial_status3
                #     )
                # )
                # print("dist_attr3:",dist_attr3)
                # dist_type3 = paddle.base.libpaddle.pir.cvt_to_dist_type(
                #     comm_op.results()[0].type(), dist_attr3
                # )

                placements2 = new_op.results()[0].placements
                print("res placements2:",placements2)
                dim_map2, partial_status2 = dist.auto_parallel.placement_type.to_dim_map(
                    placements2, len(placements2)
                )
                dim_map2 = [-1,-1,-1] # 修改输入的切分
                # partial_on_dims = new_op.dist_attr.operand(0).as_tensor_dist_attr().dims_mappings()
                print("dim_map2:",dim_map2,"partial_status2",partial_status2)
                # print("partial_on_dims:",partial_on_dims)
                dist_attr2 = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        # new_op.results()[0].process_mesh, dim_map2, [partial_on_dims]
                        # new_op.results()[0].process_mesh, dim_map2, partial_status_good
                        new_op.results()[0].process_mesh, dim_map2, {}
                    )
                )
                print("dist_attr2:",dist_attr2)
                dist_type2 = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    new_op.results()[0].type(), dist_attr2
                )
                # new_op.results()[0].set_type(dist_type2)
                # new_op.dist_attr = (
                #     paddle.base.libpaddle.pir.create_op_dist_attribute(
                #         new_op.results()[0].process_mesh, [], 
                #     )
                # )




                #  weight 
                placements0 = new_op.operand(0).source().placements
                dim_map0, partial_status0 = dist.auto_parallel.placement_type.to_dim_map(
                    placements0, len(placements0)
                )
                print("partial_status0:",partial_status0)
                print("dim_map0:",dim_map0,"partial_status0",partial_status0)
                dim_map0 = [1,-1] 
                dist_attr0 = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        new_op.operand(0).source().process_mesh, dim_map0, partial_status0
                    )
                )
                dist_type0 = paddle.base.libpaddle.pir.cvt_to_dist_type(
                    new_op.operand(0).source().type(), dist_attr0
                )
                new_op.operand(0).source().set_type(dist_type0)
            

                # X 输入
                placements = new_op.operand(1).source().placements
                dim_map, partial_status = dist.auto_parallel.placement_type.to_dim_map(
                    placements, len(placements)
                )
                print("dim_map:",dim_map,"partial_status",partial_status)
                dim_map = [-1,1] 
                dist_attr = (
                    paddle.base.libpaddle.pir.create_tensor_dist_attribute(
                        new_op.operand(1).source().process_mesh, dim_map, partial_status
                    )
                )
                print("dist_attr:",dist_attr)

                new_op.dist_attr = (
                    paddle.base.libpaddle.pir.create_op_dist_attribute(
                        new_op.operand(0).source().process_mesh, [dist_attr0, dist_attr], [dist_attr2]
                    )
                )
                print("new_op.dist_attr",new_op.dist_attr)
                print(
                    "### lzx ### op.operand(1).source() 33 placements:",
                    new_op.operand(0).source().placements,
                )
