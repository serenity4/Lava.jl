function debug_callback(msg_severity, msg_type, callback_data_ptr, user_data_ptr)
    callback_data_ptr == C_NULL && return UInt32(0)
    callback_data = unsafe_load(callback_data_ptr)
    msg = unsafe_string(callback_data.pMessage)

    # ignore messages about available device extensions
    if !startswith(msg, "Device Extension: VK")
        id_name = unsafe_string(callback_data.pMessageIdName)
        # out of date as of 1.2.190
        id_name == "VUID-VkShaderModuleCreateInfo-pCode-04147" && contains(msg, "SPV_EXT_physical_storage_buffer") && return UInt32(0)

        msg_type = @match msg_type begin
            &Vk.DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT => "General"
            &Vk.DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT => "Validation"
            &Vk.DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT => "Performance"
            _ => error("Unknown message type $msg_type")
        end

        log = string("$msg_type ($id_name): $msg")

        # defer logging to avoid problems when the callback is run inside a finalizer
        @async @match msg_severity begin
            &Vk.DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT => @debug(log)
            &Vk.DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT => @info(log)
            &Vk.DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT => @warn(log)
            &Vk.DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT => @error(log)
            _ => error("Unknown message severity $msg_severity")
        end
    end
    UInt32(0)
end

struct Instruction
    name::Symbol
    args::Vector{Any}
    kwargs::Vector{NamedTuple}
end

function Base.show(io::IO, inst::Instruction)
    printstyled(io, inst.name; color = :cyan)
    print(io, '(', join(inst.args, ", "))
    if !isempty(inst.kwargs)
        print(io, "; ", join(inst.kwargs, ", "))
    end
    print(io, ')')
end

struct SnoopCommandBuffer <: CommandBuffer
    records::Vector{Instruction}
end

SnoopCommandBuffer() = SnoopCommandBuffer([])

function Base.show(io::IO, ::MIME"text/plain", snoop::SnoopCommandBuffer)
    println(io, "SnoopCommandBuffer($(length(snoop.records)) commands recorded):")
    foreach(snoop.records) do record
        println(io, "  ", record)
    end
end

name(ex) = @match ex begin
    ::Symbol => ex
    :($x::$_) => x
end

macro snoopdef(ex)
    (args, kwargs) = @match ex begin
        :($cmd(command_buffer, $(args...); $(kwargs...))::$_) => (args, kwargs)
        :($cmd(command_buffer, $(args...))::$_) => (args, [])
    end
    @match ex begin
        :($cmd(command_buffer, $(_...))::$rtype) || :($cmd(command_buffer, $(_...); $(_...))::$rtype) => begin
            (rtype_annotation, last_stmt) = @match rtype begin
                :(ResultTypes.Result{Result,VulkanError}) => (:(Result{Vk.Result,Vk.VulkanError}), :(Vk.SUCCESS))
                :Cvoid => (:Cvoid, :nothing)
            end
            push_ex = :(push!(snoop.records, Instruction($(QuoteNode(cmd)), collect([$(name.(args)...)]), collect(kwargs))))
            esc(:(Vk.$cmd(snoop::SnoopCommandBuffer, $(args...); kwargs...)::$rtype_annotation = ($push_ex; $last_stmt)))
        end
        _ => error("Malformed expression: $ex")
    end
end

@snoopdef begin_command_buffer(command_buffer, info::Vk.CommandBufferBeginInfo)::ResultTypes.Result{Result,VulkanError}
@snoopdef end_command_buffer(command_buffer)::ResultTypes.Result{Result,VulkanError}
@snoopdef cmd_bind_pipeline(command_buffer, pipeline_bind_point::Vk.PipelineBindPoint, pipeline)::Cvoid
@snoopdef cmd_set_viewport(command_buffer, viewports::AbstractArray)::Cvoid
@snoopdef cmd_set_scissor(command_buffer, scissors::AbstractArray)::Cvoid
@snoopdef cmd_set_line_width(command_buffer, line_width::Real)::Cvoid
@snoopdef cmd_set_depth_bias(command_buffer, depth_bias_constant_factor::Real, depth_bias_clamp::Real, depth_bias_slope_factor::Real)::Cvoid
@snoopdef cmd_set_blend_constants(command_buffer, blend_constants::NTuple{4, Float32})::Cvoid
@snoopdef cmd_set_depth_bounds(command_buffer, min_depth_bounds::Real, max_depth_bounds::Real)::Cvoid
@snoopdef cmd_set_stencil_compare_mask(command_buffer, face_mask::Vk.StencilFaceFlag, compare_mask::Integer)::Cvoid
@snoopdef cmd_set_stencil_write_mask(command_buffer, face_mask::Vk.StencilFaceFlag, write_mask::Integer)::Cvoid
@snoopdef cmd_set_stencil_reference(command_buffer, face_mask::Vk.StencilFaceFlag, reference::Integer)::Cvoid
@snoopdef cmd_bind_descriptor_sets(command_buffer, pipeline_bind_point::Vk.PipelineBindPoint, layout, first_set::Integer, descriptor_sets::AbstractArray, dynamic_offsets::AbstractArray)::Cvoid
@snoopdef cmd_bind_index_buffer(command_buffer, buffer, offset::Integer, index_type::Vk.IndexType)::Cvoid
@snoopdef cmd_bind_vertex_buffers(command_buffer, buffers::AbstractArray, offsets::AbstractArray)::Cvoid
@snoopdef cmd_draw(command_buffer, vertex_count::Integer, instance_count::Integer, first_vertex::Integer, first_instance::Integer)::Cvoid
@snoopdef cmd_draw_indexed(command_buffer, index_count::Integer, instance_count::Integer, first_index::Integer, vertex_offset::Integer, first_instance::Integer)::Cvoid
@snoopdef cmd_draw_indirect(command_buffer, buffer, offset::Integer, draw_count::Integer, stride::Integer)::Cvoid
@snoopdef cmd_draw_indexed_indirect(command_buffer, buffer, offset::Integer, draw_count::Integer, stride::Integer)::Cvoid
@snoopdef cmd_dispatch(command_buffer, group_count_x::Integer, group_count_y::Integer, group_count_z::Integer)::Cvoid
@snoopdef cmd_dispatch_indirect(command_buffer, buffer, offset::Integer)::Cvoid
@snoopdef cmd_copy_buffer(command_buffer, src_buffer, dst_buffer, regions::AbstractArray)::Cvoid
@snoopdef cmd_copy_image(command_buffer, src_image, src_image_layout::Vk.ImageLayout, dst_image, dst_image_layout::Vk.ImageLayout, regions::AbstractArray)::Cvoid
@snoopdef cmd_blit_image(command_buffer, src_image, src_image_layout::Vk.ImageLayout, dst_image, dst_image_layout::Vk.ImageLayout, regions::AbstractArray, filter::Vk.Filter)::Cvoid
@snoopdef cmd_copy_buffer_to_image(command_buffer, src_buffer, dst_image, dst_image_layout::Vk.ImageLayout, regions::AbstractArray)::Cvoid
@snoopdef cmd_copy_image_to_buffer(command_buffer, src_image, src_image_layout::Vk.ImageLayout, dst_buffer, regions::AbstractArray)::Cvoid
@snoopdef cmd_update_buffer(command_buffer, dst_buffer, dst_offset::Integer, data_size::Integer, data::Ptr{Cvoid})::Cvoid
@snoopdef cmd_fill_buffer(command_buffer, dst_buffer, dst_offset::Integer, size::Integer, data::Integer)::Cvoid
@snoopdef cmd_clear_color_image(command_buffer, image, image_layout::Vk.ImageLayout, color::Vk.ClearColorValue, ranges::AbstractArray)::Cvoid
@snoopdef cmd_clear_depth_stencil_image(command_buffer, image, image_layout::Vk.ImageLayout, depth_stencil::Vk.ClearDepthStencilValue, ranges::AbstractArray)::Cvoid
@snoopdef cmd_clear_attachments(command_buffer, attachments::AbstractArray, rects::AbstractArray)::Cvoid
@snoopdef cmd_resolve_image(command_buffer, src_image, src_image_layout::Vk.ImageLayout, dst_image, dst_image_layout::Vk.ImageLayout, regions::AbstractArray)::Cvoid
@snoopdef cmd_set_event(command_buffer, event, stage_mask::Vk.PipelineStageFlag)::Cvoid
@snoopdef cmd_reset_event(command_buffer, event, stage_mask::Vk.PipelineStageFlag)::Cvoid
@snoopdef cmd_wait_events(command_buffer, events::AbstractArray, memory_barriers::AbstractArray, buffer_memory_barriers::AbstractArray, image_memory_barriers::AbstractArray; src_stage_mask = 0, dst_stage_mask = 0)::Cvoid
@snoopdef cmd_pipeline_barrier(command_buffer, src_stage_mask::Vk.PipelineStageFlag, dst_stage_mask::Vk.PipelineStageFlag, memory_barriers::AbstractArray, buffer_memory_barriers::AbstractArray, image_memory_barriers::AbstractArray; dependency_flags = 0)::Cvoid
@snoopdef cmd_begin_query(command_buffer, query_pool, query::Integer; flags = 0)::Cvoid
@snoopdef cmd_end_query(command_buffer, query_pool, query::Integer)::Cvoid
@snoopdef cmd_begin_conditional_rendering_ext(command_buffer, conditional_rendering_begin::Vk.ConditionalRenderingBeginInfoEXT)::Cvoid
@snoopdef cmd_end_conditional_rendering_ext(command_buffer)::Cvoid
@snoopdef cmd_reset_query_pool(command_buffer, query_pool, first_query::Integer, query_count::Integer)::Cvoid
@snoopdef cmd_write_timestamp(command_buffer, pipeline_stage::Vk.PipelineStageFlag, query_pool, query::Integer)::Cvoid
@snoopdef cmd_copy_query_pool_results(command_buffer, query_pool, first_query::Integer, query_count::Integer, dst_buffer, dst_offset::Integer, stride::Integer; flags = 0)::Cvoid
@snoopdef cmd_push_constants(command_buffer, layout, stage_flags::Vk.ShaderStageFlag, offset::Integer, size::Integer, values::Ptr{Cvoid})::Cvoid
@snoopdef cmd_begin_render_pass(command_buffer, render_pass_begin::Vk.RenderPassBeginInfo, contents::Vk.SubpassContents)::Cvoid
@snoopdef cmd_next_subpass(command_buffer, contents::Vk.SubpassContents)::Cvoid
@snoopdef cmd_end_render_pass(command_buffer)::Cvoid
@snoopdef cmd_execute_commands(command_buffer, command_buffers::AbstractArray)::Cvoid
@snoopdef cmd_debug_marker_begin_ext(command_buffer, marker_info::Vk.DebugMarkerMarkerInfoEXT)::Cvoid
@snoopdef cmd_debug_marker_end_ext(command_buffer)::Cvoid
@snoopdef cmd_debug_marker_insert_ext(command_buffer, marker_info::Vk.DebugMarkerMarkerInfoEXT)::Cvoid
@snoopdef cmd_execute_generated_commands_nv(command_buffer, is_preprocessed::Bool, generated_commands_info::Vk.GeneratedCommandsInfoNV)::Cvoid
@snoopdef cmd_preprocess_generated_commands_nv(command_buffer, generated_commands_info::Vk.GeneratedCommandsInfoNV)::Cvoid
@snoopdef cmd_bind_pipeline_shader_group_nv(command_buffer, pipeline_bind_point::Vk.PipelineBindPoint, pipeline, group_index::Integer)::Cvoid
@snoopdef cmd_push_descriptor_set_khr(command_buffer, pipeline_bind_point::Vk.PipelineBindPoint, layout, set::Integer, descriptor_writes::AbstractArray)::Cvoid
@snoopdef cmd_set_device_mask(command_buffer, device_mask::Integer)::Cvoid
@snoopdef cmd_dispatch_base(command_buffer, base_group_x::Integer, base_group_y::Integer, base_group_z::Integer, group_count_x::Integer, group_count_y::Integer, group_count_z::Integer)::Cvoid
@snoopdef cmd_push_descriptor_set_with_template_khr(command_buffer, descriptor_update_template, layout, set::Integer, data::Ptr{Cvoid})::Cvoid
@snoopdef cmd_set_viewport_w_scaling_nv(command_buffer, viewport_w_scalings::AbstractArray)::Cvoid
@snoopdef cmd_set_discard_rectangle_ext(command_buffer, discard_rectangles::AbstractArray)::Cvoid
@snoopdef cmd_set_sample_locations_ext(command_buffer, sample_locations_info::Vk.SampleLocationsInfoEXT)::Cvoid
@snoopdef cmd_begin_debug_utils_label_ext(command_buffer, label_info::Vk.DebugUtilsLabelEXT)::Cvoid
@snoopdef cmd_end_debug_utils_label_ext(command_buffer)::Cvoid
@snoopdef cmd_insert_debug_utils_label_ext(command_buffer, label_info::Vk.DebugUtilsLabelEXT)::Cvoid
@snoopdef cmd_write_buffer_marker_amd(command_buffer, pipeline_stage::Vk.PipelineStageFlag, dst_buffer, dst_offset::Integer, marker::Integer)::Cvoid
@snoopdef cmd_begin_render_pass_2(command_buffer, render_pass_begin::Vk.RenderPassBeginInfo, subpass_begin_info::Vk.SubpassBeginInfo)::Cvoid
@snoopdef cmd_next_subpass_2(command_buffer, subpass_begin_info::Vk.SubpassBeginInfo, subpass_end_info::Vk.SubpassEndInfo)::Cvoid
@snoopdef cmd_end_render_pass_2(command_buffer, subpass_end_info::Vk.SubpassEndInfo)::Cvoid
@snoopdef cmd_draw_indirect_count(command_buffer, buffer, offset::Integer, count_buffer, count_buffer_offset::Integer, max_draw_count::Integer, stride::Integer)::Cvoid
@snoopdef cmd_draw_indexed_indirect_count(command_buffer, buffer, offset::Integer, count_buffer, count_buffer_offset::Integer, max_draw_count::Integer, stride::Integer)::Cvoid
@snoopdef cmd_set_checkpoint_nv(command_buffer, checkpoint_marker::Ptr{Cvoid})::Cvoid
@snoopdef cmd_bind_transform_feedback_buffers_ext(command_buffer, buffers::AbstractArray, offsets::AbstractArray; sizes = C_NULL)::Cvoid
@snoopdef cmd_begin_transform_feedback_ext(command_buffer, counter_buffers::AbstractArray; counter_buffer_offsets = C_NULL)::Cvoid
@snoopdef cmd_end_transform_feedback_ext(command_buffer, counter_buffers::AbstractArray; counter_buffer_offsets = C_NULL)::Cvoid
@snoopdef cmd_begin_query_indexed_ext(command_buffer, query_pool, query::Integer, index::Integer; flags = 0)::Cvoid
@snoopdef cmd_end_query_indexed_ext(command_buffer, query_pool, query::Integer, index::Integer)::Cvoid
@snoopdef cmd_draw_indirect_byte_count_ext(command_buffer, instance_count::Integer, first_instance::Integer, counter_buffer, counter_buffer_offset::Integer, counter_offset::Integer, vertex_stride::Integer)::Cvoid
@snoopdef cmd_set_exclusive_scissor_nv(command_buffer, exclusive_scissors::AbstractArray)::Cvoid
@snoopdef cmd_bind_shading_rate_image_nv(command_buffer, image_layout::Vk.ImageLayout; image_view = C_NULL)::Cvoid
@snoopdef cmd_set_viewport_shading_rate_palette_nv(command_buffer, shading_rate_palettes::AbstractArray)::Cvoid
@snoopdef cmd_set_coarse_sample_order_nv(command_buffer, sample_order_type::Vk.CoarseSampleOrderTypeNV, custom_sample_orders::AbstractArray)::Cvoid
@snoopdef cmd_draw_mesh_tasks_nv(command_buffer, task_count::Integer, first_task::Integer)::Cvoid
@snoopdef cmd_draw_mesh_tasks_indirect_nv(command_buffer, buffer, offset::Integer, draw_count::Integer, stride::Integer)::Cvoid
@snoopdef cmd_draw_mesh_tasks_indirect_count_nv(command_buffer, buffer, offset::Integer, count_buffer, count_buffer_offset::Integer, max_draw_count::Integer, stride::Integer)::Cvoid
@snoopdef cmd_copy_acceleration_structure_nv(command_buffer, dst, src, mode::Vk.CopyAccelerationStructureModeKHR)::Cvoid
@snoopdef cmd_copy_acceleration_structure_khr(command_buffer, info::Vk.CopyAccelerationStructureInfoKHR)::Cvoid
@snoopdef cmd_copy_acceleration_structure_to_memory_khr(command_buffer, info::Vk.CopyAccelerationStructureToMemoryInfoKHR)::Cvoid
@snoopdef cmd_copy_memory_to_acceleration_structure_khr(command_buffer, info::Vk.CopyMemoryToAccelerationStructureInfoKHR)::Cvoid
@snoopdef cmd_write_acceleration_structures_properties_khr(command_buffer, acceleration_structures::AbstractArray, query_type::Vk.QueryType, query_pool, first_query::Integer)::Cvoid
@snoopdef cmd_write_acceleration_structures_properties_nv(command_buffer, acceleration_structures::AbstractArray, query_type::Vk.QueryType, query_pool, first_query::Integer)::Cvoid
@snoopdef cmd_build_acceleration_structure_nv(command_buffer, info::Vk.AccelerationStructureInfoNV, instance_offset::Integer, update::Bool, dst, scratch, scratch_offset::Integer; instance_data = C_NULL, src = C_NULL)::Cvoid
@snoopdef cmd_trace_rays_khr(command_buffer, raygen_shader_binding_table::Vk.StridedDeviceAddressRegionKHR, miss_shader_binding_table::Vk.StridedDeviceAddressRegionKHR, hit_shader_binding_table::Vk.StridedDeviceAddressRegionKHR, callable_shader_binding_table::Vk.StridedDeviceAddressRegionKHR, width::Integer, height::Integer, depth::Integer)::Cvoid
@snoopdef cmd_trace_rays_nv(command_buffer, raygen_shader_binding_table_buffer, raygen_shader_binding_offset::Integer, miss_shader_binding_offset::Integer, miss_shader_binding_stride::Integer, hit_shader_binding_offset::Integer, hit_shader_binding_stride::Integer, callable_shader_binding_offset::Integer, callable_shader_binding_stride::Integer, width::Integer, height::Integer, depth::Integer; miss_shader_binding_table_buffer = C_NULL, hit_shader_binding_table_buffer = C_NULL, callable_shader_binding_table_buffer = C_NULL)::Cvoid
@snoopdef cmd_trace_rays_indirect_khr(command_buffer, raygen_shader_binding_table::Vk.StridedDeviceAddressRegionKHR, miss_shader_binding_table::Vk.StridedDeviceAddressRegionKHR, hit_shader_binding_table::Vk.StridedDeviceAddressRegionKHR, callable_shader_binding_table::Vk.StridedDeviceAddressRegionKHR, indirect_device_address::Integer)::Cvoid
@snoopdef cmd_set_ray_tracing_pipeline_stack_size_khr(command_buffer, pipeline_stack_size::Integer)::Cvoid
@snoopdef cmd_set_performance_marker_intel(command_buffer, marker_info::Vk.PerformanceMarkerInfoINTEL)::ResultTypes.Result{Result, VulkanError}
@snoopdef cmd_set_performance_stream_marker_intel(command_buffer, marker_info::Vk.PerformanceStreamMarkerInfoINTEL)::ResultTypes.Result{Result, VulkanError}
@snoopdef cmd_set_performance_override_intel(command_buffer, override_info::Vk.PerformanceOverrideInfoINTEL)::ResultTypes.Result{Result, VulkanError}
@snoopdef cmd_set_line_stipple_ext(command_buffer, line_stipple_factor::Integer, line_stipple_pattern::Integer)::Cvoid
@snoopdef cmd_build_acceleration_structures_khr(command_buffer, infos::AbstractArray, build_range_infos::AbstractArray)::Cvoid
@snoopdef cmd_build_acceleration_structures_indirect_khr(command_buffer, infos::AbstractArray, indirect_device_addresses::AbstractArray, indirect_strides::AbstractArray, max_primitive_counts::AbstractArray)::Cvoid
@snoopdef cmd_set_cull_mode_ext(command_buffer; cull_mode = 0)::Cvoid
@snoopdef cmd_set_front_face_ext(command_buffer, front_face::Vk.FrontFace)::Cvoid
@snoopdef cmd_set_primitive_topology_ext(command_buffer, primitive_topology::Vk.PrimitiveTopology)::Cvoid
@snoopdef cmd_set_viewport_with_count_ext(command_buffer, viewports::AbstractArray)::Cvoid
@snoopdef cmd_set_scissor_with_count_ext(command_buffer, scissors::AbstractArray)::Cvoid
@snoopdef cmd_bind_vertex_buffers_2_ext(command_buffer, buffers::AbstractArray, offsets::AbstractArray; sizes = C_NULL, strides = C_NULL)::Cvoid
@snoopdef cmd_set_depth_test_enable_ext(command_buffer, depth_test_enable::Bool)::Cvoid
@snoopdef cmd_set_depth_write_enable_ext(command_buffer, depth_write_enable::Bool)::Cvoid
@snoopdef cmd_set_depth_compare_op_ext(command_buffer, depth_compare_op::Vk.CompareOp)::Cvoid
@snoopdef cmd_set_depth_bounds_test_enable_ext(command_buffer, depth_bounds_test_enable::Bool)::Cvoid
@snoopdef cmd_set_stencil_test_enable_ext(command_buffer, stencil_test_enable::Bool)::Cvoid
@snoopdef cmd_set_stencil_op_ext(command_buffer, face_mask::Vk.StencilFaceFlag, fail_op::Vk.StencilOp, pass_op::Vk.StencilOp, depth_fail_op::Vk.StencilOp, compare_op::Vk.CompareOp)::Cvoid
@snoopdef cmd_set_patch_control_points_ext(command_buffer, patch_control_points::Integer)::Cvoid
@snoopdef cmd_set_rasterizer_discard_enable_ext(command_buffer, rasterizer_discard_enable::Bool)::Cvoid
@snoopdef cmd_set_depth_bias_enable_ext(command_buffer, depth_bias_enable::Bool)::Cvoid
@snoopdef cmd_set_logic_op_ext(command_buffer, logic_op::Vk.LogicOp)::Cvoid
@snoopdef cmd_set_primitive_restart_enable_ext(command_buffer, primitive_restart_enable::Bool)::Cvoid
@snoopdef cmd_copy_buffer_2_khr(command_buffer, copy_buffer_info::Vk.CopyBufferInfo2KHR)::Cvoid
@snoopdef cmd_copy_image_2_khr(command_buffer, copy_image_info::Vk.CopyImageInfo2KHR)::Cvoid
@snoopdef cmd_blit_image_2_khr(command_buffer, blit_image_info::Vk.BlitImageInfo2KHR)::Cvoid
@snoopdef cmd_copy_buffer_to_image_2_khr(command_buffer, copy_buffer_to_image_info::Vk.CopyBufferToImageInfo2KHR)::Cvoid
@snoopdef cmd_copy_image_to_buffer_2_khr(command_buffer, copy_image_to_buffer_info::Vk.CopyImageToBufferInfo2KHR)::Cvoid
@snoopdef cmd_resolve_image_2_khr(command_buffer, resolve_image_info::Vk.ResolveImageInfo2KHR)::Cvoid
@snoopdef cmd_set_fragment_shading_rate_khr(command_buffer, fragment_size::Vk.Extent2D, combiner_ops::NTuple{2, Vk.FragmentShadingRateCombinerOpKHR})::Cvoid
@snoopdef cmd_set_fragment_shading_rate_enum_nv(command_buffer, shading_rate::Vk.FragmentShadingRateNV, combiner_ops::NTuple{2, Vk.FragmentShadingRateCombinerOpKHR})::Cvoid
@snoopdef cmd_set_vertex_input_ext(command_buffer, vertex_binding_descriptions::AbstractArray, vertex_attribute_descriptions::AbstractArray)::Cvoid
@snoopdef cmd_set_color_write_enable_ext(command_buffer, color_write_enables::AbstractArray)::Cvoid
@snoopdef cmd_set_event_2_khr(command_buffer, event, dependency_info::Vk.DependencyInfoKHR)::Cvoid
@snoopdef cmd_reset_event_2_khr(command_buffer, event, stage_mask::Integer)::Cvoid
@snoopdef cmd_wait_events_2_khr(command_buffer, events::AbstractArray, dependency_infos::AbstractArray)::Cvoid
@snoopdef cmd_pipeline_barrier_2_khr(command_buffer, dependency_info::Vk.DependencyInfoKHR)::Cvoid
@snoopdef cmd_write_timestamp_2_khr(command_buffer, stage::Integer, query_pool, query::Integer)::Cvoid
@snoopdef cmd_write_buffer_marker_2_amd(command_buffer, stage::Integer, dst_buffer, dst_offset::Integer, marker::Integer)::Cvoid
