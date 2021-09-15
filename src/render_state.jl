Base.@kwdef struct RenderState
    blending_mode = nothing
    enable_depth_testing::Bool = false
    enable_depth_bias::Bool = false
    enable_depth_clamp::Bool = false
    color_write_mask::Vk.ColorComponentFlag = Vk.COLOR_COMPONENT_R_BIT | Vk.COLOR_COMPONENT_G_BIT | Vk.COLOR_COMPONENT_B_BIT
end

Base.@kwdef struct ProgramInvocationState
    face_culling::Vk.CullModeFlag = Vk.CULL_MODE_BACK_BIT
    triangle_orientation::Vk.FrontFace = Vk.FRONT_FACE_CLOCKWISE
    enable_primitive_restart::Bool = false
    primitive_topology::Vk.PrimitiveTopology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
    vertex_input_rate::Vk.VertexInputRate = Vk.VERTEX_INPUT_RATE_VERTEX
    polygon_mode::Vk.PolygonMode = Vk.POLYGON_MODE_FILL
end