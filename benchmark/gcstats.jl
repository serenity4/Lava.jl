using Lava
using Test
using Random: Random, MersenneTwister, AbstractRNG
using Dictionaries
using SPIRV: SPIRV, ShaderInterface, U, F, @mat, image_type
using GeometryExperiments
using FixedPointNumbers
using FileIO, ImageIO, VideoIO
using Accessors
# XCB must be loaded prior to creating the instance that will use VK_KHR_xcb_surface.
using XCB: XCB, XWindowManager, current_screen, XCBWindow, resize, extent
using ImageMagick: clamp01nan, clamp01nan!

using Lava: request_index!, GlobalDescriptors, DescriptorArray, patch_descriptors!, patch_pointers!, device_address_block!, RESOURCE_TYPE_IMAGE, RESOURCE_TYPE_BUFFER, RESOURCE_TYPE_ATTACHMENT, assert_type, resource_type, descriptor_type, islogical, isphysical, DESCRIPTOR_TYPE_TEXTURE, get_descriptor_index!, delete_descriptor!, NodeID, free_unused_descriptors!, fence_status, compact!, FencePool, request_command_buffer, ShaderCache, combine_resource_uses_per_node, combine_resource_uses, isbuffer, isimage, isattachment, SynchronizationState, bake!, dependency_info!, rendering_info, PROGRAM_TYPE_GRAPHICS, PROGRAM_TYPE_COMPUTE, COMMAND_TYPE_DRAW_INDEXED, COMMAND_TYPE_DRAW_INDEXED_INDIRECT, Image

include("test/utils.jl")
instance, device = init(; with_validation = true, instance_extensions = ["VK_KHR_xcb_surface"])

function save_test_render(filename, data, h::Union{Nothing, UInt} = nothing; tmp = false, clamp = false)
  clamp && (data = clamp01nan.(data))
  filename = render_file(filename; tmp)
  ispath(filename) && rm(filename)
  save(filename, data')
  @test stat(filename).size > 0
  if !isnothing(h)
    @test hash(data) == h
  else
    hash(data)
  end
end

render_graphics(device, node::RenderNode) = render_graphics(device, node.commands[end])
render_graphics(device, command::Command) = render_graphics(device, only(command.graphics.targets.color), [command])
function render_graphics(device, color, nodes)
  render(device, nodes)
  read_data(device, color)
end

read_data(device, color) = clamp01nan!(collect(RGBA{Float16}, color.attachment.view.image, device))

include("test/examples/textures.jl")

color = attachment_resource(device, nothing; format = Vk.FORMAT_R16G16B16A16_SFLOAT, usage_flags = Vk.IMAGE_USAGE_TRANSFER_SRC_BIT | Vk.IMAGE_USAGE_TRANSFER_DST_BIT | Vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT, dims = [1920, 1080])

include("test/examples/boids.jl")

function xcb_surface(instance, win::XCBWindow)
  handle = unwrap(Vk.create_xcb_surface_khr(instance, Vk.XcbSurfaceCreateInfoKHR(win.conn.h, win.id)))
  Surface(handle, win)
end

wm = XWindowManager()
win = XCBWindow(wm; x=0, y=0, width=1920, height=1080, border_width=50, window_title="Test window", icon_title="Test", attributes=[XCB.XCB_CW_BACK_PIXEL], values=[zero(UInt32)])
cycle = FrameCycle(device, xcb_surface(instance, win))
color = attachment_resource(Vk.FORMAT_R16G16B16A16_SFLOAT, [1920, 1080])
sprite_image = read_boid_image(device)
compute_progs = (boids_forces_program(device), boids_update_program(device))
graphics_prog = boid_drawing_program(device)
parameters = BoidParameters(
  # separation_radius = 0.00001,
  # alignment_factor = 0.8,
  # cohesion_factor = 0.4,
  # awareness_radius = 0.00001,
  # alignment_strength = 0.0,
  # cohesion_strength = 0.0,
)
initial = rand(BoidAgent, 512)
for agent in initial; agent.velocity.xy .*= Vec2(0.1, 0.1); end
agents = buffer_resource(device, initial; memory_domain = MEMORY_DOMAIN_HOST, usage_flags = Vk.BUFFER_USAGE_TRANSFER_DST_BIT | Vk.BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
forces = buffer_resource(device, zeros(Vec2, 512); memory_domain = MEMORY_DOMAIN_HOST)

# initial[1] = BoidAgent(Vec(0, 0), Vec(0.1, 0.), 1.0)
# initial[2:end] .= [BoidAgent(Vec2(-100, -100), Vec2(0, 0), 1.0) for _ in 1:511]
boids = BoidSimulation(deepcopy(initial), parameters)

function draw(swapchain_image, Δt::Float32)
  # nodes = [boid_simulation_nodes(device, agents, forces, parameters, Δt, compute_progs); boid_drawing_node(device, agents, color, sprite_image, graphics_prog)]
  # popfirst!(nodes)
  next!(boids, Δt * 0.1F)
  cpu_agents = buffer_resource(device, boids.agents; memory_domain = MEMORY_DOMAIN_HOST, usage_flags = Vk.BUFFER_USAGE_TRANSFER_SRC_BIT)
  nodes = [transfer_command(cpu_agents, agents), boid_drawing_node(device, agents, color, sprite_image, graphics_prog)]
  draw_and_prepare_for_presentation(device, nodes, color, swapchain_image)
end
draw_and_present! = (cycle, Δt) -> wait(cycle!(image -> draw(image, Float32(Δt)), cycle))
draw_and_present!(cycle, 0.01)
let
  t0 = time()
  t = time()
  Δt = t - t0
  while Δt < 1
    sleep(0.01)
    draw_and_present!(cycle, Δt)
    Δt = time() - t
  end
end
