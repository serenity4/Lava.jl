struct BoidAgent
  position::Vec2
  velocity::Vec2 # facing direction is taken to be the normalized velocity
  mass::Float32
end
BoidAgent(position, velocity) = BoidAgent(position, velocity, 1)
Base.:(==)(x::BoidAgent, y::BoidAgent) = x.position == y.position && x.velocity == y.velocity && x.mass == y.mass
Base.isapprox(x::BoidAgent, y::BoidAgent) = isapprox(x.position, y.position) && isapprox(x.velocity, y.velocity) && x.mass == y.mass

struct BoidAgentSampler end
Random.Sampler(::Type{<:AbstractRNG}, ::Random.SamplerType{BoidAgent}, ::Val{1}) = BoidAgentSampler()
Base.rand(rng::AbstractRNG, ::BoidAgentSampler) = BoidAgent(Vec2(rand_square(rng, 2)...), Vec2(rand_square(rng, 2)...))
rand_square(rng, n) = 2 .* rand(rng, n) .- 1

Base.@kwdef struct BoidParameters
  separation_radius::Float32 = separation_radius = 0.02
  alignment_factor::Float32 = alignment_factor = 0.8
  cohesion_factor::Float32 = cohesion_factor = 0.4
  awareness_radius::Float32 = awareness_radius = 0.2
  alignment_strength::Float32 = alignment_strength = 1.0
  cohesion_strength::Float32 = cohesion_strength = 1.0
end

struct BoidSimulation{V<:AbstractVector{BoidAgent}}
  agents::V
  parameters::BoidParameters
end
BoidSimulation(agents) = BoidSimulation(agents, BoidParameters())

distance2(x::Vec, y::Vec) = sum(x -> x^2, y - x)
distance(x::Vec, y::Vec) = sqrt(distance2(x, y))
norm(x::Vec) = distance(x, zero(x))
normalize(x::Vec) = ifelse(iszero(x), x, x / norm(x))
function rotate(v::Vec2, angle::Float32)
  cv = v.x + v.y * im
  crot = cos(angle) + sin(angle) * im
  cv′ = cv * crot
  Vec2(real(cv′), imag(cv′))
end

const WORKGROUP_SIZE = (8U, 8U, 1U)
const DISPATCH_SIZE = (8U, 1U, 1U)
const COMPUTE_EXECUTION_OPTIONS = ComputeExecutionOptions(local_size = WORKGROUP_SIZE)

# from https://stackoverflow.com/questions/21483999/using-atan2-to-find-angle-between-two-vectors
function angle_2d(x, y)
  θ = atan(y[2], y[1]) - atan(x[2], x[1])
  θ % 2(π)F
end

# from https://en.wikipedia.org/wiki/Slerp
function slerp(x, y, t)
  θ = angle_2d(x, y)
  wx = sin((1 - t)θ)
  wy = sin(t * θ)
  normalize(x * wx + y * wy)
end
lerp(x, y, t) = x * t + (1 - t)y

function next!(boids::BoidSimulation{V}, Δt::Float32) where {V}
  forces = [compute_forces(boids.agents, boids.parameters, i, Δt) for i in eachindex(boids.agents)]
  boids.agents .= step_euler.(boids.agents, forces, Δt)
end

function compute_forces(agents, parameters::BoidParameters, i::Integer, Δt::Float32)
  forces = zero(Vec2)
  (; position, velocity, mass) = agents[i]
  average_heading = zero(Vec2)
  flock_center = zero(Vec2)
  flock_size = 0U
  direction = normalize(velocity)

  for (j, other) in enumerate(agents)
    i == j && continue
    d = distance(position, other.position)
    d < parameters.awareness_radius || continue
    flock_size += 1U
    flock_center[] = flock_center + other.position
    repulsion_strength = exp(-d^2 / (2 * parameters.separation_radius^2))
    forces[] = forces - (other.position - position) * repulsion_strength
    average_heading[] = average_heading + normalize(other.velocity)
  end

  if !iszero(flock_size)
    flock_center /= flock_size
    target_position = lerp(position, flock_center, 1 - parameters.cohesion_factor)
    forces[] = forces + (target_position - position) * parameters.cohesion_strength

    if !iszero(average_heading)
      average_heading = normalize(average_heading)
      target_heading = slerp(direction, average_heading, parameters.alignment_factor)
      forces[] = forces + (velocity - target_heading * norm(velocity)) * parameters.alignment_strength
    end
  end

  forces
end

function step_euler(agent::BoidAgent, forces, Δt)
  (; position, velocity, mass) = agent

  # Use a simple Euler integration scheme to get the next state.
  acceleration = forces / mass
  new_position = position + velocity * Δt
  new_velocity = velocity + acceleration * Δt

  BoidAgent(new_position, new_velocity, mass)
end

linearize_index((x, y, z), (nx, ny, nz)) = x + y * nx + z * nx * ny
function linearize_index(global_id, global_size, local_id, local_size)
  linearize_index(local_id, local_size) + prod(local_size) * linearize_index(global_id, global_size)
end

struct BoidsInfo
  agents::DeviceAddress
  parameters::BoidParameters
  Δt::Float32
  forces::DeviceAddress
end

function compute_forces!(data_address::DeviceAddressBlock, local_id::Vec{3,UInt32}, global_id::Vec{3,UInt32})
  i = linearize_index(global_id, DISPATCH_SIZE, local_id, WORKGROUP_SIZE)
  (; agents, parameters, Δt, forces) = @load data_address::BoidsInfo
  agents = @load agents::Arr{512,BoidAgent}
  @store forces[i]::Vec2 = compute_forces(agents, parameters, i + 1, Δt)
  nothing
end

function step_euler!(data_address::DeviceAddressBlock, local_id::Vec{3,UInt32}, global_id::Vec{3,UInt32})
  i = linearize_index(global_id, DISPATCH_SIZE, local_id, WORKGROUP_SIZE)
  (; agents, Δt, forces) = @load data_address::BoidsInfo
  agent = @load agents[i]::BoidAgent
  @store agents[i]::BoidAgent = step_euler(agent, (@load forces[i]::Vec2), Δt)
  nothing
end

function boids_forces_program(device)
  compute = @compute device compute_forces!(
    ::DeviceAddressBlock::PushConstant,
    ::Vec{3,UInt32}::Input{LocalInvocationId},
    ::Vec{3,UInt32}::Input{WorkgroupId},
  ) COMPUTE_EXECUTION_OPTIONS
  Program(compute)
end

function boids_update_program(device)
  compute = @compute device step_euler!(
    ::DeviceAddressBlock::PushConstant,
    ::Vec{3,UInt32}::Input{LocalInvocationId},
    ::Vec{3,UInt32}::Input{WorkgroupId},
  ) COMPUTE_EXECUTION_OPTIONS
  Program(compute)
end

function boid_simulation_nodes(device, agents::Resource, forces::Resource, parameters::BoidParameters, Δt::Float32)
  prog_1 = boids_forces_program(device)
  prog_2 = boids_update_program(device)
  data = @invocation_data (prog_1, prog_2) begin
    @block BoidsInfo(DeviceAddress(agents), parameters, Δt, DeviceAddress(forces))
  end
  dispatch = Dispatch(DISPATCH_SIZE)
  command_1 = compute_command(
    dispatch,
    prog_1,
    data,
    @resource_dependencies begin
      @read agents::Buffer::Physical
      @write forces::Buffer::Physical
    end
  )
  command_2 = compute_command(
    dispatch,
    prog_2,
    data,
    @resource_dependencies begin
      @read forces::Buffer::Physical
      @write agents::Buffer::Physical
    end
  )
  [command_1, command_2]
end

struct BoidDrawData
  agents::DeviceAddress
  texture_index::DescriptorIndex
  size::Float32
end

function quad_2d(center, size) # top-left, bottom-left, bottom-right, top-right
  # Negate `y` dimension to match Vulkan's device-local coordinate system.
  e1 = Vec2(size.x, -size.y) / 2F
  e2 = Vec2(-size.x, -size.y) / 2F

  # Corners are given for a coordinate system with `x` and `y` increasing along right and up directions.
  # As the `y` components have been negated, this coincides with the device-local coordinate system.
  A = center + e2 # top-left
  B = center - e1 # bottom-left
  C = center - e2 # bottom-right
  D = center + e1 # top-right

  # Return verices in triangle strip order.
  Arr(A, B, D, C)
end

function boid_vert(uv::Vec2, position::Vec4, vertex_index::UInt32, instance_index::UInt32, data::DeviceAddressBlock)
  boid_data = @load data::BoidDrawData
  agent = @load boid_data.agents[instance_index]::BoidAgent
  angle = angle_2d(Vec2(0F, 1F), agent.velocity)
  corner = quad_2d(agent.position, Vec2(boid_data.size, boid_data.size))[vertex_index]

  # Temporarily revert to using a X-right, Y-up coordinate system to apply the rotation.
  corner.y *= -1F
  v = rotate(corner, angle)
  v.y *= -1F

  uv[] = quad_2d(Vec2(0.5F, 0.5F), Vec2(1F, -1F))[vertex_index]
  position[] = Vec4(v.x, v.y, 0F, 1F)
end

function boid_frag(color::Vec4, uv::Vec2, data::DeviceAddressBlock, textures)
  (; texture_index) = @load data::BoidDrawData
  texture = textures[texture_index]
  color[] = texture(uv)
end

function boid_drawing_program(device)
  vert = @vertex device boid_vert(::Vec2::Output, ::Vec4::Output{Position}, ::UInt32::Input{VertexIndex}, ::UInt32::Input{InstanceIndex}, ::DeviceAddressBlock::PushConstant)
  frag = @fragment device boid_frag(
    ::Vec4::Output,
    ::Vec2::Input,
    ::DeviceAddressBlock::PushConstant,
    ::Arr{2048,SPIRV.SampledImage{SPIRV.image_type(SPIRV.ImageFormatRgba16f, SPIRV.Dim2D, 0, false, false, 1)}}::UniformConstant{DescriptorSet = 0, Binding = 3})
  Program(vert, frag)
end

function boid_drawing_node(device, agents::Resource, color, image)
  prog = boid_drawing_program(device)
  image_texture = texture_descriptor(Texture(image, setproperties(DEFAULT_SAMPLING, (magnification = Vk.FILTER_LINEAR, minification = Vk.FILTER_LINEAR))))
  data = @invocation_data prog begin
    @block BoidDrawData(DeviceAddress(agents), @descriptor(image_texture), 0.1)
  end
  draw = graphics_command(
    DrawIndexed(1:4; instances = 1:512),
    prog,
    data,
    RenderTargets(color),
    RenderState(),
    setproperties(ProgramInvocationState(), (;
      primitive_topology = Vk.PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP,
      triangle_orientation = Vk.FRONT_FACE_CLOCKWISE,
    )),
    @resource_dependencies begin
      @read agents::Buffer::Physical image::Texture
      @write (color => (0.08, 0.05, 0.1, 1.0))::Color
    end
  )
  draw
end

@testset "Simulation of boids" begin
  @testset "Mathematical functions" begin
    @test lerp(0.1, 0.9, 0.5) == 0.5
    @test lerp(0.1, 1.2, 0.5) == 0.65

    x = Vec2(1, 0)
    y = Vec2(0, 1)
    @test norm(x) == norm(y) == 1
    @test distance2(x, y) == 2
    @test distance(x, y) ≈ sqrt(2)
    @test slerp(x, y, 0.5) ≈ normalize(Vec2(1, 1))
    @test slerp(x, y, 1.0) == normalize(y)
    @test slerp(x, y, 0.0) == normalize(x)

    @test length(rand(BoidAgent, 5)) == 5
  end

  @testset "CPU implementation" begin
    a1 = BoidAgent(Vec2(0.5, 0.5), Vec2(0.1, 0.1))
    a2 = BoidAgent(Vec2(0.6, 0.5), Vec2(0.1, 0.5))
    Δt = 0.001F
    boids = BoidSimulation(Arr(a1, a2))
    next!(boids, Δt)
    @test boids.agents[1].position == a1.position + Δt * a1.velocity
    @test all(all(!isnan, a.position) && all(!isnan, a.velocity) for a in boids.agents)
    for _ in 1:10000
      next!(boids, Δt)
    end
    @test all(all(!isnan, a.position) && all(!isnan, a.velocity) for a in boids.agents)
    boids = BoidSimulation(rand(BoidAgent, 512))
    @test all(all(!isnan, a.position) && all(!isnan, a.velocity) for a in boids.agents)
    for _ in 1:100
      next!(boids, Δt)
    end
    @test all(all(!isnan, a.position) && all(!isnan, a.velocity) for a in boids.agents)
  end

  @testset "GPU implementation" begin
    @test linearize_index((0, 0, 0), (8, 1, 1), (0, 0, 0), (8, 8, 1)) == 0
    @test linearize_index((1, 0, 0), (8, 1, 1), (0, 0, 0), (8, 8, 1)) == 64
    @test linearize_index((1, 0, 0), (8, 1, 1), (1, 0, 0), (8, 8, 1)) == 65
    @test linearize_index((7, 0, 0), (8, 1, 1), (7, 7, 0), (8, 8, 1)) == 511

    parameters = BoidParameters()
    Δt = 0.01F
    initial = rand(MersenneTwister(1), BoidAgent, 512)
    agents = buffer_resource(device, initial; memory_domain = MEMORY_DOMAIN_HOST)
    forces = buffer_resource(device, zeros(Vec2, 512); memory_domain = MEMORY_DOMAIN_HOST)
    @test collect(BoidAgent, agents.data) == initial
    nodes = boid_simulation_nodes(device, agents, forces, parameters, Δt)
    @test render(device, nodes)
    res = collect(BoidAgent, agents.data)
    res_forces = collect(Vec2, forces.data)
    expected_forces = [compute_forces(initial, parameters, i, Δt) for i in eachindex(initial)]
    expected = next!(BoidSimulation(deepcopy(initial), parameters), Δt)
    @test res_forces ≈ expected_forces
    @test all(res .≈ expected)

    agents = buffer_resource(device, initial; memory_domain = MEMORY_DOMAIN_HOST)
    forces = buffer_resource(device, zeros(Vec2, 512); memory_domain = MEMORY_DOMAIN_HOST)
    nodes = boid_simulation_nodes(device, agents, forces, parameters, Δt)
    push!(nodes, boid_drawing_node(device, agents, color, read_boid_image(device)))
    data = render_graphics(device, color, nodes)
    save_test_render("boid_agents.png", data, 0x15323ce1756b507b)
  end
end;
