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

function compute_forces!(data_address::DeviceAddressBlock, local_id::Vec{3,UInt32}, global_id::Vec{3,UInt32})
  i = linearize_index(global_id, DISPATCH_SIZE, local_id, WORKGROUP_SIZE)
  (agents, parameters, Δt, forces) = @load data_address::Tuple{DeviceAddress, BoidParameters, Float32, DeviceAddress}
  agents = @load agents::Arr{512,BoidAgent}
  @store forces[i]::Vec2 = compute_forces(agents, parameters, i + 1, Δt)
  nothing
end

function step_euler!(data_address::DeviceAddressBlock, local_id::Vec{3,UInt32}, global_id::Vec{3,UInt32})
  i = linearize_index(global_id, DISPATCH_SIZE, local_id, WORKGROUP_SIZE)
  (agents, parameters, Δt, forces) = @load data_address::Tuple{DeviceAddress, BoidParameters, Float32, DeviceAddress}
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
    @block (DeviceAddress(agents), parameters, Δt, DeviceAddress(forces))
  end
  dispatch = Dispatch(DISPATCH_SIZE)
  invocation_forces = ProgramInvocation(
    prog_1,
    dispatch,
    data,
    @resource_dependencies begin
      @read agents::Buffer::Physical
      @write forces::Buffer::Physical
    end
  )
  invocation_update = ProgramInvocation(
    prog_2,
    dispatch,
    data,
    @resource_dependencies begin
      @read forces::Buffer::Physical
      @write agents::Buffer::Physical
    end
  )
  (compute_node(invocation_forces), compute_node(invocation_update))
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
    data = rand(BoidAgent, 512)
    agents = buffer_resource(device, data; memory_domain = MEMORY_DOMAIN_HOST)
    final_agents = buffer_resource(device, data; memory_domain = MEMORY_DOMAIN_HOST)
    forces = buffer_resource(device, zeros(Vec2, 512); memory_domain = MEMORY_DOMAIN_HOST)
    @test collect(BoidAgent, agents.data) == data
    nodes = boid_simulation_nodes(device, agents, forces, parameters, Δt)
    @test render(device, nodes)
    res = collect(BoidAgent, agents.data)
    res_forces = collect(Vec2, forces.data)
    expected_forces = [compute_forces(data, parameters, i, Δt) for i in eachindex(data)]
    expected = next!(BoidSimulation(deepcopy(data), parameters), Δt)
    @test res_forces ≈ expected_forces
    @test all(res .≈ expected)
  end
end;
