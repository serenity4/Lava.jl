struct BoidAgent
  position::Vec2
  velocity::Vec2 # facing direction is taken to be the normalized velocity
  mass::Float32
end
BoidAgent(position, velocity) = BoidAgent(position, velocity, 1)

struct BoidSimulation{V<:AbstractVector{BoidAgent}}
  agents::V
  separation_radius::Float32
  alignment_factor::Float32
  cohesion_factor::Float32
  awareness_radius::Float32
  alignment_strength::Float32
  cohesion_strength::Float32
end

distance2(x::Vec, y::Vec) = sum(x -> x^2, y - x)
distance(x::Vec, y::Vec) = sqrt(distance2(x, y))
norm(x::Vec) = distance(x, zero(x))
normalize(x::Vec) = ifelse(iszero(x), x, x / norm(x))

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
  agents = deepcopy(boids.agents)
  for (i, (; position, velocity, mass)) in enumerate(agents)
    forces = zero(Vec2)
    average_heading = zero(Vec2)
    flock_center = zero(Vec2)
    flock_size = 0U
    direction = normalize(velocity)

    for (j, other) in enumerate(agents)
      i == j && continue
      d = distance(position, other.position)
      d < boids.awareness_radius || continue
      flock_size += 1U
      flock_center += other.position
      repulsion_strength = exp(-d^2 / (2 * boids.separation_radius^2))
      forces -= (other.position - position) * repulsion_strength
      average_heading += normalize(other.velocity)
    end

    if !iszero(flock_size)
      flock_center /= flock_size
      target_position = lerp(position, flock_center, 1 - boids.cohesion_factor)
      forces += (target_position - position) * boids.cohesion_strength

      if !iszero(average_heading)
        average_heading = normalize(average_heading)
        target_heading = slerp(direction, average_heading, boids.alignment_factor)
        forces += (velocity - target_heading * norm(velocity)) * boids.alignment_strength
      end
    end

    acceleration = forces / mass

    # Use a simple Euler integration scheme to get the next state.
    new_position = position + velocity * Δt
    new_velocity = velocity + acceleration * Δt

    # Update the agent.
    boids.agents[i] = BoidAgent(new_position, new_velocity)
  end
  boids
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
  end

  a1 = BoidAgent(Vec2(0.5, 0.5), Vec2(0.1, 0.1))
  a2 = BoidAgent(Vec2(0.6, 0.5), Vec2(0.1, 0.5))
  separation_radius = 0.02F
  alignment_factor = 0.8F
  cohesion_factor = 0.4F
  awareness_radius = 0.2F
  alignment_strength = 1.0F
  cohesion_strength = 1.0F
  sim = BoidSimulation(Arr(a1, a2), separation_radius, alignment_factor, cohesion_factor, awareness_radius, alignment_strength, cohesion_strength)
  Δt = 0.001F
  next!(sim, Δt)
  @test sim.agents[1].position == a1.position + Δt * a1.velocity
  @test all(all(!isnan, a.position) && all(!isnan, a.velocity) for a in sim.agents)
  for _ in 1:10000
    next!(sim, Δt)
  end
  @test all(all(!isnan, a.position) && all(!isnan, a.velocity) for a in sim.agents)
end;
