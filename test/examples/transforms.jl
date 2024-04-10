struct Plane
  u::Vec3
  v::Vec3
  Plane(u, v) = new(normalize(convert(Vec3, u)), normalize(convert(Vec3, v)))
end

Base.:(==)(x::Plane, y::Plane) = x.u == y.u && x.v == y.v
Base.isapprox(x::Plane, y::Plane) = x.u ≈ y.u && x.v ≈ y.v

Plane(normal) = Plane(convert(Vec3, normal))
function Plane(normal::Vec3)
  iszero(normal) && return Plane(Vec3(1, 0, 0), Vec3(0, 1, 0))
  u = @ga 3 Vec3 normal::Vector × 1::e1
  iszero(u) && (u = @ga 3 Vec3 dual(normal::Vector × 1f0::e2))
  v = @ga 3 Vec3 dual(normal::Vector × u::Vector)
  Plane(u, v)
end

struct Rotation
  plane::Plane
  angle::Float32
end

Rotation(axis::Vec3) = Rotation(Plane(normalize(axis)), norm(axis))
Rotation() = Rotation(Plane(Vec3(0, 0, 1)), 0)

Base.inv(rot::Rotation) = @set rot.angle = -rot.angle
Base.iszero(rot::Rotation) = iszero(rot.angle)

function apply_rotation(p::Vec3, rotation::Rotation)
  # Define rotation bivector which encodes a rotation in the given plane by the specified angle.
  ϕ = @ga 3 Vec3 rotation.angle::Scalar ⟑ (rotation.plane.u::Vector ∧ rotation.plane.v::Vector)
  # Define rotation generator to be applied to perform the operation.
  Ω = @ga 3 Tuple exp((ϕ::Bivector) / 2f0::Scalar)
  @ga 3 Vec3 begin
    Ω::(Scalar, Bivector)
    inverse(Ω) ⟑ p::Vector ⟑ Ω
  end
end

Base.@kwdef struct Transform
  translation::Vec3 = (0, 0, 0)
  rotation::Rotation = Rotation()
  scaling::Vec3 = (1, 1, 1)
end

function apply_transform(p::Vec3, (; translation, rotation, scaling)::Transform)
  apply_rotation(p .* scaling, rotation) + translation
end

Base.inv((; translation, rotation, scaling)::Transform) = Transform(-translation, inv(rotation), inv.(scaling))

"""
The image plane is taken to be z = 0.

Projection through the camera yields a z-component which describes how far
or near the camera the point was. The resulting value is between 0 and 1,
where 0 corresponds to a point on the near clipping plane, and 1 to one on
the far clipping plane.
"""
Base.@kwdef struct PinholeCamera
  near_clipping_plane::Float32 = 0
  far_clipping_plane::Float32 = 10
  transform::Transform = Transform()
end

function orthogonal_projection(p::Vec3, camera::PinholeCamera)
  p = apply_transform(p, inv(camera.transform))
  z = remap(p.z, camera.near_clipping_plane, camera.far_clipping_plane, 0F, 1F)
  Vec3(p.x, p.y, z)
end

@testset "Transforms" begin
  @testset "Plane" begin
    n = zero(Vec3)
    p = Plane(n)
    @test norm(p.u) == norm(p.v) == 1
    @test Plane((1, 0, 0)) == Plane((0, 0, 1), (0, -1, 0))
    @test Plane((0, 0, 1)) == Plane((0, -1, 0), (1, 0, 0))
  end

  @testset "Rotation" begin
    rot = Rotation()
    @test iszero(rot)
    rot = Rotation(Plane((1, 0, 0), (0, 1, 0)), (π)F/4)
    p = Vec3(0.2, 0.2, 1.0)
    p′ = apply_rotation(p, rot)
    @test p′.z == p.z
    @test p′.xy ≈ Vec2(0, 0.2sqrt(2))
    @test apply_rotation(p, @set rot.angle = 0) == p
    rot = Rotation(Plane(Tuple(rand(3))), 1.5)
    @test apply_rotation(p, rot) ≉ p
    # Works fairly well for small angles, but will not work in general
    # because the rotation is expressed in terms of rotated axes (body frame).
    # We'd need to express rotation with respect to a fixed reference frame to obtain an inverse.
    @test_broken apply_rotation(apply_rotation(p, rot), inv(rot)) ≈ p

    @test unwrap(validate(@compile apply_rotation(::Vec3, ::Rotation)))
  end

  @testset "Camera" begin
    camera = PinholeCamera(0, 10, Transform())
    p = Vec3(0.4, 0.5, 1.7)
    p′ = orthogonal_projection(p, camera)
    @test p′.xy == Vec2(0.4, 0.5)
    @test camera.near_clipping_plane < p′.z < camera.far_clipping_plane
    p.z = camera.near_clipping_plane
    @test orthogonal_projection(p, camera).z == 0
    p.z = camera.far_clipping_plane
    @test orthogonal_projection(p, camera).z == 1

    @test unwrap(validate(@compile orthogonal_projection(::Vec3, ::PinholeCamera)))
  end
end;
