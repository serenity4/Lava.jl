abstract type PipelineType end

abstract type Compute <: PipelineType end
abstract type Graphics <: PipelineType end
abstract type RayTracing <: PipelineType end

abstract type Pipeline{T} <: LavaAbstraction end
