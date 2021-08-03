"""
Primitive that allows the synchronization between a source of type `S` and target of type `T`.
CPU to GPU synchronization (fences in Vulkan) would be subtypes of SynchronizationPrimitive{CPU,GPU}.
"""
abstract type SynchronizationPrimitive{S,T} end

abstract type DeviceType end

abstract type CPU <: DeviceType end
abstract type GPU <: DeviceType end
