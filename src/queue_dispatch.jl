struct Queue <: LavaAbstraction
  handle::Vk.Queue
  capabilities::Vk.QueueFlag
  index::Int64
  family::Int64
end

vk_handle_type(::Type{Queue}) = Vk.Queue

struct QueueDispatch
  device::Vk.Device
  queues::Dictionary{Int64,Vector{Queue}}
  """
  Build a `QueueDispatch` structure from a given device and configuration.

  !!! warning
      `device` must have been created with a consistent number of queues as requested in the provided queue configuration.
      It is highly recommended to have created the device with the result of `queue_infos(QueueDispatch, physical_device, config)`.
  """
  function QueueDispatch(device, infos)
    pdevice = physical_device(device)
    families = dictionary(map(infos) do info
      info.queue_family_index => length(info.queue_priorities)
    end)
    queues = queues_by_family(device, families)
    new(device, queues)
  end
end

function queues_by_family(device::Vk.Device, families)
  props = Vk.get_physical_device_queue_family_properties(device.physical_device)
  dictionary(map(pairs(families)) do (family, count)
    family => map(1:count) do index
      info = Vk.DeviceQueueInfo2(family, index - 1)
      Queue(Vk.get_device_queue_2(device, info), props[family + 1].queue_flags, index, family + 1)
    end
  end)
end

function queue_infos(::Type{QueueDispatch}, physical_device::Vk.PhysicalDevice, config)
  # queue family index => count
  families = Dictionary{Int64,Int64}()
  props::Vector{Union{Nothing,Vk.QueueFamilyProperties}} = Vk.get_physical_device_queue_family_properties(physical_device)
  config = deepcopy(config)

  # resolve exact matches first
  for (i, prop) in enumerate(props)
    isnothing(prop) && continue
    for (flags, count) in pairs(config)
      # exact match
      if flags == prop.queue_flags
        remaining = prop.queue_count - get(families, i - 1, 0)
        if remaining ≥ count
          delete!(config, flags)
          set!(families, i - 1, get(families, i - 1, 0) + count)
        else
          config[flags] -= remaining
          set!(families, i - 1, get(families, i - 1, 0) + remaining)
          props[i - 1] = nothing
          break
        end
      end
    end
  end

  for (i, prop) in enumerate(props)
    isnothing(prop) && continue
    for (flags, count) in pairs(config)
      # match
      if flags in prop.queue_flags
        remaining = prop.queue_count - get(families, i - 1, 0)
        if remaining ≥ count
          delete!(config, flags)
          set!(families, i - 1, get(families, i - 1, 0) + count)
        else
          config[flags] -= remaining
          set!(families, i - 1, get(families, i - 1, 0) + remaining)
          props[i - 1] = nothing
          break
        end
      end
    end
  end

  map(pairs(families)) do (index, count)
    Vk.DeviceQueueCreateInfo(index, ones(count))
  end |> collect
end

function get_queue_family(dispatch::QueueDispatch, properties::Vk.QueueFlag)
  iszero(properties) && throw(ArgumentError("At least one queue flag must be specified"))

  # Try exact match.
  for (family, queues) in pairs(dispatch.queues)
    properties == first(queues).capabilities && return family
  end

  # Try queues that contain the required properties.
  for (family, queues) in pairs(dispatch.queues)
    properties in first(queues).capabilities && return family
  end

  error("Could not find a queue matching with the required properties $properties")
end

function queue(dispatch::QueueDispatch, family_index)
  @assert family_index ≠ -1
  first(get(() -> error("Could not find queue with family index $family_index"), dispatch.queues, family_index))
end

function find_presentation_queue(dispatch::QueueDispatch, surfaces)
  (; device) = dispatch
  family_index = findfirst(keys(dispatch.queues)) do family
    all(surfaces) do surface
      unwrap(Vk.get_physical_device_surface_support_khr(device.physical_device, family, surface))
    end
  end
  !isnothing(family_index) || error("Could not find a presentation queue among the queues created with this device that is compatible with all the surfaces $surfaces. You will have to provide the surfaces when initializing the device.")
  queue = first(dispatch.queues[family_index])
end

queue_family_indices(dispatch::QueueDispatch) = collect(keys(dispatch.queues))
