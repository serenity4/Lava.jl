struct Queue <: LavaAbstraction
  handle::Vk.Queue
  capabilities::Vk.QueueFlag
  index::Int64
  family::Int64
end

vk_handle_type(::Type{Queue}) = Vk.Queue

struct QueueDispatch
  queues::Dictionary{Int64,Vector{Queue}}
  present_queue::Optional{Queue}
  """
  Build a `QueueDispatch` structure from a given device and configuration.
  If a surface is provided, then a queue that supports presentation on this surface will be filled in `present_queue`.

  !!! warning
      `device` must have been created with a consistent number of queues as requested in the provided queue configuration.
      It is highly recommended to have created the device with the result of `queue_infos(QueueDispatch, physical_device, config)`.
  """
  function QueueDispatch(device, infos; surface = nothing)
    pdevice = physical_device(device)
    families = dictionary(map(infos) do info
      info.queue_family_index => length(info.queue_priorities)
    end)
    props = Vk.get_physical_device_queue_family_properties(pdevice)
    queues = dictionary(map(pairs(families)) do (family, count)
      family => map(1:count) do index
        info = Vk.DeviceQueueInfo2(family, index - 1)
        Queue(Vk.get_device_queue_2(device, info), props[family + 1].queue_flags, index, family + 1)
      end
    end)
    present_queue = if !isnothing(surface)
      idx = findfirst(families) do family
        unwrap(Vk.get_physical_device_surface_support_khr(pdevice, family, surface))
      end
      if isnothing(idx)
        error("Could not find a queue that supports presentation on the provided surface.")
      else
        first(values(queues)[idx])
      end
    else
      nothing
    end
    new(queues, present_queue)
  end
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
  # try exact match
  for (family, queues) in pairs(dispatch.queues)
    properties == first(queues).capabilities && return family
  end

  # try queues that contain the required properties
  for (family, queues) in pairs(dispatch.queues)
    properties in first(queues).capabilities && return family
  end

  # panic
  error("Could not find a queue matching with the required properties $properties")
end

function queue(dispatch::QueueDispatch, family_index)
  haskey(dispatch.queues, family_index) && return first(dispatch.queues[family_index])
  !isnothing(dispatch.present_queue) && dispatch.present_queue.family == family_index && return dispatch.present_queue
  error("Could not find queue with family index $family_index")
end

function present(dispatch::QueueDispatch, present_info::Vk.PresentInfoKHR)
  (; present_queue) = dispatch
  if isnothing(present_queue)
    error("No presentation queue was specified for $dispatch")
  else
    Vk.queue_present_khr(present_queue, present_info)
  end
end

function queue_family_indices(dispatch::QueueDispatch; include_present = true)
  indices = collect(keys(dispatch.queues))
  !isnothing(dispatch.present_queue) && include_present && union!(indices, dispatch.present_queue.family)
  indices
end
