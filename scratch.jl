import Vulkan

import Vulkan: handle

const Vk = Vulkan
using .Vk.ResultTypes: unwrap



instance = Instance(Vk.InstanceCreateInfo([], []))
physical_device = first(unwrap(Vk.enumerate_physical_devices(instance)))
device = Device(physical_device, Vk.DeviceCreateInfo([Vk.DeviceQueueCreateInfo(0, [1.0])], [], []))

sem = Semaphore(device, Vk.SemaphoreCreateInfo())
event = Event(device)
