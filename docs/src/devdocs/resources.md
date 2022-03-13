# Resources

### Resource specification

Resource information is stored in `LogicalBuffer`, `LogicalImage` and `LogicalAttachment`.

### Resource creation

- `buffer`, `image` and `attachment` are available in two flavors:
  - When the first argument is a `Device`, it allocates and return a physical resource.
  - When it is a `RenderGraph`, it registers a resource and returns its UUID for *logical* use.

### Resource information

Resources contain information relevant to their processing as part of a render graph.

Physical resources (`PhysicalBuffer`, `PhysicalImage` and `PhysicalAttachment`) contain:
- Full resource information after analysis on the render graph of all usage, access, stages, etc.
- Vulkan handles required to live as long as the resource is used.
