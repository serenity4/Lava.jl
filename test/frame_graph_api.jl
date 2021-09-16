#=

Definition of resources.
Attachments are swapchain-sized by default.

=#

emissive = Attachment(ATTACHMENT_COLOR)
albedo = Attachment(ATTACHMENT_COLOR)
normal = Attachment(ATTACHMENT_COLOR)
pbr = Attachment(ATTACHMENT_COLOR)
depth = Attachment(ATTACHMENT_DEPTH)
output = Attachment(ATTACHMENT_COLOR)

shadow_main = Texture(...)
shadow_near = Texture(...)
average_luminance = Buffer(...)
bloom_downsample_3 = Texture(...)

#=

Definition of passes.
By default, there is only one subpass, aliased with the pass itself. If other passes are provided,
then each pass will need to be used separately in a program, and not the pass itself.

Example:
```
sp1 = SubPass()
sp2 = SubPass()
sp3 = SubPass()
pass = Pass(STAGE_GRAPHICS_ALL, [sp1, sp2, sp3])
```

We might want to make sp1, sp2 and sp3 passes and detect that they are subpasses when they are passed as arguments to a `Pass`.
This can then lead to a program of the form

```
prog = @program begin
    store = sp1(input_1, input_2)
    store_2 = sp2(input_1, input_3)
    output = sp3(store, store_2)
end
```

One interesting syntax could also be

```
prog = @program begin
    output = sp3(sp1(input_1, input_2), sp2(input_1, input_3))
end
```

Where the result of `sp1(...)` and `sp2(...)` could be assigned to a programmatically generated variable,
possibly interpreted as a transient resource.

=#

gbuffer = Pass(STAGE_GRAPHICS_ALL)
lighting = Pass(STAGE_GRAPHICS_ALL)
adapt_luminance = Pass(STAGE_COMPUTE)
combine = Pass(STAGE_GRAPHICS_ALL)

#=

Program definition.

Access to resources (read, write, read-write) is determined by the assignment/use logic.
B = f(A) means that in pass F, B is written to and A is read
B = g(A, B) means that in pass G, B is read and written to and A is read
C = h() means that in pass h, C is written to and nothing is read

Resources may be attachments, images or buffers.

Variables also have type annotations, which specify their usage in the current pass.
So `B::Color = f(A::Input)` would mean that B is a color attachment, and that a is an
input attachment. But later we may have `B::Color = g(A::Depth, B::Input), that is,
different usage in a different pass.

=#

resources = @resources begin
    shadow_main = 
    shadow_near = Texture(...)
    average_luminance = Buffer(...)
    bloom_downsample_3 = Texture(...)
end

prog = @program begin
    emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth = gbuffer(vbuffer::Buffer::Vertex, ibuffer::Buffer::Index)
    color::Color = lighting(emissive::Color, albedo::Color, normal::Color, pbr::Color, depth::Depth, shadow_main::Texture, shadow_near::Texture)
    average_luminance::Buffer::Storage = adapt_luminance(average_luminance::Buffer::Storage, bloom_downsample_3::Texture)
    output::Color = combine(color::Color, average_luminance::Texture)
end

#=

## Image layouts and usage flags

The layout and usage of every attachment or image, as well as the usage of every buffer is determined by the access type (read, write, read-write), the shader stage and the type of resource.

A vertex buffer is directly associated with a particular usage (`BUFFER_USAGE_VERTEX_BUFFER_BIT`), stage (`PIPELINE_STAGE_VERTEX_INPUT_BIT`) and access (`ACCESS_VERTEX_ATTRIBUTE_READ_BIT`).
Same story for the index buffer, but with `BUFFER_USAGE_INDEX_BUFFER_BIT`, `PIPELINE_STAGE_VERTEX_INPUT_BIT` and `ACCESS_INDEX_READ_BIT`.

Color attachment:
- usage: `IMAGE_USAGE_COLOR_ATTACHMENT_BIT`
- layout & access:
    - read-only (input attachment): `IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` with access `ACCESS_COLOR_ATTACHMENT_READ_BIT`
    - write-only (standard color output): `IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL` with access `ACCESS_COLOR_ATTACHMENT_WRITE_BIT`
    - read-write (e.g. programmable blending); provided both as color attachment and as input attachment: `IMAGE_LAYOUT_GENERAL` with access `ACCESS_COLOR_ATTACHMENT_READ_BIT` and `ACCESS_COLOR_ATTACHMENT_WRITE_BIT`.

Depth attachment:
- usage: `IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT`
- layout and access:
    - read-only: `IMAGE_LAYOUT_DEPTH_READ_ONLY_OPTIMAL`
    - write-only: `IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL`

Stencil attachment:
- usage: `IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT`
- layout and access:
    - read-only: IMAGE_LAYOUT_STENCIL_READ_ONLY_OPTIMAL
    - write-only: `IMAGE_LAYOUT_STENCIL_ATTACHMENT_OPTIMAL`

Depth + stencil attachment:
- usage: `IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT`
- layout and access:
    - depth + stencil read-only: `IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL`
    - depth write-only (input attachment), stencil read-only: `IMAGE_LAYOUT_DEPTH_ATTACHMENT_STENCIL_READ_ONLY_OPTIMAL`
    - depth read-only, stencil write-only (input attachment): `IMAGE_LAYOUT_DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL`
    - both write-only: `IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL`

Input attachment (depth/stencil/color excluded):
- usage: `IMAGE_USAGE_INPUT_ATTACHMENT_BIT`
- layout and access:
    - read-only: `IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL` with access `ACCESS_INPUT_ATTACHMENT_READ_BIT`
    - write-only: unsupported


Note: we also need to know how attachments should be loaded/stored. This info only applies to the first and last times attachments are used.
Mapping from attachment load operations + aspect to memory access masks.

For input attachments, the aspect needs to be provided. In color and depth/stencil attachments, there is no need to do so, because the attachment type already includes the aspect. We can fill `InputAttachmentAspectReference` when generating subpasses (old API) or we can use `AttachmentReference2` (new API).
Also need to know which pipeline type subpasses will be using (computde or graphics). Can get it from pass stage masks provided by the user.
For attachments, can put image layout in subpass references, done automatically with subpass dependencies.
Need to know which objects will be attachments, which will be storage images/buffers.
Possibility to create framebuffer images/image views at framebuffer creation time with `FramebufferAttachmentsCreateInfo` in `next`.

For render passes, we have the possibility to use multiview or not. Maybe we can skip it at first.
`render_area` must be coherent with the pipeline; ultimately one of the render pass or the pipeline should determine the value. However, we need the render pass to create the pipeline, so... I should probably move out from this and separate the specification of programmable stages from the fixed-function stages that are tied to render pass operations.

For each subpass, I need to know how it will be recorded: inline, or with secondary command buffers.
Image views need to be specified at render pass begin time in addition to framebuffer creation.
Subpasses separate resolve operations: each subpass will do its own resolve at the end.


Can investigate:

Is multisampling kind of orthogonal to anything? As in, can we separate the multisampling logic?

The number of samples needs to be specified both at in the render pass somewhere and in the pipeline.

Information regarding write masks for color and depth/stencil at pipeline creation need to be coherent with edge cases that arise with certain attachment combinations (in particular, when a color or depth/stencil attachment is used as both its "aspect" attachment and as input attachment). This affects synchronization for subpasses with self-dependencies.

Clearly, one has to win over the other: only one of the render pass or the pipeline (program) has to specify this information.

## Shader interface

This is where vertex input assembly is specified.

## (Graphics) Pipeline

A pipeline specifies shaders used for rendering. The whole graphics pipeline can be thought of as a function which executes shader methods.

## Program

A program is a set of computations carried out with pipelines. A program corresponds to the logic of a given pass.

OR

A program is a set of passes that perform computations. A program defines a specific execution order and access order, based on the use of resources between passes.

## Render Pass

A render pass specifies a render target, rendering options (multisampling, area) and a program that executes pipelines.
Subpasses are abstracted away as simple passes. The render graph may turn passes into subpasses wherever appropriate.

## Features

I want to make it possible to not have to care about explicit synchronization.
This requires the knowledge of the GPU work that we intend to do. All of it.

## API

Programs implement passes. Passes can be provided to the render graph without associating them with programs; the same is true for 

### Pipeline

=#

main_pipeline = Pipeline(
    vertex_shader = Shader("my_shader.vert", GLSL, (pc_offset = PushConstant(0, UInt),)),
    fragment_shader = Shader("my_shader.frag", GLSL, (image = SampledImage(my_image; sample_parameters...),)),
    (color = ColorAttachment(), depth = DepthAttachment(), normal = ColorAttachment(), pbr = ColorAttachment(), albedo = ColorAttachment());
    name = "graphics_pipeline", # optional name
)

emissive_map = Pipeline(
    vertex_shader = Shader("my_emissive_map.vert", GLSL),
    fragment_shader = Shader("my_emissive_map.frag", GLSL),
    (emissive = ColorAttachment(),),
)

#=

Pipeline specify shaders and their interfaces.
The `main_pipeline`, labeled `"graphics_pipeline"`, has its inputs determined by all its shader parameters.
Here, a sampled image `"image"` has to be provided as well as a push constant `"pc_offset"`.

The emissive map has no parameters, and so has only vertex and, optionally, index data as input.

### Passes

=#

gbuffer = Pass(:gbuffer)

draw!(gbuffer, main_pipeline, vbuffer, ibuffer[1:120], PushConstant(:pc_offset) => 20)
for obj in objects
    draw!(gbuffer, main_pipeline, vbuffer(obj), ibuffer(obj), PushConstant(:pc_offset) => 20)
end
draw!(gbuffer, emissive_map, vbuffer[51:123], ibuffer[121:2325])

#=

Note that the pipelines can be used directly or used by a reference name. For example, `main_pipeline` could have been referred to as `"graphics_pipeline"`.
Instead of binding an index buffer and specifying a draw count manually, a slice is specified instead.
It is then assumed that all of the index sub-buffer will be used for drawing, so we will draw 120 vertices starting at offset 0 here.
The index buffer will only be bound once, even if a different slice was used before; the offset will simply be passed as vertex offset in the draw commands.
Of course, this is different if indirect indexing is used, in which case the option of rebinding a sub-buffer or assuming an offset in stored indices is left to the user.

Equivalently, we can use a macro to input the same information:
TODO: explain why variable assignment can be useful

=#

gbuffer = @pass begin
    color, depth, normal, pbr, albedo = main_pipeline(vbuffer[1:50], ibuffer[1:120]; pc_offset = 20)
    for obj in objects
        color, depth, normal, pbr, albedo = main_pipeline($(vbuffer(obj)), $(ibuffer(obj)); pc_offset = 20)
    end
    emissive = emissive_map(vbuffer[51:123], ibuffer[121:2325])
end

#=

## Dependency graph (directed, acyclic)

In this graph, vertices represent passes, and edges are resource dependencies between passes.
A topological sort of this graph represents a possible execution order that respects execution dependencies.

Reusing the example above, the graph has three vertices: `gbuffer`, `lighting` and `adapt_luminance`.
`gbuffer` has five outgoing edges to `lighting`, each edge being labeled with a resource.
`lighting` has one outgoing edge to `adapt_luminance`.

## Resource graph (directed)

In this graph, vertices represent resources (attachments, textures...), and edges represent operations that use these resources.

We can improve the dependency graph by 

=#
