# Lava

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Renderer based on [Vulkan.jl](https://github.com/JuliaGPU/Vulkan.jl).

## Render graphs

This project uses a render graph to automate and optimize resource state transitions and resource synchronization. You can read more about render graphs here:
- [Render Graphs and Vulkan - a deep dive](http://themaister.net/blog/2017/08/15/render-graphs-and-vulkan-a-deep-dive/) - *Maister's Graphics Adventures (2017)*, in context of the [Granite renderer](https://github.com/Themaister/Granite)
- [FrameGraph: Extensible Rendering Architecture in Frostbite](https://www.gdcvault.com/play/1024612/FrameGraph-Extensible-Rendering-Architecture-in) - *Yuriy O'Donnell (2017)*
- [Render Dependency Graph](https://docs.unrealengine.com/5.0/en-US/render-dependency-graph-in-unreal-engine/) - *Unreal Engine 5 Manual (2023)*
- [Organizing GPU Work with Directed Acyclic Graphs](https://levelup.gitconnected.com/organizing-gpu-work-with-directed-acyclic-graphs-f3fd5f2c2af3) - *Pavlo Muratov (2020)*, in context of the [PathFinder renderer](https://github.com/man-in-black382/PathFinder)
- [Render Graphs](https://logins.github.io/graphics/2021/05/31/RenderGraphs.html) - *Riccardo Loggini (2021)*
