# Lava

[![Build Status](https://github.com/serenity4/Lava.jl/workflows/CI/badge.svg)](https://github.com/serenity4/Lava.jl/actions)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

Rewrite of the Vulkan API in a more idiomatic Julia style, without loss of functionality and with minimal overhead.

Main differences with the Vulkan API:
- Handle hierarchies and parameters are encoded in types. This should greatly simplify the number of call parameters and create info structures that you have to specify.
- API functions are rewritten with multiple dispatch in mind.
- Vulkan-related functionality is separate from the main data structures, allowing types and functions to be extendable and support other APIs.
