struct PresentCommand <: CommandImplementation
  attachment::Resource
end

resource_dependencies(present::PresentCommand) = Dictionary([present.attachment], [ResourceDependency(RESOURCE_USAGE_PRESENT, READ, nothing, 1)])
