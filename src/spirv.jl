spirv_capability(spec::Vk.SpecCapabilitySPIRV) = getproperty(SPIRV, Symbol(:Capability, spec.name))

function has_feature(features, fcond::Vk.FeatureCondition, api_version)
    !isnothing(fcond.core_version) && fcond.core_version > api_version && return false
    haskey(features, fcond.type) && getproperty(features[fcond.type], fcond.member)
end

function has_property(props, pcond::Vk.PropertyCondition, api_version)
    !isnothing(pcond.core_version) && pcond.core_version > api_version && return false
    haskey(props, pcond.type) && begin
        prop = getproperty(props[pcond.type], pcond.member)
        prop isa Bool ? prop : getproperty(Vk, pcond.bit) in prop
    end 
end

function get_required_property_types(api_version)
    props = reduce(vcat, map(Vk.SPIRV_CAPABILITIES) do cap
        filter(x -> api_version ≥ something(x.core_version, typemin(VersionNumber)), cap.enabling_properties)
    end)
    unique(getproperty.(Ref(Vk), getproperty.(props, :type)))
end

function spirv_features(physical_device, device_extensions, device_features)
    props = Vk.get_physical_device_properties_2(physical_device)
    (; api_version) = props.properties
    props = Vk.get_physical_device_properties_2(physical_device, get_required_property_types(api_version)...)

    exts = String[]
    for spec in Vk.SPIRV_EXTENSIONS
        if something(spec.promoted_to, typemin(VersionNumber)) ≤ api_version || any(ext in device_extensions for ext in spec.enabling_extensions)
            push!(exts, spec.name)
        end
    end

    caps = SPIRV.Capability[]
    feature_dict = dictionary(nameof(typeof(f)) => f for f in Vk.unchain(device_features))
    prop_dict = dictionary(nameof(typeof(p)) => p for p in Vk.unchain(props))
    for spec in Vk.SPIRV_CAPABILITIES
        if !isnothing(spec.promoted_to) && spec.promoted_to ≤ api_version ||
            any(ext in device_extensions for ext in spec.enabling_extensions) ||
            any(has_feature(feature_dict, f, api_version) for f in spec.enabling_features) ||
            any(has_property(prop_dict, p, api_version) for p in spec.enabling_properties)

            push!(caps, spirv_capability(spec))
        end
    end

    SupportedFeatures(exts, caps)
end
