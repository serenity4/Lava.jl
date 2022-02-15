spirv_capability(spec::Vk.SpecCapabilitySPIRV) = getproperty(SPIRV, Symbol(:Capability, spec.name))

function has_feature(features, fcond::Vk.FeatureCondition)
    haskey(features, fcond.type) && getproperty(features[fcond.type], fcond.member)
end

function has_property(props, pcond::Vk.PropertyCondition)
    haskey(props, pcond.type) && begin
        prop = getproperty(props[pcond.type], pcond.member)
        prop isa Bool ? prop : getproperty(Vk, pcond.bit) in prop
    end 
end

function SPIRV.SupportedFeatures(physical_device, device_extensions, device_features)
    props = Vk.get_physical_device_properties_2(physical_device)

    exts = String[]
    for spec in Vk.SPIRV_EXTENSIONS
        if spec.promoted_to ≤ props.api_version || any(ext in device_extensions for ext in spec.enabling_extensions)
            push!(exts, spec.name)
        end
    end

    caps = SPIRV.Capability[]
    feature_dict = Dictionary(nameof(typeof(f)) => f for f in unchain(device_features))
    prop_dict = Dictionary(nameof(typeof(p)) => p for p in unchain(props))
    for spec in Vk.SPIRV_CAPABILITIES
        if spec.core_version ≤ props.api_version ||
            any(ext in device_extensions for ext in spec.enabling_extensions) ||
            any(Base.Fix1(has_feature, feature_dict), spec.enabling_features) ||
            any(Base.Fix1(has_property, prop_dict), spec.enabling_properties) ||

            push!(caps, spirv_capability(spec.name))
        end
    end

    SupportedFeatures(exts, caps)
end
