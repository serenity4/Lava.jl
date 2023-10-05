using Documenter
using Literate
using Lava

function julia_files(dir)
    files = reduce(vcat, [joinpath(root, file) for (root, dirs, files) in walkdir(dir) for file in files])
    sort(filter(endswith(".jl"), files))
end

function replace_edit(content)
    haskey(ENV, "JULIA_GITHUB_ACTIONS_CI") && return content
    # Linking does not work locally, but we can make
    # the warning go away with a hard link to the repo.
    replace(
        content,
        r"EditURL = \".*<unknown>/(.*)\"" => s"EditURL = \"https://github.com/serenity4/Lava.jl/tree/main/\1\"",
    )
end

function generate_markdowns()
    dir = joinpath(@__DIR__, "src")
    Threads.@threads for file in julia_files(dir)
        Literate.markdown(
            file,
            dirname(file);
            postprocess = replace_edit,
            documenter = true,
        )
    end
end

generate_markdowns()

makedocs(;
    modules=[Lava],
    format=Documenter.HTML(prettyurls = true),
    pages=[
        "Home" => "index.md",
        "Tutorial" => [
        ],
        "How to" => [
        ],
        "Reference" => [
        ],
        "Explanations" => [
        ],
        "API" => "api.md",
        "Developer documentation" => [
            "Resources" => "devdocs/resources.md",
        ],
    ],
    repo="https://github.com/serenity4/Lava.jl/blob/{commit}{path}#L{line}",
    sitename="Lava.jl",
    authors="serenity4 <cedric.bel@hotmail.fr>",
    strict=false,
    doctest=false,
    checkdocs=:exports,
    linkcheck=:true,
)

deploydocs(
    repo = "github.com/serenity4/Lava.jl.git",
    push_preview = true,
)
