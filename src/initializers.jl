
# This file is adapted from JuliaArrays/StaticArrays.jl License is MIT:
# https://github.com/JuliaArrays/StaticArrays.jl/blob/master/LICENSE.md

"""
    TV[ elements ]
    TV{T}[ elements ]

Create `Values` literals using array construction syntax. The element type is
inferred by promoting `elements` to a common type or set to `T` when `T` is
provided explicitly.

# Examples:

* `TV[1.0, 2.0]` creates a length-2 `Values` of `Float64` elements.
* `TV{Float32}[1, 2]` creates a length-2 `Values` of `Float32` elements.

A couple of helpful type aliases are also provided:

* `TV_F64[1, 2]` creates a length-2 `Values` of `Float64` elements
* `TV_F32[1, 2]` creates a length-2 `Values` of `Float32` elements
"""
struct TV{T} ; end

const TV_F32 = TV{Float32}
const TV_F64 = TV{Float64}

@inline similar_type(::Type{TV}, ::Val{S}) where {S} = Values{S}
@inline similar_type(::Type{TV{T}}, ::Val{S}) where {T,S} = Values{S,T}

# These definitions are duplicated to avoid matching `sa === Union{}` in the
# neater-looking alternative `sa::Type{<:TV}`.
@inline Base.getindex(sa::Type{TV}, xs...)            = similar_type(sa, Val(length(xs)))(xs)
@inline Base.getindex(sa::Type{TV{T}}, xs...) where T = similar_type(sa, Val(length(xs)))(xs)

@inline Base.typed_vcat(sa::Type{TV}, xs::Number...)            = similar_type(sa, Val(length(xs)))(xs)
@inline Base.typed_vcat(sa::Type{TV{T}}, xs::Number...) where T = similar_type(sa, Val(length(xs)))(xs)
