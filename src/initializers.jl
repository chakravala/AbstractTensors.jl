
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

* `TV_F64[1, 2]` creates a lenght-2 `Values` of `Float64` elements
* `TV_F32[1, 2]` creates a lenght-2 `Values` of `Float32` elements
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

#=@inline Base.typed_hcat(sa::Type{TV}, xs::Number...)            = similar_type(sa, Val(length(xs)))(xs)
@inline Base.typed_hcat(sa::Type{TV{T}}, xs::Number...) where T = similar_type(sa, Val(length(xs)))(xs)

Base.@pure function _TV_hvcat_transposed_size(rows)
    M = rows[1]
    if any(r->r != M, rows)
        # @pure may not throw... probably. See
        # https://discourse.julialang.org/t/can-pure-functions-throw-an-error/18459
        return nothing
    end
    Val(length(rows))
end

@inline function _TV_typed_hvcat(sa, rows, xs)
    msize = _TV_hvcat_transposed_size(rows)
    if msize === nothing
        throw(ArgumentError("TV[...] matrix rows of length $rows are inconsistent"))
    end
    # hvcat lowering is row major ordering, so we must transpose
    transpose(similar_type(sa, msize)(xs))
end

@inline Base.typed_hvcat(sa::Type{TV}, rows::Dims, xs::Number...)            = _TV_typed_hvcat(sa, rows, xs)
@inline Base.typed_hvcat(sa::Type{TV{T}}, rows::Dims, xs::Number...) where T = _TV_typed_hvcat(sa, rows, xs)
=#
