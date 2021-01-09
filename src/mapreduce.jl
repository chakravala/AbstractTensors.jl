
"""
    _InitialValue

A singleton type for representing "universal" initial value (identity element).

The idea is that, given `op` for `mapfoldl`, virtually, we define an "extended"
version of it by

    op′(::_InitialValue, x) = x
    op′(acc, x) = op(acc, x)

This is just a conceptually useful model to have in mind and we don't actually
define `op′` here  (yet?).  But see `Base.BottomRF` for how it might work in
action.

(It is related to that you can always turn a semigroup without an identity into
a monoid by "adjoining" an element that acts as the identity.)
"""
struct _InitialValue end

@inline _first(a1, as...) = a1

################
## map / map! ##
################

# In 0.6 the three methods below could be replaced with
# `map(f, as::Union{<:StaticArray,AbstractArray}...)` which included at least one `StaticArray`
# this is not the case on 0.7 and we instead hope to find a StaticArray in the first two arguments.
@inline function Base.map(f, a1::TupleVector, as::AbstractArray...)
    _map(f, a1, as...)
end
@inline function Base.map(f, a1::AbstractArray, a2::TupleVector, as::AbstractArray...)
    _map(f, a1, a2, as...)
end
@inline function Base.map(f, a1::TupleVector, a2::TupleVector, as::AbstractArray...)
    _map(f, a1, a2, as...)
end

@generated function _map(f, a::AbstractArray...)
    first_tuplevector = findfirst(ai -> ai <: TupleVector, a)
    if first_tuplevector === nothing
        return :(throw(ArgumentError("No TupleVector found in argument list")))
    end
    # Passing the Val as an argument to _map leads to inference issues when
    # recursively mapping over nested TupleVectors (see issue #593). Calling
    # Val in the generator here is valid because a[first_staticarray] is known to be a
    # TupleVector for which the default Val method is correct. If wrapped
    # TupleVector (with a custom Val method) are to be supported, this will
    # no longer be valid.
    S = length(a[first_tuplevector])

    if S == 0
        # In the empty case only, use inference to try figuring out a sensible
        # eltype, as is done in Base.collect and broadcast.
        # See https://github.com/JuliaArrays/StaticArrays.jl/issues/528
        eltys = [:(eltype(a[$i])) for i ∈ 1:length(a)]
        return quote
            @_inline_meta
            S = same_size(a...)
            T = Core.Compiler.return_type(f, Tuple{$(eltys...)})
            @inbounds return similar_type(a[$first_staticarray], T, S)()
        end
    end

    exprs = Vector{Expr}(undef, S)
    for i ∈ 1:S
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        exprs[i] = :(f($(tmp...)))
    end

    return quote
        @_inline_meta
        S = same_size(a...)
        @inbounds elements = tuple($(exprs...))
        @inbounds return similar_type(typeof(_first(a...)), eltype(elements), S)(elements)
    end
end

@inline function Base.map!(f, dest::TupleVector{n}, a::TupleVector{n}...) where n
    _map!(f, dest, Val(n), a...)
end

# Ambiguities with Base:
@inline function map!(f, dest::TupleVector{n}, a::TupleVector{n}) where n
    _map!(f, dest, Val(n), a)
end
@inline function map!(f,dest::TupleVector{n},a::TupleVector{n},b::TupleVector{n}) where n
    _map!(f, dest, Val(n), a, b)
end


@generated function _map!(f, dest, ::Val{S}, a::TupleVector...) where {S}
    exprs = Vector{Expr}(undef, S)
    for i ∈ 1:S
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        exprs[i] = :(dest[$i] = f($(tmp...)))
    end
    return quote
        @_inline_meta
        @inbounds $(Expr(:block, exprs...))
    end
end

###############
## mapreduce ##
###############

@inline function Base.mapreduce(f, op, a::TupleVector, b::TupleVector...; dims=:, init = _InitialValue())
    _mapreduce(f, op, dims, init, same_size(a, b...), a, b...)
end

@inline _mapreduce(args::Vararg{Any,N}) where N = _mapfoldl(args...)

@generated function _mapfoldl(f, op, dims::Colon, init, ::Val{S}, a::TupleVector...) where {S}
    if S == 0
        if init === _InitialValue
            if length(a) == 1
                return :(Base.mapreduce_empty(f, op, $(eltype(a[1]))))
            else
                return :(throw(ArgumentError("reducing over an empty collection is not allowed")))
            end
        else
            return :init
        end
    end
    tmp = [:(a[$j][1]) for j ∈ 1:length(a)]
    expr = :(f($(tmp...)))
    if init === _InitialValue
        expr = :(Base.reduce_first(op, $expr))
    else
        expr = :(op(init, $expr))
    end
    for i ∈ 2:S
        tmp = [:(a[$j][$i]) for j ∈ 1:length(a)]
        expr = :(op($expr, f($(tmp...))))
    end
    return quote
        @_inline_meta
        @inbounds return $expr
    end
end

@inline function _mapreduce(f, op, D::Int, init, sz::Val{S}, a::TupleVector) where {S}
    # Body of this function is split because constant propagation (at least
    # as of Julia 1.2) can't always correctly propagate here and
    # as a result the function is not type stable and very slow.
    # This makes it at least fast for three dimensions but people should use
    # for example any(a; dims=Val(1)) instead of any(a; dims=1) anyway.
    if D == 1
        return _mapreduce(f, op, Val(1), init, sz, a)
    else
        return _mapreduce(f, op, Val(D), init, sz, a)
    end
end

@generated function _mapfoldl(f, op, dims::Val{1}, init,
                               ::Val{S}, a::TupleVector) where {S,D}

    exprs = Array{Expr}(undef, 1)
    for i ∈ Base.product(1:1)
        expr = :(f(a[$(i...)]))
        if init === _InitialValue
            expr = :(Base.reduce_first(op, $expr))
        else
            expr = :(op(init, $expr))
        end

        exprs[i...] = expr
    end

    return quote
        @_inline_meta
        @inbounds elements = tuple($(exprs...))
        @inbounds return similar_type(a, eltype(elements), Val(1))(elements)
    end
end

############
## reduce ##
############

@inline Base.reduce(op, a::TupleVector; dims = :, init = _InitialValue()) =
    _reduce(op, a, dims, init)

# disambiguation
Base.reduce(::typeof(vcat), A::TupleVector{n,<:AbstractVecOrMat}) where n =
    Base._typed_vcat(mapreduce(eltype, promote_type, A), A)
Base.reduce(::typeof(vcat), A::TupleVector{n,<:TupleVectorLike}) where n =
    _reduce(vcat, A, :, _InitialValue())

Base.reduce(::typeof(hcat), A::TupleVector{n,<:AbstractVecOrMat}) where n =
    Base._typed_hcat(mapreduce(eltype, promote_type, A), A)
Base.reduce(::typeof(hcat), A::TupleVector{n,<:TupleVectorLike}) where n =
    _reduce(hcat, A, :, _InitialValue())

@inline _reduce(op, a::TupleVector, dims, init = _InitialValue()) =
    _mapreduce(identity, op, dims, init, length_val(a), a)

################
## (map)foldl ##
################

# Using `where {R}` to force specialization. See:
# https://docs.julialang.org/en/v1.5-dev/manual/performance-tips/#Be-aware-of-when-Julia-avoids-specializing-1
# https://github.com/JuliaLang/julia/pull/33917

@inline Base.mapfoldl(f::F, op::R, a::TupleVector; init = _InitialValue()) where {F,R} =
    _mapfoldl(f, op, :, init, length_val(a), a)
@inline Base.foldl(op::R, a::TupleVector; init = _InitialValue()) where {R} =
    _foldl(op, a, :, init)
@inline _foldl(op::R, a, dims, init = _InitialValue()) where {R} =
    _mapfoldl(identity, op, dims, init, length_val(a), a)

#######################
## related functions ##
#######################

# These are all similar in Base but not @inline'd
#
# Implementation notes:
#
# 1. mapreduce and mapreducedim usually do not take initial value, because we don't
# always know the return type of an arbitrary mapping function f.  (We usually want to use
# some initial value such as one(T) or zero(T), where T is the return type of f, but
# if users provide type-unstable f, its return type cannot be known.)  Therefore, mapped
# versions of the functions implemented below usually require the collection to have at
# least two entries.
#
# 2. Exceptions are the ones that require Boolean mapping functions.  For example, f in
# all and any must return Bool, so we know the appropriate initial value is true and false,
# respectively.  Therefore, all(f, ...) and any(f, ...) are implemented by mapreduce(f, ...)
# with an initial value v0 = true and false.
#
# TODO: change to use Base.reduce_empty/Base.reduce_first
@inline Base.iszero(a::TupleVector{N,T}) where {N,T} = reduce((x,y) -> x && iszero(y), a, init=true)

@inline Base.sum(a::TupleVector{N,T}; dims=:) where {N,T} = _reduce(+, a, dims)
@inline Base.sum(f, a::TupleVector{N,T}; dims=:) where {N,T} = _mapreduce(f, +, dims, _InitialValue(), Val(N), a)
@inline Base.sum(f::Union{Function, Type}, a::TupleVector{N,T}; dims=:) where {N,T} = _mapreduce(f, +, dims, _InitialValue(), Val(N), a) # avoid ambiguity

@inline Base.prod(a::TupleVector{N,T}; dims=:) where {N,T} = _reduce(*, a, dims)
@inline Base.prod(f, a::TupleVector{N,T}; dims=:) where {N,T} = _mapreduce(f, *, dims, _InitialValue(), Val(N), a)
@inline Base.prod(f::Union{Function, Type}, a::TupleVector{N,T}; dims=:) where {N,T} = _mapreduce(f, *, dims, _InitialValue(), Val(N), a)

@inline Base.count(a::TupleVector{N,Bool}; dims=:) where N = _reduce(+, a, dims)
@inline Base.count(f, a::TupleVector{N}; dims=:) where N = _mapreduce(x->f(x)::Bool, +, dims, _InitialValue(), Val(N), a)

@inline Base.all(a::TupleVector{N,Bool}; dims=:) where N = _reduce(&, a, dims, true)  # non-branching versions
@inline Base.all(f::Function, a::TupleVector{N}; dims=:) where N = _mapreduce(x->f(x)::Bool, &, dims, true, Val(N), a)

@inline Base.any(a::TupleVector{N,Bool}; dims=:) where N = _reduce(|, a, dims, false) # (benchmarking needed)
@inline Base.any(f::Function, a::TupleVector{N}; dims=:) where N = _mapreduce(x->f(x)::Bool, |, dims, false, Val(N), a) # (benchmarking needed)

@inline Base.in(x, a::TupleVector{N}) where N = _mapreduce(==(x), |, :, false, Val(N), a)

#=_mean_denom(a, dims::Colon) = length(a)
_mean_denom(a, dims::Int) = size(a, dims)
_mean_denom(a, ::Val{D}) where {D} = size(a, D)
_mean_denom(a, ::Type{Val{D}}) where {D} = size(a, D)

@inline Statistics.mean(a::TupleVector; dims=:) = _reduce(+, a, dims) / _mean_denom(a, dims)
@inline Statistics.mean(f::Function, a::TupleVector{N}; dims=:) = _mapreduce(f, +, dims, _InitialValue(), Val(N), a) / _mean_denom(a, dims)=#

@inline Base.minimum(a::TupleVector; dims=:) = _reduce(min, a, dims) # base has mapreduce(idenity, scalarmin, a)
@inline Base.minimum(f::Function, a::TupleVector{N}; dims=:) where N = _mapreduce(f, min, dims, _InitialValue(), Val{N}, a)

@inline Base.maximum(a::TupleVector; dims=:) = _reduce(max, a, dims) # base has mapreduce(idenity, scalarmax, a)
@inline Base.maximum(f::Function, a::TupleVector; dims=:) = _mapreduce(f, max, dims, _InitialValue(), Val(N), a)

# Diff is slightly different
@inline LinearAlgebra.diff(a::TupleVector{N};dims=Val(1)) where N = _diff(Val(N),a,dims)

@inline function _diff(sz::Val, a::TupleVector, D::Int)
    _diff(sz,a,Val(D))
end
@generated function _diff(::Val{N}, a::TupleVector, ::Val{1}) where N
    Snew = N-1
    exprs = Array{Expr}(undef, Snew)
    for i1 = Base.product(1:Snew)
        i2 = copy([i1...])
        i2[1] = i1[1] + 1
        exprs[i1...] = :(a[$(i2...)] - a[$(i1...)])
    end
    return quote
        @_inline_meta
        elements = tuple($(exprs...))
        @inbounds return similar_type(a, eltype(elements), Val($Snew))(elements)
    end
end

_maybe_val(dims::Integer) = Val(Int(dims))
_maybe_val(dims) = dims
_valof(::Val{D}) where D = D

@inline Base.accumulate(op::F, a::TupleVector; dims = :, init = _InitialValue()) where {F} =
    _accumulate(op, a, _maybe_val(dims), init)

@inline function _accumulate(op::F, a::TupleVector, dims::Union{Val,Colon}, init) where {F}
    # Adjoin the initial value to `op` (one-line version of `Base.BottomRF`):
    rf(x, y) = x isa _InitialValue ? Base.reduce_first(op, y) : op(x, y)

    if isempty(a)
        T = return_type(rf, Tuple{typeof(init), eltype(a)})
        return similar_type(a, T)()
    end

    results = _foldl(
        a,
        dims,
        (similar_type(a, Union{}, Val(0))(), init),
    ) do (ys, acc), x
        y = rf(acc, x)
        # Not using `push(ys, y)` here since we need to widen element type as
        # we iterate.
        (vcat(ys, TV[y]), y)
    end
    dims === (:) && return first(results)

    ys = map(first, results)
    # Now map over all indices of `a`.  Since `_map` needs at least
    # one `TupleVector` to be passed, we pass `a` here, even though
    # the values of `a` are not used.
    data = _map(a, CartesianIndices(a)) do _, CI
        D = _valof(dims)
        I = Tuple(CI)
        J = Base.setindex(I, 1, D)
        ys[J...][I[D]]
    end
    return similar_type(a, eltype(data))(data)
end

@inline Base.cumsum(a::TupleVector; kw...) = accumulate(Base.add_sum, a; kw...)
@inline Base.cumprod(a::TupleVector; kw...) = accumulate(Base.mul_prod, a; kw...)
