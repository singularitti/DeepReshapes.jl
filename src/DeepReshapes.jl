module DeepReshapes

export structure
export deep_length
export deep_reshape
export flatten
export pack

# This defines where to stop recursing and where to descend further when
# producing values.
ScalarDefault = Union(Number, String, Nothing)
DeepDefault = Union(Array, Tuple)

# structure() recursively describes a deep-reshapable object
#   1. so that reshapes can be reversed and
#   2. to determine the number of Scalar values contained within in the
#      deep_length() function.
#
# In the untyped version (typed = false) this is a generalization of
# Base.size(): the structure of
#   - a Scalar is an empty tuple,
#   - an Array of Scalars is the Tuple of dimensions,
#   - an Array of non-Scalars is the Array{Any} of structures of its elements,
#   - a non-empty Tuple is the Tuple of structures of its elements,
#   - an empty Tuple is nothing.
#
# When typed = true, the structure of
#   - a Scalar is its type,
#   - an Array of Scalars is the Tuple of its element type concatenated with its
#     dimensions,
#   - and of all other objects is the same as in the untyped case.
#
# NOTE: Deep and Scalar can only be passed to deep_reshape, all other functions
#       in this module assume the defaults. This is because the structure
#       description format and deep_length behavior would need to be extended
#       to arbitrary types to make this possible.

structure{T <: ScalarDefault}(x::T; typed::Bool = true) = typed ? T : ()

structure(x::(); typed::Bool = true) = nothing

structure(x::(Any, Any...); typed::Bool = true) =
  tuple([structure(e, typed = typed) for e in x]...)

structure{T <: ScalarDefault}(x::Array{T}; typed::Bool = true) =
  typed ? tuple(T, Base.size(x)...) : Base.size(x)

structure(x::Array; typed::Bool = true) =
  map!(e -> structure(e, typed = typed), similar(x, Any), x)
  # ^ We use map! here because we want to define the element type of the result.


# A deep_reshape(x, structure) iterates over all the Scalars (recursively)
# contained within x, and copies them into a now object whose structure is
# defined in the same format that structure() returns.
#
# This happens in two independent processes which are executed in parallel:
#   - produce_scalars() recursively walks the input and yields the scalars, and
#   - consume_stucture() recursively builds the result structure from them.

deep_reshape(x, structure; Scalar = ScalarDefault, Deep = DeepDefault) =
  consume_stucture(@task(produce_scalars(x, Scalar, Deep)), structure)

deep_reshape(x, structure...) = deep_reshape(x, structure)

# If x is a Scalar, produce it, no matter whether it is also of a Deep type or
# not.
produce_scalars{T}(x::T, Scalar::Type{T}, Deep::Type) = produce(x)
produce_scalars{T}(x::T, Scalar::Type{T}, Deep::Type{T}) = produce(x)

# If x is not a Scalar and of a Deep type, recurse.
produce_scalars{T}(x::T, Scalar::Type, Deep::Type{T}) =
  for e in x; produce_scalars(e, Scalar, Deep) end

  
function consume_stucture(
  producer::Task,
  MemberType::DataType,
  dimensions::Integer...
)
  if dimensions == ()
    convert(MemberType, consume(producer))
  else
    result = Array(MemberType, dimensions...)
    copy_object(i...) = result[i...] = consume(producer)
    cartesianmap(copy_object, dimensions)
  
    result
  end
end

consume_stucture(producer::Task, structure::(Integer...)) =
  consume_stucture(producer, Any, structure...)

consume_stucture(producer::Task, structure::(DataType, Integer...)) =
  consume_stucture(producer, structure...)

consume_stucture(producer::Task, structure::Tuple) =
  tuple([consume_stucture(producer::Task, element) for element in structure]...)

function consume_stucture(producer::Task, structure::Array)
  result = Array(Any, size(structure)...)
  
  copy_into_result(i...) = result[i...] =
    consume_stucture(producer, structure[i...])
  cartesianmap(copy_into_result, size(structure))
  
  result
end

consume_stucture(producer::Task, _::Nothing) = ()

# This returns the count of Scalar values (recursively) contained within x.
deep_length(x) = structure_length(structure(x, typed = false))

structure_length(_::Nothing) = 0
structure_length(_::()) = 1
structure_length(x::(Integer...)) = prod(x)
structure_length(x) = sum(structure_length, x)

# This is just a convenience wrapper to reshape any deep-reshapable into a flat
# Array, optionally with a fixed result element type.

flatten(MemberType::DataType, x...) =
  deep_reshape(x, MemberType, deep_length(x))

flatten(x...) = flatten(Any, x...)

# A convenience wrapper to ease converting between deep and flat representations
# of the same data.

function pack(MemberType::DataType, data...)
  s = structure(data, typed = false)
  l = structure_length(s)
  (deep_reshape(data, MemberType, l), s)
end

pack(x...) = pack(Any, x...)

end # module
