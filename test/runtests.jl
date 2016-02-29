using DeepReshapes
using Base.Test

a = [1, 2, 3]

b = Real[1.0, 2, 6//2]

C = [1.0 2.0 3.0;
     4.0 5.0 6.0]

D = [1.0 2.0 3.0 4.0]

E = (1, 2.0, 1//2)

F = (a, b, C, (D, E))

g = [1, (2, 3)]

h = [1 (2, 3)]

I = Vector[a, b]

# Untyped describe

@test describe(1, typed = false) == ()
@test describe(a, typed = false) == (3,)
@test describe(b, typed = false) == (3,)
@test describe(C, typed = false) == (2,3)
@test describe(D, typed = false) == (1,4)
@test describe(E, typed = false) == ((), (), ())
@test describe(F, typed = false) == ((3,), (3,), (2, 3), ((1, 4), ((), (), ())))
@test describe(g, typed = false) == [(), ((), ())]
@test describe(h, typed = false) == [() ((), ())]
@test describe(I, typed = false) == [(3,), (3,)]


# Typed (default) describe

@test describe(1) == Int
@test describe(a) == (Int, 3)
@test describe(b) == (Real, 3)
@test describe(C) == (Float64, 2,3)
@test describe(D) == (Float64, 1,4)
@test describe(E) == (Int, Float64, Rational{Int})
@test describe(F) == ((Int, 3), (Real, 3), (Float64, 2,3), ((Float64, 1,4), (Int, Float64, Rational{Int})))
@test describe(g) == [Int, (Int, Int)]
@test describe(h) == [Int (Int, Int)]
@test describe(I) == [(Int, 3), (Real, 3)]


# Length determination

@test deep_length(1) == 1
@test deep_length(a) == 3
@test deep_length(b) == 3
@test deep_length(C) == 6
@test deep_length(D) == 4
@test deep_length(E) == 3
@test deep_length(F) == 19
@test deep_length(g) == 3
@test deep_length(h) == 3
@test deep_length(I) == 6


# Flattening reshapes

@test deep_reshape(1, (Int,)) == 1
@test deep_reshape(a, (Int, 3)) == [1, 2, 3]
@test deep_reshape(b, (Real, 3)) == [1.0, 2, 3//1]
@test deep_reshape(C, (Float64, 6)) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
@test deep_reshape(D, (Float64, 4)) == [1.0, 2.0, 3.0, 4.0]
@test deep_reshape(E, (Real, 3)) == [1, 2.0, 1//2]
@test deep_reshape(F, (Real, 19)) == [1, 2, 3, 1, 2.0, 3//1, 1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 1.0, 2.0, 3.0, 4.0, 1, 2.0, 1//2]
@test deep_reshape(g, (Int, 3)) == [1, 2, 3]
@test deep_reshape(h, (Int, 3)) == [1, 2, 3]
@test deep_reshape(I, (Real, 6)) == [1, 2, 3, 1, 2.0, 3//1]


# Structuring reshapes

@test deep_reshape(1, ()) == 1
@test deep_reshape(1, (Int,)) == 1
@test deep_reshape(1, (Float64,)) == 1.0

@test deep_reshape([1, 2, 3], (Int, Int, Int)) == (1, 2, 3)
@test deep_reshape([1, 2, 3], (Int, Float64, Rational{Int})) == (1, 2.0, 3//1)
@test deep_reshape([1, 2, 3], [Int, Int, Int]) == [1, 2, 3]
@test deep_reshape([1, 2, 3], (Int, (Int, Int))) == (1, (2, 3))
@test deep_reshape([1, 2, 3], ((Int, Int), Int)) == ((1, 2), 3)
@test deep_reshape([1, 2, 3], (Int, [Int, Int])) == (1, [2, 3])
@test deep_reshape([1, 2, 3], [Int, (Int, Int)]) == [1, (2, 3)]

@test deep_reshape(collect(1:10), (2, 5)) == [1 3 5 7 9; 2 4 6 8 10]
@test deep_reshape(collect(1:10), (Int, 2, 5)) == [1 3 5 7 9; 2 4 6 8 10]
@test eltype(deep_reshape(collect(1:10), (Int, 2, 5))) == Int
@test deep_reshape(collect(1:10), (Float64, 2, 5)) == [1.0 3.0 5.0 7.0 9.0; 2.0 4.0 6.0 8.0 10.0]
@test eltype(deep_reshape(collect(1:10), (Float64, 2, 5))) == Float64

@test deep_reshape(collect(1:25), ((3, 3), (4, 4))) == ([1 4 7; 2 5 8; 3 6 9], [10  14  18  22; 11  15  19  23; 12  16  20  24; 13  17  21  25])
@test deep_reshape([1, 2, 3, 1.0, 2, 3//1], [(3,), (3,)]) == I


# Arbitrary reshapes

@test deep_reshape((C, D), (Float64, 2, 5)) == [1.0 2.0 3.0 1.0 3.0; 4.0 5.0 6.0 2.0 4.0]
@test deep_reshape((collect(1:10), [11 13 15; 12 14 16]), (2, 2, 2, 2)) == reshape(collect(1:16), 2, 2, 2, 2)


# Reshapes with non-default recursion behavior (experimental)

@test deep_reshape(1:6, (2, 3); Deep = Range) == [1 3 5; 2 4 6]
@test deep_reshape((1:3, 4:6, 7, 8:10), (10,); Deep = Union{Tuple, Range}) == collect(1:10)

# Convenience wrappers

@test deep_reshape(collect(1:100), 10, 10) == reshape(1:100, 10, 10)
@test deep_reshape([1, 2, 3], Float64, Int, Rational{Int}) == (1.0, 2, 3//1)

@test flatten(I) == [1, 2, 3, 1, 2.0, 3//1]
@test flatten(Float64, C, D) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 1.0, 2.0, 3.0, 4.0]
@test eltype(flatten(Float64, C, D)) == Float64

@test deep_reshape(pack(C, D)...) == (C, D)
@test deep_reshape(pack(Float64, C, D)...) == (C, D)
