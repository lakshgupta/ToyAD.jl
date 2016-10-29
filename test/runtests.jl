using ToyAD
using Base.Test

# ======================
# scalar
# ======================
x1 = ToyAD.AD(2.0)
y1 = ToyAD.AD(1)
r1 = (x1+y1)*(y1 + ToyAD.AD(1))
bp1 = ToyAD.backprop(r1, true)
@test r1.value == ones(1,1)*6
@test x1.grad == ones(1,1)*2
@test y1.grad == ones(1,1)*5

# ======================
# matrix
# ======================
m1 = ToyAD.AD(zeros(3,2))
m1.value[1,1] = 1.; m1.value[1,2] = 2.
m1.value[2,1] = 3.; m1.value[2,2] = 4.
m1.value[3,1] = 5.; m1.value[3,2] = 6.

m2 = ToyAD.AD(zeros(2,3))
m2.value[1,1] = 2.; m2.value[1,2] = 3.; m2.value[1,3] = 4.
m2.value[2,1] = 5.; m2.value[2,2] = 6.; m2.value[2,3] = 7.

# addition
m3 = m1 + m1
## delete =>
m3.grad[1,1] = .1; m3.grad[1,2] = .2
m3.grad[2,1] = .3; m3.grad[2,2] = .4
m3.grad[3,1] = .5; m3.grad[3,2] = .6
#println(m3)
rmAdd = ToyAD.backprop(m3, false)
#println(rmAdd)
@test m1.grad[1,1] == 0.2
@test m1.grad[1,2] == 0.4
@test m1.grad[2,1] == 0.6
@test m1.grad[2,2] == 0.8
@test m1.grad[3,1] == 1.0
@test m1.grad[3,2] == 1.2

## <= delete
#@test m3.value[1,1] == 2.
#@test m3.value[1,2] == 4.
#@test m3.value[2,1] == 6.
#@test m3.value[2,2] == 8.
#@test m3.value[3,1] == 10.
#@test m3.value[3,2] == 12.

#rm1 = ToyAD.backprop(m3)

#@test m1.grad[1,1] == 2
#@test m1.grad[1,2] == 2
#@test m1.grad[2,1] == 2
#@test m1.grad[2,2] == 2
#@test m1.grad[3,1] == 2
#@test m1.grad[3,2] == 2

# multiplication
## delete =>
# mul test
m1.grad[:] = 0. # reset gradient matrices
m2.grad[:] = 0. # reset  gradient matrices
m3 = m1* m2
@test m3.value[1,1] == 12.
@test m3.value[1,2] == 15.
@test m3.value[1,3] == 18.
@test m3.value[2,1] == 26.
@test m3.value[2,2] == 33.
@test m3.value[2,3] == 40.
@test m3.value[3,1] == 40.
@test m3.value[3,2] == 51.
@test m3.value[3,3] == 62.

m3.grad[1,1] = .1; m3.grad[1,2] = .2; m3.grad[1,3] = .3
m3.grad[2,1] = .4; m3.grad[2,2] = .5; m3.grad[2,3] = .6
m3.grad[3,1] = .7; m3.grad[3,2] = .8; m3.grad[3,3] = .9

#println(m3)
rmMul = ToyAD.backprop(m3, false)
#println(rmMul)

# m1 gradient tests
@test m1.grad[1,1] == 2.
@test m1.grad[1,2] == 3.8000000000000003
@test m1.grad[2,1] == 4.699999999999999
@test m1.grad[2,2] == 9.2
@test m1.grad[3,1] == 7.4
@test m1.grad[3,2] == 14.600000000000001

# m2 gradient tests
@test m2.grad[1,1] == 4.800000000000001
@test m2.grad[1,2] == 5.7
@test m2.grad[1,3] == 6.6
@test m2.grad[2,1] == 5.999999999999999
@test m2.grad[2,2] == 7.200000000000001
@test m2.grad[2,3] == 8.4

# reul() tests
m4 = ToyAD.AD(zeros(3,2))
m4.value[1,1] = 1.; m4.value[1,2] =-2.
m4.value[2,1] =-3.; m4.value[2,2] = 4.
m4.value[3,1] = 5.; m4.value[3,2] =-6.
m5 = ToyAD.relu(m4)
@test m5.value[1,1] == 1.
@test m5.value[1,2] == 0.
@test m5.value[2,1] == 0.
@test m5.value[2,2] == 4.
@test m5.value[3,1] == 5.
@test m5.value[3,2] == 0.

m5.grad[1,1] =-.1; m5.grad[1,2] = .2
m5.grad[2,1] = .3; m5.grad[2,2] = .4
m5.grad[3,1] = .5; m5.grad[3,2] = .6

#println(m5)
rmRelu = ToyAD.backprop(m5, false)
#println(rmRelu)

@test m4.grad[1,1] == -0.1
@test m4.grad[1,2] == 0.
@test m4.grad[2,1] == 0.
@test m4.grad[2,2] == 0.4
@test m4.grad[3,1] == 0.5
@test m4.grad[3,2] == 0.

# softmax tests
m6 = ToyAD.AD(zeros(5,1))
m6.value[1,1] = 0.3
m6.value[2,1] = 0.1
m6.value[3,1] = 0.6
m6.value[4,1] = 0.002
m6.value[5,1] = 0.00001

m7 = ToyAD.softmax(m6)
#println("m7:")
#println(m7)
# @test_approx_eq
#@test_approx_eq m7.value[1,1] 0.214940#50089813527
#@test_approx_eq m7.value[2,1] 0.175978#39816728894
#@test_approx_eq m7.value[3,1] 0.290139#3282421457
#@test_approx_eq m7.value[4,1] 0.159550#6217827435
#@test_approx_eq m7.value[5,1] 0.159391#15090968653
## <== delete

