import numpy as np
import csdl_alpha as csdl


def main():
    rec = csdl.build_new_recorder(inline=True, debug=True)
    rec.start()

    x = csdl.Variable(name='x', value=np.array([1.0, 2.0, 3.0]))
    idx = csdl.Variable(name='idx', value=np.array([2.0]))  # choose branch 1

    def branch0(v):
        return 2*v + 10.0

    def branch1(v):
        return v * 5.0

    def branch2(v):
        return 3*v - 7.0

    y = csdl.switch(idx, [branch0, branch1, branch2], x)
    
    dy_dx = csdl.derivative(y, x)
    dy2_dx = csdl.derivative(dy_dx, x)  # second derivative
    dy_didx = csdl.derivative(y, idx)

    print('Selected index:', int(idx.value[0]))
    print('x =', x.value)
    print('y =', y.value)
    print('dy/dx =', dy_dx.value)
    # should be [[5.0, 0.0, 0.0],
    #            [0.0, 5.0, 0.0],
    #            [0.0, 0.0, 5.0]]
    print('d2y/dx2 =', dy2_dx.value)
    # should be all zeros

    print('dy/didx =', dy_didx.value)

def main_two_inputs():
    rec = csdl.build_new_recorder(inline=True, debug=True)
    rec.start()

    x1 = csdl.Variable(name='x1', value=np.array([1.0, 2.0, 3.0]))
    x2 = csdl.Variable(name='x2', value=np.array([4.0, 5.0, 6.0]))
    idx = csdl.Variable(name='idx', value=np.array([1.0]))  # choose branch 1

    def branch0(v1, v2):
        return 2*v1 + 10.0 + 3*v2 - 5.0

    def branch1(v1, v2):
        return v1 * 5.0 + v2 + 1.0

    def branch2(v1, v2):
        return 3*v1 - 7.0 + 4*v2 + 2.0

    y = csdl.switch(idx, [branch0, branch1, branch2], x1, x2)

    dy_dx1 = csdl.derivative(y, [x1, x2])[x1]
    dy_dx2 = csdl.derivative(y, x2)
    d2y_dx12 = csdl.derivative(dy_dx1, x1)  # second derivative
    dy_didx = csdl.derivative(y, idx)

    print('Selected index:', int(idx.value[0]))
    print('x1 =', x1.value)
    print('x2 =', x2.value)
    print('y =', y.value)
    print('dy/dx1 =', dy_dx1.value)
    print('dy/dx2 =', dy_dx2.value)
    # print('d2y/dx2 =', dy2_dx.value)
    # should be all zeros
    print('dy/didx =', dy_didx.value)

if __name__ == '__main__':
    # main()
    main_two_inputs()
