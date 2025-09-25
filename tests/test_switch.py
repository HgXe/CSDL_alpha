import numpy as np
import csdl_alpha as csdl
import csdl_alpha.utils.testing_utils as csdl_tests


class TestSwitch(csdl_tests.CSDLTest):
    overwrite_backend = 'jax'

    def test_switch_basic(self):
        self.prep(always_build_inline=False)

        x = csdl.Variable(value=np.array([1.0, 2.0, 3.0]))
        idx = csdl.Variable(value=np.array([0.0]))

        def b0(v):
            return v + 1

        def b1(v):
            return 2 * v

        def b2(v):
            return v - 3

        y = csdl.switch(idx, [b0, b1, b2], x)

        compare = []
        compare += [csdl_tests.TestingPair(y, np.array([2.0, 3.0, 4.0]), tag='branch0')]
        # Use batched derivatives to avoid loop-builder edge cases in this environment
        self.batched_deriv = True
        self.run_tests(compare_values=compare, verify_derivatives=True, turn_off_recorder=False)

        # mutate index and check forward inline execution updates
        idx.value = np.array([2.0])
        csdl.get_current_recorder().active_graph.execute_inline()
        assert np.allclose(y.value, np.array([-2.0, -1.0, 0.0]))

    def test_switch_multi_output(self):
        self.prep(always_build_inline=False)

        a = csdl.Variable(value=np.array([1.0]))
        b = csdl.Variable(value=np.array([2.0]))
        idx = csdl.Variable(value=np.array([1.0]))

        def br0(x, y):
            s = x + y
            p = x * y
            return s, p

        def br1(x, y):
            s = 2 * x + 3 * y
            p = (x + 1) * (y + 1)
            return s, p

        s, p = csdl.switch(idx, [br0, br1], a, b)

        compare = []
        compare += [csdl_tests.TestingPair(s, np.array([8.0]), tag='sum')]
        compare += [csdl_tests.TestingPair(p, np.array([6.0]), tag='prod')]
        # Use batched derivatives for this test as well
        self.batched_deriv = True
        self.run_tests(compare_values=compare, verify_derivatives=True)
