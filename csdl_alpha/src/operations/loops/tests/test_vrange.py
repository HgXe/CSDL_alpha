import csdl_alpha.utils.test_utils as csdl_tests
import pytest

class TestVRange(csdl_tests.CSDLTest):
    def test_simple_loop(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import vrange
        import numpy as np

        a = csdl.Variable(value=2, name='a')
        b = b0 = csdl.Variable(value=3, name='b')
        for i in vrange(0, 10):
            b2 = a + b
            c = a*2
        
        x = a+b2+b0+c

        self.run_tests(
            compare_values=[
                csdl_tests.TestingPair(a, np.array([2])),
                csdl_tests.TestingPair(b, np.array([3])),
                csdl_tests.TestingPair(b0, np.array([3])),
                csdl_tests.TestingPair(b2, np.array([5])),
                csdl_tests.TestingPair(c, np.array([2*2])),
                csdl_tests.TestingPair(x, np.array([2+5+3+4]))
            ]
        )

    def test_range_inputs(self):
        self.prep()
        import csdl_alpha as csdl
        from csdl_alpha.api import vrange

        with pytest.raises(ValueError):
            vrange(10, 0)

        v_range = vrange(vals=[1, 2, 3, 4, 5])
        assert v_range.vals == [1, 2, 3, 4, 5]



# Tests with feedback - turned off right now
# class TestVRange(csdl_tests.CSDLTest):
#     def test_simple_loop(self):
#         self.prep()
#         import csdl_alpha as csdl
#         from csdl_alpha.api import vrange
#         import numpy as np

#         a = csdl.Variable(value=2, name='a')
#         b = b0 = csdl.Variable(value=3, name='b')
#         for i in vrange(0, 10):
#             b = a + b
#             c = a*2
        
#         x = a+b+b0+c

#         self.run_tests(
#             compare_values=[
#                 csdl_tests.TestingPair(a, np.array([2])),
#                 csdl_tests.TestingPair(b, np.array([3+2*10])),
#                 csdl_tests.TestingPair(b0, np.array([3])),
#                 csdl_tests.TestingPair(c, np.array([2*2])),
#                 csdl_tests.TestingPair(x, np.array([2+(3+2*10)+3+2*2]))
#             ]
#         )

#     def test_range_inputs(self):
#         self.prep()
#         import csdl_alpha as csdl
#         from csdl_alpha.api import vrange

#         with pytest.raises(ValueError):
#             vrange(10, 0)

#         v_range = vrange(vals=[1, 2, 3, 4, 5])
#         assert v_range.vals == [1, 2, 3, 4, 5]