#!/usr/bin/env python3
"""
Example usage of the CSDL select operation.

This example demonstrates how to use the select operation that wraps jax.lax.select
through jax.numpy.where.
"""

import numpy as np
import csdl_alpha as csdl

def main():
    print("CSDL Select Operation Example")
    print("=" * 40)
    
    # Example 1: Basic conditional selection
    print("\n1. Basic conditional selection:")
    
    recorder = csdl.build_new_recorder(inline=True)
    recorder.start()
    
    # Create some data
    condition = csdl.Variable(value=np.array([True, False, True, False, True]))
    values_if_true = csdl.Variable(value=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    values_if_false = csdl.Variable(value=np.array([10.0, 20.0, 30.0, 40.0, 50.0]))
    
    # Use select operation
    result = csdl.select(condition, values_if_true, values_if_false)
    
    recorder.stop()
    
    print(f"Condition:      {condition.value}")
    print(f"If True:        {values_if_true.value}")  
    print(f"If False:       {values_if_false.value}")
    print(f"Result:         {result.value}")
    print(f"Expected:       {np.where(condition.value, values_if_true.value, values_if_false.value)}")
    
    # Example 2: Broadcasting
    print("\n2. Broadcasting example:")
    
    recorder = csdl.build_new_recorder(inline=True)
    recorder.start()
    
    # Condition vector that will broadcast
    condition = csdl.Variable(value=np.array([True, False]))
    # 2D arrays
    values_if_true = csdl.Variable(value=np.array([[1.0, 2.0], [3.0, 4.0]]))
    values_if_false = csdl.Variable(value=np.array([[10.0, 20.0], [30.0, 40.0]]))
    
    result = csdl.select(condition, values_if_true, values_if_false)
    
    recorder.stop()
    
    print(f"Condition shape: {condition.value.shape}, value: {condition.value}")
    print(f"Arrays shape: {values_if_true.value.shape}")
    print(f"Result:\n{result.value}")
    print(f"Expected:\n{np.where(condition.value, values_if_true.value, values_if_false.value)}")
    
    # Example 3: Use in optimization (with derivatives)
    print("\n3. Optimization example (with derivatives):")
    
    recorder = csdl.build_new_recorder(inline=False)
    recorder.start()
    
    # Design variables
    x = csdl.Variable(value=np.array([0.5, -0.3, 0.8]))
    
    # Create condition: x > 0
    condition = x > 0.0
    
    # Different penalty functions based on condition
    penalty_positive = x**2  # quadratic penalty for positive values
    penalty_negative = -2*x  # linear penalty for negative values
    
    # Select appropriate penalty
    penalty = csdl.select(condition, penalty_positive, penalty_negative)
    
    # Compute derivatives for optimization
    dpenaulty_dx = csdl.derivative(penalty, x)
    
    recorder.stop()
    
    # Run simulation
    from csdl_alpha.backends.simulator import PySimulator
    sim = PySimulator(recorder)
    sim.run()
    
    print(f"x values:           {sim[x]}")
    print(f"x > 0:              {sim[condition]}")
    print(f"Penalty:            {sim[penalty]}")
    print(f"Penalty gradient:   diagonal of jacobian = {np.diag(sim[dpenaulty_dx])}")
    
    # Verify gradients manually
    x_val = sim[x]
    expected_grad = np.where(x_val > 0, 2*x_val, -2.0)
    print(f"Expected gradient:  {expected_grad}")
    print(f"Gradients match:    {np.allclose(np.diag(sim[dpenaulty_dx]), expected_grad)}")

def ex_select_n():
    rec = csdl.Recorder(inline=True)
    rec.start()

    a = csdl.Variable(value=np.array([1.0, 2.0, 3.0]))
    b = a**2
    c = a**3
    # b = csdl.Variable(value=np.array([10.0, 20.0, 30.0]))
    # c = csdl.Variable(value=np.array([100.0, 200.0, 300.0]))
    cond = csdl.Variable(value=np.array([0, 1, 2]))

    result = csdl.select_n(cond, a, b, c)

    dr_da = csdl.derivative(result, a)
    dr_db = csdl.derivative(result, b)
    dr_dc = csdl.derivative(result, c)

    d2r_da = csdl.derivative(dr_da, a)
    d2r_db = csdl.derivative(dr_db, b)
    d2r_dc = csdl.derivative(dr_dc, c)

    print("\nSelect_n example:")
    print(f"Condition:      {cond.value}")
    print(f"Array A:       {a.value}")
    print(f"Array B:       {b.value}")
    print(f"Array C:       {c.value}")
    print(f"Result:       {result.value}")
    print("="*20)
    print(f"Derivative wrt A:\n{dr_da.value}")
    print(f"Derivative wrt B:\n{dr_db.value}")
    print(f"Derivative wrt C:\n{dr_dc.value}")
    print("="*20)
    print(f"Second Derivative wrt A:\n{d2r_da.value}")
    print(f"Second Derivative wrt B:\n{d2r_db.value}")
    print(f"Second Derivative wrt C:\n{d2r_dc.value}")

if __name__ == "__main__":
    ex_select_n()