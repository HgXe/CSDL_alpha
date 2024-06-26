def test_data(tmp_path):
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    import os
    os.chdir(tmp_path)

    import csdl_alpha as csdl
    from csdl_alpha.src.graph.variable import Variable
    recorder = csdl.Recorder(inline = True)
    recorder.start()

    a = Variable(value=2, name='a')
    a.add_tag('test')
    a.hierarchy = 1
    b = Variable(value=3)
    csdl.enter_namespace('test1')
    c = Variable(value=4)
    csdl.exit_namespace()
    csdl.save_all_variables()

    dv = Variable(value=4)
    constraint = dv*2
    constraint.is_input = False
    objective = dv*3
    objective.is_input = False
    dv.set_as_design_variable()
    constraint.set_as_constraint()
    objective.set_as_objective()

    csdl.save_optimization_variables()

    csdl.inline_export('test_data')

    variables = csdl.inline_import('test_data.hdf5', 'inline')
    assert variables['a'].value == a.value
    assert variables['a'].tags == a.tags
    assert variables['a'].hierarchy == a.hierarchy
    assert variables['a'].name == a.name

def test_csv(tmp_path):
    from csdl_alpha.utils.hard_reload import hard_reload
    hard_reload()
    import os
    os.chdir(tmp_path)

    import csdl_alpha as csdl
    from csdl_alpha.src.graph.variable import Variable

    recorder = csdl.Recorder(inline = True)
    recorder.start()
    a = Variable(value=2, name='a')
    a.add_tag('test')
    a.hierarchy = 1
    b = Variable(value=3)
    csdl.enter_namespace('test1')
    c = Variable(value=4)
    csdl.exit_namespace()
    csdl.save_all_variables()

    dv = Variable(value=4)
    constraint = dv*2
    constraint.is_input = False
    objective = dv*3
    objective.is_input = False
    dv.set_as_design_variable()
    constraint.set_as_constraint()
    objective.set_as_objective()

    csdl.save_optimization_variables()

    csdl.inline_export('test_data', summary_csv=True)

if __name__ == '__main__':
    import csdl_alpha as csdl
    from csdl_alpha import Variable
    import numpy as np
    recorder = csdl.Recorder(inline=True)
    recorder.start()
    matrix = csdl.Variable(value=np.array([[1, 2], [3, 4]]), name='matrix')
    tensor = csdl.Variable(value=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), name='tensor')
    with csdl.namespace('test0'):
        a = Variable(value=2, name='a')
        a.add_tag('test')
        a.hierarchy = 1
        b = Variable(value=3)

        csdl.enter_namespace('test1')
        c = Variable(value=4)
        csdl.exit_namespace()
        csdl.save_all_variables()

        dv = Variable(value=4)
        constraint = dv*2
        constraint.is_input = False
        objective = dv*3
        objective.is_input = False
        dv.set_as_design_variable()
        constraint.set_as_constraint()
        objective.set_as_objective()




    csdl.save_optimization_variables()
    recorder.visualize_graph('test_data')

    csdl.inline_csv_save('test_data_2')
    csdl.inline_save('test_data_2')
