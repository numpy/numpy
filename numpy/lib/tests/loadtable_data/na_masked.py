np.ma.MaskedArray([(999999, 3.4, 300.0, 'blah', 'sing', 'a'), (5, 1e+20, 24.6, 'N/A', 'zoomzoom', 'N/'),
       (999999, 42.0, 1e+20, 'posies', 'N/A', 'a'), (999999, 1e+20, 1e+20, 'little', 'lamb', 'ab')], 
				mask = [(True, False, False, False, False, False),
				       (False, True, False, True, False, True),
							 (True, False, True, False, True, False),
				       (True, True, True, False, False, False)],
			 dtype=[('f0', '<i8'), ('f1', '<f8'), ('f2', '<f8'), ('f3', 'S6'), ('f4', 'S8'), ('f5', 'S2')])
