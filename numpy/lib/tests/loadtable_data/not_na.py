np.ma.MaskedArray([('A', 'B', 'N/A'), ('NAC', 'NAB', 'NAC'),
                        ('D', 'N/A', 'N/A'), ('NAD', 'NAE', 'NAF')],
             mask = [(False, False, True), (False, False, False),
                     (False, True, True), (False, False, False)],
             dtype=[('f0', 'S3'), ('f1', 'S3'), ('f2', 'S3')])
