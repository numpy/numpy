np.ma.MaskedArray(data=[(True, 6L, 1e+20, '2', (6+0j), None),
       (True, -6L, 5.2400000000000002, 'BAH!', 2j, datetime.date(1992, 4, 29)),
       (False, 999999L, 40000.0, 'Yellow Helicopter', (1.0000000200408773e+20+0j), datetime.date(3003, 4, 1)),
       (True, 323L, -0.029999999999999999, 'N/A', (-3.4000000953674316-420j), datetime.date(1676, 5, 7))],
       mask = [(False, False, True, False, False, True),
        (True, False, False, False, False, False),
        (False, True, False, False, True, False),
        (False, False, False, True, False, False)],
      dtype=[('bool', '?'), ('int', '<i8'), ('float', '<f8'), ('string', 'S19'), ('complex', '<c8'), ('datetime', '<M8[D]')])
