np.ma.MaskedArray([(datetime.date(1991,11,3), np.datetime64('NaT') ), 
                    (np.datetime64('NaT'), datetime.date(1021,12,25)),
       (datetime.date(2001, 3, 10), datetime.date(3013, 6, 6))],
       mask = [(False, True), (True, False), (False, False)],
      dtype=[('date1', '<M8[D]'), ('date2', '<M8[D]')])
