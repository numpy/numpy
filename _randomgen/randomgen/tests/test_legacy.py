import pickle

from randomgen.legacy import LegacyGenerator


def test_pickle():
    lg = LegacyGenerator()
    lg.random_sample(100)
    lg.standard_normal()
    lg2 = pickle.loads(pickle.dumps(lg))
    assert lg.standard_normal() == lg2.standard_normal()
    assert lg.random_sample() == lg2.random_sample()
