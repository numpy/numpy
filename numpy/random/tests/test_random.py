from numpy.testing import TestCase, run_module_suite, assert_,\
        assert_raises
from numpy import random
from numpy.compat import asbytes
import numpy as np


class TestMultinomial(TestCase):
    def test_basic(self):
        random.multinomial(100, [0.2, 0.8])

    def test_zero_probability(self):
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def test_int_negative_interval(self):
        assert_( -5 <= random.randint(-5,-1) < -1)
        x = random.randint(-5,-1,5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))


class TestSetState(TestCase):
    def setUp(self):
        self.seed = 1234567890
        self.prng = random.RandomState(self.seed)
        self.state = self.prng.get_state()

    def test_basic(self):
        old = self.prng.tomaxint(16)
        self.prng.set_state(self.state)
        new = self.prng.tomaxint(16)
        assert_(np.all(old == new))

    def test_gaussian_reset(self):
        """ Make sure the cached every-other-Gaussian is reset.
        """
        old = self.prng.standard_normal(size=3)
        self.prng.set_state(self.state)
        new = self.prng.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_gaussian_reset_in_media_res(self):
        """ When the state is saved with a cached Gaussian, make sure the cached
        Gaussian is restored.
        """
        self.prng.standard_normal()
        state = self.prng.get_state()
        old = self.prng.standard_normal(size=3)
        self.prng.set_state(state)
        new = self.prng.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_backwards_compatibility(self):
        """ Make sure we can accept old state tuples that do not have the cached
        Gaussian value.
        """
        old_state = self.state[:-2]
        x1 = self.prng.standard_normal(size=16)
        self.prng.set_state(old_state)
        x2 = self.prng.standard_normal(size=16)
        self.prng.set_state(self.state)
        x3 = self.prng.standard_normal(size=16)
        assert_(np.all(x1 == x2))
        assert_(np.all(x1 == x3))

    def test_negative_binomial(self):
        """ Ensure that the negative binomial results take floating point
        arguments without truncation.
        """
        self.prng.negative_binomial(0.5, 0.5)

class TestRandomDist(TestCase):
    """ Make sure the random distrobution return the correct value for a
    given seed
    """
    def setUp(self):
        self.seed = 1234567890

    def test_rand(self):
        np.random.seed(self.seed)
        actual = np.random.rand(3, 2)
        desired = np.array([[ 0.61879477158567997,  0.59162362775974664],
                         [ 0.88868358904449662,  0.89165480011560816],
                         [ 0.4575674820298663 ,  0.7781880808593471 ]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_randn(self):
        np.random.seed(self.seed)
        actual = np.random.randn(3, 2)
        desired = np.array([[ 1.34016345771863121,  1.73759122771936081],
                         [ 1.498988344300628  , -0.2286433324536169 ],
                         [ 2.031033998682787  ,  2.17032494605655257]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_randint(self):
        np.random.seed(self.seed)
        actual = np.random.randint(-99, 99, size=(3,2))
        desired = np.array([[ 31,   3],
                         [-52,  41],
                         [-48, -66]])
        np.testing.assert_array_equal(actual, desired)

    def test_random_integers(self):
        np.random.seed(self.seed)
        actual = np.random.random_integers(-99, 99, size=(3,2))
        desired = np.array([[ 31,   3],
                         [-52,  41],
                         [-48, -66]])
        np.testing.assert_array_equal(actual, desired)

    def test_random_sample(self):
        np.random.seed(self.seed)
        actual = np.random.random_sample((3, 2))
        desired = np.array([[ 0.61879477158567997,  0.59162362775974664],
                         [ 0.88868358904449662,  0.89165480011560816],
                         [ 0.4575674820298663 ,  0.7781880808593471 ]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_bytes(self):
        np.random.seed(self.seed)
        actual = np.random.bytes(10)
        desired = asbytes('\x82Ui\x9e\xff\x97+Wf\xa5')
        np.testing.assert_equal(actual, desired)

    def test_shuffle(self):
        np.random.seed(self.seed)
        alist = [1,2,3,4,5,6,7,8,9,0]
        np.random.shuffle(alist)
        actual = alist
        desired = [0, 1, 9, 6, 2, 4, 5, 8, 7, 3]
        np.testing.assert_array_equal(actual, desired)

    def test_beta(self):
        np.random.seed(self.seed)
        actual = np.random.beta(.1, .9, size=(3, 2))
        desired = np.array([[  1.45341850513746058e-02,   5.31297615662868145e-04],
                         [  1.85366619058432324e-06,   4.19214516800110563e-03],
                         [  1.58405155108498093e-04,   1.26252891949397652e-04]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_binomial(self):
        np.random.seed(self.seed)
        actual = np.random.binomial(100.123, .456, size=(3, 2))
        desired = np.array([[37, 43],
                         [42, 48],
                         [46, 45]])
        np.testing.assert_array_equal(actual, desired)

    def test_chisquare(self):
        np.random.seed(self.seed)
        actual = np.random.chisquare(50, size=(3, 2))
        desired = np.array([[ 63.87858175501090585,  68.68407748911370447],
                            [ 65.77116116901505904,  47.09686762438974483],
                            [ 72.3828403199695174 ,  74.18408615260374006]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=13)

    def test_dirichlet(self):
        np.random.seed(self.seed)
        alpha = np.array([51.72840233779265162,  39.74494232180943953])
        actual = np.random.mtrand.dirichlet(alpha, size=(3, 2))
        desired = np.array([[[ 0.54539444573611562,  0.45460555426388438],
                             [ 0.62345816822039413,  0.37654183177960598]],
                            [[ 0.55206000085785778,  0.44793999914214233],
                             [ 0.58964023305154301,  0.41035976694845688]],
                            [[ 0.59266909280647828,  0.40733090719352177],
                             [ 0.56974431743975207,  0.43025568256024799]]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_exponential(self):
        np.random.seed(self.seed)
        actual = np.random.exponential(1.1234, size=(3, 2))
        desired = np.array([[ 1.08342649775011624,  1.00607889924557314],
                         [ 2.46628830085216721,  2.49668106809923884],
                         [ 0.68717433461363442,  1.69175666993575979]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_f(self):
        np.random.seed(self.seed)
        actual = np.random.f(12, 77, size=(3, 2))
        desired = np.array([[ 1.21975394418575878,  1.75135759791559775],
                            [ 1.44803115017146489,  1.22108959480396262],
                            [ 1.02176975757740629,  1.34431827623300415]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_gamma(self):
        np.random.seed(self.seed)
        actual = np.random.gamma(5, 3, size=(3, 2))
        desired = np.array([[ 24.60509188649287182,  28.54993563207210627],
                             [ 26.13476110204064184,  12.56988482927716078],
                             [ 31.71863275789960568,  33.30143302795922011]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=14)

    def test_geometric(self):
        np.random.seed(self.seed)
        actual = np.random.geometric(.123456789, size=(3, 2))
        desired = np.array([[ 8,  7],
                         [17, 17],
                         [ 5, 12]])
        np.testing.assert_array_equal(actual, desired)

    def test_gumbel(self):
        np.random.seed(self.seed)
        actual = np.random.gumbel(loc = .123456789, scale = 2.0, size = (3, 2))
        desired = np.array([[ 0.19591898743416816,  0.34405539668096674],
                         [-1.4492522252274278 , -1.47374816298446865],
                         [ 1.10651090478803416, -0.69535848626236174]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_hypergeometric(self):
        np.random.seed(self.seed)
        actual = np.random.hypergeometric(10.1, 5.5, 14, size=(3, 2))
        desired = np.array([[10, 10],
                         [10, 10],
                         [ 9,  9]])
        np.testing.assert_array_equal(actual, desired)

    def test_laplace(self):
        np.random.seed(self.seed)
        actual = np.random.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[ 0.66599721112760157,  0.52829452552221945],
                         [ 3.12791959514407125,  3.18202813572992005],
                         [-0.05391065675859356,  1.74901336242837324]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_logistic(self):
        np.random.seed(self.seed)
        actual = np.random.logistic(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[ 1.09232835305011444,  0.8648196662399954 ],
                         [ 4.27818590694950185,  4.33897006346929714],
                         [-0.21682183359214885,  2.63373365386060332]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_lognormal(self):
        np.random.seed(self.seed)
        actual = np.random.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
        desired = np.array([[ 16.50698631688883822,  36.54846706092654784],
                         [ 22.67886599981281748,   0.71617561058995771],
                         [ 65.72798501792723869,  86.84341601437161273]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=13)

    def test_logseries(self):
        np.random.seed(self.seed)
        actual = np.random.logseries(p=.923456789, size=(3, 2))
        desired = np.array([[ 2,  2],
                         [ 6, 17],
                         [ 3,  6]])
        np.testing.assert_array_equal(actual, desired)

    def test_multinomial(self):
        np.random.seed(self.seed)
        actual = np.random.multinomial(20, [1/6.]*6, size=(3, 2))
        desired = np.array([[[4, 3, 5, 4, 2, 2],
                          [5, 2, 8, 2, 2, 1]],
                         [[3, 4, 3, 6, 0, 4],
                          [2, 1, 4, 3, 6, 4]],
                         [[4, 4, 2, 5, 2, 3],
                          [4, 3, 4, 2, 3, 4]]])
        np.testing.assert_array_equal(actual, desired)

    def test_multivariate_normal(self):
        np.random.seed(self.seed)
        mean= (.123456789, 10)
        cov = [[1,0],[1,0]]
        size = (3, 2)
        actual = np.random.multivariate_normal(mean, cov, size)
        desired = np.array([[[ -1.47027513018564449,  10.                 ],
                          [ -1.65915081534845532,  10.                 ]],
                         [[ -2.29186329304599745,  10.                 ],
                          [ -1.77505606019580053,  10.                 ]],
                         [[ -0.54970369430044119,  10.                 ],
                          [  0.29768848031692957,  10.                 ]]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_negative_binomial(self):
        np.random.seed(self.seed)
        actual = np.random.negative_binomial(n = 100, p = .12345, size = (3, 2))
        desired = np.array([[848, 841],
                         [892, 611],
                         [779, 647]])
        np.testing.assert_array_equal(actual, desired)

    def test_noncentral_chisquare(self):
        np.random.seed(self.seed)
        actual = np.random.noncentral_chisquare(df = 5, nonc = 5, size = (3, 2))
        desired = np.array([[ 23.91905354498517511,  13.35324692733826346],
                         [ 31.22452661329736401,  16.60047399466177254],
                         [  5.03461598262724586,  17.94973089023519464]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f(self):
        np.random.seed(self.seed)
        actual = np.random.noncentral_f(dfnum = 5, dfden = 2, nonc = 1,
                                        size = (3, 2))
        desired = np.array([[ 1.40598099674926669,  0.34207973179285761],
                         [ 3.57715069265772545,  7.92632662577829805],
                         [ 0.43741599463544162,  1.1774208752428319 ]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=14)

    def test_normal(self):
        np.random.seed(self.seed)
        actual = np.random.normal(loc = .123456789, scale = 2.0, size = (3, 2))
        desired = np.array([[ 2.80378370443726244,  3.59863924443872163],
                         [ 3.121433477601256  , -0.33382987590723379],
                         [ 4.18552478636557357,  4.46410668111310471]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_pareto(self):
        np.random.seed(self.seed)
        actual = np.random.pareto(a =.123456789, size = (3, 2))
        desired = np.array([[  2.46852460439034849e+03,   1.41286880810518346e+03],
                         [  5.28287797029485181e+07,   6.57720981047328785e+07],
                         [  1.40840323350391515e+02,   1.98390255135251704e+05]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_poisson(self):
        np.random.seed(self.seed)
        actual = np.random.poisson(lam = .123456789, size=(3, 2))
        desired = np.array([[0, 0],
                         [1, 0],
                         [0, 0]])
        np.testing.assert_array_equal(actual, desired)

    def test_poisson_exceptions(self):
        lambig = np.iinfo('l').max
        lamneg = -1
        assert_raises(ValueError, np.random.poisson, lamneg)
        assert_raises(ValueError, np.random.poisson, [lamneg]*10)
        assert_raises(ValueError, np.random.poisson, lambig)
        assert_raises(ValueError, np.random.poisson, [lambig]*10)

    def test_power(self):
        np.random.seed(self.seed)
        actual = np.random.power(a =.123456789, size = (3, 2))
        desired = np.array([[ 0.02048932883240791,  0.01424192241128213],
                         [ 0.38446073748535298,  0.39499689943484395],
                         [ 0.00177699707563439,  0.13115505880863756]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_rayleigh(self):
        np.random.seed(self.seed)
        actual = np.random.rayleigh(scale = 10, size = (3, 2))
        desired = np.array([[ 13.8882496494248393 ,  13.383318339044731  ],
                         [ 20.95413364294492098,  21.08285015800712614],
                         [ 11.06066537006854311,  17.35468505778271009]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=14)

    def test_standard_cauchy(self):
        np.random.seed(self.seed)
        actual = np.random.standard_cauchy(size = (3, 2))
        desired = np.array([[ 0.77127660196445336, -6.55601161955910605],
                         [ 0.93582023391158309, -2.07479293013759447],
                         [-4.74601644297011926,  0.18338989290760804]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_exponential(self):
        np.random.seed(self.seed)
        actual = np.random.standard_exponential(size = (3, 2))
        desired = np.array([[ 0.96441739162374596,  0.89556604882105506],
                         [ 2.1953785836319808 ,  2.22243285392490542],
                         [ 0.6116915921431676 ,  1.50592546727413201]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_gamma(self):
        np.random.seed(self.seed)
        actual = np.random.standard_gamma(shape = 3, size = (3, 2))
        desired = np.array([[ 5.50841531318455058,  6.62953470301903103],
                         [ 5.93988484943779227,  2.31044849402133989],
                         [ 7.54838614231317084,  8.012756093271868  ]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=14)

    def test_standard_normal(self):
        np.random.seed(self.seed)
        actual = np.random.standard_normal(size = (3, 2))
        desired = np.array([[ 1.34016345771863121,  1.73759122771936081],
                         [ 1.498988344300628  , -0.2286433324536169 ],
                         [ 2.031033998682787  ,  2.17032494605655257]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_t(self):
        np.random.seed(self.seed)
        actual = np.random.standard_t(df = 10, size = (3, 2))
        desired = np.array([[ 0.97140611862659965, -0.08830486548450577],
                         [ 1.36311143689505321, -0.55317463909867071],
                         [-0.18473749069684214,  0.61181537341755321]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_triangular(self):
        np.random.seed(self.seed)
        actual = np.random.triangular(left = 5.12, mode = 10.23, right = 20.34,
                                      size = (3, 2))
        desired = np.array([[ 12.68117178949215784,  12.4129206149193152 ],
                         [ 16.20131377335158263,  16.25692138747600524],
                         [ 11.20400690911820263,  14.4978144835829923 ]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=14)

    def test_uniform(self):
        np.random.seed(self.seed)
        actual = np.random.uniform(low = 1.23, high=10.54, size = (3, 2))
        desired = np.array([[ 6.99097932346268003,  6.73801597444323974],
                         [ 9.50364421400426274,  9.53130618907631089],
                         [ 5.48995325769805476,  8.47493103280052118]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)


    def test_vonmises(self):
        np.random.seed(self.seed)
        actual = np.random.vonmises(mu = 1.23, kappa = 1.54, size = (3, 2))
        desired = np.array([[ 2.28567572673902042,  2.89163838442285037],
                         [ 0.38198375564286025,  2.57638023113890746],
                         [ 1.19153771588353052,  1.83509849681825354]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_wald(self):
        np.random.seed(self.seed)
        actual = np.random.wald(mean = 1.23, scale = 1.54, size = (3, 2))
        desired = np.array([[ 3.82935265715889983,  5.13125249184285526],
                         [ 0.35045403618358717,  1.50832396872003538],
                         [ 0.24124319895843183,  0.22031101461955038]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=14)

    def test_weibull(self):
        np.random.seed(self.seed)
        actual = np.random.weibull(a = 1.23, size = (3, 2))
        desired = np.array([[ 0.97097342648766727,  0.91422896443565516],
                         [ 1.89517770034962929,  1.91414357960479564],
                         [ 0.67057783752390987,  1.39494046635066793]])
        np.testing.assert_array_almost_equal(actual, desired, decimal=15)

    def test_zipf(self):
        np.random.seed(self.seed)
        actual = np.random.zipf(a = 1.23, size = (3, 2))
        desired = np.array([[66, 29],
                         [ 1,  1],
                         [ 3, 13]])
        np.testing.assert_array_equal(actual, desired)

if __name__ == "__main__":
    run_module_suite()
