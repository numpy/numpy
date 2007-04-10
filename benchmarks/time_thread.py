import numpy
import timeit


def run_trial(array_sizes, code, init_template, repeats=1):
    results = []
    for size in array_sizes:
        timer = timeit.Timer(code,init_template%(size,size))
        # limit the number of iterations that are run to 10000.
        iters = int(min(array_sizes[-1]/size,1e3))
        results.append(min(timer.repeat(repeats,iters)))

    return numpy.array(results)

def compare_threads(code, init_template, thread_counts, array_sizes=None):

    if array_sizes is None:
        array_sizes = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

    use_threads = [0] + [1]*len(thread_counts)
    thread_counts = [1] + thread_counts
    element_threshold = [1] * len(thread_counts)
    thread_settings = zip(use_threads, thread_counts, element_threshold)

    results = []
    numpy.set_printoptions(precision=2)
    for i, settings in enumerate(thread_settings):
        numpy.setthreading(*settings)
        times = run_trial(array_sizes, code, init_template, repeats=3)
        results.append(times)
        speed_up = results[0]/times
        if i == 0:
           print 'times:', times
        # print out the speed up along with the thread settings.
        print code, settings[0], settings[1], speed_up 

    return results

init_template = "from numpy import ones, sin, cos, sqrt, arctanh;a=ones(%f);b=ones(%f)"
mult = "a*b"
dist = "sqrt(a*a+b*b)"
finite_difference = "b[:1] * a[1:] - a[:-1]"
trig = "sin(a);"
trig_expr = "sin(a)+cos(b);"

for code in [mult, dist, finite_difference, trig, trig_expr]:
    compare_threads(code, init_template, [1,2,4,8], [1e4, 2e4, 5e4, 1e5, 2e5, 5e5, 1e6, 1e7])
    #compare_threads([1,2,4,8], [1e7])
