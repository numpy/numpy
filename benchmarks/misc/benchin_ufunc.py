#!/usr/bin/env python3
"""
bench_ufunc.py [OPTIONS] [-- ARGS]

Standalone benchmark script for the inner loops of ufunc

  This script only measuring the performance of inner loops
  of ufunc, the idea behind it is to remove umath object calls
  from the equation, in order to reduce the number of noises and
  provides stable ratios.

Examples::
    $ benchin_ufunc.py --filter "square.*f" --export opt_square.json
    $ benchin_ufunc.py --filter "square.*f" --compare opt_square.json --output current.md
    $ benchin_ufunc.py --filter "square.*f" --compare opt_square.json --only-changed 0.05
"""

import os, sys, re, itertools, functools, json, argparse, multiprocessing, time
import numpy as np
import numpy.core._umath_tests as utests

class Colored:
    # FG codes
    RED = 31
    GREEN = 32
    YELLOW = 33
    IS_TTYOUT = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    IS_TTYERR = hasattr(sys.stderr, 'isatty') and sys.stderr.isatty()

    _colored_ansi = lambda txt, fg: (
        "\033[%dm%s\033[0m" % (fg, txt)
    ) if os.name != "nt" else lambda txt, fg: txt

    _colored_out = _colored_ansi if IS_TTYOUT \
                   else lambda txt, fg: txt

    _colored_err = _colored_ansi if IS_TTYERR \
                   else lambda txt, fg: txt

    @staticmethod
    def text(txt, color_id):
        return Colored._colored_ansi(txt, color_id)

    @staticmethod
    def text_tty(txt, color_id):
        return Colored._colored_out(txt, color_id)

    @staticmethod
    def ok(txt, **kwargs):
        print(Colored._colored_out(txt, Colored.GREEN), **kwargs)

    @staticmethod
    def notify(txt, **kwargs):
        print(Colored._colored_out(txt, Colored.YELLOW), **kwargs)

    @staticmethod
    def fatal(txt):
        raise SystemExit(Colored._colored_err(txt, Colored.RED))

class Table:
    FORMATS = ('md', 'txt', 'ansi')
    ALIGN_TXT = {
        'left': '<',
        'right': '>',
        'center': '^'
    }
    ALIGN_MD = {
        'left': '---',
        'right': '---',
        'center': ':-:'
    }
    def __init__(self, tformat, field_names, ratios_indexes=[], align_indexes={}):
        assert(tformat in self.FORMATS)
        assert(all([a in self.ALIGN_TXT for a in align_indexes.values()]))
        if tformat == "md":
            self._highlight = self._highlight_md
            self.get_string = self._md_str
        elif tformat == "ansi":
            self._highlight = self._highlight_ansi
            self.get_string = self._pretty_str
        else:
            self._highlight = lambda f: f
            self.get_string = self._pretty_str

        self._rindexes = ratios_indexes
        self._aindexes = align_indexes
        self._fields_len = len(field_names)
        self._fields_mwidth = [0 for _ in range(self._fields_len)]
        self._rows = []
        self.add_row(field_names)

    def __str__(self):
        return self.get_string()

    def add_row(self, fields):
        assert(len(fields) == self._fields_len)
        final = []
        mwidth = self._fields_mwidth
        for c, f in enumerate(fields):
            if f is None:
                f = "N/A"
            elif not isinstance(f, str):
                if c in self._rindexes:
                    f = str(self._highlight(round(f, 2)))
                else:
                    f = "{0:6.4f}".format(f)
            flen = len(f)
            if flen > mwidth[c]:
                mwidth[c] = flen
            final.append(f)
        self._rows.append(final)

    def _highlight_md(self, f):
        if f > 1.05:
            return "**`%s`**" % f
        elif f < 0.95:
            return "*`%s`*" % f
        return f

    def _highlight_ansi(self, f):
        if f > 1.05:
            return Colored.text(f, Colored.GREEN)
        elif f < 0.95:
            return Colored.text(f, Colored.RED)
        return f

    def _pretty_str(self):
        def pretty_row(margin, mwidth, joinc, row):
            ansi_pad = lambda x: 9 if x.startswith("\033[") else 0
            text_align = lambda c: self.ALIGN_TXT.get(
                self._aindexes.get(c, 'center')
            )
            return ''.join([
                joinc + (
                    "{margin}{val:%s%d}{margin}" % (
                        text_align(c), mwidth[c] + ansi_pad(val)
                    )
                ).format(margin=' '*margin, val=val)
                for c, val in enumerate(row)
            ]) + joinc

        list_str = []
        padding = 1
        margin = 1
        mwidth = [w + padding for w in self._fields_mwidth]
        # the header
        list_str.append(pretty_row(
            0, mwidth, '+', [('-' * (w + margin*2)) for w in mwidth]
        ))
        list_str.append(pretty_row(
            margin, mwidth, '|', self._rows[0]
        ))
        list_str.append(list_str[0])
        # get the rest
        for row in self._rows[1:]:
            list_str.append(pretty_row(
                margin, mwidth, '|', row
            ))
        # the footer
        list_str.append(list_str[0])
        return "\n".join(list_str)

    def _md_str(self):
        ensure_space = lambda row: [
            f.replace(' ', r'&nbsp; ') if ' '*2 in f else f for f in row
        ]
        md_row = lambda row: '|' + '|'.join(ensure_space(row)) + '|'
        fields_align = md_row([
            self.ALIGN_MD.get(
                self._aindexes.get(c, 'center')
            )
            for c in range(self._fields_len)
        ])
        list_str = [md_row(self._rows[0]), fields_align]
        for row in self._rows[1:]:
            list_str.append(md_row(row))
        return '\n'.join(list_str)

class Timing:
    SEC = 1e0
    MS  = 1e3
    US  = 1e6

    @staticmethod
    def set_affinity(*CPUs):
        utests.ctiming_set_affinity(CPUs)

    @staticmethod
    def to_unit(ncycles, scale):
        # ctiming_frequency contains the number of clock-cycles per second
        return (ncycles * scale) / utests.ctiming_frequency

    def __init__(self, ufunc, nsamples, iteration, warmup, msleep):
        self._ufunc = utests.ctiming(ufunc, iteration, warmup)
        self._csamples = utests.ctiming_elapsed(self._ufunc)
        self._samples = []
        self._nsamples = nsamples
        self._iteration = iteration
        self._warmup = warmup
        self._msleep = msleep

    def metrics(self):
        samples = self._samples
        lx = np.log(samples)
        gmean = np.exp(lx.sum()/len(samples))
        gstd = np.exp(np.std(lx))
        mean = np.mean(samples)
        median = np.median(samples)
        return dict(gmean=gmean, gstd=gstd, mean=mean, median=median)

    def run(self, *args, **kwargs):
        self._samples = self._run(*args, **kwargs)

    def _run(self, *args, **kwargs):
        # clear any previous C samples
        self._csamples.clear()
        if self._msleep > 0:
            ssleep = self._msleep / 1000
            for _ in range(self._nsamples):
                time.sleep(ssleep)
                self._ufunc(*args, **kwargs)
        else:
            for _ in range(self._nsamples):
                self._ufunc(*args, **kwargs)

        s =  np.array(self._csamples).astype("float64")
        s /= self._iteration
        # remove outliers
        q_25, q_75 = np.percentile(s, [25, 75])
        iqr = q_75 - q_25
        half = iqr * 1.5
        low = s >= (q_25 - half)
        high = s <= (q_75 + half)
        idx = low & high
        return s[idx]

class Benchmark:
    ASCII = 0
    MARKDOWN = 1

    def __init__(self, **kwargs):
        for attr, dval in (
            ("filter", ".*"),
            ("strides", [1]),
            ("sizes", [1024]),
            ("nsamples", 100),
            ("iteration", 1),
            ("warmup", 0),
            ("msleep", 0),
            ("unit_scale", 1000),
            ("metric", "gmean"),
            ("rand_range", None)
        ):
            setattr(self, '_' + attr, kwargs.pop(attr, dval))


    def generate_tests(self):
        tests = dict()
        filter_rgx = re.compile(self._filter)

        for ufunc_name in dir(np):
            ufunc = getattr(np, ufunc_name)
            if not isinstance(ufunc, np.ufunc):
                continue

            nin = ufunc.nin; nout = ufunc.nout; utypes = ufunc.types
            permutes = [self._strides] * (nin + nout)
            permutes = list(itertools.product(*permutes))

            for tsym in utypes:
                tsym = tsym.split('->'); tin = tsym[0]; tout = tsym[1]
                for p in permutes:
                    str_in  = ' '.join([
                        "%s::%d" % (tin[c], s)  for c, s in enumerate(p[:nin])
                    ])
                    str_out = ' '.join([
                        "%s::%d" % (tout[c], s) for c, s in enumerate(p[nin:])
                    ])
                    for size in self._sizes:
                        case_name = "{ufunc_name}::{size:<6} {str_in} -> {str_out}".format(
                            ufunc_name=ufunc_name, size=size, str_in=str_in, str_out=str_out
                        )
                        if not filter_rgx.match(case_name):
                            continue
                        cases = tests.setdefault(ufunc_name, {})
                        cases[case_name] = dict(
                            size=size, strides=p, types=tin+tout
                        )
        return tests

    @staticmethod
    def timing_ufunc(queue, test_cases, ufunc_name, nsamples, iteration, warmup, msleep, rand_range):
        @functools.lru_cache(maxsize=1024)
        def rand(size, dtype, prevent_overlap=0):
            if dtype == '?':
                return np.random.randint(0, 1, size=size, dtype=dtype)
            elif not rand_range:
                if dtype in 'bBhHiIlLqQ':
                    return np.random.randint(1, 127, size=size, dtype=dtype)
                else:
                    return np.array(np.random.rand(size), dtype=dtype)
            else:
                ltype = dtype.lower() if dtype in 'GDF' else dtype
                lrange = len(rand_range)//2
                rand = np.empty(size, dtype=ltype)

                for i, r in enumerate(zip(rand_range[0::2], rand_range[1::2])):
                    rmin, rmax = np.array(r, dtype=ltype)
                    rand[i::lrange] = np.random.uniform(low=rmin, high=rmax, size=size//lrange).astype(ltype)

                return rand + rand*1j if dtype in 'GDF' else rand

        timing = Timing(
            ufunc=getattr(np, ufunc_name), nsamples=nsamples,
            iteration=iteration, warmup=warmup, msleep=msleep
        )
        result = {}
        for name, prob in test_cases.items():
            size, strides, types = prob["size"], prob["strides"], prob["types"]
            try:
                timing.run(*[
                    rand(size * strides[c], t, c)[::strides[c]]
                    for c, t in enumerate(types)
                ])
                print('.', end='', flush=True)
            except KeyboardInterrupt:
                break
            except Exception as err:
                Colored.notify(f"Escape test '{name}' -> {err}")
                continue
            result[name] = timing.metrics()
        queue.put(result)
        print("done", flush=True)

    def run(self, tests):
        multiprocessing.set_start_method('spawn')
        for ufunc_name, test_cases in tests.items():
            print("Benchmarking ufunc %s, %d cases " % (ufunc_name, len(test_cases)), end='', flush=True)
            queue = multiprocessing.Queue()
            p = multiprocessing.Process(target=self.timing_ufunc, args=(
                queue, test_cases, ufunc_name, self._nsamples,
                self._iteration, self._warmup, self._msleep, self._rand_range)
            )
            p.start()
            p.join()
            result = queue.get()
            if not result:
                continue
            for name, metrics in result.items():
                test_cases[name].update(metrics)

    def generate_table(self, tests_names, tests, only_changed=0, tformat="txt"):
        assert(len(tests_names) == len(tests))
        field_names = ["name of test"] + tests_names + [
            "%s vs %s" % (t, tests_names[0]) for t in tests_names[1:]
        ]
        ratios_fields = list(range(len(tests_names) + 1, len(tests_names)*2))
        table = Table(tformat, field_names, ratios_fields, {0:'left'})

        if len(tests) > 1 and only_changed != 0:
            factor_l = 1.0 - only_changed; factor_h = 1.0 + only_changed
            factor_falls = lambda f: f <= factor_l or f >= factor_h
            self._compare(tests[0], tests[1:],
                lambda case_name, metrics, ratios: table.add_row(
                    [case_name] + metrics + ratios
                ) if any([f and factor_falls(round(f, 2)) for f in ratios]) else None
            )
        else:
            self._compare(tests[0], tests[1:],
                lambda case_name, metrics, ratios: table.add_row(
                    [case_name] + metrics + ratios
                )
            )
        return str(table)

    def _compare(self, tests, cmp_tests, append_to):
        for ufunc_name, test_cases in tests.items():
            test_cases = sorted(test_cases.items(), key=lambda k: (
                k[1]["types"], k[1]["strides"], k[1]["size"]
            ))
            for case_name, case in test_cases:
                metric = case.get(self._metric)
                if not metric:
                    cmplen = len(cmp_tests)
                    append_to(case_name, [None]*(cmplen+1), [None]*cmplen)
                    continue

                metric = Timing.to_unit(metric, self._unit_scale)
                cmp_metrics, cmp_ratios = [], []
                for cmp_test in cmp_tests:
                    cmp_case = cmp_test.get(ufunc_name, {}).get(case_name, {})
                    cmp_metric = cmp_case.get(self._metric)
                    if not cmp_metric:
                        cmp_metrics.append(None)
                        cmp_ratios.append(None)
                        continue
                    cmp_metric = Timing.to_unit(cmp_metric, self._unit_scale)
                    try:
                        ratio = metric/cmp_metric
                    except ZeroDivisionError:
                        ratio = None
                    cmp_ratios.append(ratio)
                    cmp_metrics.append(cmp_metric)
                append_to(case_name, [metric] + cmp_metrics, cmp_ratios)

def arg_output(arg):
    output_path = os.path.abspath(arg)
    output_name = os.path.splitext(os.path.basename(arg))
    output_type, output_name = output_name[1][1:], output_name[0]

    if not output_type or not output_name:
        raise argparse.ArgumentTypeError(f"Invalid output path -> '{arg}'")
    if output_type not in Table.FORMATS:
        raise argparse.ArgumentTypeError(
            f"Unsupported output format {output_type}, expected a path ends with one of ({Table.FORMATS}) -> '{arg}'"
        )
    return dict(path=output_path, name=output_name, type=output_type)

def arg_compare(arg):
    path = os.path.abspath(arg)
    test_name = os.path.splitext(os.path.basename(path))[0]
    data = {}
    try:
        with open(path, "r") as fd:
            data = json.load(fd)
    except IOError as err:
        raise argparse.ArgumentTypeError(f"Unable to load JSON file -> {str(err)}")
    except json.JSONDecodeError as err:
        raise argparse.ArgumentTypeError(f"Invalid JSON file -> {path}, {str(err)}")
    return (test_name, data)

def arg_only_changed(arg):
    try:
        a = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Expected a floating point number")
    if a and a < 0.0 or a > 1.0:
        raise argparse.ArgumentTypeError("The value must falls between 0.0 and 1.0")
    return a

class arg_rand_range(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if len(values) not in (2, 4, 8):
            print(parser.error)
            parser.error(f"argument {option_string}: Invalid range")
        setattr(args, self.dest, values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-o", "--output", nargs="+", type=arg_output,
                        metavar=f"[PATH.({', '.join(Table.FORMATS)})]", default=[],
                        help=("store the benchmark test results to the given file, "
                              "the extension determines the output format."))
    parser.add_argument("-f", "--filter", metavar="REGEX", default=".*",
                        help="regex to filter benchmark tests")
    parser.add_argument("-c", "--compare", nargs="+", type=arg_compare, metavar="FILE", default=[],
                        help="list of exported JSON files to compared with")
    parser.add_argument("-d", "--export", metavar="PATH", default=None,
                        help="store the result into JSON file, to be used later with --compare")
    parser.add_argument("--only-changed", type=arg_only_changed, default=0,
                        help="show only changed results depend of the given factor and"
                             "value must falls between 0.0 and 1.0. "
                             "NOTE: this option works with '--compare'")

    parser.add_argument("-n", "--nsamples", type=int, default=100,
                        help="number of samples to be collected for each benchmark")

    parser.add_argument("--iteration", type=int, default=10,
                        help="number of iteration for each collected sample")
    parser.add_argument("--warmup", type=int, default=5,
                        help="uncollected iterations for each collected sample")

    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2],
                        help="strides for input and output arrays")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 2048, 4096],
                        help="array sizes")

    cpu_count = multiprocessing.cpu_count()
    parser.add_argument("--cpu-affinity", type=int, nargs="+", choices=range(0, cpu_count), default=None,
                        help="Set the CPU affinity for the running ufunc, only supported on Linux")

    metric_choices = ["gmean", "gstd", "mean", "median"]
    parser.add_argument("--metric", choices=metric_choices, default="gmean",
                       help="output metric")
    units_choices = dict(sec=Timing.SEC, ms=Timing.MS, us=Timing.US)
    parser.add_argument("--units", default="ms", choices=list(units_choices.keys()),
                       help="units of the output values")
    parser.add_argument("--msleep", type=float, default=0.1,
                       help="suspends execution of the calling thread before collecting each sample "
                            "for at least milliseconds")
    parser.add_argument("--rand-range", type=float, nargs="+", default=None, action=arg_rand_range,
                       help="specify a rand range for the generated arrays. "
                            "NOTE: multiple ranges will be interleaved")
    args = parser.parse_args()
    # 1- initialize
    bench = Benchmark(filter=args.filter, strides=args.strides, sizes=args.sizes,
                      nsamples=args.nsamples, iteration=args.iteration,
                      warmup=args.warmup, msleep=args.msleep, metric=args.metric,
                      unit_scale=units_choices[args.units],
                      rand_range=args.rand_range)

    # 2- fetch the generated tests,
    from numpy._pytesttester import _show_numpy_info
    _show_numpy_info()
    Colored.ok("Discovering benchmarks")
    running_tests = bench.generate_tests()
    if len(running_tests) < 1:
        Colored.fatal("No benchmarks selected")

    total_tests = 0
    for ufunc, cases in running_tests.items():
        total_tests += len(cases)
    Colored.ok("Running %d total benchmarks from %d ufuncs" % (
        total_tests, len(running_tests)
    ))
    if total_tests > 1024 * 10:
        desc = input(Colored.text((
            "Which is a huge amount of benchmarks, "
            "you may need to use '--filter' to reduce them.\n"
            "Do you want to continue? y or n? "
        ), Colored.YELLOW))
        desc = desc.strip().lower()
        while(1):
            if desc.startswith('y'):
                break
            if desc.startswith('n'):
                sys.exit(1)
            desc = input(Colored.text(('y or n? '), Colored.RED)).strip().lower()

    # 3- Set CPU affinity for running the benchmark
    if args.cpu_affinity:
        Timing.set_affinity(*args.cpu_affinity[:cpu_count])

    # 4- unleash the tests
    bench.run(running_tests)

    # 5- print the results
    cmp_tests_name, cmp_tests_data = zip(*args.compare) if args.compare else ([], [])
    all_tests = [running_tests] + list(cmp_tests_data)
    all_names = ["current"] + list(cmp_tests_name)

    final_result = bench.generate_table(
        all_names, all_tests, only_changed=args.only_changed,
        tformat=("ansi" if Colored.IS_TTYOUT else "txt")
    )
    print(final_result)

    # 6- store results into JSON file
    if args.export:
        export_path = os.path.abspath(args.export)
        Colored.notify(f"Exporting benchmarking result into '{export_path}'")
        try:
            with open(export_path, "w") as fd:
                json.dump(running_tests, fd)
        except IOError as err:
            Colored.fatal(f"Failed to export benchmarking result, '{str(err)}'")

    # 7- store the results into files
    for out in args.output:
        Colored.notify(f"Writing benchmarking result into {out['path']}")
        all_names[0] = out["name"]
        final_result = bench.generate_table(
            all_names, all_tests, only_changed=args.only_changed, tformat=out["type"]
        )
        with open(out["path"], "w") as fd:
            fd.write(f"metric: {args.metric}, units: {args.units}\n")
            fd.write(final_result)
