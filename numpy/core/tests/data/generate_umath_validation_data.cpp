#include<math.h>
#include<stdio.h>
#include<iostream>
#include<algorithm>
#include<vector>
#include<random>
#include<fstream>
#include<time.h>

static std::vector<std::string> funcnames = {"sin", "cos", "tan", "arcsin", "arccos", "arctan", "sinh", "cosh", "tanh", "arcsinh", "arccosh", "arctanh", "cbrt", "exp2", "expm1", "log10", "log1p", "log2"};
static std::vector<double (*) (double)> f32funcs = {sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, cbrt, exp2, expm1, log10, log1p, log2};
static std::vector<long double (*) (long double)> f64funcs = {sin, cos, tan, asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh, cbrt, exp2, expm1, log10, log1p, log2};

template<typename T>
T RandomFloat(T a, T b) {
    T random = ((T) rand()) / (T) RAND_MAX;
    T diff = b - a;
    T r = random * diff;
    return a + r;
}

template<typename T>
void append_random_array(std::vector<T>& arr, T min, T max, size_t N)
{
    for (size_t ii = 0; ii < N; ++ii)
        arr.emplace_back(RandomFloat<T>(min, max));
}

template<typename T1, typename T2>
std::vector<T1> computeTrueVal(const std::vector<T1>& in, T2(*mathfunc)(T2)) {
    std::vector<T1> out;
    for (T1 elem : in) {
        T2 elem_d = (T2) elem;
        T1 out_elem = (T1) mathfunc(elem_d);
        out.emplace_back(out_elem);
    }
    return out;
}

/*
 * FP range:
 * [-inf, -maxflt, -1., -minflt, -minden, 0., minden, minflt, 1., maxflt, inf]
 */

#define MINDEN std::numeric_limits<T>::denorm_min()
#define MINFLT std::numeric_limits<T>::min()
#define MAXFLT std::numeric_limits<T>::max()
#define INF    std::numeric_limits<T>::infinity()
#define qNAN   std::numeric_limits<T>::quiet_NaN()
#define sNAN   std::numeric_limits<T>::signaling_NaN()

template<typename T>
std::vector<T> generate_input_vector(int funcindex) {
    std::vector<T> input = {MINDEN, -MINDEN, MINFLT, -MINFLT, MAXFLT, -MAXFLT,
                            INF, -INF, qNAN, sNAN, -1.0, 1.0, 0.0, -0.0};
    std::string func = funcnames[funcindex];

    // [-1.0, 1.0]
    if ((func == "arcsinput") || (func == "arccos") || (func == "arctanh")){
        append_random_array<T>(input, -1.0, 1.0, 700);
    }
    // (0.0, INF]
    else if ((func == "log2") || (func == "log10")) {
        append_random_array<T>(input, 0.0, 1.0, 200);
        append_random_array<T>(input, MINDEN, MINFLT, 200);
        append_random_array<T>(input, MINFLT, 1.0, 200);
        append_random_array<T>(input, 1.0, MAXFLT, 200);
    }
    // (-1.0, INF]
    else if (func == "log1p") {
        append_random_array<T>(input, -1.0, 1.0, 200);
        append_random_array<T>(input, -MINFLT, -MINDEN, 100);
        append_random_array<T>(input, -1.0, -MINFLT, 100);
        append_random_array<T>(input, MINDEN, MINFLT, 100);
        append_random_array<T>(input, MINFLT, 1.0, 100);
        append_random_array<T>(input, 1.0, MAXFLT, 100);
    }
    // [1.0, INF]
    else if (func == "arccosh") {
        append_random_array<T>(input, 1.0, 2.0, 400);
        append_random_array<T>(input, 2.0, MAXFLT, 300);
    }
    // [-INF, INF]
    else {
        append_random_array<T>(input, -1.0, 1.0, 100);
        append_random_array<T>(input, MINDEN, MINFLT, 100);
        append_random_array<T>(input, -MINFLT, -MINDEN, 100);
        append_random_array<T>(input, MINFLT, 1.0, 100);
        append_random_array<T>(input, -1.0, -MINFLT, 100);
        append_random_array<T>(input, 1.0, MAXFLT, 100);
        append_random_array<T>(input, -MAXFLT, -100.0, 100);
    }

    std::random_shuffle(input.begin(), input.end());
    return input;
}

int main() {
    srand (42);
    for (int ii = 0; ii < f32funcs.size(); ++ii) {
         // ignore sin/cos
        if ((funcnames[ii] != "sin") && (funcnames[ii] != "cos")) {
            std::string fileName = "umath-validation-set-" + funcnames[ii] + ".csv";
            std::ofstream txtOut;
            txtOut.open (fileName, std::ofstream::trunc);
            txtOut << "dtype,input,output,ulperrortol" << std::endl;

            // Single Precision
            auto f32in = generate_input_vector<float>(ii);
            auto f32out = computeTrueVal<float, double>(f32in, f32funcs[ii]);
            for (int ii = 0; ii < f32in.size(); ++ii) {
                txtOut << "np.float32" << std::hex <<
                          ",0x" << *reinterpret_cast<uint32_t*>(&f32in[ii]) <<
                          ",0x" << *reinterpret_cast<uint32_t*>(&f32out[ii]) <<
                          ",4" << std::endl;
            }

            // Double Precision
            auto f64in = generate_input_vector<double>(ii);
            auto f64out = computeTrueVal<double, long double>(f64in, f64funcs[ii]);
            for (int ii = 0; ii < f64in.size(); ++ii) {
                txtOut << "np.float64" << std::hex <<
                          ",0x" << *reinterpret_cast<uint64_t*>(&f64in[ii]) <<
                          ",0x" << *reinterpret_cast<uint64_t*>(&f64out[ii]) <<
                          ",4" << std::endl;
            }
            txtOut.close();
        }
    }
    return 0;
}
