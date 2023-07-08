#include <string>
#include <algorithm>
#include <hwy/targets.h>

#include <Python.h> // for PyObject
#include "numpy/numpyconfig.h" // for NPY_VISIBILITY_HIDDEN

namespace np {

static PyObject* create_target_list(std::vector<int64_t> targets) {
    PyObject *list = PyList_New(targets.size());
    for(std::size_t i = 0; i < targets.size(); ++i) {
        int64_t target = targets[i];
        PyList_SetItem(list, i, PyUnicode_FromString(hwy::TargetName(target)));
    }
    return list;
}

static std::string create_target_string(std::vector<std::string> targets) {
    std::string target_string;
    if (targets.size() == 0) {
        return "";
    }

    for(std::size_t i = 0; i < targets.size(); ++i) {
        std::string target = targets[i];
        target_string.append(" ");
        target_string.append(target);
    }
    
    // Skip first space
    return target_string.substr(1);
}

static std::string create_target_string(std::vector<int64_t> targets) {
    std::string target_string;
    if (targets.size() == 0) {
        return "";
    }

    for(std::size_t i = 0; i < targets.size(); ++i) {
        int64_t target = targets[i];
        target_string.append(" ");
        target_string.append(hwy::TargetName(target));
    }
    
    // Skip first space
    return target_string.substr(1);
}

static std::vector<int64_t> baseline_targets() {
    std::vector<int64_t> baseline_targets;
    for (int64_t targets = HWY_BASELINE_TARGETS; targets != 0; targets = targets & (targets - 1)) {
        int64_t current_target = targets & ~(targets - 1);
        baseline_targets.push_back(current_target);
    }
    return baseline_targets;
}

static std::vector<int64_t> dispatch_targets() {
    std::vector<int64_t> dispatch_targets;
    for (int64_t targets = hwy::SupportedTargets() & HWY_TARGETS; targets != 0;
        targets = targets & (targets - 1)) {
        int64_t current_target = targets & ~(targets - 1);
        if (!(HWY_BASELINE_TARGETS & current_target)) {
            dispatch_targets.push_back(current_target);
        }
    }
    return dispatch_targets;
}

static bool is_dispatch_target(int64_t target_id) {
    std::vector<int64_t> dispatchable_targets = dispatch_targets();
    return std::find(dispatchable_targets.begin(), dispatchable_targets.end(), target_id) != dispatchable_targets.end();
}


static int64_t get_target_id(const char *feature)
{
    std::vector<int64_t> baseline_targets;
    for (int64_t targets = HWY_TARGETS; targets != 0; targets = targets & (targets - 1)) {
        int64_t current_target = targets & ~(targets - 1);
        if (strcmp(hwy::TargetName(current_target), feature) == 0) {
            return current_target;
        }
    }
    return 0;
}

static int check_env(int disable, char *env) {
    static const char *names[] = {
        "enable", "disable",
        "During parsing environment variable: 'NPY_ENABLE_CPU_FEATURES':\n",
        "During parsing environment variable: 'NPY_DISABLE_CPU_FEATURES':\n"
    };
    disable = disable ? 1 : 0;
    const char *act_name = names[disable];
    const char *err_head = names[disable + 2];

    const char *delim = ", \t\v\r\n\f";

    int64_t enable_targets = 0;
    int64_t disable_targets = 0;
    std::vector<std::string> not_in_dispatch;
    std::vector<std::string> not_in_supported;

    char *feature = strtok(env, delim);
    while (feature) {
        int64_t target_id = get_target_id(feature);
        if (target_id == 0) {
            // Got no target back
            not_in_dispatch.push_back(feature);
        } else {
            if (disable && (target_id & HWY_BASELINE_TARGETS)) {
                // Escape early when trying to disable baseline features
                PyErr_Format(PyExc_RuntimeError,
                    "%s"
                    "You cannot disable CPU feature '%s', since it is part of "
                    "the baseline optimizations:\n"
                    "(%s).",
                    err_head, feature, create_target_string(baseline_targets()).c_str()
                );
                return -1;
            } else if (is_dispatch_target(target_id)) {
                if (disable) {
                    disable_targets |= target_id;
                } else {
                    enable_targets |= target_id;
                }
            } else {
                if (target_id & hwy::SupportedTargets()) {
                    not_in_dispatch.push_back(feature);
                } else if (!disable) {
                    not_in_supported.push_back(feature);
                }
            }
        }

        feature = strtok(NULL, delim);
    }

    if (not_in_dispatch.size()) {
        if (PyErr_WarnFormat(
            PyExc_ImportWarning, 0,
            "%sYou cannot %s CPU features (%s), since "
            "they are not part of the dispatched optimizations\n"
            "(%s).",
            err_head, act_name,
            create_target_string(not_in_dispatch).c_str(),
            create_target_string(dispatch_targets()).c_str()
        ) < 0) {
            return -1;
        }

        return 0;
    }
    if (not_in_supported.size()) {
        PyErr_Format(PyExc_RuntimeError,
            "%s" \
            "You cannot %s CPU features (%s), since " \
            "they are not supported by your machine.", \
            err_head, act_name, 
            create_target_string(not_in_supported).c_str()
        );

        return -1;
    }

    // disable any targets which are not marked as enabled
    if (disable && disable_targets != 0) {
        hwy::DisableTargets(disable_targets);
    }
    if (!disable && enable_targets != 0) {
        hwy::DisableTargets(~enable_targets);
    }

    return 0;
}

extern "C" {

NPY_VISIBILITY_HIDDEN int
npy_hwy_init(void)
{
    char *enable_env = getenv("NPY_ENABLE_CPU_FEATURES");
    char *disable_env = getenv("NPY_DISABLE_CPU_FEATURES");
    int is_enable = enable_env && enable_env[0];
    int is_disable = disable_env && disable_env[0];
    if (is_enable & is_disable) {
        PyErr_Format(PyExc_ImportError,
            "Both NPY_DISABLE_CPU_FEATURES and NPY_ENABLE_CPU_FEATURES "
            "environment variables cannot be set simultaneously."
        );
        return -1;
    }

    if (is_enable | is_disable) {
        if (check_env(is_disable, is_disable ? disable_env : enable_env) < 0) {
            return -1;
        }
    }
       
    return 0;
}

NPY_VISIBILITY_HIDDEN PyObject *
npy_hwy_baseline_list(void) {
    std::vector<int64_t> baseline_targets;
    for (int64_t targets = HWY_BASELINE_TARGETS; targets != 0; targets = targets & (targets - 1)) {
        int64_t current_target = targets & ~(targets - 1);
        baseline_targets.push_back(current_target);
    }
    return create_target_list(baseline_targets);
}

NPY_VISIBILITY_HIDDEN PyObject *
npy_hwy_features_dict(void) {
    PyObject *dict = PyDict_New();
    if (dict) {
        for (int64_t targets = HWY_ATTAINABLE_TARGETS; targets != 0; targets = targets & (targets - 1)) {
            int64_t current_target = targets & ~(targets - 1);
            PyDict_SetItemString(
                dict,
                hwy::TargetName(current_target),
                (hwy::SupportedTargets() & current_target) ? Py_True : Py_False
            );
        }
    }
    return dict;
}

NPY_VISIBILITY_HIDDEN PyObject *
npy_hwy_dispatch_list(void)
{
    std::vector<int64_t> dispatch_targets;
    for (int64_t targets = hwy::SupportedTargets() & HWY_TARGETS; targets != 0;
        targets = targets & (targets - 1)) {
        int64_t current_target = targets & ~(targets - 1);
        if (!(HWY_BASELINE_TARGETS & current_target)) {
            dispatch_targets.push_back(current_target);
        }
    }
    return create_target_list(dispatch_targets);
}
}
}
