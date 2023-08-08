# Copyright (c) 2023, NumPy Developers.
import os
from typing import (
    Dict, Set, Tuple, List, Callable, Optional,
    Union, Any, cast, TYPE_CHECKING
)
from ... import mlog, build
from ...compilers import Compiler
from ...mesonlib import File, MesonException
from ...interpreter.type_checking import NoneType
from ...interpreterbase.decorators import (
    noKwargs, KwargInfo, typed_kwargs, typed_pos_args,
    ContainerTypeInfo, permittedKwargs
)
from .. import ModuleInfo, NewExtensionModule, ModuleObject
from .feature import FeatureObject, ConflictAttr
from .utils import test_code, get_compiler, generate_hash

if TYPE_CHECKING:
    from typing import TypedDict
    from ...interpreterbase import TYPE_var, TYPE_kwargs
    from .. import ModuleState
    from .feature import FeatureKwArgs

    class TestKwArgs(TypedDict):
        compiler: Optional[Compiler]
        force_args: Optional[List[str]]
        anyfet: bool
        cached: bool

    class TestResultKwArgs(TypedDict):
        target_name: str
        prevalent_features: List[str]
        features: List[str]
        args: List[str]
        detect: List[str]
        defines: List[str]
        undefines: List[str]
        is_supported: bool
        is_disabled: bool
        fail_reason: str

class TargetsObject(ModuleObject):
    def __init__(self) -> None:
        super().__init__()
        self._targets: Dict[
            Union[FeatureObject, Tuple[FeatureObject, ...]],
            List[build.StaticLibrary]
        ] = {}
        self._baseline: List[build.StaticLibrary] = []
        self.methods.update({
            'static_lib': self.static_lib_method,
            'extend': self.extend_method
        })

    def extend_method(self, state: 'ModuleState',
                      args: List['TYPE_var'],
                      kwargs: 'TYPE_kwargs') -> 'TargetsObject':

        @typed_pos_args('feature.TargetsObject.extend', TargetsObject)
        @noKwargs
        def test_args(state: 'ModuleState',
                      args: Tuple[TargetsObject],
                      kwargs: 'TYPE_kwargs') -> TargetsObject:
            return args[0]
        robj: TargetsObject = test_args(state, args, kwargs)
        self._baseline.extend(robj._baseline)
        for features, robj_targets in robj._targets.items():
            targets: List[build.StaticLibrary] = self._targets.setdefault(features, [])
            targets += robj_targets
        return self

    @typed_pos_args('features.TargetsObject.static_lib', str)
    @noKwargs
    def static_lib_method(self, state: 'ModuleState', args: Tuple[str],
                          kwargs: 'TYPE_kwargs'
                          ) -> Any:
        # The linking order must be based on the lowest interested features,
        # to ensures that the linker prioritizes any duplicate weak global symbols
        # of the lowest interested features over the highest ones,
        # starting with the baseline to avoid any possible crashes due
        # to any involved optimizations that may generated based
        # on the highest interested features.
        link_whole = [] + self._baseline
        tcast = Union[FeatureObject, Tuple[FeatureObject, ...]]
        for features in FeatureObject.sorted_multi(self._targets.keys()):
            link_whole += self._targets[cast(tcast, features)]
        if not link_whole:
            return []
        static_lib = state._interpreter.func_static_lib(
            None, [args[0]], {
                'link_whole': link_whole
            }
        )
        return static_lib

    def add_baseline_target(self, target: build.StaticLibrary) -> None:
        self._baseline.append(target)

    def add_target(self, features: Union[FeatureObject, List[FeatureObject]],
                   target: build.StaticLibrary) -> None:
        tfeatures = (
            features if isinstance(features, FeatureObject)
            else tuple(sorted(features))
        )
        targets: List[build.StaticLibrary] = self._targets.setdefault(
            tfeatures, cast(List[build.StaticLibrary], []))  # type: ignore
        targets.append(target)

class Module(NewExtensionModule):
    INFO = ModuleInfo('features', '0.1.0')
    def __init__(self) -> None:
        super().__init__()
        self.methods.update({
            'new': self.new_method,
            'test': self.test_method,
            'implicit': self.implicit_method,
            'implicit_c': self.implicit_c_method,
            'sort': self.sort_method,
            'multi_targets': self.multi_targets_method,
        })

    def new_method(self, state: 'ModuleState',
                   args: List['TYPE_var'],
                   kwargs: 'TYPE_kwargs') -> FeatureObject:
        return FeatureObject(state, args, kwargs)

    def _cache_dict(self, state: 'ModuleState'
                    ) -> Dict[str, 'TestResultKwArgs']:
        coredata = state.environment.coredata
        attr_name = 'module_features_cache'
        if not hasattr(coredata, attr_name):
            setattr(coredata, attr_name, {})
        return getattr(coredata, attr_name, {})

    def _get_cache(self, state: 'ModuleState', key: str
                   ) -> Optional['TestResultKwArgs']:
        return self._cache_dict(state).get(key)

    def _set_cache(self, state: 'ModuleState', key: str,
                   val: 'TestResultKwArgs') -> None:
        self._cache_dict(state)[key] = val

    @typed_pos_args('features.test', varargs=FeatureObject, min_varargs=1)
    @typed_kwargs('features.test',
        KwargInfo('compiler', (NoneType, Compiler)),
        KwargInfo('anyfet', bool, default = False),
        KwargInfo('cached', bool, default = True),
        KwargInfo(
            'force_args', (NoneType, str, ContainerTypeInfo(list, str)),
            listify=True
        ),
    )
    def test_method(self, state: 'ModuleState',
                    args: Tuple[List[FeatureObject]],
                    kwargs: 'TestKwArgs'
                    ) -> List[Union[bool, 'TestResultKwArgs']]:

        features = args[0]
        features_set = set(features)
        anyfet = kwargs['anyfet']
        cached = kwargs['cached']
        compiler = kwargs.get('compiler')
        if not compiler:
            compiler = get_compiler(state)

        force_args = kwargs['force_args']
        if force_args is not None:
            # removes in empty strings
            force_args = [a for a in force_args if a]

        test_cached, test_result = self.cached_test(
            state, features=features_set,
            compiler=compiler,
            anyfet=anyfet,
            cached=cached,
            force_args=force_args
        )
        if not test_result['is_supported']:
            if test_result['is_disabled']:
                label = mlog.yellow('disabled')
            else:
                label = mlog.yellow('Unsupported')
        else:
            label = mlog.green('Supported')
            if anyfet:
                unsupported = ' '.join([
                    fet.name for fet in sorted(features_set)
                    if fet.name not in test_result['features']
                ])
                if unsupported:
                    label = mlog.green(f'Parial support, missing({unsupported})')

        features_names = ' '.join([f.name for f in features])
        log_prefix = f'Test features "{mlog.bold(features_names)}" :'
        cached_msg = f'({mlog.blue("cached")})' if test_cached else ''
        if not test_result['is_supported']:
            mlog.log(log_prefix, label, 'due to', test_result['fail_reason'])
        else:
            mlog.log(log_prefix, label, cached_msg)
        return [test_result['is_supported'], test_result]

    def cached_test(self, state: 'ModuleState',
                    features: Set[FeatureObject],
                    compiler: 'Compiler',
                    force_args: Optional[List[str]],
                    anyfet: bool, cached: bool,
                    _caller: Optional[Set[FeatureObject]] = None
                    ) -> Tuple[bool, 'TestResultKwArgs']:

        if cached:
            test_hash = generate_hash(
                sorted(features), compiler,
                anyfet, force_args
            )
            test_result = self._get_cache(state, test_hash)
            if test_result is not None:
                return True, test_result

        if anyfet:
            test_func = self.test_any
        else:
            test_func = self.test

        test_result = test_func(
            state, features=features,
            compiler=compiler,
            force_args=force_args,
            cached=cached,
            _caller=_caller
        )
        if cached:
            self._set_cache(state, test_hash, test_result)
        return False, test_result

    def test_any(self, state: 'ModuleState', features: Set[FeatureObject],
                 compiler: 'Compiler',
                 force_args: Optional[List[str]],
                 cached: bool,
                 # dummy no need for recrusive guard
                 _caller: Optional[Set[FeatureObject]] = None,
                 ) -> 'TestResultKwArgs':

        _, test_any_result = self.cached_test(
            state, features=features,
            compiler=compiler,
            anyfet=False,
            cached=cached,
            force_args=force_args,
        )
        if test_any_result['is_supported']:
            return test_any_result

        all_features = sorted(FeatureObject.get_implicit_combine_multi(features))
        features_any = set()
        for fet in all_features:
            _, test_any_result = self.cached_test(
                state, features={fet,},
                compiler=compiler,
                cached=cached,
                anyfet=False,
                force_args=force_args,
            )
            if test_any_result['is_supported']:
                features_any.add(fet)

        _, test_any_result = self.cached_test(
            state, features=features_any,
            compiler=compiler,
            cached=cached,
            anyfet=False,
            force_args=force_args,
        )
        return test_any_result

    def test(self, state: 'ModuleState', features: Set[FeatureObject],
             compiler: 'Compiler',
             force_args: Optional[List[str]] = None,
             cached: bool = True,
             _caller: Optional[Set[FeatureObject]] = None
             ) -> 'TestResultKwArgs':

        implied_features = FeatureObject.get_implicit_multi(features)
        all_features = sorted(implied_features.union(features))
        # For multiple features, it important to erase any features
        # implied by another to avoid duplicate testing since
        # implied features already tested also we use this set to genrate
        # unque target name that can be used for multiple targets
        # build.
        prevalent_features = sorted(features.difference(implied_features))
        if len(prevalent_features) == 0:
            # It happens when all features imply each other.
            # Set the highest interested feature
            prevalent_features = sorted(features)[-1:]

        prevalent_names =  [fet.name for fet in prevalent_features]
        # prepare the result dict
        test_result: 'TestResultKwArgs' = {
            'target_name': '__'.join(prevalent_names),
            'prevalent_features': prevalent_names,
            'features': [fet.name for fet in all_features],
            'args': [],
            'detect': [],
            'defines': [],
            'undefines': [],
            'is_supported': True,
            'is_disabled': False,
            'fail_reason': '',
        }
        def fail_result(fail_reason: str, is_disabled: bool = False
                        ) -> 'TestResultKwArgs':
            test_result.update({
                'features': [],
                'args': [],
                'detect': [],
                'defines': [],
                'undefines': [],
                'is_supported': False,
                'is_disabled': is_disabled,
                'fail_reason': fail_reason,
            })
            return test_result

        # test any of prevalent features wither they disabled or not
        for fet in prevalent_features:
            if fet.disable:
                return fail_result(
                    f'{fet.name} is disabled due to "{fet.disable}"',
                    True
                )

        # since we allows features to imply each other
        # items of `features` may part of `implied_features`
        if _caller is None:
            _caller = set()
        _caller = _caller.union(prevalent_features)
        predecessor_features = implied_features.difference(_caller)
        for fet in sorted(predecessor_features):
            _, pred_result = self.cached_test(
                state, features={fet,},
                compiler=compiler,
                cached=cached,
                anyfet=False,
                force_args=force_args,
                _caller=_caller,
            )
            if not pred_result['is_supported']:
                reason = f'Implied feature "{fet.name}" '
                pred_disabled = pred_result['is_disabled']
                if pred_disabled:
                    fail_reason = reason + 'is disabled'
                else:
                    fail_reason = reason + 'is not supported'
                return fail_result(fail_reason, pred_disabled)

            for k in ['defines', 'undefines']:
                def_values = test_result[k]  # type: ignore
                pred_values = pred_result[k]  # type: ignore
                def_values += [v for v in pred_values if v not in def_values]

        # Sort based on the lowest interest to deal with conflict attributes
        # when combine all attributes togathers
        conflict_attrs = ['detect']
        if force_args is None:
            conflict_attrs += ['args']
        else:
            test_result['args'] = force_args

        for fet in all_features:
            for attr in conflict_attrs:
                values: List[ConflictAttr] = getattr(fet, attr)
                accumulate_values = test_result[attr]  # type: ignore
                for conflict in values:
                    if not conflict.match:
                        accumulate_values.append(conflict.val)
                        continue
                    conflict_vals: List[str] = []
                    # select the acc items based on the match
                    new_acc: List[str] = []
                    for acc in accumulate_values:
                        # not affected by the match so we keep it
                        if not conflict.match.match(acc):
                            new_acc.append(acc)
                            continue
                        # no filter so we totaly escape it
                        if not conflict.mfilter:
                            continue
                        filter_val = conflict.mfilter.findall(acc)
                        filter_val = [
                            conflict.mjoin.join([i for i in val if i])
                            if isinstance(val, tuple) else val
                            for val in filter_val if val
                        ]
                        # no filter match so we totaly escape it
                        if not filter_val:
                            continue
                        conflict_vals.append(conflict.mjoin.join(filter_val))
                    new_acc.append(conflict.val + conflict.mjoin.join(conflict_vals))
                    test_result[attr] = new_acc  # type: ignore

        test_args = compiler.has_multi_arguments
        args = test_result['args']
        if args:
            supported_args, test_cached = test_args(args, state.environment)
            if not supported_args:
                return fail_result(
                    f'Arguments "{", ".join(args)}" are not supported'
                )

        for fet in prevalent_features:
            if fet.test_code:
                _, tested_code, _ = test_code(
                    state, compiler, args, fet.test_code
                )
                if not tested_code:
                    return fail_result(
                        f'Compiler fails against the test code of "{fet.name}"'
                    )

            test_result['defines'] += [fet.name] + fet.group
            for extra_name, extra_test in fet.extra_tests.items():
                _, tested_code, _ = test_code(
                    state, compiler, args, extra_test
                )
                k = 'defines' if tested_code else 'undefines'
                test_result[k].append(extra_name)  # type: ignore
        return test_result

    @permittedKwargs(build.known_stlib_kwargs | {
        'dispatch', 'baseline', 'prefix', 'cached', 'keep_sort'
    })
    @typed_pos_args('features.multi_targets', str, min_varargs=1, varargs=(
        str, File, build.CustomTarget, build.CustomTargetIndex,
        build.GeneratedList, build.StructuredSources, build.ExtractedObjects,
        build.BuildTarget
    ))
    @typed_kwargs('features.multi_targets',
        KwargInfo(
            'dispatch', (
                ContainerTypeInfo(list, (FeatureObject, list)),
            ),
            default=[]
        ),
        KwargInfo(
            'baseline', (
                NoneType,
                ContainerTypeInfo(list, FeatureObject)
            )
        ),
        KwargInfo('prefix', str, default=''),
        KwargInfo('compiler', (NoneType, Compiler)),
        KwargInfo('cached', bool, default = True),
        KwargInfo('keep_sort', bool, default = False),
        allow_unknown=True
    )
    def multi_targets_method(self, state: 'ModuleState',
                            args: Tuple[str], kwargs: 'TYPE_kwargs'
                            ) -> TargetsObject:
        config_name = args[0]
        sources = args[1]  # type: ignore
        dispatch: List[Union[FeatureObject, List[FeatureObject]]] = (
            kwargs.pop('dispatch') # type: ignore
        )
        baseline: Optional[List[FeatureObject]] = (
            kwargs.pop('baseline')  # type: ignore
        )
        prefix: str = kwargs.pop('prefix')  # type: ignore
        cached: bool = kwargs.pop('cached')  # type: ignore
        compiler: Optional[Compiler] = kwargs.pop('compiler')  # type: ignore
        if not compiler:
            compiler = get_compiler(state)

        baseline_features : Set[FeatureObject] = set()
        has_baseline = baseline is not None
        if has_baseline:
            baseline_features = FeatureObject.get_implicit_combine_multi(baseline)
            _, baseline_test_result = self.cached_test(
                state, features=set(baseline),
                anyfet=True, cached=cached,
                compiler=compiler,
                force_args=None
            )

        enabled_targets_names: List[str] = []
        enabled_targets_features: List[Union[
            FeatureObject, List[FeatureObject]
        ]] = []
        enabled_targets_tests: List['TestResultKwArgs'] = []
        skipped_targets: List[Tuple[
            Union[FeatureObject, List[FeatureObject]], str
        ]] = []
        for d in dispatch:
            if isinstance(d, FeatureObject):
                target = {d,}
                is_base_part = d in baseline_features
            else:
                target = set(d)
                is_base_part = all(f in baseline_features for f in d)

            if is_base_part:
                skipped_targets.append((d, "part of baseline features"))
                continue
            _, test_result = self.cached_test(
                state=state, features=target,
                anyfet=False, cached=cached,
                compiler=compiler,
                force_args=None
            )
            if not test_result['is_supported']:
                skipped_targets.append(
                    (d, test_result['fail_reason'])
                )
                continue
            target_name = test_result['target_name']
            if target_name in enabled_targets_names:
                skipped_targets.append((
                    d, f'Dublicate target name "{target_name}"'
                ))
                continue
            enabled_targets_names.append(target_name)
            enabled_targets_features.append(d)
            enabled_targets_tests.append(test_result)

        if not kwargs.pop('keep_sort'):
            enabled_targets_sorted = FeatureObject.sorted_multi(enabled_targets_features, reverse=True)
            if enabled_targets_features != enabled_targets_sorted:
                log_targets = FeatureObject.features_names(enabled_targets_features)
                log_targets_sorted = FeatureObject.features_names(enabled_targets_sorted)
                raise MesonException(
                    'The enabled dispatch features should be sorted based on the highest interest:\n'
                    f'Expected: {log_targets_sorted}\n'
                    f'Got: {log_targets}\n'
                    'Note: This validation may not be accurate when dealing with multi-features '
                    'per single target.\n'
                    'You can keep the current sort and bypass this validation by passing '
                    'the argument "keep_sort: true".'
                )

        config_path = self.gen_config(
            state,
            config_name=config_name,
            targets=enabled_targets_tests,
            prefix=prefix,
            has_baseline=has_baseline
        )
        mtargets_obj = TargetsObject()
        if has_baseline:
            mtargets_obj.add_baseline_target(
                self.gen_target(
                    state=state, config_name=config_name,
                    sources=sources, test_result=baseline_test_result,
                    prefix=prefix, is_baseline=True,
                    stlib_kwargs=kwargs
                )
            )
        for features_objects, target_test in zip(enabled_targets_features, enabled_targets_tests):
            static_lib = self.gen_target(
                state=state, config_name=config_name,
                sources=sources, test_result=target_test,
                prefix=prefix, is_baseline=False,
                stlib_kwargs=kwargs
            )
            mtargets_obj.add_target(features_objects, static_lib)

        skipped_targets_info: List[str] = []
        skipped_tab = ' '*4
        for skipped, reason in skipped_targets:
            name = ', '.join(
                [skipped.name] if isinstance(skipped, FeatureObject)
                else [fet.name for fet in skipped]
            )
            skipped_targets_info.append(f'{skipped_tab}"{name}": "{reason}"')

        target_info: Callable[[str, 'TestResultKwArgs'], str] = lambda target_name, test_result: (
            f'{skipped_tab}"{target_name}":\n' + '\n'.join([
                f'{skipped_tab*2}"{k}": {v}'
                for k, v in test_result.items()
            ])
        )
        enabled_targets_info: List[str] = [
            target_info(test_result['target_name'], test_result)
            for test_result in enabled_targets_tests
        ]
        if has_baseline:
            enabled_targets_info.append(target_info(
                f'baseline({baseline_test_result["target_name"]})',
                baseline_test_result
            ))
            enabled_targets_names += ['baseline']

        mlog.log(
            f'Generating multi-targets for "{mlog.bold(config_name)}"',
            '\n  Enabled targets:',
            mlog.green(', '.join(enabled_targets_names))
        )
        mlog.debug(
            f'Generating multi-targets for "{config_name}"',
            '\n  Config path:', config_path,
            '\n  Enabled targets:',
            '\n'+'\n'.join(enabled_targets_info),
            '\n  Skipped targets:',
            '\n'+'\n'.join(skipped_targets_info),
            '\n'
        )
        return mtargets_obj

    def gen_target(self, state: 'ModuleState', config_name: str,
                   sources: List[Union[
                      str, File, build.CustomTarget, build.CustomTargetIndex,
                      build.GeneratedList, build.StructuredSources, build.ExtractedObjects,
                      build.BuildTarget
                   ]],
                   test_result: 'TestResultKwArgs',
                   prefix: str, is_baseline: bool,
                   stlib_kwargs: Dict[str, Any]
                   ) -> build.StaticLibrary:

        target_name = 'baseline' if is_baseline else test_result['target_name']
        args = [f'-D{prefix}HAVE_{df}' for df in test_result['defines']]
        args += test_result['args']
        if is_baseline:
            args.append(f'-D{prefix}MTARGETS_BASELINE')
        else:
            args.append(f'-D{prefix}MTARGETS_CURRENT={target_name}')
        stlib_kwargs = stlib_kwargs.copy()
        stlib_kwargs.update({
            'sources': sources,
            'c_args': stlib_kwargs.get('c_args', []) + args,
            'cpp_args': stlib_kwargs.get('cpp_args', []) + args
        })
        static_lib: build.StaticLibrary = state._interpreter.func_static_lib(
            None, [config_name + '_' + target_name],
            stlib_kwargs
        )
        return static_lib

    def gen_config(self, state: 'ModuleState', config_name: str,
                   targets: List['TestResultKwArgs'],
                   prefix: str, has_baseline: bool
                   ) -> str:

        dispatch_calls: List[str] = []
        for test in targets:
            c_detect = '&&'.join([
                f'TEST_CB({d})' for d in test['detect']
            ])
            if c_detect:
                c_detect = f'({c_detect})'
            else:
                c_detect = '1'
            dispatch_calls.append(
                f'{prefix}_MTARGETS_EXPAND('
                    f'EXEC_CB({c_detect}, {test["target_name"]}, __VA_ARGS__)'
                ')'
            )

        config_file = [
            '/* Autogenerated by the Meson features module. */',
            '/* Do not edit, your changes will be lost. */',
            '',
            f'#undef {prefix}_MTARGETS_EXPAND',
            f'#define {prefix}_MTARGETS_EXPAND(X) X',
            '',
            f'#undef {prefix}MTARGETS_CONF_BASELINE',
            f'#define {prefix}MTARGETS_CONF_BASELINE(EXEC_CB, ...) ' + (
                f'{prefix}_MTARGETS_EXPAND(EXEC_CB(__VA_ARGS__))'
                if has_baseline else ''
            ),
            '',
            f'#undef {prefix}MTARGETS_CONF_DISPATCH',
            f'#define {prefix}MTARGETS_CONF_DISPATCH(TEST_CB, EXEC_CB, ...) \\',
            ' \\\n'.join(dispatch_calls),
            '',
        ]

        build_dir = state.environment.build_dir
        sub_dir = state.subdir
        if sub_dir:
            build_dir = os.path.join(build_dir, sub_dir)
        config_path = os.path.abspath(os.path.join(build_dir, config_name))

        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding='utf-8') as cout:
            cout.write('\n'.join(config_file))

        return config_path

    @typed_pos_args('features.sort', varargs=FeatureObject, min_varargs=1)
    @typed_kwargs('features.sort',
        KwargInfo('reverse', bool, default = False),
    )
    def sort_method(self, state: 'ModuleState',
                    args: Tuple[List[FeatureObject]],
                    kwargs: Dict[str, bool]
                    ) -> List[FeatureObject]:
        return sorted(args[0], reverse=kwargs['reverse'])

    @typed_pos_args('features.implicit', varargs=FeatureObject, min_varargs=1)
    @noKwargs
    def implicit_method(self, state: 'ModuleState',
                        args: Tuple[List[FeatureObject]],
                        kwargs: 'TYPE_kwargs'
                        ) -> List[FeatureObject]:

        features = args[0]
        return sorted(FeatureObject.get_implicit_multi(features))

    @typed_pos_args('features.implicit', varargs=FeatureObject, min_varargs=1)
    @noKwargs
    def implicit_c_method(self, state: 'ModuleState',
                          args: Tuple[List[FeatureObject]],
                          kwargs: 'TYPE_kwargs'
                          ) -> List[FeatureObject]:
        return sorted(FeatureObject.get_implicit_combine_multi(args[0]))
