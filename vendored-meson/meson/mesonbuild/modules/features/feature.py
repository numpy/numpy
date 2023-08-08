# Copyright (c) 2023, NumPy Developers.
# All rights reserved.
import re
from typing import (
    Dict, Set, Tuple, List, Callable, Optional,
    Union, Any, Iterable, cast, TYPE_CHECKING
)
from dataclasses import dataclass, field
from ...mesonlib import File, MesonException
from ...interpreter.type_checking import NoneType
from ...interpreterbase.decorators import (
    noKwargs, noPosargs, KwargInfo, typed_kwargs, typed_pos_args,
    ContainerTypeInfo
)
from .. import ModuleObject

if TYPE_CHECKING:
    from typing import TypedDict
    from typing_extensions import NotRequired
    from ...interpreterbase import TYPE_var, TYPE_kwargs
    from ...compilers import Compiler
    from .. import ModuleState

@dataclass(unsafe_hash=True, order=True)
class ConflictAttr:
    """
    Data class representing an feature attribute that may conflict
    with other features attributes.

    The reason behind this class to clear any possible conflicts with
    compiler arguments when they joined together due gathering
    the implied features or concatenate non-implied features.

    Attributes:
        val: The value of the feature attribute.
        match: Regular expression pattern for matching conflicted values
               (optional).
        mfilter: Regular expression pattern for filtering these conflicted values
               (optional).
        mjoin: String used to join filtered values (optional)

    """
    val: str = field(hash=True, compare=True)
    match: Union[re.Pattern, None] = field(
        default=None, hash=False, compare=False
    )
    mfilter: Union[re.Pattern, None] = field(
        default=None, hash=False, compare=False
    )
    mjoin: str = field(default='', hash=False, compare=False)

    def copy(self) -> 'ConflictAttr':
        return ConflictAttr(**self.__dict__)

    def to_dict(self) -> Dict[str, str]:
        ret: Dict[str, str] = {}
        for attr in ('val', 'mjoin'):
            ret[attr] = getattr(self, attr)
        for attr in ('match', 'mfilter'):
            val = getattr(self, attr)
            if not val:
                val = ''
            else:
                val = str(val)
            ret[attr] = val
        return ret

class KwargConfilctAttr(KwargInfo):
    def __init__(self, func_name: str, opt_name: str, default: Any = None):
        types = (
            NoneType, str, ContainerTypeInfo(dict, str),
            ContainerTypeInfo(list, (dict, str))
        )
        super().__init__(
            opt_name, types,
            convertor = lambda values: self.convert(
                func_name, opt_name, values
            ),
            default = default
        )

    @staticmethod
    def convert(func_name:str, opt_name: str, values: 'IMPLIED_ATTR',
                ) -> Union[None, List[ConflictAttr]]:
        if values is None:
            return None
        ret: List[ConflictAttr] = []
        values = [values] if isinstance(values, (str, dict)) else values
        accepted_keys = ('val', 'match', 'mfilter', 'mjoin')
        for edict in values:
            if isinstance(edict, str):
                if edict:
                    ret.append(ConflictAttr(val=edict))
                continue
            if not isinstance(edict, dict):
                # It shouldn't happen
                # TODO: need exception here
                continue
            unknown_keys = [k for k in edict.keys() if k not in accepted_keys]
            if unknown_keys:
                raise MesonException(
                    f'{func_name}: unknown keys {unknown_keys} in '
                    f'option {opt_name}'
                )
            val = edict.get('val')
            if val is None:
                raise MesonException(
                    f'{func_name}: option "{opt_name}" requires '
                    f'a dictionary with key "val" to be set'
                )
            implattr = ConflictAttr(val=val, mjoin=edict.get('mjoin', ''))
            for cattr in ('match', 'mfilter'):
                cval = edict.get(cattr)
                if not cval:
                    continue
                try:
                    ccval = re.compile(cval)
                except Exception as e:
                    raise MesonException(
                        '{func_name}: unable to '
                        f'compile the regex in option "{opt_name}"\n'
                        f'"{cattr}:{cval}" -> {str(e)}'
                    )
                setattr(implattr, cattr, ccval)
            ret.append(implattr)
        return ret

if TYPE_CHECKING:
    IMPLIED_ATTR = Union[
        None, str, Dict[str, str], List[
            Union[str, Dict[str, str]]
        ]
    ]
    class FeatureKwArgs(TypedDict):
        #implies: Optional[List['FeatureObject']]
        implies: NotRequired[List[Any]]
        group: NotRequired[List[str]]
        detect: NotRequired[List[ConflictAttr]]
        args: NotRequired[List[ConflictAttr]]
        test_code: NotRequired[Union[str, File]]
        extra_tests: NotRequired[Dict[str, Union[str, File]]]
        disable: NotRequired[str]

    class FeatureUpdateKwArgs(FeatureKwArgs):
        name: NotRequired[str]
        interest: NotRequired[int]

class FeatureObject(ModuleObject):
    name: str
    interest: int
    implies: Set['FeatureObject']
    group: List[str]
    detect: List[ConflictAttr]
    args: List[ConflictAttr]
    test_code: Union[str, File]
    extra_tests: Dict[str, Union[str, File]]
    disable: str

    def __init__(self, state: 'ModuleState',
                 args: List['TYPE_var'],
                 kwargs: 'TYPE_kwargs') -> None:

        super().__init__()

        @typed_pos_args('features.new', str, int)
        @typed_kwargs('features.new',
            KwargInfo(
                'implies',
                (FeatureObject, ContainerTypeInfo(list, FeatureObject)),
                default=[], listify=True
            ),
            KwargInfo(
                'group', (str, ContainerTypeInfo(list, str)),
                default=[], listify=True
            ),
            KwargConfilctAttr('features.new', 'detect', default=[]),
            KwargConfilctAttr('features.new', 'args', default=[]),
            KwargInfo('test_code', (str, File), default=''),
            KwargInfo(
                'extra_tests', (ContainerTypeInfo(dict, (str, File))),
                default={}
            ),
            KwargInfo('disable', (str), default=''),
        )
        def init_attrs(state: 'ModuleState',
                       args: Tuple[str, int],
                       kwargs: 'FeatureKwArgs') -> None:
            self.name = args[0]
            self.interest = args[1]
            self.implies = set(kwargs['implies'])
            self.group = kwargs['group']
            self.detect = kwargs['detect']
            self.args = kwargs['args']
            self.test_code = kwargs['test_code']
            self.extra_tests = kwargs['extra_tests']
            self.disable: str = kwargs['disable']
            if not self.detect:
                if self.group:
                    self.detect = [ConflictAttr(val=f) for f in self.group]
                else:
                    self.detect = [ConflictAttr(val=self.name)]

        init_attrs(state, args, kwargs)
        self.methods.update({
            'update': self.update_method,
            'get': self.get_method,
        })

    def update_method(self, state: 'ModuleState', args: List['TYPE_var'],
                      kwargs: 'TYPE_kwargs') -> 'FeatureObject':
        @noPosargs
        @typed_kwargs('features.FeatureObject.update',
            KwargInfo('name', (NoneType, str)),
            KwargInfo('interest', (NoneType, int)),
            KwargInfo(
                'implies', (
                    NoneType, FeatureObject,
                    ContainerTypeInfo(list, FeatureObject)
                ),
                listify=True
            ),
            KwargInfo(
                'group', (NoneType, str, ContainerTypeInfo(list, str)),
                listify=True
            ),
            KwargConfilctAttr('features.FeatureObject.update', 'detect'),
            KwargConfilctAttr('features.FeatureObject.update', 'args'),
            KwargInfo('test_code', (NoneType, str, File)),
            KwargInfo(
                'extra_tests', (
                    NoneType, ContainerTypeInfo(dict, (str, File)))
            ),
            KwargInfo('disable', (NoneType, str)),
        )
        def update(state: 'ModuleState', args: List['TYPE_var'],
                   kwargs: 'FeatureUpdateKwArgs') -> None:
            for k, v in kwargs.items():
                if v is not None and k != 'implies':
                    setattr(self, k, v)
            implies = kwargs.get('implies')
            if implies is not None:
                self.implies = set(implies)
        update(state, args, kwargs)
        return self

    @noKwargs
    @typed_pos_args('features.FeatureObject.get', str)
    def get_method(self, state: 'ModuleState', args: Tuple[str],
                   kwargs: 'TYPE_kwargs') -> 'TYPE_var':

        impl_lst = lambda lst: [v.to_dict() for v in lst]
        noconv = lambda v: v
        dfunc = {
            'name': noconv,
            'interest': noconv,
            'group': noconv,
            'implies': lambda v: [fet.name for fet in sorted(v)],
            'detect': impl_lst,
            'args': impl_lst,
            'test_code': noconv,
            'extra_tests': noconv,
            'disable': noconv
        }
        cfunc: Optional[Callable[[str], 'TYPE_var']] = dfunc.get(args[0])
        if cfunc is None:
            raise MesonException(f'Key {args[0]!r} is not in the feature.')
        val = getattr(self, args[0])
        return cfunc(val)

    def get_implicit(self, _caller: Set['FeatureObject'] = None
                     ) -> Set['FeatureObject']:
        # infinity recursive guard since
        # features can imply each other
        _caller = {self, } if not _caller else _caller.union({self, })
        implies = self.implies.difference(_caller)
        ret = self.implies
        for sub_fet in implies:
            ret = ret.union(sub_fet.get_implicit(_caller))
        return ret

    @staticmethod
    def get_implicit_multi(features: Iterable['FeatureObject']) -> Set['FeatureObject']:
        implies = set().union(*[f.get_implicit() for f in features])
        return implies

    @staticmethod
    def get_implicit_combine_multi(features: Iterable['FeatureObject']) -> Set['FeatureObject']:
        return FeatureObject.get_implicit_multi(features).union(features)

    @staticmethod
    def sorted_multi(features: Iterable[Union['FeatureObject', Iterable['FeatureObject']]],
                     reverse: bool = False
                     ) -> List[Union['FeatureObject', Iterable['FeatureObject']]]:
        def sort_cb(k: Union[FeatureObject, Iterable[FeatureObject]]) -> int:
            if isinstance(k, FeatureObject):
                return k.interest
            # keep prevalent features and erase any implied features
            implied_features = FeatureObject.get_implicit_multi(k)
            prevalent_features = set(k).difference(implied_features)
            if len(prevalent_features) == 0:
                # It happens when all features imply each other.
                # Set the highest interested feature
                return sorted(k)[-1].interest
            # multiple features
            rank = max(f.interest for f in prevalent_features)
            # FIXME: that's not a safe way to increase the rank for
            # multi features this why this function isn't considerd
            # accurate.
            rank += len(prevalent_features) -1
            return rank
        return sorted(features, reverse=reverse, key=sort_cb)

    @staticmethod
    def features_names(features: Iterable[Union['FeatureObject', Iterable['FeatureObject']]]
                       ) -> List[Union[str, List[str]]]:
        return [
            fet.name if isinstance(fet, FeatureObject)
            else [f.name for f in fet]
            for fet in features
        ]

    def __repr__(self) -> str:
        args = ', '.join([
            f'{attr} = {str(getattr(self, attr))}'
            for attr in [
                'group', 'implies',
                'detect', 'args',
                'test_code', 'extra_tests',
                'disable'
            ]
        ])
        return f'FeatureObject({self.name}, {self.interest}, {args})'

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, robj: object) -> bool:
        if not isinstance(robj, FeatureObject):
            return False
        return self is robj and self.name == robj.name

    def __lt__(self, robj: object) -> Any:
        if not isinstance(robj, FeatureObject):
            return NotImplemented
        return self.interest < robj.interest

    def __le__(self, robj: object) -> Any:
        if not isinstance(robj, FeatureObject):
            return NotImplemented
        return self.interest <= robj.interest

    def __gt__(self, robj: object) -> Any:
        return robj < self

    def __ge__(self, robj: object) -> Any:
        return robj <= self
