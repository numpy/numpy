"""Here we define the exported functions, types, etc... which need to be
exported through a global C pointer.

Each dictionary contains name -> index pair.

Whenever you change one index, you break the ABI (and the ABI version number
should be incremented). Whenever you add an item to one of the dict, the API
needs to be updated in both numpy/core/meson.build and by adding an appropriate
entry to cversion.txt (generate the hash via "python cversions.py").

When adding a function, make sure to use the next integer not used as an index
(in case you use an existing index or jump, the build will stop and raise an
exception, so it should hopefully not get unnoticed).

"""

import importlib.util
import os


def get_annotations():
    # Convoluted because we can't import from numpy.distutils
    # (numpy is not yet built)
    genapi_py = os.path.join(os.path.dirname(__file__), 'genapi.py')
    spec = importlib.util.spec_from_file_location('conv_template', genapi_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.StealRef, mod.MinVersion


StealRef, MinVersion = get_annotations()
#from code_generators.genapi import StealRef

# index, type
multiarray_global_vars = {
    'NPY_NUMUSERTYPES':             (7, 'int'),
    'NPY_DEFAULT_ASSIGN_CASTING':   (292, 'NPY_CASTING'),
    'PyDataMem_DefaultHandler':     (306, 'PyObject*'),
}

multiarray_scalar_bool_values = {
    '_PyArrayScalar_BoolValues':    (9,)
}

# index, annotations
# please mark functions that have been checked to not need any annotations
multiarray_types_api = {
    # Slot 1 was never meaningfully used by NumPy
    'PyArray_Type':                     (2,),
    # Internally, PyArrayDescr_Type is a PyArray_DTypeMeta,
    # the following also defines PyArrayDescr_TypeFull (Full appended)
    'PyArrayDescr_Type':                (3, "PyArray_DTypeMeta"),
    # Unused slot 4, was `PyArrayFlags_Type`
    'PyArrayIter_Type':                 (5,),
    'PyArrayMultiIter_Type':            (6,),
    'PyBoolArrType_Type':               (8,),
    'PyGenericArrType_Type':            (10,),
    'PyNumberArrType_Type':             (11,),
    'PyIntegerArrType_Type':            (12,),
    'PySignedIntegerArrType_Type':      (13,),
    'PyUnsignedIntegerArrType_Type':    (14,),
    'PyInexactArrType_Type':            (15,),
    'PyFloatingArrType_Type':           (16,),
    'PyComplexFloatingArrType_Type':    (17,),
    'PyFlexibleArrType_Type':           (18,),
    'PyCharacterArrType_Type':          (19,),
    'PyByteArrType_Type':               (20,),
    'PyShortArrType_Type':              (21,),
    'PyIntArrType_Type':                (22,),
    'PyLongArrType_Type':               (23,),
    'PyLongLongArrType_Type':           (24,),
    'PyUByteArrType_Type':              (25,),
    'PyUShortArrType_Type':             (26,),
    'PyUIntArrType_Type':               (27,),
    'PyULongArrType_Type':              (28,),
    'PyULongLongArrType_Type':          (29,),
    'PyFloatArrType_Type':              (30,),
    'PyDoubleArrType_Type':             (31,),
    'PyLongDoubleArrType_Type':         (32,),
    'PyCFloatArrType_Type':             (33,),
    'PyCDoubleArrType_Type':            (34,),
    'PyCLongDoubleArrType_Type':        (35,),
    'PyObjectArrType_Type':             (36,),
    'PyStringArrType_Type':             (37,),
    'PyUnicodeArrType_Type':            (38,),
    'PyVoidArrType_Type':               (39,),
    # End 1.5 API
    'PyTimeIntegerArrType_Type':        (214,),
    'PyDatetimeArrType_Type':           (215,),
    'PyTimedeltaArrType_Type':          (216,),
    'PyHalfArrType_Type':               (217,),
    'NpyIter_Type':                     (218,),
    # End 1.6 API
    # NOTE: The Slots 320-360 are defined in `_experimental_dtype_api.h`
    #       and filled explicitly outside the code generator as the metaclass
    #       makes them tricky to expose.  (This may be refactored.)
    # Slot 366, 367, 368 are the abstract DTypes
    # End 2.0 API
}

# define NPY_NUMUSERTYPES (*(int *)PyArray_API[6])
# define PyBoolArrType_Type (*(PyTypeObject *)PyArray_API[7])
# define _PyArrayScalar_BoolValues ((PyBoolScalarObject *)PyArray_API[8])

multiarray_funcs_api = {
    '__unused_indices__': (
        [1, 4, 40, 41, 66, 67, 68, 81, 82, 83,
         103, 115, 117, 122, 163, 164, 171, 173, 197,
         201, 202, 208, 219, 220, 221, 222, 278,
         291, 293, 294, 295, 301]
        # range/slots reserved DType classes (see _public_dtype_api_table.h):
        + list(range(320, 361)) + [366, 367, 368]
        ),
    'PyArray_GetNDArrayCVersion':           (0,),
    # Unused slot 40, was `PyArray_SetNumericOps`
    # Unused slot 41, was `PyArray_GetNumericOps`,
    'PyArray_INCREF':                       (42,),
    'PyArray_XDECREF':                      (43,),
    # `PyArray_SetStringFunction` was stubbed out
    # and should be removed in the future.
    'PyArray_SetStringFunction':            (44,),
    'PyArray_DescrFromType':                (45,),
    'PyArray_TypeObjectFromType':           (46,),
    'PyArray_Zero':                         (47,),
    'PyArray_One':                          (48,),
    'PyArray_CastToType':                   (49, StealRef(2)),
    'PyArray_CopyInto':                     (50,),
    'PyArray_CopyAnyInto':                  (51,),
    'PyArray_CanCastSafely':                (52,),
    'PyArray_CanCastTo':                    (53,),
    'PyArray_ObjectType':                   (54,),
    'PyArray_DescrFromObject':              (55,),
    'PyArray_ConvertToCommonType':          (56,),
    'PyArray_DescrFromScalar':              (57,),
    'PyArray_DescrFromTypeObject':          (58,),
    'PyArray_Size':                         (59,),
    'PyArray_Scalar':                       (60,),
    'PyArray_FromScalar':                   (61, StealRef(2)),
    'PyArray_ScalarAsCtype':                (62,),
    'PyArray_CastScalarToCtype':            (63,),
    'PyArray_CastScalarDirect':             (64,),
    'PyArray_Pack':                         (65, MinVersion("2.0")),
    # Unused slot 66, was `PyArray_GetCastFunc`
    # Unused slot 67, was `PyArray_FromDims`
    # Unused slot 68, was `PyArray_FromDimsAndDataAndDescr`
    'PyArray_FromAny':                      (69, StealRef(2)),
    'PyArray_EnsureArray':                  (70, StealRef(1)),
    'PyArray_EnsureAnyArray':               (71, StealRef(1)),
    'PyArray_FromFile':                     (72,),
    'PyArray_FromString':                   (73,),
    'PyArray_FromBuffer':                   (74,),
    'PyArray_FromIter':                     (75, StealRef(2)),
    'PyArray_Return':                       (76, StealRef(1)),
    'PyArray_GetField':                     (77, StealRef(2)),
    'PyArray_SetField':                     (78, StealRef(2)),
    'PyArray_Byteswap':                     (79,),
    'PyArray_Resize':                       (80,),
    # Unused slot 81, was `PyArray_MoveInto``
    # Unused slot 82 was `PyArray_CopyInto` (which replaces `..._CastTo`)
    # Unused slot 82 was `PyArray_CopyAnyInto` (which replaces `..._CastAnyTo`)
    'PyArray_CopyObject':                   (84,),
    'PyArray_NewCopy':                      (85,),
    'PyArray_ToList':                       (86,),
    'PyArray_ToString':                     (87,),
    'PyArray_ToFile':                       (88,),
    'PyArray_Dump':                         (89,),
    'PyArray_Dumps':                        (90,),
    'PyArray_ValidType':                    (91,),
    'PyArray_UpdateFlags':                  (92,),
    'PyArray_New':                          (93,),
    'PyArray_NewFromDescr':                 (94, StealRef(2)),
    'PyArray_DescrNew':                     (95,),
    'PyArray_DescrNewFromType':             (96,),
    'PyArray_GetPriority':                  (97,),
    'PyArray_IterNew':                      (98,),
    'PyArray_MultiIterNew':                 (99,),
    'PyArray_PyIntAsInt':                   (100,),
    'PyArray_PyIntAsIntp':                  (101,),
    'PyArray_Broadcast':                    (102,),
    # Unused slot 103, was `PyArray_FillObjectArray`
    'PyArray_FillWithScalar':               (104,),
    'PyArray_CheckStrides':                 (105,),
    'PyArray_DescrNewByteorder':            (106,),
    'PyArray_IterAllButAxis':               (107,),
    'PyArray_CheckFromAny':                 (108, StealRef(2)),
    'PyArray_FromArray':                    (109, StealRef(2)),
    'PyArray_FromInterface':                (110,),
    'PyArray_FromStructInterface':          (111,),
    'PyArray_FromArrayAttr':                (112,),
    'PyArray_ScalarKind':                   (113,),
    'PyArray_CanCoerceScalar':              (114,),
    # Unused slot 115, was `PyArray_NewFlagsObject`
    'PyArray_CanCastScalar':                (116,),
    # Unused slot 117, was `PyArray_CompareUCS4`
    'PyArray_RemoveSmallest':               (118,),
    'PyArray_ElementStrides':               (119,),
    'PyArray_Item_INCREF':                  (120,),
    'PyArray_Item_XDECREF':                 (121,),
    # Unused slot 122, was `PyArray_FieldNames`
    'PyArray_Transpose':                    (123,),
    'PyArray_TakeFrom':                     (124,),
    'PyArray_PutTo':                        (125,),
    'PyArray_PutMask':                      (126,),
    'PyArray_Repeat':                       (127,),
    'PyArray_Choose':                       (128,),
    'PyArray_Sort':                         (129,),
    'PyArray_ArgSort':                      (130,),
    'PyArray_SearchSorted':                 (131,),
    'PyArray_ArgMax':                       (132,),
    'PyArray_ArgMin':                       (133,),
    'PyArray_Reshape':                      (134,),
    'PyArray_Newshape':                     (135,),
    'PyArray_Squeeze':                      (136,),
    'PyArray_View':                         (137, StealRef(2)),
    'PyArray_SwapAxes':                     (138,),
    'PyArray_Max':                          (139,),
    'PyArray_Min':                          (140,),
    'PyArray_Ptp':                          (141,),
    'PyArray_Mean':                         (142,),
    'PyArray_Trace':                        (143,),
    'PyArray_Diagonal':                     (144,),
    'PyArray_Clip':                         (145,),
    'PyArray_Conjugate':                    (146,),
    'PyArray_Nonzero':                      (147,),
    'PyArray_Std':                          (148,),
    'PyArray_Sum':                          (149,),
    'PyArray_CumSum':                       (150,),
    'PyArray_Prod':                         (151,),
    'PyArray_CumProd':                      (152,),
    'PyArray_All':                          (153,),
    'PyArray_Any':                          (154,),
    'PyArray_Compress':                     (155,),
    'PyArray_Flatten':                      (156,),
    'PyArray_Ravel':                        (157,),
    'PyArray_MultiplyList':                 (158,),
    'PyArray_MultiplyIntList':              (159,),
    'PyArray_GetPtr':                       (160,),
    'PyArray_CompareLists':                 (161,),
    'PyArray_AsCArray':                     (162, StealRef(5)),
    # Unused slot 163, was `PyArray_As1D`
    # Unused slot 164, was `PyArray_As2D`
    'PyArray_Free':                         (165,),
    'PyArray_Converter':                    (166,),
    'PyArray_IntpFromSequence':             (167,),
    'PyArray_Concatenate':                  (168,),
    'PyArray_InnerProduct':                 (169,),
    'PyArray_MatrixProduct':                (170,),
    # Unused slot 171, was `PyArray_CopyAndTranspose`
    'PyArray_Correlate':                    (172,),
    # Unused slot 173, was `PyArray_TypestrConvert`
    'PyArray_DescrConverter':               (174,),
    'PyArray_DescrConverter2':              (175,),
    'PyArray_IntpConverter':                (176,),
    'PyArray_BufferConverter':              (177,),
    'PyArray_AxisConverter':                (178,),
    'PyArray_BoolConverter':                (179,),
    'PyArray_ByteorderConverter':           (180,),
    'PyArray_OrderConverter':               (181,),
    'PyArray_EquivTypes':                   (182,),
    'PyArray_Zeros':                        (183, StealRef(3)),
    'PyArray_Empty':                        (184, StealRef(3)),
    'PyArray_Where':                        (185,),
    'PyArray_Arange':                       (186,),
    'PyArray_ArangeObj':                    (187,),
    'PyArray_SortkindConverter':            (188,),
    'PyArray_LexSort':                      (189,),
    'PyArray_Round':                        (190,),
    'PyArray_EquivTypenums':                (191,),
    'PyArray_RegisterDataType':             (192,),
    'PyArray_RegisterCastFunc':             (193,),
    'PyArray_RegisterCanCast':              (194,),
    'PyArray_InitArrFuncs':                 (195,),
    'PyArray_IntTupleFromIntp':             (196,),
    # Unused slot 197, was `PyArray_TypeNumFromName`
    'PyArray_ClipmodeConverter':            (198,),
    'PyArray_OutputConverter':              (199,),
    'PyArray_BroadcastToShape':             (200,),
    # Unused slot 201, was `_PyArray_SigintHandler`
    # Unused slot 202, was `_PyArray_GetSigintBuf`
    'PyArray_DescrAlignConverter':          (203,),
    'PyArray_DescrAlignConverter2':         (204,),
    'PyArray_SearchsideConverter':          (205,),
    'PyArray_CheckAxis':                    (206,),
    'PyArray_OverflowMultiplyList':         (207,),
    # Unused slot 208, was `PyArray_CompareString`
    'PyArray_MultiIterFromObjects':         (209,),
    'PyArray_GetEndianness':                (210,),
    'PyArray_GetNDArrayCFeatureVersion':    (211,),
    'PyArray_Correlate2':                   (212,),
    'PyArray_NeighborhoodIterNew':          (213,),
    # End 1.5 API
    # Unused slot 219, was `PyArray_SetDatetimeParseFunction`
    # Unused slot 220, was `PyArray_DatetimeToDatetimeStruct`
    # Unused slot 221, was `PyArray_TimedeltaToTimedeltaStruct`
    # Unused slot 222, was `PyArray_DatetimeStructToDatetime`
    # NDIter API
    'NpyIter_GetTransferFlags':             (223, MinVersion("2.3")),
    'NpyIter_New':                          (224,),
    'NpyIter_MultiNew':                     (225,),
    'NpyIter_AdvancedNew':                  (226,),
    'NpyIter_Copy':                         (227,),
    'NpyIter_Deallocate':                   (228,),
    'NpyIter_HasDelayedBufAlloc':           (229,),
    'NpyIter_HasExternalLoop':              (230,),
    'NpyIter_EnableExternalLoop':           (231,),
    'NpyIter_GetInnerStrideArray':          (232,),
    'NpyIter_GetInnerLoopSizePtr':          (233,),
    'NpyIter_Reset':                        (234,),
    'NpyIter_ResetBasePointers':            (235,),
    'NpyIter_ResetToIterIndexRange':        (236,),
    'NpyIter_GetNDim':                      (237,),
    'NpyIter_GetNOp':                       (238,),
    'NpyIter_GetIterNext':                  (239,),
    'NpyIter_GetIterSize':                  (240,),
    'NpyIter_GetIterIndexRange':            (241,),
    'NpyIter_GetIterIndex':                 (242,),
    'NpyIter_GotoIterIndex':                (243,),
    'NpyIter_HasMultiIndex':                (244,),
    'NpyIter_GetShape':                     (245,),
    'NpyIter_GetGetMultiIndex':             (246,),
    'NpyIter_GotoMultiIndex':               (247,),
    'NpyIter_RemoveMultiIndex':             (248,),
    'NpyIter_HasIndex':                     (249,),
    'NpyIter_IsBuffered':                   (250,),
    'NpyIter_IsGrowInner':                  (251,),
    'NpyIter_GetBufferSize':                (252,),
    'NpyIter_GetIndexPtr':                  (253,),
    'NpyIter_GotoIndex':                    (254,),
    'NpyIter_GetDataPtrArray':              (255,),
    'NpyIter_GetDescrArray':                (256,),
    'NpyIter_GetOperandArray':              (257,),
    'NpyIter_GetIterView':                  (258,),
    'NpyIter_GetReadFlags':                 (259,),
    'NpyIter_GetWriteFlags':                (260,),
    'NpyIter_DebugPrint':                   (261,),
    'NpyIter_IterationNeedsAPI':            (262,),
    'NpyIter_GetInnerFixedStrideArray':     (263,),
    'NpyIter_RemoveAxis':                   (264,),
    'NpyIter_GetAxisStrideArray':           (265,),
    'NpyIter_RequiresBuffering':            (266,),
    'NpyIter_GetInitialDataPtrArray':       (267,),
    'NpyIter_CreateCompatibleStrides':      (268,),
    #
    'PyArray_CastingConverter':             (269,),
    'PyArray_CountNonzero':                 (270,),
    'PyArray_PromoteTypes':                 (271,),
    'PyArray_MinScalarType':                (272,),
    'PyArray_ResultType':                   (273,),
    'PyArray_CanCastArrayTo':               (274,),
    'PyArray_CanCastTypeTo':                (275,),
    'PyArray_EinsteinSum':                  (276,),
    'PyArray_NewLikeArray':                 (277, StealRef(3)),
    # Unused slot 278, was `PyArray_GetArrayParamsFromObject`
    'PyArray_ConvertClipmodeSequence':      (279,),
    'PyArray_MatrixProduct2':               (280,),
    # End 1.6 API
    'NpyIter_IsFirstVisit':                 (281,),
    'PyArray_SetBaseObject':                (282, StealRef(2)),
    'PyArray_CreateSortedStridePerm':       (283,),
    'PyArray_RemoveAxesInPlace':            (284,),
    'PyArray_DebugPrint':                   (285,),
    'PyArray_FailUnlessWriteable':          (286,),
    'PyArray_SetUpdateIfCopyBase':          (287, StealRef(2)),
    'PyDataMem_NEW':                        (288,),
    'PyDataMem_FREE':                       (289,),
    'PyDataMem_RENEW':                      (290,),
    # Unused slot 291, was `PyDataMem_SetEventHook`
    # Unused slot 293, was `PyArray_MapIterSwapAxes`
    # Unused slot 294, was `PyArray_MapIterArray`
    # Unused slot 295, was `PyArray_MapIterNext`
    # End 1.7 API
    'PyArray_Partition':                    (296,),
    'PyArray_ArgPartition':                 (297,),
    'PyArray_SelectkindConverter':          (298,),
    'PyDataMem_NEW_ZEROED':                 (299,),
    # End 1.8 API
    # End 1.9 API
    'PyArray_CheckAnyScalarExact':          (300,),
    # End 1.10 API
    # Unused slot 301, was `PyArray_MapIterArrayCopyIfOverlap`
    # End 1.13 API
    'PyArray_ResolveWritebackIfCopy':       (302,),
    'PyArray_SetWritebackIfCopyBase':       (303,),
    # End 1.14 API
    'PyDataMem_SetHandler':                 (304, MinVersion("1.22")),
    'PyDataMem_GetHandler':                 (305, MinVersion("1.22")),
    # End 1.22 API
    'NpyDatetime_ConvertDatetime64ToDatetimeStruct': (307, MinVersion("2.0")),
    'NpyDatetime_ConvertDatetimeStructToDatetime64': (308, MinVersion("2.0")),
    'NpyDatetime_ConvertPyDateTimeToDatetimeStruct': (309, MinVersion("2.0")),
    'NpyDatetime_GetDatetimeISO8601StrLen':          (310, MinVersion("2.0")),
    'NpyDatetime_MakeISO8601Datetime':               (311, MinVersion("2.0")),
    'NpyDatetime_ParseISO8601Datetime':              (312, MinVersion("2.0")),
    'NpyString_load':                                (313, MinVersion("2.0")),
    'NpyString_pack':                                (314, MinVersion("2.0")),
    'NpyString_pack_null':                           (315, MinVersion("2.0")),
    'NpyString_acquire_allocator':                   (316, MinVersion("2.0")),
    'NpyString_acquire_allocators':                  (317, MinVersion("2.0")),
    'NpyString_release_allocator':                   (318, MinVersion("2.0")),
    'NpyString_release_allocators':                  (319, MinVersion("2.0")),
    # Slots 320-360 reserved for DType classes (see comment in types)
    'PyArray_GetDefaultDescr':                       (361, MinVersion("2.0")),
    'PyArrayInitDTypeMeta_FromSpec':                 (362, MinVersion("2.0")),
    'PyArray_CommonDType':                           (363, MinVersion("2.0")),
    'PyArray_PromoteDTypeSequence':                  (364, MinVersion("2.0")),
    # The actual public API for this is the inline function
    # `PyDataType_GetArrFuncs` checks for the NumPy runtime version.
    '_PyDataType_GetArrFuncs':                       (365,),
    # End 2.0 API
    # NpyIterGetTransferFlags (slot 223) added.
    # End 2.3 API
}

ufunc_types_api = {
    'PyUFunc_Type':                             (0,),
}

ufunc_funcs_api = {
    '__unused_indices__': [3, 25, 26, 29, 32],
    'PyUFunc_FromFuncAndData':                  (1,),
    'PyUFunc_RegisterLoopForType':              (2,),
    # Unused slot 3, was `PyUFunc_GenericFunction`
    'PyUFunc_f_f_As_d_d':                       (4,),
    'PyUFunc_d_d':                              (5,),
    'PyUFunc_f_f':                              (6,),
    'PyUFunc_g_g':                              (7,),
    'PyUFunc_F_F_As_D_D':                       (8,),
    'PyUFunc_F_F':                              (9,),
    'PyUFunc_D_D':                              (10,),
    'PyUFunc_G_G':                              (11,),
    'PyUFunc_O_O':                              (12,),
    'PyUFunc_ff_f_As_dd_d':                     (13,),
    'PyUFunc_ff_f':                             (14,),
    'PyUFunc_dd_d':                             (15,),
    'PyUFunc_gg_g':                             (16,),
    'PyUFunc_FF_F_As_DD_D':                     (17,),
    'PyUFunc_DD_D':                             (18,),
    'PyUFunc_FF_F':                             (19,),
    'PyUFunc_GG_G':                             (20,),
    'PyUFunc_OO_O':                             (21,),
    'PyUFunc_O_O_method':                       (22,),
    'PyUFunc_OO_O_method':                      (23,),
    'PyUFunc_On_Om':                            (24,),
    # Unused slot 25, was `PyUFunc_GetPyValues`
    # Unused slot 26, was `PyUFunc_checkfperr`
    'PyUFunc_clearfperr':                       (27,),
    'PyUFunc_getfperr':                         (28,),
    # Unused slot 29, was `PyUFunc_handlefperr`
    'PyUFunc_ReplaceLoopBySignature':           (30,),
    'PyUFunc_FromFuncAndDataAndSignature':      (31,),
    # Unused slot 32, was `PyUFunc_SetUsesArraysAsData`
    # End 1.5 API
    'PyUFunc_e_e':                              (33,),
    'PyUFunc_e_e_As_f_f':                       (34,),
    'PyUFunc_e_e_As_d_d':                       (35,),
    'PyUFunc_ee_e':                             (36,),
    'PyUFunc_ee_e_As_ff_f':                     (37,),
    'PyUFunc_ee_e_As_dd_d':                     (38,),
    # End 1.6 API
    'PyUFunc_DefaultTypeResolver':              (39,),
    'PyUFunc_ValidateCasting':                  (40,),
    # End 1.7 API
    'PyUFunc_RegisterLoopForDescr':             (41,),
    # End 1.8 API
    'PyUFunc_FromFuncAndDataAndSignatureAndIdentity': (42, MinVersion("1.16")),
    # End 1.16 API
    'PyUFunc_AddLoopFromSpec':                       (43, MinVersion("2.0")),
    'PyUFunc_AddPromoter':                           (44, MinVersion("2.0")),
    'PyUFunc_AddWrappingLoop':                       (45, MinVersion("2.0")),
    'PyUFunc_GiveFloatingpointErrors':               (46, MinVersion("2.0")),
    # End 2.0 API
    'PyUFunc_AddLoopsFromSpecs':                     (47, MinVersion("2.4")),
}

# List of all the dicts which define the C API
# XXX: DO NOT CHANGE THE ORDER OF TUPLES BELOW !
multiarray_api = (
        multiarray_global_vars,
        multiarray_scalar_bool_values,
        multiarray_types_api,
        multiarray_funcs_api,
)

ufunc_api = (
        ufunc_funcs_api,
        ufunc_types_api
)

full_api = multiarray_api + ufunc_api
