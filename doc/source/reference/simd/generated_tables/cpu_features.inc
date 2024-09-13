.. generated via /numpy/numpy/./doc/source/reference/simd/gen_features.py

On x86
~~~~~~
.. table::
    :align: left

    ============== ========================================================================================================================================================================================== =====================================================
    Name           Implies                                                                                                                                                                                    Gathers                                              
    ============== ========================================================================================================================================================================================== =====================================================
    ``SSE``        ``SSE2``                                                                                                                                                                                                                                        
    ``SSE2``       ``SSE``                                                                                                                                                                                                                                         
    ``SSE3``       ``SSE`` ``SSE2``                                                                                                                                                                                                                                
    ``SSSE3``      ``SSE`` ``SSE2`` ``SSE3``                                                                                                                                                                                                                       
    ``SSE41``      ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3``                                                                                                                                                                                                             
    ``POPCNT``     ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41``                                                                                                                                                                                                   
    ``SSE42``      ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT``                                                                                                                                                                                        
    ``AVX``        ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42``                                                                                                                                                                              
    ``XOP``        ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX``                                                                                                                                                                      
    ``FMA4``       ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX``                                                                                                                                                                      
    ``F16C``       ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX``                                                                                                                                                                      
    ``FMA3``       ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C``                                                                                                                                                             
    ``AVX2``       ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C``                                                                                                                                                             
    ``AVX512F``    ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2``                                                                                                                                           
    ``AVX512CD``   ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F``                                                                                                                               
    ``AVX512_KNL`` ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD``                                                             ``AVX512ER`` ``AVX512PF``                            
    ``AVX512_KNM`` ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD`` ``AVX512_KNL``                                              ``AVX5124FMAPS`` ``AVX5124VNNIW`` ``AVX512VPOPCNTDQ``
    ``AVX512_SKX`` ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD``                                                             ``AVX512VL`` ``AVX512BW`` ``AVX512DQ``               
    ``AVX512_CLX`` ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD`` ``AVX512_SKX``                                              ``AVX512VNNI``                                       
    ``AVX512_CNL`` ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD`` ``AVX512_SKX``                                              ``AVX512IFMA`` ``AVX512VBMI``                        
    ``AVX512_ICL`` ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD`` ``AVX512_SKX`` ``AVX512_CLX`` ``AVX512_CNL``                ``AVX512VBMI2`` ``AVX512BITALG`` ``AVX512VPOPCNTDQ`` 
    ``AVX512_SPR`` ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD`` ``AVX512_SKX`` ``AVX512_CLX`` ``AVX512_CNL`` ``AVX512_ICL`` ``AVX512FP16``                                       
    ============== ========================================================================================================================================================================================== =====================================================

On IBM/POWER big-endian
~~~~~~~~~~~~~~~~~~~~~~~
.. table::
    :align: left

    ======== =========================
    Name     Implies                  
    ======== =========================
    ``VSX``                           
    ``VSX2`` ``VSX``                  
    ``VSX3`` ``VSX`` ``VSX2``         
    ``VSX4`` ``VSX`` ``VSX2`` ``VSX3``
    ======== =========================

On IBM/POWER little-endian
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. table::
    :align: left

    ======== =========================
    Name     Implies                  
    ======== =========================
    ``VSX``  ``VSX2``                 
    ``VSX2`` ``VSX``                  
    ``VSX3`` ``VSX`` ``VSX2``         
    ``VSX4`` ``VSX`` ``VSX2`` ``VSX3``
    ======== =========================

On ARMv7/A32
~~~~~~~~~~~~
.. table::
    :align: left

    ============== ===========================================================
    Name           Implies                                                    
    ============== ===========================================================
    ``NEON``                                                                  
    ``NEON_FP16``  ``NEON``                                                   
    ``NEON_VFPV4`` ``NEON`` ``NEON_FP16``                                     
    ``ASIMD``      ``NEON`` ``NEON_FP16`` ``NEON_VFPV4``                      
    ``ASIMDHP``    ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD``            
    ``ASIMDDP``    ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD``            
    ``ASIMDFHM``   ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD`` ``ASIMDHP``
    ============== ===========================================================

On ARMv8/A64
~~~~~~~~~~~~
.. table::
    :align: left

    ============== ===========================================================
    Name           Implies                                                    
    ============== ===========================================================
    ``NEON``       ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD``                     
    ``NEON_FP16``  ``NEON`` ``NEON_VFPV4`` ``ASIMD``                          
    ``NEON_VFPV4`` ``NEON`` ``NEON_FP16`` ``ASIMD``                           
    ``ASIMD``      ``NEON`` ``NEON_FP16`` ``NEON_VFPV4``                      
    ``ASIMDHP``    ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD``            
    ``ASIMDDP``    ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD``            
    ``ASIMDFHM``   ``NEON`` ``NEON_FP16`` ``NEON_VFPV4`` ``ASIMD`` ``ASIMDHP``
    ============== ===========================================================

On IBM/ZSYSTEM(S390X)
~~~~~~~~~~~~~~~~~~~~~
.. table::
    :align: left

    ======== ==============
    Name     Implies       
    ======== ==============
    ``VX``                 
    ``VXE``  ``VX``        
    ``VXE2`` ``VX`` ``VXE``
    ======== ==============

