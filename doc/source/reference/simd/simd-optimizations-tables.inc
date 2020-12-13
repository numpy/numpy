.. generated via source/reference/simd/simd-optimizations.py

x86 - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~
.. table::
    :align: left

    ============ =================================================================================================================
    Name         Implies                                                                                                          
    ============ =================================================================================================================
    ``SSE``      ``SSE2``                                                                                                         
    ``SSE2``     ``SSE``                                                                                                          
    ``SSE3``     ``SSE`` ``SSE2``                                                                                                 
    ``SSSE3``    ``SSE`` ``SSE2`` ``SSE3``                                                                                        
    ``SSE41``    ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3``                                                                              
    ``POPCNT``   ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41``                                                                    
    ``SSE42``    ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT``                                                         
    ``AVX``      ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42``                                               
    ``XOP``      ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX``                                       
    ``FMA4``     ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX``                                       
    ``F16C``     ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX``                                       
    ``FMA3``     ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C``                              
    ``AVX2``     ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C``                              
    ``AVX512F``  ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2``            
    ``AVX512CD`` ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F``
    ============ =================================================================================================================

x86 - Group names
~~~~~~~~~~~~~~~~~
.. table::
    :align: left

    ============== ===================================================== ===========================================================================================================================================================================
    Name           Gather                                                Implies                                                                                                                                                                    
    ============== ===================================================== ===========================================================================================================================================================================
    ``AVX512_KNL`` ``AVX512ER`` ``AVX512PF``                             ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD``                                             
    ``AVX512_KNM`` ``AVX5124FMAPS`` ``AVX5124VNNIW`` ``AVX512VPOPCNTDQ`` ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD`` ``AVX512_KNL``                              
    ``AVX512_SKX`` ``AVX512VL`` ``AVX512BW`` ``AVX512DQ``                ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD``                                             
    ``AVX512_CLX`` ``AVX512VNNI``                                        ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD`` ``AVX512_SKX``                              
    ``AVX512_CNL`` ``AVX512IFMA`` ``AVX512VBMI``                         ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD`` ``AVX512_SKX``                              
    ``AVX512_ICL`` ``AVX512VBMI2`` ``AVX512BITALG`` ``AVX512VPOPCNTDQ``  ``SSE`` ``SSE2`` ``SSE3`` ``SSSE3`` ``SSE41`` ``POPCNT`` ``SSE42`` ``AVX`` ``F16C`` ``FMA3`` ``AVX2`` ``AVX512F`` ``AVX512CD`` ``AVX512_SKX`` ``AVX512_CLX`` ``AVX512_CNL``
    ============== ===================================================== ===========================================================================================================================================================================

IBM/POWER big-endian - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. table::
    :align: left

    ======== ================
    Name     Implies         
    ======== ================
    ``VSX``                  
    ``VSX2`` ``VSX``         
    ``VSX3`` ``VSX`` ``VSX2``
    ======== ================

IBM/POWER little-endian - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. table::
    :align: left

    ======== ================
    Name     Implies         
    ======== ================
    ``VSX``  ``VSX2``        
    ``VSX2`` ``VSX``         
    ``VSX3`` ``VSX`` ``VSX2``
    ======== ================

ARMv7/A32 - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

ARMv8/A64 - CPU feature names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

