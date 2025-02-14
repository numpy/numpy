#cython: binding=True

import numpy as np
cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t
from numpy.random cimport BitGenerator, SeedSequence
from numpy.random._common cimport uint64_to_double, wrap_int

np.import_array()


from libc.stdint cimport uint32_t as uint32_t
from libc.stdint cimport uint64_t as uint64_t

cdef extern from "src/RomuRandom/RomuRandom.h" nogil:      
    struct s_romuquad_state:
        uint64_t[4] state
        int has_uint32
        uint32_t uinteger
    
    ctypedef s_romuquad_state romuquad_state 
    
    uint64_t romuquad_next(uint64_t*)        
    uint64_t romuquad_next64(romuquad_state*)
    uint32_t romuquad_next32(romuquad_state*)
    
    struct s_romutrio_state:
        uint64_t[3] state
        int has_uint32
        uint32_t uinteger
    
    ctypedef s_romutrio_state romutrio_state 
    
    uint64_t romutrio_next(uint64_t*)        
    uint64_t romutrio_next64(romutrio_state*)
    uint32_t romutrio_next32(romutrio_state*)
    
    struct s_romuduo_state:
        uint64_t[2] state
        int has_uint32
        uint32_t uinteger
    
    ctypedef s_romuduo_state romuduo_state   
    
    uint64_t romuduo_next(uint64_t*)
    uint64_t romuduo_next64(romuduo_state*)
    uint32_t romuduo_next32(romuduo_state*)

    struct s_romuduojr_state:
        uint64_t[2] state
        int has_uint32
        uint32_t uinteger

    ctypedef s_romuduojr_state romuduojr_state
    
    uint64_t romuduojr_next(uint64_t*)
    uint64_t romuduojr_next64(romuduojr_state*)
    uint32_t romuduojr_next32(romuduojr_state*)

    struct s_romuquad32_state:
        uint32_t[4] state
    
    ctypedef s_romuquad32_state romuquad32_state
    
    uint32_t romuquad32_next(uint32_t*)
    uint64_t romuquad32_next64(romuquad32_state*)
    uint32_t romuquad32_next32(romuquad32_state*)

    struct s_romutrio32_state:
        uint32_t[3] state
  
    ctypedef s_romutrio32_state romutrio32_state
    
    uint32_t romutrio32_next(uint32_t*)
    uint64_t romutrio32_next64(romutrio32_state*)
    uint32_t romutrio32_next32(romutrio32_state*)


cdef uint64_t romuquad_uint64(void* st) noexcept nogil:
    return romuquad_next64(<romuquad_state *>st)

cdef uint32_t romuquad_uint32(void *st) noexcept nogil:
    return romuquad_next32(<romuquad_state *> st)

cdef double romuquad_double(void* st) noexcept nogil:
    return uint64_to_double(romuquad_next64(<romuquad_state *>st))

cdef uint64_t romuquad_raw(void *st) noexcept nogil:
    return <uint64_t>romuquad_next32(<romuquad_state *> st)

cdef uint64_t romutrio_uint64(void* st) noexcept nogil:
    return romutrio_next64(<romutrio_state *>st)

cdef uint32_t romutrio_uint32(void *st) noexcept nogil:
    return romutrio_next32(<romutrio_state *> st)

cdef double romutrio_double(void* st) noexcept nogil:
    return uint64_to_double(romutrio_next64(<romutrio_state *>st))

cdef uint64_t romutrio_raw(void *st) noexcept nogil:
    return <uint64_t>romutrio_next32(<romutrio_state *> st)

cdef uint64_t romuduo_uint64(void* st) noexcept nogil:
    return romuduo_next64(<romuduo_state *>st)

cdef uint32_t romuduo_uint32(void *st) noexcept nogil:
    return romuduo_next32(<romuduo_state *> st)

cdef double romuduo_double(void* st) noexcept nogil:
    return uint64_to_double(romuduo_next64(<romuduo_state *>st))

cdef uint64_t romuduo_raw(void *st) noexcept nogil:
    return <uint64_t>romuduo_next32(<romuduo_state *> st)

cdef uint64_t romuduojr_uint64(void* st) noexcept nogil:
    return romuduojr_next64(<romuduojr_state *>st)

cdef uint32_t romuduojr_uint32(void *st) noexcept nogil:
    return romuduojr_next32(<romuduojr_state *> st)

cdef double romuduojr_double(void* st) noexcept nogil:
    return uint64_to_double(romuduojr_next64(<romuduojr_state *>st))

cdef uint64_t romuduojr_raw(void *st) noexcept nogil:
    return <uint64_t>romuduojr_next32(<romuduojr_state *> st)

cdef uint64_t romuquad32_uint64(void* st) noexcept nogil:
    return romuquad32_next64(<romuquad32_state *>st)

cdef uint32_t romuquad32_uint32(void *st) noexcept nogil:
    return romuquad32_next32(<romuquad32_state *> st)

cdef double romuquad32_double(void* st) noexcept nogil:
    return uint64_to_double(romuquad32_next64(<romuquad32_state *>st))

cdef uint64_t romuquad32_raw(void *st) noexcept nogil:
    return <uint64_t>romuquad32_next32(<romuquad32_state *> st)

cdef uint64_t romutrio32_uint64(void* st) noexcept nogil:
    return romutrio32_next64(<romutrio32_state *>st)

cdef uint32_t romutrio32_uint32(void *st) noexcept nogil:
    return romutrio32_next32(<romutrio32_state *> st)

cdef double romutrio32_double(void* st) noexcept nogil:
    return uint64_to_double(romutrio32_next64(<romutrio32_state *>st))

cdef uint64_t romutrio32_raw(void *st) noexcept nogil:
    return <uint64_t>romutrio32_next32(<romutrio32_state *> st)


# TODO (Vizonex) Optimize when I have the time to.

cdef class RomuQuad(BitGenerator):
    """
    RomuQuad
    --------

    More robust than anyone could need, but uses more registers than RomuTrio.
    Est. capacity >= 2^90 bytes. Register pressure = 8 (high). State size = 256 bits.
    """

    cdef romuquad_state rng_state

    def __init__(self, seed=None) -> None:
        BitGenerator.__init__(self, seed)
        
        val = self._seed_seq.generate_state(4, np.uint64)
        
        for i in range(4):
            self.rng_state.state[i] = val
        

        self._bitgen.state = &self.rng_state
        self._bitgen.next_uint32 = &romuquad_uint32      
        self._bitgen.next_uint64 = &romuquad_uint64      
        self._bitgen.next_double = &romuquad_double      
        self._bitgen.next_raw = &romuquad_raw
    
    @property
    def state(self):
        cdef np.ndarray key = np.zeros(4, dtype=np.uint64)
        for i in range(4):
            key[i] = self.rng_state.state[i]

        return {'bit_generator': self.__class__.__name__, 'state':{'key':key}}
    
    @state.setter
    def state(self, value):
        if isinstance(value, tuple):
            if value[0] != 'RomuQuad' or len(value) != 2:
                raise ValueError('state is not a RomuQuad state')
            value = {'bit_generator': 'RomuQuad', 'state': {'key': value[1]}}

        if not isinstance(value, dict):
            raise TypeError('state must be a dict')

        bitgen = value.get('bit_generator', '')
        
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        key = value['state']['key']
        
        for i in range(4):
            self.rng_state.key[i] = key[i]




cdef class RomuTrio(BitGenerator):
    """
    RomuTrio
    --------

    Great for general purpose work, including huge jobs.
    Est. capacity = 2^75 bytes. Register pressure = 6. State size = 192 bits.

    """

    cdef romutrio_state rng_state

    def __init__(self, seed=None) -> None:
        BitGenerator.__init__(self, seed)

        val = self._seed_seq.generate_state(3, np.uint64)

        for i in range(3):
            self.rng_state.state[i] = val


        self._bitgen.state = &self.rng_state
        self._bitgen.next_uint32 = &romutrio_uint32
        self._bitgen.next_uint64 = &romutrio_uint64
        self._bitgen.next_double = &romutrio_double
        self._bitgen.next_raw = &romutrio_raw

    @property
    def state(self):
        key = np.zeros(3, dtype=np.uint64)
        for i in range(3):
            key[i] = self.rng_state.state[i]

        return {'bit_generator': self.__class__.__name__, 'state':{'key':key}}
    
    @state.setter
    def state(self, value):
        if isinstance(value, tuple):
            if value[0] != 'RomuTrio' or len(value) != 2:
                raise ValueError('state is not a RomuTrio state')
            value = {'bit_generator': 'RomuTrio', 'state': {'key': value[1]}}

        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        key = value['state']['key']
        
        for i in range(3):
            self.rng_state.key[i] = key[i]


cdef class RomuDuo(BitGenerator):
    """
    RomuDuo
    -------

    Might be faster than RomuTrio due to using fewer registers, but might struggle with massive jobs.
    Est. capacity = 2^61 bytes. Register pressure = 5. State size = 128 bits.
    """
    
    cdef romuduo_state rng_state

    def __init__(self, seed=None) -> None:
        BitGenerator.__init__(self, seed)

        val = self._seed_seq.generate_state(2, np.uint64)

        for i in range(2):
            self.rng_state.state[i] = val


        self._bitgen.state = &self.rng_state
        self._bitgen.next_uint32 = &romuduo_uint32
        self._bitgen.next_uint64 = &romuduo_uint64
        self._bitgen.next_double = &romuduo_double
        self._bitgen.next_raw = &romuduo_raw

    @property
    def state(self):
        cdef np.ndarray key = np.zeros(2, dtype=np.uint64)
        for i in range(2):
            key[i] = self.rng_state.state[i]

        return {'bit_generator': self.__class__.__name__, 'state':{'key':key}}
    
    @state.setter
    def state(self, value):
        if isinstance(value, tuple):
            if value[0] != 'RomuDuo' or len(value) != 2:
                raise ValueError('state is not a RomuDuo state')
            value = {'bit_generator': 'RomuDuo', 'state': {'key': value[1]}}

        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        key = value['state']['key']
        
        for i in range(2):
            self.rng_state.key[i] = key[i]


cdef class RomuDuoJR(BitGenerator):
    """

    RomuDuoJR
    ---------

    The fastest generator using 64-bit arith., but not suited for huge jobs.
    Est. capacity = 2^51 bytes. Register pressure = 4. State size = 128 bits.
    """
    cdef romuduojr_state rng_state

    def __init__(self, seed=None) -> None:
        BitGenerator.__init__(self, seed)

        val = self._seed_seq.generate_state(2, np.uint64)

        for i in range(2):
            self.rng_state.state[i] = val


        self._bitgen.state = &self.rng_state
        self._bitgen.next_uint32 = &romuduojr_uint32
        self._bitgen.next_uint64 = &romuduojr_uint64
        self._bitgen.next_double = &romuduojr_double
        self._bitgen.next_raw = &romuduojr_raw

    @property
    def state(self):
        cdef np.ndarray key = np.zeros(2, dtype=np.uint64)
        for i in range(2):
            key[i] = self.rng_state.state[i]

        return {'bit_generator': self.__class__.__name__, 'state':{'key':key}}
    
    @state.setter
    def state(self, value):
        if isinstance(value, tuple):
            if value[0] != 'RomuDuoJR' or len(value) != 2:
                raise ValueError('state is not a RomuDuoJR state')
            value = {'bit_generator': 'RomuDuoJR', 'state': {'key': value[1]}}

        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        key = value['state']['key']
        
        for i in range(4):
            self.rng_state.key[i] = key[i]


cdef class RomuQuad32(BitGenerator):
    """
    RomuQuad32
    ----------

    32-bit arithmetic: Good for general purpose use.
    Est. capacity >= 2^62 bytes. Register pressure = 7. State size = 128 bits.
    """

    cdef romuquad32_state rng_state

    def __init__(self, seed=None) -> None:
        BitGenerator.__init__(self, seed)

        val = self._seed_seq.generate_state(4, np.uint32)

        for i in range(4):
            self.rng_state.state[i] = val


        self._bitgen.state = &self.rng_state
        self._bitgen.next_uint32 = &romuquad32_uint32
        self._bitgen.next_uint64 = &romuquad32_uint64
        self._bitgen.next_double = &romuquad32_double
        self._bitgen.next_raw = &romuquad32_raw

    @property
    def state(self):
        cdef np.ndarray key = np.zeros(4, dtype=np.uint32)
        for i in range(4):
            key[i] = self.rng_state.state[i]

        return {
            'bit_generator': self.__class__.__name__, 
            'state': {'key':key}
        }
    
    @state.setter
    def state(self, value):
        if isinstance(value, tuple):
            if value[0] != 'RomuQuad32' or len(value) != 2:
                raise ValueError('state is not a RomuQuad32 state')
            value = {'bit_generator': 'RomuQuad32', 'state': {'key': value[1]}}

        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        key = value['state']['key']
        
        for i in range(4):
            self.rng_state.key[i] = key[i]


cdef class RomuTrio32(BitGenerator):
    """
    RomuTrio32
    ----------

    32-bit arithmetic: Good for general purpose use, except for huge jobs.
    Est. capacity >= 2^53 bytes. Register pressure = 5. State size = 96 bits.
    """

    cdef romutrio32_state rng_state

    def __init__(self, seed=None) -> None:
        BitGenerator.__init__(self, seed)

        val = self._seed_seq.generate_state(3, np.uint32)

        for i in range(3):
            self.rng_state.state[i] = val


        self._bitgen.state = &self.rng_state
        self._bitgen.next_uint32 = &romutrio32_uint32
        self._bitgen.next_uint64 = &romutrio32_uint64
        self._bitgen.next_double = &romutrio32_double
        self._bitgen.next_raw = &romutrio32_raw

    @property
    def state(self):
        cdef np.ndarray key = np.zeros(3, dtype=np.uint32)
        for i in range(3):
            key[i] = self.rng_state.state[i]
        
        return {'bit_generator': self.__class__.__name__, 'state':{'key':key}}
    
    @state.setter
    def state(self, value):
        if isinstance(value, tuple):
            if value[0] != 'RomuTrio32' or len(value) != 2:
                raise ValueError('state is not a RomuTrio32 state')
            value = {'bit_generator': 'RomuTrio32', 'state': {'key': value[1]}}

        if not isinstance(value, dict):
            raise TypeError('state must be a dict')
        bitgen = value.get('bit_generator', '')
        if bitgen != self.__class__.__name__:
            raise ValueError('state must be for a {0} '
                             'PRNG'.format(self.__class__.__name__))
        key = value['state']['key']
        
        for i in range(4):
            self.rng_state.key[i] = key[i]


