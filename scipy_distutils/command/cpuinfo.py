#!/usr/bin/env python
"""
cpuinfo

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the 
terms of the SciPy (BSD style) license.  See LICENSE.txt that came with
this distribution for specifics.

Note:  This should be merged into proc at some point.  Perhaps proc should
be returning classes like this instead of using dictionaries.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Revision$
$Date$
Pearu Peterson
"""

__version__ = "$Id$"

__all__ = ['cpuinfo']

import sys,string,re,types

class cpuinfo_base:
    """Holds CPU information and provides methods for requiring
    the availability of various CPU features.
    """

    def _try_call(self,func):
        try:
            return func()
        except:
            pass

    def __getattr__(self,name):
        if name[0]!='_':
            if hasattr(self,'_'+name):
                attr = getattr(self,'_'+name)
                if type(attr) is types.MethodType:
                    return lambda func=self._try_call,attr=attr : func(attr)
            else:
                return lambda : None
        raise AttributeError,name    


class linux_cpuinfo(cpuinfo_base):

    info = None
    
    def __init__(self):
        if self.info is not None:
            return
        info = []
        try:
            for line in open('/proc/cpuinfo').readlines():
                name_value = map(string.strip,string.split(line,':',1))
                if len(name_value)!=2:
                    continue
                name,value = name_value
                if not info or info[-1].has_key(name): # next processor
                    info.append({})
                info[-1][name] = value
        except:
            print sys.exc_value,'(ignoring)'
        self.__class__.info = info

    def _not_impl(self): pass

    # Athlon

    def _is_AMD(self):
        return self.info[0]['vendor_id']=='AuthenticAMD'

    def _is_AthlonK6(self):
        return re.match(r'.*?AMD-K6',self.info[0]['model name']) is not None

    def _is_AthlonK7(self):
        return re.match(r'.*?AMD-K7',self.info[0]['model name']) is not None

    # Alpha

    def _is_Alpha(self):
        return self.info[0]['cpu']=='Alpha'

    def _is_EV4(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV4'

    def _is_EV5(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV5'

    def _is_EV56(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'EV56'

    def _is_PCA56(self):
        return self.is_Alpha() and self.info[0]['cpu model'] == 'PCA56'

    # Intel

    #XXX
    _is_i386 = _not_impl

    def _is_Intel(self):
        return self.info[0]['vendor_id']=='GenuineIntel'

    def _is_i486(self):
        return self.info[0]['cpu']=='i486'

    def _is_i586(self):
        return self.is_Intel() and self.info[0]['cpu family'] == '5'

    def _is_i686(self):
        return self.is_Intel() and self.info[0]['cpu family'] == '6'

    def _is_Celeron(self):
        return re.match(r'.*?Celeron',
                        self.info[0]['model name']) is not None

    def _is_Pentium(self):
        return re.match(r'.*?Pentium',
                        self.info[0]['model name']) is not None

    def _is_PentiumII(self):
        return re.match(r'.*?Pentium II\b',
                        self.info[0]['model name']) is not None

    _is_PentiumPro = _not_impl

    def _is_PentiumIII(self):
        return re.match(r'.*?Pentium III\b',
                        self.info[0]['model name']) is not None

    def _is_PentiumIV(self):
        return re.match(r'.*?Pentium IV\b',
                        self.info[0]['model name']) is not None

    # Varia

    def _is_singleCPU(self):
        return len(self.info) == 1

    def _has_fdiv_bug(self):
        return self.info[0]['fdiv_bug']=='yes'

    def _has_f00f_bug(self):
        return self.info[0]['f00f_bug']=='yes'

    def _has_mmx(self):
        return re.match(r'.*?\bmmx\b',self.info[0]['flags']) is not None

    def _has_sse(self):
        return re.match(r'.*?\bsse\b',self.info[0]['flags']) is not None

    def _has_sse2(self):
        return re.match(r'.*?\bsse2\b',self.info[0]['flags']) is not None

    def _has_3dnow(self):
        return re.match(r'.*?\b3dnow\b',self.info[0]['flags']) is not None

if sys.platform[:5] == 'linux': # variations: linux2,linux-i386 (any others?)
    cpuinfo = linux_cpuinfo
#XXX: other OS's. Eg. use _winreg on Win32. Or os.uname on unices.
else:
    cpuinfo = cpuinfo_base


"""
laptop:
[{'cache size': '256 KB', 'cpu MHz': '399.129', 'processor': '0', 'fdiv_bug': 'no', 'coma_bug': 'no', 'model': '6', 'cpuid level': '2', 'model name': 'Mobile Pentium II', 'fpu_exception': 'yes', 'hlt_bug': 'no', 'bogomips': '796.26', 'vendor_id': 'GenuineIntel', 'fpu': 'yes', 'wp': 'yes', 'cpu family': '6', 'f00f_bug': 'no', 'stepping': '13', 'flags': 'fpu vme de pse tsc msr pae mce cx8 sep mtrr pge mca cmov pat pse36 mmx fxsr'}]

kev:
[{'cache size': '512 KB', 'cpu MHz': '350.799', 'processor': '0', 'fdiv_bug': 'no', 'coma_bug': 'no', 'model': '5', 'cpuid level': '2', 'model name': 'Pentium II (Deschutes)', 'fpu_exception': 'yes', 'hlt_bug': 'no', 'bogomips': '699.59', 'vendor_id': 'GenuineIntel', 'fpu': 'yes', 'wp': 'yes', 'cpu family': '6', 'f00f_bug': 'no', 'stepping': '3', 'flags': 'fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 mmx fxsr'}, {'cache size': '512 KB', 'cpu MHz': '350.799', 'processor': '1', 'fdiv_bug': 'no', 'coma_bug': 'no', 'model': '5', 'cpuid level': '2', 'model name': 'Pentium II (Deschutes)', 'fpu_exception': 'yes', 'hlt_bug': 'no', 'bogomips': '701.23', 'vendor_id': 'GenuineIntel', 'fpu': 'yes', 'wp': 'yes', 'cpu family': '6', 'f00f_bug': 'no', 'stepping': '3', 'flags': 'fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 mmx fxsr'}]

ath:
[{'cache size': '512 KB', 'cpu MHz': '503.542', 'processor': '0', 'fdiv_bug': 'no', 'coma_bug': 'no', 'model': '1', 'cpuid level': '1', 'model name': 'AMD-K7(tm) Processor', 'fpu_exception': 'yes', 'hlt_bug': 'no', 'bogomips': '1002.70', 'vendor_id': 'AuthenticAMD', 'fpu': 'yes', 'wp': 'yes', 'cpu family': '6', 'f00f_bug': 'no', 'stepping': '2', 'flags': 'fpu vme de pse tsc msr pae mce cx8 sep mtrr pge mca cmov pat mmx syscall mmxext 3dnowext 3dnow'}]

fiasco:
[{'max. addr. space #': '127', 'cpu': 'Alpha', 'cpu serial number': 'Linux_is_Great!', 'kernel unaligned acc': '0 (pc=0,va=0)', 'system revision': '0', 'system variation': 'LX164', 'cycle frequency [Hz]': '533185472', 'system serial number': 'MILO-2.0.35-c5.', 'timer frequency [Hz]': '1024.00', 'cpu model': 'EV56', 'platform string': 'N/A', 'cpu revision': '0', 'BogoMIPS': '530.57', 'cpus detected': '0', 'phys. address bits': '40', 'user unaligned acc': '1340 (pc=2000000ec90,va=20001156da4)', 'page size [bytes]': '8192', 'system type': 'EB164', 'cpu variation': '0'}]
"""

if __name__ == "__main__":
    cpu = cpuinfo()

    cpu.is_blaa()
    cpu.is_Intel()
    cpu.is_Alpha()

    print 'CPU information:',
    for name in dir(cpuinfo):
        if name[0]=='_' and name[1]!='_' and getattr(cpu,name[1:])():
            print name[1:],
    print
