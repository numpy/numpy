#!/usr/bin/env python
"""

Copyright 2001 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>          
Permission to use, modify, and distribute this software is given under the
terms of the LGPL.  See http://www.fsf.org

Note:  This should be merged into proc at some point.  Perhaps proc should
be returning classes like this instead of using dictionaries.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
$Revision$
$Date$
Pearu Peterson
"""

__version__ = "$Id$"

import sys,string,re

class cpuinfo:
    """Holds CPU information and provides methods for requiring
    the availability of CPU features.
    """

    info = None
    
    def __init__(self):
        if self.info is not None:
            return
        info = []
        try:
            if sys.platform == 'linux2':
                info.append({})
                for line in open('/proc/cpuinfo').readlines():
                    name_value = map(string.strip,string.split(line,':',1))
                    if len(name_value)!=2:
                        continue
                    name,value = name_value
                    if info[-1].has_key(name): # next processor
                        info.append({})
                    info[-1][name] = value
            #XXX How to obtain CPU information on other platforms?
        except e,m:
            print '%s: %s (ignoring)' % (e,m)
        self.info = info

    def is_Pentium(self):
        #XXX
        pass

    def is_PentiumPro(self):
        #XXX
        pass

    def is_PentiumII(self):
        try:
            return re.match(r'.*?Pentium II[^I]',
                            self.info[0]['model name']) is not None
        except:
            pass

    def is_PentiumIII(self):
        #XXX
        pass

    def is_PentiumIV(self):
        #XXX
        pass

    def is_AthlonK6(self):
        try:
            return re.match(r'.*?AMD-K6',self.info[0]['model name']) is not None
        except:
            pass

    def is_AthlonK7(self):
        try:
            return re.match(r'.*?AMD-K7',self.info[0]['model name']) is not None
        except:
            pass

    def is_AMD(self):
        try:
            return self.info[0]['vendor_id']=='AuthenticAMD'
        except:
            pass

    def is_Intel(self):
        try:
            return self.info[0]['vendor_id']=='GenuineIntel'
        except:
            pass

    def is_Alpha(self):
        try:
            return self.info[0]['cpu']=='Alpha'
        except:
            pass

    def is_i386(self):
        #XXX
        pass

    def is_i486(self):
        #XXX
        pass

    def is_i586(self):
        if self.is_Intel():
            try:
                return self.info[0]['model'] == '5'
            except:
                pass

    def is_i686(self):
        if self.is_Intel():
            try:
                return self.info[0]['model'] == '6'
            except:
                pass

    def is_singleCPU(self):
        if self.info:
            return len(self.info) == 1

    def has_fdiv_bug(self):
        try:
            return self.info[0]['fdiv_bug']=='yes'
        except:
            pass

    def has_f00f_bug(self):
        try:
            return self.info[0]['f00f_bug']=='yes'
        except:
            pass

    def has_mmx(self):
        try:
            return re.match(r'.*?\bmmx',self.info[0]['flags']) is not None
        except:
            pass

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
