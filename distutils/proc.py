#! /usr/bin/python
# Currently only implemented for linux...need to do win32
# I thought of making OS specific modules that were chosen 
# based on OS much like path works, but I think this will
# break pickling when process objects are passed between
# two OSes because the module name is saved in the pickle.
# A Unix process info object passed to a Windows machine
# would attempt to import the linux proc module and fail.
# Using a single module with if-then-else I think prevents
# this name problem (but it isn't as pretty...)

import sys,string

if string.find(sys.platform,'linux') != -1:
    # begin linux specific
    import string,os,stat
    import pwd, grp
    
    hertz = 100. #standard value for jiffies (in seconds) on Linux.
    states = {'R':'RUNNING','S':'SLEEPING','Z':'ZOMBIE',
              'T':'TRACED','D':'DEEPSLEEP'}
    
    import socket
    long_machine_name = socket.gethostname()
    machine_name = string.split(long_machine_name,'.')[0]
    
    def uptime_info():
        f = open("/proc/uptime")
        line = f.readline()
        field = string.split(line)        
        info={}
        info['uptime'] = eval(field[0])
        info['idle'] = eval(field[1])
        return info
        
    def cpu_info():
        """* We'll assume SMP machines have identical processors
        
        *"""
        f = open("/proc/cpuinfo")
        lines = f.readlines()
        pairs = map(lambda x,y=':':string.split(x,y),lines)
        info={}
        for line in pairs:
            if len(line) > 1:            
                key = string.strip(line[0])
                value = string.strip(line[1])
                try: 
                    info[key] = eval(value)
                except: 
                    info[key] = value
        sub_info = {}
        sub_info['cpu_count'] = info['processor'] + 1
        sub_info['cpu_type'] = filter_name(info['model name'])
        sub_info['cpu_cache'] = eval(string.split(info['cache size'])[0])
        sub_info['cpu_speed'] = round(info['cpu MHz']/1000.,1) # in GHz
        sub_info['cpu_bogomips'] = info['bogomips']    
        #print sub_info       
        return sub_info
    
    def filter_name(name):
        """* Try to shorten the verbose names in cpuinfo *"""
        if string.find(name,'Pentium III') != -1:
            return 'P3'
        elif string.find(name,'Athlon') != -1:
            return 'Athlon'
        elif string.find(name,'Pentium II') != -1:
            return 'P2'
        return name
            
    
    def mem_info():
        f = open("/proc/meminfo")
        l = f.readlines()
        x = string.split(l[1])
        total = eval(x[1])
        used = eval(x[2])
        free = eval(x[3])
        shared = eval(x[4])
        buffers = eval(x[5])
        cached = eval(x[6])
    
        x = string.split(l[2])
        stotal = eval(x[1])
        sused = eval(x[2])
        sfree = eval(x[3])
    
        memtotal = os.stat("/proc/kcore")[stat.ST_SIZE]
    
        m = 1024*1024
        #print "Memory RAM   : %d MB  (%d MB kernel, %d MB free)" % (memtotal/m, (memtotal-total)/m, free/m)
        #print "    usage    : %d MB shared   %d MB buffers   %d MB cached" % (shared/m, buffers/m, cached/m)
        #print "Swap area    : %d MB  (%d MB used)" % (stotal/m, sused/m)
        info = {}
        info['mem_total'] = memtotal/m
        info['mem_kernel'] = (memtotal-total)/m
        info['mem_free'] = free/m
        info['swap_total'] = stotal/m
        info['swap_free'] = (stotal-sused)/m
        return info
        
    def load_avg():
        f = open("/proc/loadavg")
        line = f.readline()
        field = string.split(line)        
        info={}
        info['load_1'] = eval(field[0])
        info['load_5'] = eval(field[1])
        info['load_15'] = eval(field[2])
        return info
    
    def machine_info():
        all = load_avg()
        all.update(cpu_info())
        all.update(mem_info())
        all['long_name'] = long_machine_name
        all['name'] = machine_name
        return all
    
    
    uid_cache = {}
    gid_cache = {}
    def user_from_uid(uid):
        global uid_cache
        try:
           user = uid_cache[uid]
        except:
           user = pwd.getpwuid(uid)[0]
           uid_cache[uid] = user
        return user   
    
    def group_from_gid(gid):
        global gid_cache
        try:
           group =  gid_cache[gid]
        except:
           group = grp.getgrgid(gid)[0]
           gid_cache[gid] =group 
        return group   
    
    # Yes, this is slow.  It also works and is pure Python (easy).
    # If your looking to do a real top, or fast ps command use SIWG to wrap the
    # ps and top code.  This would give much better performance, be more robust,
    # and make the generally world a happier place.  Please send it in when 
    # your finished!
    # This is all pretty much stolen from readproc.c in the procps 
    # package (scour the web).
    
    class process:
        def __init__(self,pid,seconds_since_boot=None,total_memory=None):
            self.info(int(pid),seconds_since_boot,total_memory)
        def info(self,pid,seconds_since_boot=None,total_memory=None):
            self.status2proc(pid)
            self.stat2proc(pid)
            self.statm2proc(pid)
            self.get_cmdline(pid)
            #self.get_environ(pid)
            if self.state == 'Z':
                self.cmd = self.cmd + "<defunct>"
            self.beautify(seconds_since_boot,total_memory)
            self.pid = pid
            self.machine = machine_name
            self.long_machine = long_machine_name
        def status2proc(self,pid):
            f = open("/proc/%d/status" % pid)
            lines = f.readlines()
            #line = map(string.split,lines)
            id = map(string.split,lines[4:6])
            
            #self.cmd = line[0][1]
            #self.state = line[1][2]
            self.ruid,self.euid,self.suid,self.fuid = map(int,id[0][1:])
            self.rgid,self.egid,self.sgid,self.fgid = map(int,id[1][1:])
            #self.euid = int(id[0][1])
            #self.egid = int(id[1][1])
            #self.vm_size = long(line[7][1])
            #self.vm_lock = long(line[8][1])
            #self.vm_rss = long(line[9][1])
            #self.vm_data = long(line[10][1])
            #self.vm_stack = long(line[11][1])
            #self.vm_exe = long(line[12][1])
            #self.vm_lib = long(line[13][1])
            
            #translate id's to user and group names
            #self.ruser = user_from_uid(self.ruid)
            #self.suser = user_from_uid(self.suid)
            self.euser = user_from_uid(self.euid)
            #self.fuser = user_from_uid(self.fuid)
            #self.rgroup = group_from_gid(self.rgid)
            #self.sgroup = group_from_gid(self.sgid)
            self.egroup = group_from_gid(self.egid)
            #self.fgroup = group_from_gid(self.fgid)
            
        def stat2proc(self,pid):
            # oh, to have sscanf... (I'm sure it is somewhere)
            f = open("/proc/%d/stat" % pid)
            s = f.read(-1)
            f.close()
            cmd = string.rfind(s,'(') + 1
            begin = string.rfind(s,')')
            self.cmd = s[cmd:begin]
            begin = begin + 2
            field = string.split(s[begin:])
            self.state = field[0]
            self.ppid = int(field[1])
            #self.pgrp = int(field[2])
            #self.session = int(field[3])
            self.tty = int(field[4])
            #self.tpgid = eval(field[5])
            #self.flags = eval(field[6])
            #self.min_flt = long(field[7])
            #self.cmin_flt = long(field[8])
            #self.maj_flt = long(field[9])
            #self.cmaj_flt = long(field[10])
            self.utime = long(field[11])
            self.stime = long(field[12])
            self.cutime = int(field[13])
            self.cstime = int(field[14])
            self.priority = int(field[15])
            self.nice = int(field[16])
            #self.timeout = eval(field[17])
            #self.it_real_value = eval(field[18])
            self.start_time = long(field[19])
            self.vsize = long(field[20])
            self.rss = long(field[21])
            self.rss_rlim = long(field[22])
            #self.start_code = long(field[23])
            #self.end_code = long(field[24])
            self.start_stack = long(field[25])
            #self.kstk_esp = long(field[26])
            #self.kstk_eip = long(field[27])
            #self.wchan = long(field[29])
            #self.nswap = long(field[30])
            self.cnswap = long(field[31])
            if self.tty == 0: self.tty=-1
        def statm2proc(self,pid):
            f = open("/proc/%d/statm" % pid)
            s = f.read(-1)
            f.close()
            field = string.split(s)
            self.size = int(field[0])
            self.resident = int(field[1])
            #self.share = int(field[2])
            #self.trs = int(field[3])
            #self.lrs = int(field[4])
            #self.drs = int(field[5])
            #self.dt = int(field[6])
        def get_cmdline(self,pid):
            f = open("/proc/%d/cmdline" % pid)
            self.cmdline = f.read(-1)
            f.close()             
        def get_environ(self,pid):
            f = open("/proc/%d/environ" % pid)
            self.cmdline = f.read(-1)
            f.close()
    
        def beautify(self,seconds_since_boot=None,total_memory=None):
            """* total_memory in MB                     
            *"""
            if seconds_since_boot is None:
                seconds_since_boot = uptime_info()['uptime']
            if total_memory is None:
                total_memory = mem_info()['mem_total']
            self.beautify_user()
            self.beautify_cpu(seconds_since_boot)
            self.beautify_memory(total_memory)
            self.beautify_state()
        def beautify_user(self):    
            self.uid = self.euid
            self.user = self.euser
            self.gid = self.egid
            self.group = self.egroup
        
        def beautify_cpu(self,seconds_since_boot):
            include_dead_children = 0    
            self.total_time = (self.utime + self.stime) / hertz        
            self.wall_time = seconds_since_boot - self.start_time /hertz
            if include_dead_children:
                self.total_time = self.total_time + \
                                  (self.cutime + self.cstime) / hertz
            self.pcpu = 0
            if self.wall_time:
                self.pcpu = self.total_time * 1000. / self.wall_time
            if self.pcpu > 999: 
                self.pcpu = 999.
            
            self.cpu_percent = self.pcpu / 10.
            #foramt time into a days:hours:minutes:seconds string
            t = long(self.wall_time)
            t,ss = divmod(t,60)
            t,mm = divmod(t,60)
            t,hh = divmod(t,24)
            dd = t
            self.wall_time2 = "%2d:%2d:%2d:%2d" % (dd,hh,mm,ss)
            t = long(self.total_time)  
            t,ss = divmod(t,60)
            t,mm = divmod(t,60)
            t,hh = divmod(t,24)
            dd = t
            self.total_time2 = "%2d:%2d:%2d:%2d" % (dd,hh,mm,ss)
            
        def beautify_memory(self,total_memory):
            """* translate memory values to MB, and percentage
            *"""
            self.total_memory = self.size * 4 / 1024.
            self.resident_memory = self.resident * 4/ 1024.
            self.memory_percent = self.resident_memory / total_memory * 100
        def beautify_state(self):
            self.condition= states[self.state]
            self.cmdline = string.replace(self.cmdline,'\x00',' ')
            self.cmdline = string.strip(self.cmdline)
    
        #ps_default = ['user','pid','cpu_percent','total_memory','resident_memory',
        #              'state','start_time','cmdline']
        ps_default = ['user','pid','cpu_percent','total_memory','resident_memory',
                      'state','total2_time','cmdline']
        def labels(self):
            s = "%-8s %5s  %4s  %4s %8s %8s %1s %10s %3s" % \
                 ('USER','PID','%CPU', '%MEM', 'TOTAL MB', ' RES MB',
                  'ST', 'RT-D:H:M:S', 'CMD')
            return s                      
        def labels_with_name(self):
            s = "%-6s %s" % ('MACHINE',self.labels())
            return s
        def str_with_name(self):
            s = "%-6s %-8s  %5d  %4.1f  %4.1f %8.3f %8.3f %1s %s " % \
                 (self.machine[-6:], self.user,self.pid,self.cpu_percent,
                  self.memory_percent, self.total_memory, self.resident_memory,
                  self.state, self.total_time2)        
            bytes_left = 80 - len(s) - 1
            if len(self.cmdline) > bytes_left:
                s = s +  self.cmdline[:6] + '...' + self.cmdline[-(bytes_left-9):]
            else:    
                s = s +  self.cmdline
            return s
        def __str__(self):
            s = "%-8s %5d %4.1f %4.1f %8.3f %8.3f %1s %s " % \
                 (self.user,self.pid,self.cpu_percent,self.memory_percent,
                   self.total_memory, self.resident_memory, self.state, 
                  self.total_time2)        
            bytes_left = 80 - len(s) - 1
            if len(self.cmdline) > bytes_left:
                s = s +  self.cmdline[:6] + '...' + self.cmdline[-(bytes_left-9):]
            else:    
                s = s +  self.cmdline
            return 
            
    
    def ps_list(sort_by='cpu',**filters):
        import os, glob
        current = os.path.abspath('.')
        os.chdir('/proc')   
        procs = glob.glob('[0-9]*')
        results = []
        seconds_since_boot = uptime_info()['uptime']
        total_memory = mem_info()['mem_total']            
        for proc in procs:
            results.append(process(proc,seconds_since_boot,total_memory))
        os.chdir(current)
        return ps_sort(results,sort_by,**filters)
    # end linux specific
else:
    # punt.  At least there exist a class so that unpickling won't fail.
    def uptime_info():
        raise NotImplemented, 'not implemented on this architecture'
    def cpu_info():
        raise NotImplemented, 'not implemented on this architecture'
    def filter_name(name):
        raise NotImplemented, 'not implemented on this architecture'    
    def mem_info():
        raise NotImplemented, 'not implemented on this architecture'
    def load_avg():
        raise NotImplemented, 'not implemented on this architecture'
    def machine_info():
        raise NotImplemented, 'not implemented on this architecture'
        
    uid_cache = {}
    gid_cache = {}
    def user_from_uid(uid):
        raise NotImplemented, 'not implemented on this architecture'
    def group_from_gid(gid):
        raise NotImplemented, 'not implemented on this architecture'
    def ps_list(sort_by='cpu',**filters):
        raise NotImplemented, 'not implemented on this architecture'                
    class process:
        def labels(self):
            s = "%-8s %5s  %4s  %4s %8s %8s %1s %10s %3s" % \
                 ('USER','PID','%CPU', '%MEM', 'TOTAL MB', ' RES MB',
                  'ST', 'RT-D:H:M:S', 'CMD')
            return s                      
        def labels_with_name(self):
            s = "%-6s %s" % ('MACHINE',self.labels())
            return s
        def str_with_name(self):
            s = "%-6s %-8s  %5d  %4.1f  %4.1f %8.3f %8.3f %1s %s " % \
                 (self.machine[-6:], self.user,self.pid,self.cpu_percent,
                  self.memory_percent, self.total_memory, self.resident_memory,
                  self.state, self.total_time2)        
            bytes_left = 80 - len(s) - 1
            if len(self.cmdline) > bytes_left:
                s = s +  self.cmdline[:6] + '...' + self.cmdline[-(bytes_left-9):]
            else:    
                s = s +  self.cmdline
            return s
             
        def __str__(self):
            s = "%-8s %5d %4.1f %4.1f %8.3f %8.3f %1s %s " % \
                 (self.user,self.pid,self.cpu_percent,self.memory_percent,
                   self.total_memory, self.resident_memory, self.state, 
                  self.total_time2)        
            bytes_left = 80 - len(s) - 1
            if len(self.cmdline) > bytes_left:
                s = s +  self.cmdline[:6] + '...' + self.cmdline[-(bytes_left-9):]
            else:    
                s = s +  self.cmdline
            return 

# these are all general to any OS

def cmp_pid(x,y):
    return cmp(x.pid,y.pid)
def cmp_cpu(x,y):
    return -cmp(x.cpu_percent,y.cpu_percent)
def cmp_user(x,y):
    return cmp(x.user,y.user)
def cmp_machine(x,y):
    return cmp(x.machine,y.machine)
def cmp_memory(x,y):
    return -cmp(x.memory_percent,y.memory_percent)
def cmp_state(x,y):
    return cmp(x.state,y.state)
def cmp_command(x,y):
    return cmp(x.cmdline,y.cmdline)
    
ps_cmp={}
ps_cmp['pid'] = cmp_pid    
ps_cmp['cpu'] = cmp_cpu    
ps_cmp['user'] = cmp_user
ps_cmp['machine'] = cmp_machine
ps_cmp['memory'] = cmp_memory
ps_cmp['state'] = cmp_state
ps_cmp['command'] = cmp_command

from fnmatch import fnmatch

def filter_machine(x,filt):
    return (fnmatch(x.machine,filt) or fnmatch(x.long_machine,filt))
def filter_user(x,filt):
    return (fnmatch(x.user,filt))   
def filter_state(x,filt):
    return (fnmatch(x.state,filt))
def filter_command(x,filt):
    return (fnmatch(x.cmdline,filt))
def filter_cpu(x,filt):
    return eval(str(x.cpu_percent) + filt)
def filter_memory(x,filt):
    return eval(str(x.memory_percent) + filt)
def filter_mb(x,filt):
    return eval(str(x.total_memory) + filt)
    
ps_filter={}
ps_filter['user'] = filter_user
ps_filter['machine'] = filter_machine
ps_filter['state'] = filter_state
ps_filter['command'] = filter_command
ps_filter['memory'] = filter_memory
ps_filter['mb'] = filter_mb
ps_filter['cpu'] = filter_cpu

def ps(sort_by='cpu',**filters):
    psl = ps_list(sort_by,**filters)
    if len(psl):
        print psl[0].labels_with_name()
    for i in psl: 
        print i
    
def ps_sort(psl,sort_by='cpu',**filters):
    for f in filters.keys():
        try:
            filt = ps_filter[f]
            filt_str = filters[f]
            psl = filter(lambda x,filt=filt,y=filt_str:filt(x,y),psl)
        except KeyError:
            print 'warning: "', f, '"is an invalid key for filtering command.'
            print '         ', 'use one of the following:', str(ps_filter.keys())
    try:
        compare = ps_cmp[sort_by]    
        psl.sort(compare)
    except KeyError:
        print 'warning: "', sort_by, '"is an invalid choice for sorting.'
        print '         ', 'use one of the following:', str(ps_cmp.keys())
    return psl


