"""
Storing actual strings instead of their md5 value appears to 
be about 10 times faster.

>>> md5_speed.run(200,50000)
md5 build(len,sec): 50000 0.870999932289
md5 retrv(len,sec): 50000 0.680999994278
std build(len,sec): 50000 0.259999990463
std retrv(len,sec): 50000 0.0599999427795

This test actually takes several minutes to generate the random
keys used to populate the dictionaries.  Here is a smaller run,
but with longer keys.

>>> md5_speed.run(1000,4000)
md5 build(len,sec,per): 4000 0.129999995232 3.24999988079e-005
md5 retrv(len,sec,per): 4000 0.129999995232 3.24999988079e-005
std build(len,sec,per): 4000 0.0500000715256 1.25000178814e-005
std retrv(len,sec,per): 4000 0.00999999046326 2.49999761581e-006

Results are similar, though not statistically to good because of
the short times used and the available clock resolution.

Still, I think it is safe to say that, for speed, it is better 
to store entire strings instead of using md5 versions of 
their strings.  Yeah, the expected result, but it never hurts
to check...

"""
import random, md5, time, cStringIO

def speed(n,m):
    s = 'a'*n
    t1 = time.time()            
    for i in range(m):
        q= md5.new(s).digest()
    t2 = time.time()
    print (t2 - t1) / m

#speed(50,1e6)

def generate_random(avg_length,count):
    all_str = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    lo,hi = [30,avg_length*2+30]
    for i in range(count):
        new_str = cStringIO.StringIO()
        l = random.randrange(lo,hi)
        for i in range(l):
            new_str.write(random.choice(alphabet))
        all_str.append(new_str.getvalue())
    return all_str    
    
def md5_dict(lst):
    catalog = {}
    t1 = time.time()
    for s in lst:
        key= md5.new(s).digest()
        catalog[key] = None
    t2 = time.time()    
    print 'md5 build(len,sec,per):', len(lst), t2 - t1, (t2-t1)/len(lst)
    
    t1 = time.time()
    for s in lst:
        key= md5.new(s).digest()
        val = catalog[key]
    t2 = time.time()    
    print 'md5 retrv(len,sec,per):', len(lst), t2 - t1, (t2-t1)/len(lst)

def std_dict(lst):
    catalog = {}
    t1 = time.time()
    for s in lst:
        catalog[s] = None
    t2 = time.time()    
    print 'std build(len,sec,per):', len(lst), t2 - t1, (t2-t1)/len(lst)
    
    t1 = time.time()
    for s in lst:
        val = catalog[s]
    t2 = time.time()    
    print 'std retrv(len,sec,per):', len(lst), t2 - t1, (t2-t1)/len(lst)

def run(m=200,n=10):
    lst = generate_random(m,n)
    md5_dict(lst)
    std_dict(lst)

run(2000,100)    