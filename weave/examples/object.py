# h:\wrk\scipy\weave\examples>python object.py
# initial val: 1
# inc result: 2
# after set attr: 5

import weave

#----------------------------------------------------------------------------
# get/set attribute and call methods example
#----------------------------------------------------------------------------

class foo:
    def __init__(self):
        self.val = 1
    def inc(self,amount):
        self.val += 1
        return self.val
obj = foo()
code = """
       int i = obj.attr("val");
       std::cout << "initial val: " << i << std::endl;
       
       py::tuple args(1);
       args[0] = 2; 
       i = obj.mcall("inc",args);
       std::cout << "inc result: " << i << std::endl;
       
       obj.set_attr("val",5);
       i = obj.attr("val");
       std::cout << "after set attr: " << i << std::endl;
       """
weave.inline(code,['obj'])       
       
#----------------------------------------------------------------------------
# indexing of values.
#----------------------------------------------------------------------------
from UserList import UserList
obj = UserList([1,[1,2],"hello"])
code = """
       int i;
       // find obj length and accesss each of its items
       std::cout << "UserList items: ";
       for(i = 0; i < obj.length(); i++)
           std::cout << obj[i] << " ";
       std::cout << std::endl;
       // assign new values to each of its items
       for(i = 0; i < obj.length(); i++)
           obj[i] = "goodbye";
       """
weave.inline(code,['obj'])       
print "obj with new values:", obj