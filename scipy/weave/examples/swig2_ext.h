#include <iostream>
class A {
public:
   void f() {std::cout << "A::f()\n";}
};

A* foo()
{
   A* a = new A;
   return a;
}
