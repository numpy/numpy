#include <iostream>
#include <string>

class operand2 {
public:
    operand2(const char* name, const char* templateName1, 
        const char* templateName2, const char* iterator, const char* numtype, 
        const char* initialization)
      : name_(name), template1_(templateName1), template2_(templateName2),
        iterator_(iterator), numtype_(numtype), initialization_(initialization)
    { }

    int haveTemplate1() const
    { return template1_ != 0; }

    int haveTemplate2() const
    { return template2_ != 0; }

    void setOperandNum(int i) 
    { operandNum_ = i; }

    virtual void printName(std::ostream& os)
    { 
        os << name_;           
        printTemplates(os);
    }

    virtual void printTemplates(std::ostream& os)
    {
        if (haveTemplate1())
        {
            os << "<" << template1_ << operandNum_;
            if (haveTemplate2())
            {
                os << ", " << template2_ << operandNum_;
            }
            os << ">";
        }
    }

    virtual void printTemplate1(std::ostream& os)
    {
        if (haveTemplate1())
            os << template1_ << operandNum_;
    }

    virtual void printTemplate2(std::ostream& os)
    {
        if (haveTemplate2())
            os << template2_ << operandNum_;
    }

    virtual void printTemplateType1(std::ostream& os)
    {
        if (haveTemplate1())
            os << "class";
    }

    virtual void printTemplateType2(std::ostream& os)
    {
        if (haveTemplate2())
            os << "class";
    }

    virtual int numTemplateParameters() const
    {
        if (haveTemplate1() && haveTemplate2())
            return 2;
        else if (haveTemplate1())
            return 1;
        else
            return 0;
    }

    virtual int isScalar() const
    { return 0; }

    virtual int isInteger() const
    { return 0; }

    virtual int isComplex() const
    { return 0; }

    virtual void printArgument(std::ostream& os)
    {
        printName(os);
        os << " d" << operandNum_;      
    }

    virtual void printIterator(std::ostream& os)
    {
        os << iterator_;
        printTemplates(os);
    }

    virtual void printNumtype(std::ostream& os)
    {
        os << numtype_ << operandNum_;
    }

    virtual void printInitialization(std::ostream& os)
    {
        os << "d" << operandNum_;       
        if (initialization_ != 0)
            os << initialization_;
    }

protected:
    const char* name_;
    const char* template1_;
    const char* template2_;
    const char* iterator_;
    const char* numtype_;
    const char* initialization_;
    int operandNum_;
};

class MatOperand : public operand2 {
public:
    MatOperand()
      : operand2("Matrix", "P_numtype", "P_struct", "_bz_MatrixRef",
           "P_numtype", "._bz_getRef()")
    { }

    virtual void printArgument(std::ostream& os)
    {
        os << "const ";
        printName(os);
        os << "& d" << operandNum_;
    }

};

class MatExprOperand : public operand2 {
public:
    MatExprOperand()
      : operand2("_bz_MatExpr", "P_expr", 0, "_bz_MatExpr",
           0, 0)
    { }

    virtual void printNumtype(std::ostream& os)
    {
        os << "typename P_expr" << operandNum_ << "::T_numtype";
    }
};

class ScalarOperand2 : public operand2 {
public:
    ScalarOperand2(const char* name)
      : operand2(name, 0, 0, 0, name, 0)
    { }

    virtual void printIterator(std::ostream& os)
    {
        os << "_bz_MatExprConstant<" << name_ << ">";
    }

    virtual void printInitialization(std::ostream& os)
    {
        os << "_bz_MatExprConstant<" << name_ << ">(d"
           << operandNum_ << ")";
    }

    virtual void printNumtype(std::ostream& os)
    {
        os << name_;
    }

    virtual int isScalar() const
    { return 1; }

    virtual int isInteger() const
    { return !strcmp(name_, "int"); }

};

class ComplexOperand2 : public operand2 {
public:
    ComplexOperand2()
      : operand2("complex", "T", 0, 0, "complex", 0)
    { }

    virtual int isComplex() const
    { return 1; }

    virtual void printIterator(std::ostream& os)
    {
        os << "_bz_MatExprConstant<";
        printNumtype(os);
        os << "> ";
    }

    virtual void printInitialization(std::ostream& os)
    {
        printIterator(os);
        os << "(d" << operandNum_ << ")";
    }

    virtual void printNumtype(std::ostream& os)
    {
        os << "complex<";
        printTemplate1(os);
        os << "> ";
    }

    virtual int isScalar() const
    { return 1; }

    virtual int isInteger() const
    { return !strcmp(name_, "int"); }
};

/**************************************************************************
 *
 * Operand Set
 *
 **************************************************************************/

#define BZ_MATEXPR_NUM_OPERANDS 7

class operandSet2 {
public:
    operandSet2()
    {
        operands_[0] = new MatOperand;
        operands_[1] = new MatExprOperand;
        operands_[2] = new ScalarOperand2("int");
        operands_[3] = new ScalarOperand2("float");
        operands_[4] = new ScalarOperand2("double");
        operands_[5] = new ScalarOperand2("long double");
        operands_[6] = new ComplexOperand2;
    }

    ~operandSet2()
    {
        for (int i=0; i < BZ_MATEXPR_NUM_OPERANDS; ++i)
            delete operands_[i];
    }

    operand2& operator[](int i) 
    { return *operands_[i]; }

    int numOperands() const
    { return BZ_MATEXPR_NUM_OPERANDS; }

    void setOperandNum(int num)
    {
        for (int i=0; i < BZ_MATEXPR_NUM_OPERANDS; ++i)
            operands_[i]->setOperandNum(num);
    }

private:
    operandSet2(const operandSet2&);
    void operator=(const operandSet2&);

    operand2 * operands_[BZ_MATEXPR_NUM_OPERANDS];
};
