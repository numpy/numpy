#include <iostream>
#include <string>
#include <fstream>

class operand {
public:
    operand(const char* name, const char* templateName, const char* iterator, 
        const char* numtype, const char* initialization)
      : name_(name), template_(templateName), iterator_(iterator), 
        numtype_(numtype), initialization_(initialization)
    { }

    virtual int haveTemplate() const
    { return template_ != 0; }

    void setOperandNum(int i) 
    { operandNum_ = i; }

    virtual void printName(std::ostream& os)
    { 
        os << name_;           
        printTemplateList(os);
    }

    virtual void printTemplate(std::ostream& os, int n=0)
    {
        if (haveTemplate())
            os << template_ << operandNum_;
    }

    virtual void printTemplateType(std::ostream& os, int n=0)
    {
        if (haveTemplate())
            os << "class";
    }

    virtual int numTemplateParameters() const
    {
        if (haveTemplate())
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

    virtual int isRange() const
    { return 0; }

    virtual int passByReference() const
    { return 0; }

    virtual void printArgument(std::ostream& os)
    {
        if (passByReference())
            os << "const ";
        printName(os);
        if (passByReference())
            os << "&";

        os << " d" << operandNum_;      
    }

    virtual void printIterator(std::ostream& os)
    {
        os << iterator_;
        printTemplateList(os);
    }

    virtual void printTemplateList(std::ostream& os)
    {
        if (haveTemplate())
        {
            os << "<";
            for (int i=0; i < numTemplateParameters(); ++i)
            {
                // printTemplateType(os,i);
                // os << " ";
                printTemplate(os,i);
                if (i != numTemplateParameters() - 1)
                    os << ", ";
            }
            os << ">";
        }
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
    const char* template_;
    const char* iterator_;
    const char* numtype_;
    const char* initialization_;
    int operandNum_;
};

class ArrayOperand : public operand {
public:
    ArrayOperand()
      : operand("Array", 0, "ArrayIterator", "T_numtype", ".begin()")
    { }

    virtual void printTemplate(std::ostream& os, int n)
    {
        switch(n)
        {
            case 0: os << "T_numtype" << operandNum_; break;
            case 1: os << "N_rank" << operandNum_; break;
        }
    }

    virtual int passByReference() const
    { return 1; }

    virtual void printTemplateType(std::ostream& os, int n=0)
    {
        switch(n)
        {
            case 0: os << "class"; break;
            case 1: os << "int"; break;
        }
    }

    virtual int numTemplateParameters() const
    {
        return 2;
    }

    virtual int haveTemplate() const
    { return 1; }
};

class ArrayExprOperand : public operand {
public:
    ArrayExprOperand()
      : operand("_bz_ArrayExpr", "P_expr", "_bz_ArrayExpr",
           0, 0)
    { }

    virtual void printNumtype(std::ostream& os)
    {
        os << "typename P_expr" << operandNum_ << "::T_numtype";
    }
};

class IndexOperand : public operand {
public:
    IndexOperand()
      : operand("IndexPlaceholder", "N_index", "IndexPlaceholder", 0, 0)
    { }

    virtual void printNumtype(std::ostream& os)
    {
        os << "int";
    }

    virtual void printTemplateType(std::ostream& os, int n=0)
    {
        if (haveTemplate())
            os << "int";
    }
};

class ScalarOperand : public operand {
public:
    ScalarOperand(const char* name)
      : operand(name, 0, 0, name, 0)
    { }

    virtual void printIterator(std::ostream& os)
    {
        os << "_bz_ArrayExprConstant<" << name_ << ">";
    }

    virtual void printInitialization(std::ostream& os)
    {
        os << "_bz_ArrayExprConstant<" << name_ << ">(d"
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

class ComplexOperand : public operand {
public:
    ComplexOperand()
      : operand("complex", "T", 0, "complex", 0)
    { }

    virtual int isComplex() const
    { return 1; }

    virtual void printIterator(std::ostream& os)
    {
        os << "_bz_ArrayExprConstant<";
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
        printTemplate(os);
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

#define BZ_ARREXPR_NUM_OPERANDS 8

class operandSet {
public:
    operandSet()
    {
        operands_[0] = new ArrayOperand;
        operands_[1] = new ArrayExprOperand;
        operands_[2] = new IndexOperand;
        operands_[3] = new ScalarOperand("int");
        operands_[4] = new ScalarOperand("float");
        operands_[5] = new ScalarOperand("double");
        operands_[6] = new ScalarOperand("long double");
        operands_[7] = new ComplexOperand;
    }

    ~operandSet()
    {
        for (int i=0; i < BZ_ARREXPR_NUM_OPERANDS; ++i)
            delete operands_[i];
    }

    operand& operator[](int i) 
    { return *operands_[i]; }

    int numOperands() const
    { return BZ_ARREXPR_NUM_OPERANDS; }

    void setOperandNum(int num)
    {
        for (int i=0; i < BZ_ARREXPR_NUM_OPERANDS; ++i)
            operands_[i]->setOperandNum(num);
    }

private:
    operandSet(const operandSet&);
    void operator=(const operandSet&);

    operand * operands_[BZ_ARREXPR_NUM_OPERANDS];
};


