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

    int haveTemplate() const
    { return template_ != 0; }

    void setOperandNum(int i) 
    { operandNum_ = i; }

    virtual void printName(std::ostream& os)
    { 
        os << name_;           
        if (haveTemplate())
        {
            os << "<" << template_ << operandNum_ << ">";
        }
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

    virtual void printArgument(std::ostream& os)
    {
        printName(os);
        os << " d" << operandNum_;      
    }

    virtual void printIterator(std::ostream& os)
    {
        os << iterator_;

        if (haveTemplate())
            os << "<" << template_ << operandNum_ << ">";
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

class VecOperand : public operand {
public:
    VecOperand()
      : operand("Vector", "P_numtype", "VectorIterConst",
           "P_numtype", ".beginFast()")
    { }

    virtual void printArgument(std::ostream& os)
    {
        os << "const ";
        printName(os);
        os << "& d" << operandNum_;
    }

};

class TinyVecOperand : public operand {
public:
    TinyVecOperand()
      : operand("TinyVector", "P_numtype", "TinyVectorIterConst",
           "P_numtype", ".beginFast()")
    { }

    virtual void printArgument(std::ostream& os)
    {
        os << "const ";
        printName(os);
        os << "& d" << operandNum_;
    }

    virtual void printName(std::ostream& os)
    {
        os << "TinyVector<P_numtype" << operandNum_
           << ", N_length" << operandNum_ << ">";
    }

    virtual void printTemplate(std::ostream& os, int n)
    {
        if (n == 0)
        {
            os << "P_numtype" << operandNum_;
        }
        else if (n == 1)
        {
            os << "N_length" << operandNum_;
        }
    }

    virtual void printTemplateType(std::ostream& os, int n)
    {
        if (n == 0)
            os << "class";
        else if (n == 1)
            os << "int";
    }

    virtual int numTemplateParameters() const
    {
        return 2;
    }

    virtual void printIterator(std::ostream& os)
    {
        os << "TinyVectorIterConst<P_numtype" << operandNum_
           << ", N_length" << operandNum_ << ">";
    }


};


class VecPickOperand : public operand  {
public:
    VecPickOperand()
      : operand("VectorPick", "P_numtype", "VectorPickIterConst",
            "P_numtype", ".beginFast()")
    { }

    virtual void printArgument(std::ostream& os)
    {
        os << "const ";
        printName(os);
        os << "& d" << operandNum_;
    }
};

class VecExprOperand : public operand {
public:
    VecExprOperand()
      : operand("_bz_VecExpr", "P_expr", "_bz_VecExpr",
           0, 0)
    { }

    virtual void printNumtype(std::ostream& os)
    {
        os << "typename P_expr" << operandNum_ << "::T_numtype";
    }
};

class RangeOperand : public operand {
public:
    RangeOperand()
      : operand("Range", 0, "Range", 0, 0)
    { }

    virtual void printNumtype(std::ostream& os)
    {
        os << "int";
    }

    virtual int isRange() const
    { return 1; }
};

class ScalarOperand : public operand {
public:
    ScalarOperand(const char* name)
      : operand(name, 0, 0, name, 0)
    { }

    virtual void printIterator(std::ostream& os)
    {
        os << "_bz_VecExprConstant<" << name_ << ">";
    }

    virtual void printInitialization(std::ostream& os)
    {
        os << "_bz_VecExprConstant<" << name_ << ">(d"
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
        os << "_bz_VecExprConstant<";
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

#define BZ_VECEXPR_NUM_OPERANDS 10

class operandSet {
public:
    operandSet()
    {
        operands_[0] = new VecOperand;
        operands_[1] = new VecExprOperand;
        operands_[2] = new VecPickOperand;
        operands_[3] = new RangeOperand;
        operands_[4] = new TinyVecOperand;
        operands_[5] = new ScalarOperand("int");
        operands_[6] = new ScalarOperand("float");
        operands_[7] = new ScalarOperand("double");
        operands_[8] = new ScalarOperand("long double");
        operands_[9] = new ComplexOperand;

//        operands_[8] = new ScalarOperand("complex<float> ");
//        operands_[9] = new ScalarOperand("complex<double> ");
//        operands_[10] = new ScalarOperand("complex<long double> ");
    }

    ~operandSet()
    {
        for (int i=0; i < BZ_VECEXPR_NUM_OPERANDS; ++i)
            delete operands_[i];
    }

    operand& operator[](int i) 
    { return *operands_[i]; }

    int numOperands() const
    { return BZ_VECEXPR_NUM_OPERANDS; }

    void setOperandNum(int num)
    {
        for (int i=0; i < BZ_VECEXPR_NUM_OPERANDS; ++i)
            operands_[i]->setOperandNum(num);
    }

private:
    operandSet(const operandSet&);
    void operator=(const operandSet&);

    operand * operands_[BZ_VECEXPR_NUM_OPERANDS];
};


