#include "operands.h"

class OperandTuple {

public:
    OperandTuple(int n)
    {
        operands_ = new operandSet[n];
        index_ = new int[n];
        numOperands_ = n;

        for (int i=0; i < numOperands_; ++i)
        {
            index_[i] = 0;
            operands_[i].setOperandNum(i+1);
        }

        done_ = 0;

        numSpecializations_ = 0;
    }

    ~OperandTuple()
    {
        delete [] operands_;
        delete [] index_;
    }

    int numSpecializations() const
    { return numSpecializations_; }
  
    int operandIndex(int i) const
    { return index_[i]; }
 
    operand& operator[](int i)
    {
        return operands_[i][index_[i]];
    }

    operator int()
    {
        return !done_;
    }

    int operator++()
    {
        // This version is like increment(), but it checks to make
        // sure the operand tuple is valid.  For example, an operand
        // tuple of all scalars is not permitted, since this
        // would interfere with built-in versions of +, -, etc.
        do {
            increment();
        } while (!done_ && !isValidTuple());

        ++numSpecializations_;

        return !done_;
    }

    int increment()
    {
        for (int j=numOperands_ - 1; j >= 0; --j)
        {
            if (++index_[j] != operands_[j].numOperands())
                break;

            if (j == 0)
            {
                done_ = 1;
                index_[j] = 0;
                break;
            }

            index_[j] = 0;
        }

        return !done_;
    }

    int isValidTuple()
    {
        // Count the number of scalar operands
        int numScalars = 0;

        for (int i=0; i < numOperands_; ++i)
        {
            if (operands_[i][index_[i]].isScalar())
                ++numScalars;
        }

        if (numScalars == numOperands_)
            return 0;

        return 1;
    }

    int anyComplex()
    {
        for (int i=0; i < numOperands_; ++i)
            if ((*this)[i].isComplex())
                return 1;

        return 0;
    }

    void reset()
    {
        done_ = 0;

        for (int i=0; i < numOperands_; ++i)
            index_[i] = 0;
    }

    int numTemplates() 
    {
        int countTemplates = 0;
        for (int i=0; i < numOperands_; ++i)
            countTemplates += operands_[i][index_[i]].numTemplateParameters();
        return countTemplates;
    }

    void printTemplates(std::ostream& os)
    {
        if (!numTemplates())
            return;

        os << "template<";

        int templatesWritten = 0;

        for (int i=0; i < numOperands_; ++i)
        {
            for (int j=0; j < (*this)[i].numTemplateParameters(); ++j)
            {
                if (templatesWritten)
                    os << ", ";
                (*this)[i].printTemplateType(os, j);
                os << " ";
                (*this)[i].printTemplate(os, j);
                ++templatesWritten;
            }

        }

        os << ">";
    }

    void printTypes(std::ostream& os, int feedFlag = 0)
    {
        for (int i=0; i < numOperands_; ++i)
        {
            if (i > 0)
            {
                os << ", ";
                if (feedFlag)
                    os << std::endl << "      ";
            }

            (*this)[i].printName(os);
        }
    }

    void printIterators(std::ostream& os, int feedFlag = 0)
    {
        for (int i=0; i < numOperands_; ++i)
        {
            if (i > 0)
            {
                os << ", ";
                if (feedFlag)
                    os << std::endl << "      ";
            }

            (*this)[i].printIterator(os);
        }
    }

    void printArgumentList(std::ostream& os, int feedFlag = 0)
    {
        for (int i=0; i < numOperands_; ++i)
        {
            if (i > 0)
            {
                os << ", ";
                if (feedFlag)
                    os << std::endl << "      ";
            }

            (*this)[i].printArgument(os);
        }
    }

    void printInitializationList(std::ostream& os, int feedFlag = 0)
    {
        for (int i=0; i < numOperands_; ++i)
        {
            if (i > 0)
            {
                os << ", ";
                if (feedFlag)
                    os << std::endl << "      ";
            }

            (*this)[i].printInitialization(os);
        }
    }
 
private:
    OperandTuple() { }
    OperandTuple(const OperandTuple&) { }
    void operator=(const OperandTuple&) { };

    operandSet* operands_;
    int* index_;
    int numOperands_;
    int done_;
    int numSpecializations_;
};

