#ifndef CUDAPIM_DRIVER_H
#define CUDAPIM_DRIVER_H

#include "constants.h"

namespace pim
{

    /**
     * Read macro-instruction
     * @param crossbar
     * @param row
     * @param reg
     * @return
     */
    template <class T>
    T read(size_t crossbar, size_t row, size_t reg);

    /**
     * Write macro-instruction
     * @param crossbar
     * @param row
     * @param reg
     * @param data
     * @return
     */
    template <class T>
    void write(size_t crossbar, size_t row, size_t reg, T data);

    /**
     * Write macro-instruction to potentially several rows
     * @param crossbars
     * @param rows
     * @param reg
     * @param data
     */
    template <class T>
    void writeMulti(RangeMask crossbars, RangeMask rows, size_t reg, T data);

    /**
     * Performs addition on the given registers
     * @param regX
     * @param regY
     * @param regZ
     * @param crossbars
     * @param rows
     */
    template <class T>
    void add(size_t regX, size_t regY, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs negation on the given register
     * @param regX
     * @param regZ
     * @param crossbars
     * @param rows
     */
    template <class T>
    void negate(size_t regX, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs absolute value on the given register
     * @param regX
     * @param regZ
     * @param crossbars
     * @param rows
     */
    template <class T>
    void absolute(size_t regX, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs subtraction on the given registers
     * @param regX
     * @param regY
     * @param regZ
     * @param crossbars
     * @param rows
     */
    template <class T>
    void subtract(size_t regX, size_t regY, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs multiplication on the given registers
     * @param regX
     * @param regY
     * @param regZ
     * @param crossbars
     * @param rows
     */
    template <class T>
    void multiply(size_t regX, size_t regY, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs division on the given registers
     * @param regX
     * @param regY
     * @param regZ
     * @param crossbars
     * @param rows
     */
    template <class T>
    void divide(size_t regX, size_t regY, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs modulo division on the given registers
     * @param regX
     * @param regY
     * @param regZ
     * @param crossbars
     * @param rows
     */
    template <class T>
    void modulo(size_t regX, size_t regY, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Returns the sign of the given register
     * @param regX
     * @param regZ
     * @param crossbars
     * @param rows
     */
    template <class T>
    void sign(size_t regX, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Returns whether the given register is all-zero
     * @param regX
     * @param regZ
     * @param crossbars
     * @param rows
     */
    template <class T>
    void zero(size_t regX, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs bitwise NOT on the given register
     * @param regX
     * @param regZ
     * @param crossbars
     * @param rows
     */
    void bitwiseNot(size_t regX, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs bitwise AND on the given registers
     * @param regX
     * @param regY
     * @param regZ
     * @param crossbars
     * @param rows
     */
    void bitwiseAnd(size_t regX, size_t regY, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs bitwise XOR on the given registers
     * @param regX
     * @param regY
     * @param regZ
     * @param crossbars
     * @param rows
     */
    void bitwiseXor(size_t regX, size_t regY, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs bitwise OR on the given registers
     * @param regX
     * @param regY
     * @param regZ
     * @param crossbars
     * @param rows
     */
    void bitwiseOr(size_t regX, size_t regY, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs bitwise mux on the given registers according to z = mux_c(x, y).
     * @param regC
     * @param regX
     * @param regY
     * @param regZ
     * @param crossbars
     * @param rows
     */
    void bitwiseMux(size_t regC, size_t regX, size_t regY, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Performs copy on the given registers
     * @param regX
     * @param regZ
     * @param crossbars
     * @param rows
     */
    void copy(size_t regX, size_t regZ, RangeMask crossbars, RangeMask rows);

    /**
     * Parallel move operation
     * @param srcRow
     * @param srcReg
     * @param dstRow
     * @param dstReg
     * @param crossbarDistance
     * @param crossbars
     */
    void move(size_t srcRow, size_t srcReg, size_t dstRow, size_t dstReg, size_t crossbarDistance, RangeMask crossbars);

    /**
     * Returns the number of threads in a warp
     * @return
     */
    size_t warpSize();

    /**
     * Returns the exponent base of the warp cluster sizes
     * @return
     */
    size_t warpClusterBase();

    /**
     * Returns the number of warps
     * @return
     */
    size_t numWarps();

    /**
     * Returns the number of registers per warp
     * @return
     */
    size_t numRegs();

}

#endif // CUDAPIM_DRIVER_H
