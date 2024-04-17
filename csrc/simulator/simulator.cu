#include <stdexcept>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "simulator.cuh"

namespace pim
{

    /** The number of threads per block in the CUDA kernels */
    constexpr size_t SIM_THREADS_PER_BLOCK = 256;

    /** The size of the logic operation buffer */
    constexpr size_t SIM_LOGIC_BUFFER_SIZE = 1024;

    /** Represents the current memory state */
    thrust::device_vector<dtype> memory(NUM_CROSSBARS * CROSSBAR_R * CROSSBAR_HEIGHT, 0);

    /** The latest crossbar mask */
    RangeMask crossbarMask = {0, NUM_CROSSBARS - 1, 1};
    /** The latest row mask */
    RangeMask rowMask = {0, CROSSBAR_HEIGHT - 1, 1};

    /** Represents the buffer of logic operations */
    thrust::host_vector<otype> logicBuffer(SIM_LOGIC_BUFFER_SIZE);
    /** Represents the buffer of logic operations (device memory) */
    thrust::device_vector<otype> d_logicBuffer(SIM_LOGIC_BUFFER_SIZE);
    /** The current index in the logic buffer */
    size_t logicBufferIdx = 0;

    /** The metrics collected (split according to operation type) */
    Metrics metrics[NUM_MICROOPERATION_TYPES];

    /**
     * Maps the given address to the address in the memory vector
     * @param crossbar
     * @param index
     * @param row
     * @return
     */
    __forceinline__ __host__ __device__ size_t mapAddress(size_t crossbar, size_t index, size_t row)
    {
        return crossbar * CROSSBAR_R * CROSSBAR_HEIGHT + index * CROSSBAR_HEIGHT + row;
    }

    /**
     * Generates a number with support {start, start + step, ..., stop}
     * @param start
     * @param stop
     * @param step
     * @return
     */
    __forceinline__ __host__ __device__ dtype genBitwiseMask(size_t start, size_t stop, size_t step)
    {
        return ((((uint64_t)(1) << ((stop - start) + step)) - 1) / ((1 << step) - 1)) << start;
    }

    /**
     * CUDA kernel that performs the given logic operations.
     * Each CUDA block represents a single *active* crossbar (num blocks = num activate crossbars).
     * @param operations
     * @param numOperations
     * @param currCrossbarMask
     * @param currRowMask
     * @param memory_ptr
     */
    __global__ void __logic(const otype *operations, size_t numOperations, RangeMask currCrossbarMask, RangeMask currRowMask, dtype *memory_ptr)
    {

        // Each block represents a single *active* crossbar
        size_t crossbar = currCrossbarMask.start + blockIdx.x * currCrossbarMask.step;

        // Iterate over the operations in the buffer
        for (size_t operationIdx = 0; operationIdx < numOperations; operationIdx++)
        {
            otype operation = operations[operationIdx];

            // Parse the operation
            if (operation & 0x1)
            { // Vertical logic operation
                operation >>= 1;

                // Gate type
                size_t gateType = operation & 0x3;
                operation >>= 2;

                // Input row
                size_t input = operation & CROSSBAR_HEIGHT_MASK;
                operation >>= LOG_CROSSBAR_HEIGHT;
                // Output row
                size_t output = operation & CROSSBAR_HEIGHT_MASK;
                operation >>= LOG_CROSSBAR_HEIGHT;
                // Index
                size_t index = operation & CROSSBAR_R_MASK;
                operation >>= LOG_CROSSBAR_R;

                // Perform the operation
                if (gateType == GateType::INIT0)
                {
                    memory_ptr[mapAddress(crossbar, index, output)] = 0;
                }
                else if (gateType == GateType::INIT1)
                {
                    memory_ptr[mapAddress(crossbar, index, output)] = 0xFFFFFFFF;
                }
                else if (gateType == GateType::NOT)
                {
                    memory_ptr[mapAddress(crossbar, index, output)] = memory_ptr[mapAddress(crossbar, index, output)] &
                                                                      (~memory_ptr[mapAddress(crossbar, index, input)]);
                }
            }
            else
            { // Horizontal logic operation
                operation >>= 1;

                // Gate type
                size_t gateType = operation & 0x3;
                operation >>= 2;

                // Input A (intra-partition and partition address)
                size_t inA = operation & CROSSBAR_R_MASK;
                operation >>= LOG_CROSSBAR_R;
                size_t pA = operation & CROSSBAR_N_MASK;
                operation >>= LOG_CROSSBAR_N;

                // Input B (intra-partition and partition address)
                size_t inB = operation & CROSSBAR_R_MASK;
                operation >>= LOG_CROSSBAR_R;
                size_t pB = operation & CROSSBAR_N_MASK;
                operation >>= LOG_CROSSBAR_N;

                // Output (intra-partition and partition address)
                size_t out = operation & CROSSBAR_R_MASK;
                operation >>= LOG_CROSSBAR_R;
                size_t pOut = operation & CROSSBAR_N_MASK;
                operation >>= LOG_CROSSBAR_N;

                // The pattern for the opcode repetition
                size_t pEnd = operation & CROSSBAR_N_MASK;
                operation >>= LOG_CROSSBAR_N;
                size_t pStep = operation & CROSSBAR_N_MASK;
                operation >>= LOG_CROSSBAR_N;

                // Construct a mask corresponding to the partitions containing an output
                dtype outputMask = genBitwiseMask(pOut, pEnd, pStep);

                // Iterate over the activated rows
                for (size_t i = threadIdx.x; i <= ((currRowMask.stop - currRowMask.start) / currRowMask.step); i += blockDim.x)
                {
                    size_t row = currRowMask.start + i * currRowMask.step;

                    // Perform the operation
                    if (gateType == GateType::INIT0)
                    {
                        memory_ptr[mapAddress(crossbar, out, row)] &= ~outputMask;
                    }
                    else if (gateType == GateType::INIT1)
                    {
                        memory_ptr[mapAddress(crossbar, out, row)] |= outputMask;
                    }
                    else if (gateType == GateType::NOT)
                    {
                        dtype oldVal = memory_ptr[mapAddress(crossbar, out, row)];
                        dtype newVal = ((pOut - pA) >= 0 ? (~memory_ptr[mapAddress(crossbar, inA, row)]) << (pOut - pA) : (~memory_ptr[mapAddress(crossbar, inA, row)]) >> (pA - pOut)) & oldVal;
                        memory_ptr[mapAddress(crossbar, out, row)] = (~outputMask & oldVal) | (outputMask & newVal);
                    }
                    else if (gateType == GateType::NOR)
                    {
                        dtype oldVal = memory_ptr[mapAddress(crossbar, out, row)];
                        dtype newVal = ((pOut - pA) >= 0 ? (~(memory_ptr[mapAddress(crossbar, inA, row)] | (memory_ptr[mapAddress(crossbar, inB, row)] >> (pB - pA)))) << (pOut - pA) : (~(memory_ptr[mapAddress(crossbar, inA, row)] | (memory_ptr[mapAddress(crossbar, inB, row)] >> (pB - pA)))) >> (pA - pOut)) & oldVal;
                        memory_ptr[mapAddress(crossbar, out, row)] = (~outputMask & oldVal) | (outputMask & newVal);
                    }
                }
            }

            __syncthreads();
        }
    }

    /**
     * Flushes the logic operations in the buffer
     */
    void flushLogic()
    {

        if (logicBufferIdx > 0)
        {

            // Allocate the kernel
            size_t activeCrossbars = (crossbarMask.stop - crossbarMask.start) / crossbarMask.step + 1;
            d_logicBuffer = logicBuffer;
            __logic<<<activeCrossbars, SIM_THREADS_PER_BLOCK>>>(
                thrust::raw_pointer_cast(d_logicBuffer.data()), logicBufferIdx, crossbarMask, rowMask,
                thrust::raw_pointer_cast(memory.data()));
        }
        logicBufferIdx = 0;
    }

    /**
     * Receives the given logic operation
     * @param operation
     */
    void logic(otype operation)
    {

        // Check for input correctness
        otype operationCopy = operation;
        if (operationCopy & 0x1)
        { // Vertical logic operation
            operationCopy >>= 1;

            // Gate type
            size_t gateType = operationCopy & 0x3;
            operationCopy >>= 2;

            // Input row
            size_t input = operationCopy & CROSSBAR_HEIGHT_MASK;
            operationCopy >>= LOG_CROSSBAR_HEIGHT;
            // Output row
            size_t output = operationCopy & CROSSBAR_HEIGHT_MASK;
            operationCopy >>= LOG_CROSSBAR_HEIGHT;
            // Index
            size_t index = operationCopy & CROSSBAR_R_MASK;
            operationCopy >>= LOG_CROSSBAR_R;

            if (gateType != GateType::INIT0 && gateType != GateType::INIT1 && gateType != GateType::NOT)
            {
                throw std::runtime_error("Logic operation: invalid gate type.");
            }
            if (input < 0 || input >= CROSSBAR_HEIGHT)
            {
                throw std::runtime_error("Logic operation: invalid input.");
            }
            if (output < 0 || output >= CROSSBAR_HEIGHT)
            {
                throw std::runtime_error("Logic operation: invalid output.");
            }
            if (index < 0 || index >= CROSSBAR_R)
            {
                throw std::runtime_error("Logic operation: invalid index.");
            }

            metrics[MicrooperationType::LOGIC].latency += 1;
            metrics[MicrooperationType::LOGIC].energy += ((crossbarMask.stop - crossbarMask.start) / crossbarMask.step + 1) * CROSSBAR_N;

#ifdef VERBOSE
            std::cerr << "Simulator: VLogic(" << (GateType)gateType << ", " << input << ", " << output << ", " << index << ")" << std::endl;
#endif
        }
        else
        { // Horizontal logic operation
            operationCopy >>= 1;

            // Gate type
            size_t gateType = operationCopy & 0x3;
            operationCopy >>= 2;

            // Input A (intra-partition and partition address)
            size_t inA = operationCopy & CROSSBAR_R_MASK;
            operationCopy >>= LOG_CROSSBAR_R;
            size_t pA = operationCopy & CROSSBAR_N_MASK;
            operationCopy >>= LOG_CROSSBAR_N;

            // Input B (intra-partition and partition address)
            size_t inB = operationCopy & CROSSBAR_R_MASK;
            operationCopy >>= LOG_CROSSBAR_R;
            size_t pB = operationCopy & CROSSBAR_N_MASK;
            operationCopy >>= LOG_CROSSBAR_N;

            // Output (intra-partition and partition address)
            size_t out = operationCopy & CROSSBAR_R_MASK;
            operationCopy >>= LOG_CROSSBAR_R;
            size_t pOut = operationCopy & CROSSBAR_N_MASK;
            operationCopy >>= LOG_CROSSBAR_N;

            // The pattern for the opcode repetition
            size_t pEnd = operationCopy & CROSSBAR_N_MASK;
            operationCopy >>= LOG_CROSSBAR_N;
            size_t pStep = operationCopy & CROSSBAR_N_MASK;
            operationCopy >>= LOG_CROSSBAR_N;

            if (gateType != GateType::INIT0 && gateType != GateType::INIT1 && gateType != GateType::NOT && gateType != GateType::NOR)
            {
                throw std::runtime_error("Logic operation: invalid gate type.");
            }
            if (inA < 0 || inA >= CROSSBAR_R ||
                pA < 0 || pA >= CROSSBAR_N)
            {
                throw std::runtime_error("Logic operation: invalid input A.");
            }
            if (inB < 0 || inB >= CROSSBAR_R ||
                pB < 0 || pB >= CROSSBAR_N)
            {
                throw std::runtime_error("Logic operation: invalid input B.");
            }
            if (out < 0 || out >= CROSSBAR_R ||
                pOut < 0 || pOut >= CROSSBAR_N)
            {
                throw std::runtime_error("Logic operation: invalid output.");
            }
            if (pEnd < pOut || pEnd >= CROSSBAR_N ||
                pStep <= 0 || (pEnd - pOut) % pStep != 0)
            {
                throw std::runtime_error("Logic operation: invalid pattern.");
            }
            if (pA > pB)
            {
                throw std::runtime_error("Logic operation: the partition of input A should be to the left of that of input B.");
            }

            metrics[MicrooperationType::LOGIC].latency += 1;
            metrics[MicrooperationType::LOGIC].energy += ((crossbarMask.stop - crossbarMask.start) / crossbarMask.step + 1) *
                      ((rowMask.stop - rowMask.start) / rowMask.step + 1) * ((pEnd - pOut) / pStep + 1);
        }

        // Add the operation to the buffer
        logicBuffer[logicBufferIdx++] = operation;
        if (logicBufferIdx == SIM_LOGIC_BUFFER_SIZE)
            flushLogic();
    }

    /**
     * Sets the current crossbar mask
     * @param operation
     */
    void setCrossbarMask(otype operation)
    {

        // Start, stop, step
        size_t start = operation & NUM_CROSSBARS_MASK;
        operation >>= LOG_NUM_CROSSBARS;
        size_t stop = operation & NUM_CROSSBARS_MASK;
        operation >>= LOG_NUM_CROSSBARS;
        size_t step = operation & NUM_CROSSBARS_MASK;
        operation >>= LOG_NUM_CROSSBARS;

        if (start == stop)
            step = 1;

        // Check for input correctness
        if (start < 0 || start >= NUM_CROSSBARS)
        {
            throw std::runtime_error("Set Crossbar Mask: invalid crossbar start.");
        }
        if (stop < 0 || stop >= NUM_CROSSBARS)
        {
            throw std::runtime_error("Set Crossbar Mask: invalid crossbar stop.");
        }
        if (start > stop || step <= 0 || (stop - start) % step != 0)
        {
            throw std::runtime_error("Set Crossbar Mask: invalid crossbar mask.");
        }

#ifdef VERBOSE
        std::cerr << "Simulator: CrossbarMask(" << start << ", " << stop << ", " << step << ")" << std::endl;
#endif

        metrics[MicrooperationType::MASK].latency += 1;

        flushLogic();
        crossbarMask = {start, stop, step};

    }

    /**
     * Sets the current row mask
     * @param operation
     */
    void setRowMask(otype operation)
    {

        // Start, stop, step
        size_t start = operation & CROSSBAR_HEIGHT_MASK;
        operation >>= LOG_CROSSBAR_HEIGHT;
        size_t stop = operation & CROSSBAR_HEIGHT_MASK;
        operation >>= LOG_CROSSBAR_HEIGHT;
        size_t step = operation & CROSSBAR_HEIGHT_MASK;
        operation >>= LOG_CROSSBAR_HEIGHT;

        if (start == stop)
            step = 1;

        // Check for input correctness
        if (start < 0 || start >= CROSSBAR_HEIGHT)
        {
            throw std::runtime_error("Set Row Mask: invalid row start.");
        }
        if (stop < 0 || stop >= CROSSBAR_HEIGHT)
        {
            throw std::runtime_error("Set Row Mask: invalid row stop.");
        }
        if (start > stop || step <= 0 || (stop - start) % step != 0)
        {
            throw std::runtime_error("Set Row Mask: invalid row mask.");
        }

#ifdef VERBOSE
        std::cerr << "Simulator: RowMask(" << start << ", " << stop << ", " << step << ")" << std::endl;
#endif

        metrics[MicrooperationType::MASK].latency += 1;

        flushLogic();
        rowMask = {start, stop, step};

    }

    /**
     * Performs a mask operation
     * @param operation
     */
    void mask(otype operation)
    {

        if (operation & 0x1)
        {
            setRowMask(operation >> 1);
        }
        else
        {
            setCrossbarMask(operation >> 1);
        }
    }

    /**
     * Performs a read operation
     * @param operation
     * @return
     */
    dtype read(otype operation)
    {

        // Index
        size_t index = (size_t)operation;

        // Verify that only a single row in a single crossbar is selected
        if ((crossbarMask.start != crossbarMask.stop) || (rowMask.start != rowMask.stop))
        {
            throw std::runtime_error("Read operation: multiple rows selected.");
        }

        // Verify that a valid index is provided
        if (index < 0 || index >= CROSSBAR_R)
        {
            throw std::runtime_error("Read operation: invalid index.");
        }

#ifdef VERBOSE
        std::cerr << "Simulator: Read(" << index << ")" << std::endl;
#endif

        metrics[MicrooperationType::READ_WRITE].latency += 1;
        metrics[MicrooperationType::READ_WRITE].energy += CROSSBAR_N;

        // Access the selected row
        flushLogic();
        return memory[mapAddress(crossbarMask.start, index, rowMask.start)];
    }

    /**
     * CUDA kernel that writes to several rows
     * Each CUDA block represents a single *active* crossbar (num blocks = num activate crossbars).
     * @param index
     * @param data
     * @param currCrossbarMask
     * @param currRowMask
     * @param memory_ptr
     */
    __global__ void __writeMulti(size_t index, size_t data, RangeMask currCrossbarMask, RangeMask currRowMask, dtype *memory_ptr)
    {

        // Each block represents a single *active* crossbar
        size_t crossbar = currCrossbarMask.start + blockIdx.x * currCrossbarMask.step;

        // Iterate over the activated rows
        for (size_t i = threadIdx.x; i <= ((currRowMask.stop - currRowMask.start) / currRowMask.step); i += blockDim.x)
        {
            size_t row = currRowMask.start + i * currRowMask.step;

            // Perform the operation
            memory_ptr[mapAddress(crossbar, index, row)] = data;
        }
    }

    /**
     * Performs a write operation
     * @param operation
     * @return
     */
    void write(otype operation)
    {

        // Index, data
        size_t index = operation & CROSSBAR_R_MASK;
        operation >>= LOG_CROSSBAR_R;
        size_t data = (size_t)operation;

        // Verify that a valid index is provided
        if (index < 0 || index >= CROSSBAR_R)
        {
            throw std::runtime_error("Write operation: invalid index.");
        }

#ifdef VERBOSE
        std::cerr << "Simulator: Write(" << index << ", " << data << ")" << std::endl;
#endif

        flushLogic();

        // If more than a single row is selected, use __writeMulti
        if ((crossbarMask.start != crossbarMask.stop) || (rowMask.start != rowMask.stop))
        {

            metrics[MicrooperationType::READ_WRITE].latency += 1;
            metrics[MicrooperationType::READ_WRITE].energy += (((crossbarMask.stop - crossbarMask.start) / crossbarMask.step + 1) *
                      ((rowMask.stop - rowMask.start) / rowMask.step + 1)) * CROSSBAR_N;

            // Allocate the kernel
            size_t activeCrossbars = (crossbarMask.stop - crossbarMask.start) / crossbarMask.step + 1;
            __writeMulti<<<activeCrossbars, SIM_THREADS_PER_BLOCK>>>(index, data, crossbarMask, rowMask, thrust::raw_pointer_cast(memory.data()));
        }
        // Otherwise, write directly
        else
        {

            metrics[MicrooperationType::READ_WRITE].latency += 1;
            metrics[MicrooperationType::READ_WRITE].energy += CROSSBAR_N;

            // Access the selected row
            memory[mapAddress(crossbarMask.start, index, rowMask.start)] = data;
        }
    }

    /**
     * Performs either a read or a write operation
     */
    dtype readWrite(otype operation)
    {

        if (operation & 0x1)
        {
            write(operation >> 1);
            return 0;
        }
        else
        {
            return read(operation >> 1);
        }
    }

    /**
     * CUDA kernel that performs parallel data movement
     * Each CUDA block represents a single *active* source crossbar (num blocks = num activate crossbars).
     * @param srcRow
     * @param srcIndex
     * @param dstRow
     * @param dstIndex
     * @param crossbarDistance
     * @param currCrossbarMask
     * @param memory_ptr
     */
    __global__ void __move(size_t srcRow, size_t srcIndex, size_t dstRow, size_t dstIndex, size_t crossbarDistance,
                           RangeMask currCrossbarMask, dtype *memory_ptr)
    {

        // Each block represents a single *active* crossbar
        size_t srcCrossbar = currCrossbarMask.start + blockIdx.x * currCrossbarMask.step;
        size_t dstCrossbar = srcCrossbar + crossbarDistance;

        memory_ptr[mapAddress(dstCrossbar, dstIndex, dstRow)] = memory_ptr[mapAddress(srcCrossbar, srcIndex, srcRow)];
    }

    /**
     * Performs an parallel move operation
     */
    void move(otype operation)
    {

        // Source row
        size_t srcRow = operation & CROSSBAR_HEIGHT_MASK;
        operation >>= LOG_CROSSBAR_HEIGHT;
        // Source index
        size_t srcIndex = operation & CROSSBAR_R_MASK;
        operation >>= LOG_CROSSBAR_R;

        // Destination row
        size_t dstRow = operation & CROSSBAR_HEIGHT_MASK;
        operation >>= LOG_CROSSBAR_HEIGHT;
        // Destination index
        size_t dstIndex = operation & CROSSBAR_R_MASK;
        operation >>= LOG_CROSSBAR_R;

        // Crossbar distance
        size_t dstCrossbar = operation & NUM_CROSSBARS_MASK;
        operation >>= LOG_NUM_CROSSBARS;
        size_t distance = dstCrossbar - crossbarMask.start;

        if (srcRow < 0 || srcRow >= CROSSBAR_HEIGHT)
        {
            throw std::runtime_error("Move operation: invalid source row.");
        }
        if (srcIndex < 0 || srcIndex >= CROSSBAR_R)
        {
            throw std::runtime_error("Move operation: invalid source index.");
        }

        if (dstRow < 0 || dstRow >= CROSSBAR_HEIGHT)
        {
            throw std::runtime_error("Move operation: invalid destination row.");
        }
        if (dstIndex < 0 || dstIndex >= CROSSBAR_R)
        {
            throw std::runtime_error("Move operation: invalid destination index.");
        }

        if (distance <= -NUM_CROSSBARS || distance >= NUM_CROSSBARS)
        {
            throw std::runtime_error("Move operation: invalid distance.");
        }

        metrics[MicrooperationType::MOVE].latency += 1;
        metrics[MicrooperationType::MOVE].energy += ((crossbarMask.stop - crossbarMask.start) / crossbarMask.step + 1) * CROSSBAR_N;

        flushLogic();
        size_t activeCrossbars = (crossbarMask.stop - crossbarMask.start) / crossbarMask.step + 1;
        __move<<<activeCrossbars, SIM_THREADS_PER_BLOCK>>>(srcRow, srcIndex, dstRow, dstIndex, distance,
                                                           crossbarMask, thrust::raw_pointer_cast(memory.data()));
    }

    dtype perform(otype operation)
    {

        // Switch according to the operation type
        switch (operation & 0x3)
        {

        case MicrooperationType::MASK:
            mask(operation >> 2);
            return 0;

        case MicrooperationType::READ_WRITE:
            return readWrite(operation >> 2);

        case MicrooperationType::LOGIC:
            logic(operation >> 2);
            return 0;

        case MicrooperationType::MOVE:
            move(operation >> 2);
            return 0;

        default:
            throw std::runtime_error("Perform: invalid operation type.");
        }
    }

    void resetMetrics(MicrooperationType type){
        metrics[type] = Metrics();
    }

    Metrics getMetrics(MicrooperationType type){
        return metrics[type];
    }

}