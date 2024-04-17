#ifndef CUDAPIM_SIMULATOR_H
#define CUDAPIM_SIMULATOR_H

#include "constants.h"

namespace pim{

    /**
     * Performs the given micro-operation
     */
    dtype perform(otype operation);

    /**
     * Reset the metrics for the given operation type
    */
    void resetMetrics(MicrooperationType type);

    /**
     * Returns the metrics for the given operation type
    */
    Metrics getMetrics(MicrooperationType type);


}

#endif // CUDAPIM_SIMULATOR_H
