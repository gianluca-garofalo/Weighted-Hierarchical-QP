#ifndef _Options_
#define _Options_

#ifndef TESTS
    #define TESTS true
#endif

#ifndef DEBUG
    #define DEBUG   true
    #define LOGFILE "hqp.log"
#endif

// Performance tuning options
#ifndef HQP_MAX_ITERATIONS
    #define HQP_MAX_ITERATIONS 2000
#endif

#ifndef HQP_TIMEOUT_SECONDS
    #define HQP_TIMEOUT_SECONDS 30.0
#endif

#ifndef HQP_ANTI_CYCLING_BUFFER_SIZE
    #define HQP_ANTI_CYCLING_BUFFER_SIZE 50
#endif

#ifndef HQP_STAGNATION_THRESHOLD
    #define HQP_STAGNATION_THRESHOLD 10
#endif

#ifndef HQP_ADAPTIVE_TOLERANCE_FACTOR
    #define HQP_ADAPTIVE_TOLERANCE_FACTOR 1e-12
#endif

#endif  // _Options_
