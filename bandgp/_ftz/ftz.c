#include <xmmintrin.h>
/* set flush-to-zero (bit 15) and denormals-are-zero (bit 6) in MXCSR for the calling thread */
void set_ftz_daz(void) { _mm_setcsr(_mm_getcsr() | 0x8040); }
