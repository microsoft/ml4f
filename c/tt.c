// from https://stackoverflow.com/questions/29381117/which-exponentiation-algorithms-do-cpu-programming-languages-use

#include <math.h> /* import fmaf(), ldexpf() */

/* Like rintf(), but -0.0f -> +0.0f, and |a| must be < 2**22 */
float quick_and_dirty_rintf (float a)
{
    float cvt_magic = 0x1.800000p+23f;
    return (a + cvt_magic) - cvt_magic;
}

/* Approximate exp(a) on the interval [log(sqrt(0.5)), log(sqrt(2.0))]. */
float expf_poly (float a)
{ 
    float r;

    r =             0x1.694000p-10f;  // 1.37805939e-3
    r = fmaf (r, a, 0x1.125edcp-07f); // 8.37312452e-3
    r = fmaf (r, a, 0x1.555b5ap-05f); // 4.16695364e-2
    r = fmaf (r, a, 0x1.555450p-03f); // 1.66664720e-1
    r = fmaf (r, a, 0x1.fffff6p-02f); // 4.99999851e-1
    r = fmaf (r, a, 0x1.000000p+00f); // 1.00000000e+0
    r = fmaf (r, a, 0x1.000000p+00f); // 1.00000000e+0
    return r;
}

/* Compute exponential base e. Maximum ulp error = 0.86565 */
float my_expf (float a)
{
    float t, r;
    int i;

    t = a * 0x1.715476p+0f;            // 1/log(2); 1.442695
    t = quick_and_dirty_rintf (t);
    i = (int)t;
    r = fmaf (t, -0x1.62e400p-01f, a); // log_2_hi; -6.93145752e-1
    r = fmaf (t, -0x1.7f7d1cp-20f, r); // log_2_lo; -1.42860677e-6
    t = expf_poly (r);
    r = ldexpf (t, i);
    if (a < -105.0f) r = 0.0f;
    if (a >  105.0f) r = 1.0f/0.0f;    // +INF
    return r;
}




void foo(float *d, float *a) {
  if (a[0] >= a[1])
    d[0] = a[0];
  else
    d[0] = a[1];
}

float expf(float x);
void softmax(float *arr, unsigned len) {
  for (unsigned i = 0; i < len; ++i)
    arr[i] = my_expf(arr[i]);
}

void _exit() {}

int main() {
  float f[12];
  softmax(f,12);
  foo(f,f);
}
