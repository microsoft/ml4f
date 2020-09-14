void foo(float *d, float *a) {
  if (a[0] >= a[1])
    d[0] = a[0];
  else
    d[0] = a[1];
}
