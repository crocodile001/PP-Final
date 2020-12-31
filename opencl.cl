#define RAND_MAX 4611686014132420610

unsigned int rand_generator(unsigned int *seed) {
  unsigned int x = *seed;
  x ^= (x << 13);
  x ^= (x >> 17);
  x ^= (x << 4);
  *seed = x;
  return x;
}

__kernel void run(const int nx, const int ny, const int ns,
                  __global int *output) {
  int i = get_global_id(0);
  int j = ny - 1 - get_global_id(1);

  /*vec3 col(0, 0, 0);
 for (int k = 0; k < ns; k += 1) {
   double u = double(i + (double)rand() / (RAND_MAX + 1.0)) / double(nx);
   double v = double(j + (double)rand() / (RAND_MAX + 1.0)) / double(ny);
   ray r = cam.get_ray(u, v);
   col += color(r, world, 0);
 }

 col /= double(ns);
 col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
 int ir = int(255.99 * col[0]);
 int ig = int(255.99 * col[1]);
 int ib = int(255.99 * col[2]);
 row = i * get_global_id(1) * 3;
 output[row] = ir;
 output[row + 1] = ig;
 output[row + 2] = ib;
 // file << ir << " " << ig << " " << ib << "\n";*/
}
