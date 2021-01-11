#define RAND_MAX 1 // undo

struct {
  double A[3];
  double B[3];
  double _time;
} ray;

void get_ray(const int nx, const int ny,double s, double t, ray *r) {
  // vec3 rd = 0
  // vec3 offset = 0;
  double time0 = 0.0;
  double time1 = 1.0;
  double time = time0 + rand() / (RAND_MAX + 1.0) * (time1 - time0);
  r->_time = time;
  double origin[3] = {13, 2, 3};
  r->A[0] = origin[0];
  r->A[1] = origin[1];
  r->A[2] = origin[2];
  double aspect = double(nx) / double(ny);
  double theta = 20 * M_PI / 180;
  double half_height = tan(theta / 2);
  double half_width = aspect * half_height;

  lower_left_corner = origin - half_width * focus_dist * u -
                      half_height * focus_dist * v - focus_dist * w;
  /*return ray(origin + offset,
             lower_left_corner + s * horizontal + t * vertical - origin -
                 offset,
             time);*/
}

double rand() { return 1; } // undo

__kernel void run(const int nx, const int ny, const int ns,
                  __global int *output) {
  int i = get_global_id(0);
  int j = get_global_id(1);

  unsigned int seed = i * j;

  double col[3] = {0, 0, 0};
  for (int k = 0; k < ns; k += 1) {
    double u = i + rand() / (RAND_MAX + 1.0) / nx;
    double v = (ny - 1 - j) + rand() / (RAND_MAX + 1.0) / ny;
    /*
    ray r;
    get_ray(nx,ny,u, v,r);
    col += color(r, world,0);
    */
  }

  col[0] = sqrt(col[0] / ns);
  col[1] = sqrt(col[1] / ns);
  col[2] = sqrt(col[2] / ns);
  int ir = convert_int(255.99 * col[0]);
  int ig = convert_int(255.99 * col[1]);
  int ib = convert_int(255.99 * col[2]);
  int row = i * j * 3;
  output[row] = ir;
  output[row + 1] = ig;
  output[row + 2] = ib;
  // file << ir << " " << ig << " " << ib << "\n";*/
}
