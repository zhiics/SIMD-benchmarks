#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include <emmintrin.h>
#include <smmintrin.h>

#define WIDTH        512
#define HEIGHT       512
#define NSUBSAMPLES  4
#define NAO_SAMPLES  16

typedef struct _vec
{
    double x;
    double y;
    double z;
} vec;


typedef struct _Isect
{
    double t;
    vec    p;
    vec    n;
    int    hit; 
} Isect;

typedef struct _Sphere
{
    vec    center;
    double radius;

} Sphere;

typedef struct _Plane
{
    vec    p;
    vec    n;

} Plane;

typedef struct _Ray
{
    vec    org;
    vec    dir;
} Ray;

Sphere spheres[3];
Plane  plane;

#define SIGNMASK _mm_castsi128_pd(_mm_set1_epi64x(0x8000000000000000))
//#define SIGNMASK _mm_castsi128_pd(_mm_set1_epi64((__m64)0x8000000000000000))
#define _MM_NEGATE_(v) (_mm_xor_pd(v, SIGNMASK))
#define _MM_ABSVAL_(v) (_mm_andnot_pd(SIGNMASK, v))

static double vdot(vec v0, vec v1)
{
    return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z;
}

static void vcross(vec *c, vec v0, vec v1)
{
    
    c->x = v0.y * v1.z - v0.z * v1.y;
    c->y = v0.z * v1.x - v0.x * v1.z;
    c->z = v0.x * v1.y - v0.y * v1.x;
}

static void vnormalize(vec *c)
{
    double length = sqrt(vdot((*c), (*c)));

    if (fabs(length) > 1.0e-17) {
        c->x /= length;
        c->y /= length;
        c->z /= length;
    }
}

void
ray_sphere_intersect(Isect *isect, const Ray *ray, const Sphere *sphere)
{
    vec rs;

    rs.x = ray->org.x - sphere->center.x;
    rs.y = ray->org.y - sphere->center.y;
    rs.z = ray->org.z - sphere->center.z;

    double B = vdot(rs, ray->dir);
    double C = vdot(rs, rs) - sphere->radius * sphere->radius;
    double D = B * B - C;

    if (D > 0.0) {
        double t = -B - sqrt(D);
        
        if ((t > 0.0) && (t < isect->t)) {
            isect->t = t;
            isect->hit = 1;
            
            isect->p.x = ray->org.x + ray->dir.x * t;
            isect->p.y = ray->org.y + ray->dir.y * t;
            isect->p.z = ray->org.z + ray->dir.z * t;

            isect->n.x = isect->p.x - sphere->center.x;
            isect->n.y = isect->p.y - sphere->center.y;
            isect->n.z = isect->p.z - sphere->center.z;

            vnormalize(&(isect->n));
        }
    }
}

static inline void
ray_sphere_intersect_simd(__m128d *t, __m128d *hit,
                          __m128d *px, __m128d *py, __m128d *pz,
                          __m128d *nx, __m128d *ny, __m128d *nz,
                          const __m128d dirx, const __m128d diry, const __m128d dirz,
                          const __m128d orgx, const __m128d orgy, const __m128d orgz,
                          const Sphere *sphere)
{
    __m128d rsx = _mm_sub_pd(orgx, _mm_set1_pd(sphere->center.x));
    __m128d rsy = _mm_sub_pd(orgy, _mm_set1_pd(sphere->center.y));
    __m128d rsz = _mm_sub_pd(orgz, _mm_set1_pd(sphere->center.z));
    
    __m128d B = _mm_add_pd(_mm_mul_pd(rsx, dirx), 
                          _mm_add_pd(_mm_mul_pd(rsy, diry), _mm_mul_pd(rsz, dirz)));
    __m128d C = _mm_sub_pd(_mm_add_pd(_mm_mul_pd(rsx, rsx), 
                                     _mm_add_pd(_mm_mul_pd(rsy, rsy), _mm_mul_pd(rsz, rsz))),
                          _mm_set1_pd(sphere->radius * sphere->radius));
    __m128d D = _mm_sub_pd(_mm_mul_pd(B, B), C);
    
    __m128d cond1 = _mm_cmpgt_pd(D, _mm_set1_pd(0.0));
    if (_mm_movemask_pd(cond1)) {
        __m128d t2 = _mm_sub_pd(_MM_NEGATE_(B), _mm_sqrt_pd(D));
        __m128d cond2 = _mm_and_pd(_mm_cmpgt_pd(t2, _mm_set1_pd(0.0)), _mm_cmplt_pd(t2, *t));
        if (_mm_movemask_pd(cond2)) {
            *t = _mm_or_pd(_mm_and_pd(cond2, t2), _mm_andnot_pd(cond2, *t));
            *hit = _mm_or_pd(cond2, *hit);
            
            *px = _mm_or_pd(_mm_and_pd(cond2, _mm_add_pd(orgx, _mm_mul_pd(dirx, *t))), 
                            _mm_andnot_pd(cond2, *px));
            *py = _mm_or_pd(_mm_and_pd(cond2, _mm_add_pd(orgy, _mm_mul_pd(diry, *t))), 
                            _mm_andnot_pd(cond2, *py));
            *pz = _mm_or_pd(_mm_and_pd(cond2, _mm_add_pd(orgz, _mm_mul_pd(dirz, *t))), 
                            _mm_andnot_pd(cond2, *pz));

            *nx = _mm_or_pd(_mm_and_pd(cond2, _mm_sub_pd(*px, _mm_set1_pd(sphere->center.x))), 
                            _mm_andnot_pd(cond2, *nx));
            *ny = _mm_or_pd(_mm_and_pd(cond2, _mm_sub_pd(*py, _mm_set1_pd(sphere->center.y))), 
                            _mm_andnot_pd(cond2, *ny));
            *nz = _mm_or_pd(_mm_and_pd(cond2, _mm_sub_pd(*pz, _mm_set1_pd(sphere->center.z))), 
                            _mm_andnot_pd(cond2, *nz));
    
            __m128d lengths = _mm_sqrt_pd(_mm_add_pd(_mm_mul_pd(*nx, *nx),
                                                    _mm_add_pd(_mm_mul_pd(*ny, *ny), 
                                                               _mm_mul_pd(*nz, *nz))));
            __m128d cond3 = _mm_cmpgt_pd(_MM_ABSVAL_(lengths), _mm_set1_pd(1.0e-17));
            *nx = _mm_or_pd(_mm_and_pd(cond3, _mm_div_pd(*nx, lengths)), _mm_andnot_pd(cond3, *nx));
            *ny = _mm_or_pd(_mm_and_pd(cond3, _mm_div_pd(*ny, lengths)), _mm_andnot_pd(cond3, *ny));
            *nz = _mm_or_pd(_mm_and_pd(cond3, _mm_div_pd(*nz, lengths)), _mm_andnot_pd(cond3, *nz));
        }
    }
}


void
ray_plane_intersect(Isect *isect, const Ray *ray, const Plane *plane)
{
    double d = -vdot(plane->p, plane->n);
    double v = vdot(ray->dir, plane->n);
    
    if (fabs(v) < 1.0e-17) return;
    
    double t = -(vdot(ray->org, plane->n) + d) / v;
    
    if ((t > 0.0) && (t < isect->t)) {
        isect->t = t;
        isect->hit = 1;
        
        isect->p.x = ray->org.x + ray->dir.x * t;
        isect->p.y = ray->org.y + ray->dir.y * t;
        isect->p.z = ray->org.z + ray->dir.z * t;

        isect->n = plane->n;
    }
}

static inline void
ray_plane_intersect_simd(__m128d *t, __m128d *hit,
                         __m128d *px, __m128d *py, __m128d *pz,
                         __m128d *nx, __m128d *ny, __m128d *nz,
                         const __m128d dirx, const __m128d diry, const __m128d dirz,
                         const __m128d orgx, const __m128d orgy, const __m128d orgz,
                         const Plane *plane)
{
    __m128d d = _MM_NEGATE_(_mm_add_pd(_mm_mul_pd(_mm_set1_pd(plane->p.x), _mm_set1_pd(plane->n.x)), 
                                      _mm_add_pd(_mm_mul_pd(_mm_set1_pd(plane->p.y), 
                                                            _mm_set1_pd(plane->n.y)), 
                                                 _mm_mul_pd(_mm_set1_pd(plane->p.z), 
                                                            _mm_set1_pd(plane->n.z)))));
    __m128d v = _mm_add_pd(_mm_mul_pd(dirx, _mm_set1_pd(plane->n.x)), 
                          _mm_add_pd(_mm_mul_pd(diry, _mm_set1_pd(plane->n.y)), 
                                     _mm_mul_pd(dirz, _mm_set1_pd(plane->n.z))));
    
    __m128d cond1 = _mm_cmpgt_pd(_MM_ABSVAL_(v), _mm_set1_pd(1.0e-17));
    __m128d dp = _mm_add_pd(_mm_mul_pd(orgx, _mm_set1_pd(plane->n.x)), 
                           _mm_add_pd(_mm_mul_pd(orgy, _mm_set1_pd(plane->n.y)), 
                                      _mm_mul_pd(orgz, _mm_set1_pd(plane->n.z))));
    __m128d t2 = _mm_and_pd(cond1, _mm_div_pd(_MM_NEGATE_(_mm_add_pd(dp, d)), v));
    __m128d cond2 = _mm_and_pd(_mm_cmpgt_pd(t2, _mm_set1_pd(0.0)), _mm_cmplt_pd(t2, *t));
    if (_mm_movemask_pd(cond2)) {
        *t = _mm_or_pd(_mm_and_pd(cond2, t2), _mm_andnot_pd(cond2, *t));
        *hit = _mm_or_pd(cond2, *hit);
    
        *px = _mm_or_pd(_mm_and_pd(cond2, _mm_add_pd(orgx, _mm_mul_pd(dirx, *t))), 
                        _mm_andnot_pd(cond2, *px));
        *py = _mm_or_pd(_mm_and_pd(cond2, _mm_add_pd(orgy, _mm_mul_pd(diry, *t))), 
                        _mm_andnot_pd(cond2, *py));
        *pz = _mm_or_pd(_mm_and_pd(cond2, _mm_add_pd(orgz, _mm_mul_pd(dirz, *t))), 
                        _mm_andnot_pd(cond2, *pz));

        *nx = _mm_or_pd(_mm_and_pd(cond2, _mm_set1_pd(plane->n.x)), 
                        _mm_andnot_pd(cond2, *nx));
        *ny = _mm_or_pd(_mm_and_pd(cond2, _mm_set1_pd(plane->n.y)), 
                        _mm_andnot_pd(cond2, *ny));
        *nz = _mm_or_pd(_mm_and_pd(cond2, _mm_set1_pd(plane->n.z)), 
                        _mm_andnot_pd(cond2, *nz));
    }
}

void
orthoBasis(vec *basis, vec n)
{
    basis[2] = n;
    basis[1].x = 0.0; basis[1].y = 0.0; basis[1].z = 0.0;

    if ((n.x < 0.6) && (n.x > -0.6)) {
        basis[1].x = 1.0;
    } else if ((n.y < 0.6) && (n.y > -0.6)) {
        basis[1].y = 1.0;
    } else if ((n.z < 0.6) && (n.z > -0.6)) {
        basis[1].z = 1.0;
    } else {
        basis[1].x = 1.0;
    }

    vcross(&basis[0], basis[1], basis[2]);
    vnormalize(&basis[0]);

    vcross(&basis[1], basis[2], basis[0]);
    vnormalize(&basis[1]);
}


void ambient_occlusion(vec *col, const Isect *isect)
{
    int    i, j;
    int    ntheta = NAO_SAMPLES;
    int    nphi   = NAO_SAMPLES;
    double eps = 0.0001;
    
    vec p;
    
    p.x = isect->p.x + eps * isect->n.x;
    p.y = isect->p.y + eps * isect->n.y;
    p.z = isect->p.z + eps * isect->n.z;
    
    vec basis[3];
    orthoBasis(basis, isect->n);
    
    double occlusion = 0.0;
    __m128d occlusionx4 = _mm_set1_pd(0.0);
    
    for (j = 0; j < ntheta; j++) {
        
        assert((nphi % 2) == 0);
        for (i = 0; i < nphi; i += 2) {
            
            __m128d theta = _mm_sqrt_pd(_mm_set_pd(drand48(), drand48()));
            double phi0 = 2 * M_PI * drand48();
            double phi1 = 2 * M_PI * drand48();
            __m128d sinphi = _mm_set_pd(sin(phi0), sin(phi1));
            __m128d cosphi = _mm_set_pd(cos(phi0), cos(phi1));

            __m128d x = _mm_mul_pd(cosphi, theta);
            __m128d y = _mm_mul_pd(sinphi, theta);
            __m128d z = _mm_sqrt_pd(_mm_sub_pd(_mm_set1_pd(1.0), _mm_mul_pd(theta, theta)));
            
            // ray
            __m128d dirx = _mm_add_pd(_mm_mul_pd(x, _mm_set1_pd(basis[0].x)),
                                     _mm_add_pd(_mm_mul_pd(y, _mm_set1_pd(basis[1].x)),
                                                _mm_mul_pd(z, _mm_set1_pd(basis[2].x))));
            __m128d diry = _mm_add_pd(_mm_mul_pd(x, _mm_set1_pd(basis[0].y)),
                                     _mm_add_pd(_mm_mul_pd(y, _mm_set1_pd(basis[1].y)),
                                                _mm_mul_pd(z, _mm_set1_pd(basis[2].y))));
            __m128d dirz = _mm_add_pd(_mm_mul_pd(x, _mm_set1_pd(basis[0].z)),
                                     _mm_add_pd(_mm_mul_pd(y, _mm_set1_pd(basis[1].z)),
                                                _mm_mul_pd(z, _mm_set1_pd(basis[2].z))));
            __m128d orgx = _mm_set1_pd(p.x);
            __m128d orgy = _mm_set1_pd(p.y);
            __m128d orgz = _mm_set1_pd(p.z);
            
            // isect
            __m128d t =  _mm_set1_pd(1.0e+17);
            __m128d hit = _mm_set1_pd(0.0);
            __m128d px, py, pz;
            __m128d nx, ny, nz;
            
            ray_sphere_intersect_simd(&t, &hit, &px, &py, &pz, &nx, &ny, &nz,
                                      dirx, diry, dirz, orgx, orgy, orgz, &spheres[0]);
            ray_sphere_intersect_simd(&t, &hit, &px, &py, &pz, &nx, &ny, &nz,
                                      dirx, diry, dirz, orgx, orgy, orgz, &spheres[1]);
            ray_sphere_intersect_simd(&t, &hit, &px, &py, &pz, &nx, &ny, &nz,
                                      dirx, diry, dirz, orgx, orgy, orgz, &spheres[2]);
            ray_plane_intersect_simd (&t, &hit, &px, &py, &pz, &nx, &ny, &nz,
                                      dirx, diry, dirz, orgx, orgy, orgz, &plane);
            
            occlusionx4 = _mm_add_pd(occlusionx4, _mm_and_pd(hit, _mm_set1_pd(1.0f)));
            
        }
    }
    
    double __attribute__ ((__aligned__(16))) occlusionTmp[4];
    _mm_store_pd(occlusionTmp, occlusionx4);
    occlusion = occlusionTmp[0] + occlusionTmp[1] + occlusionTmp[2] + occlusionTmp[3];
    occlusion = (ntheta * nphi - occlusion) / (double)(ntheta * nphi);

#if DBG
    fprintf(stderr, ".2%f\n", occlusion);
#endif

    col->x = occlusion;
    col->y = occlusion;
    col->z = occlusion;
}

unsigned char
clamp(double f)
{
    
  int i = (int)(f * 255.5);

  if (i < 0) i = 0;
  if (i > 255) i = 255;

  return (unsigned char)i;
}


void
render(unsigned char *img, int w, int h, int nsubsamples)
{
    int x, y;
    int u, v;

    double *fimg = (double *)malloc(sizeof(double) * w * h * 3);
    memset((void *)fimg, 0, sizeof(double) * w * h * 3);

    for (y = 0; y < h; y++) {
        for (x = 0; x < w; x++) {

                for (v = 0; v < nsubsamples; v++) {
                for (u = 0; u < nsubsamples; u++) {
                    
                    double px = (x + (u / (double)nsubsamples) - (w / 2.0)) / (w / 2.0);
                    double py = -(y + (v / (double)nsubsamples) - (h / 2.0)) / (h / 2.0);

                    Ray ray;

                    ray.org.x = 0.0;
                    ray.org.y = 0.0;
                    ray.org.z = 0.0;

                    ray.dir.x = px;
                    ray.dir.y = py;
                    ray.dir.z = -1.0;
                    vnormalize(&(ray.dir));
                    
                    Isect isect;
                    isect.t   = 1.0e+17;
                    isect.hit = 0;
                    
                    ray_sphere_intersect(&isect, &ray, &spheres[0]);
                    ray_sphere_intersect(&isect, &ray, &spheres[1]);
                    ray_sphere_intersect(&isect, &ray, &spheres[2]);
                    ray_plane_intersect (&isect, &ray, &plane);
                    
                    if (isect.hit) {
                        vec col;
#if DBG
                        fprintf(stderr, "%d %d %d %d\t", y, x, v, u);
#endif
                        ambient_occlusion(&col, &isect);
                        
                        fimg[3 * (y * w + x) + 0] += col.x;
                        fimg[3 * (y * w + x) + 1] += col.y;
                        fimg[3 * (y * w + x) + 2] += col.z;
                    }

                }
            }

            fimg[3 * (y * w + x) + 0] /= (double)(nsubsamples * nsubsamples);
            fimg[3 * (y * w + x) + 1] /= (double)(nsubsamples * nsubsamples);
            fimg[3 * (y * w + x) + 2] /= (double)(nsubsamples * nsubsamples);
                    
            img[3 * (y * w + x) + 0] = clamp(fimg[3 *(y * w + x) + 0]);
            img[3 * (y * w + x) + 1] = clamp(fimg[3 *(y * w + x) + 1]);
            img[3 * (y * w + x) + 2] = clamp(fimg[3 *(y * w + x) + 2]);
            
        }
    }
}

void
init_scene()
{
    spheres[0].center.x = -2.0;
    spheres[0].center.y =  0.0;
    spheres[0].center.z = -3.5;
    spheres[0].radius = 0.5;
    
    spheres[1].center.x = -0.5;
    spheres[1].center.y =  0.0;
    spheres[1].center.z = -3.0;
    spheres[1].radius = 0.5;
    
    spheres[2].center.x =  1.0;
    spheres[2].center.y =  0.0;
    spheres[2].center.z = -2.2;
    spheres[2].radius = 0.5;

    plane.p.x = 0.0;
    plane.p.y = -0.5;
    plane.p.z = 0.0;

    plane.n.x = 0.0;
    plane.n.y = 1.0;
    plane.n.z = 0.0;

}

void
saveppm(const char *fname, int w, int h, unsigned char *img)
{
    FILE *fp;

    fp = fopen(fname, "wb");
    assert(fp);

    fprintf(fp, "P6\n");
    fprintf(fp, "%d %d\n", w, h);
    fprintf(fp, "255\n");
    fwrite(img, w * h * 3, 1, fp);
    fclose(fp);
}

int
main(int argc, char **argv)
{
    unsigned char *img = (unsigned char *)malloc(WIDTH * HEIGHT * 3);

    init_scene();

#ifndef DBG
    clock_t start = clock();
#endif
    render(img, WIDTH, HEIGHT, NSUBSAMPLES);
#ifndef DBG
    clock_t elapsed = clock() - start;
    printf("%.2f sec\n", ((double) elapsed)/CLOCKS_PER_SEC);
#endif

    saveppm("ao.ppm", WIDTH, HEIGHT, img); 

    return 0;
}
