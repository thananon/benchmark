/* ----------------------------------------------------------------------
   miniMD is a simple, parallel molecular dynamics (MD) code.   miniMD is
   an MD microapplication in the Mantevo project at Sandia National
   Laboratories ( http://www.mantevo.org ). The primary
   authors of miniMD are Steve Plimpton (sjplimp@sandia.gov) , Paul Crozier
   (pscrozi@sandia.gov) and Christian Trott (crtrott@sandia.gov).

   Copyright (2008) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This library is free software; you
   can redistribute it and/or modify it under the terms of the GNU Lesser
   General Public License as published by the Free Software Foundation;
   either version 3 of the License, or (at your option) any later
   version.

   This library is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with this software; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
   USA.  See also: http://www.gnu.org/licenses/lgpl.txt .

   For questions, contact Paul S. Crozier (pscrozi@sandia.gov) or
   Christian Trott (crtrott@sandia.gov).

   Please read the accompanying README and LICENSE files.
---------------------------------------------------------------------- */

#include "stdio.h"
#include "math.h"
#include "force_lj.h"
#include "openmp.h"

#ifdef __INTEL_COMPILER
#include <ia32intrin.h>
#include <xmmintrin.h>
#include <zmmintrin.h>
#endif

ForceLJ::ForceLJ()
{
  cutforce = 0.0;
  cutforcesq = 0.0;
  use_oldcompute = 0;
  reneigh = 1;
  style = FORCELJ;
  epsilon = 1.0;
  sigma6 = 1.0;
  sigma = 1.0;
  use_sse = 1;
}
ForceLJ::~ForceLJ() {}

void ForceLJ::setup()
{
  cutforcesq = cutforce * cutforce;
}


void ForceLJ::compute(Atom &atom, Neighbor &neighbor, Comm &comm, int me)
{
  eng_vdwl = 0;
  virial = 0;

  if(evflag) {
    if(use_oldcompute)
      return compute_original<1>(atom, neighbor, me);

    if(neighbor.halfneigh) {
      if(neighbor.ghost_newton) {
        if(threads->omp_num_threads > 1)
          return compute_halfneigh_threaded<1, 1>(atom, neighbor, me);
        else
          return compute_halfneigh<1, 1>(atom, neighbor, me);
      } else {
        if(threads->omp_num_threads > 1)
          return compute_halfneigh_threaded<1, 0>(atom, neighbor, me);
        else
          return compute_halfneigh<1, 0>(atom, neighbor, me);
      }      
#ifdef KNC_FORCE_INTRINSIC_SWGS
//optimised version of compute with INTRINSIC
//Ashish Jha, ashish.jha@intel.com, Intel Corporation
    } else {
	#pragma noinline 
	compute_fullneigh_intrinsic_SWGS<1>(atom, neighbor, me);
	return;
    }
#else
    } else {
	return compute_fullneigh<1>(atom, neighbor, me);
    }
#endif    
  } else {
    if(use_oldcompute)
      return compute_original<0>(atom, neighbor, me);

    if(neighbor.halfneigh) {
      if(neighbor.ghost_newton) {
        if(threads->omp_num_threads > 1)
          return compute_halfneigh_threaded<0, 1>(atom, neighbor, me);
        else
          return compute_halfneigh<0, 1>(atom, neighbor, me);
      } else {
        if(threads->omp_num_threads > 1)
          return compute_halfneigh_threaded<0, 0>(atom, neighbor, me);
        else
          return compute_halfneigh<0, 0>(atom, neighbor, me);
      }       
#ifdef KNC_FORCE_INTRINSIC_SWGS
//optimised version of compute with INTRINSIC
//Ashish Jha, ashish.jha@intel.com, Intel Corporation
    } else {
	#pragma noinline 
	compute_fullneigh_intrinsic_SWGS<0>(atom, neighbor, me);
	return;
    }
#else
    } else {
	return compute_fullneigh<0>(atom, neighbor, me);
    }
#endif
  }
}

//original version of force compute in miniMD
//  -MPI only
//  -not vectorizable
template<int EVFLAG>
void ForceLJ::compute_original(Atom &atom, Neighbor &neighbor, int me)
{
  int i, j, k, nlocal, nall, numneigh;
  MMD_float xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  MMD_float sr2, sr6, force;
  int* neighs;
  MMD_float** x, **f;

  nlocal = atom.nlocal;
  nall = atom.nlocal + atom.nghost;
  x = atom.x;
  f = atom.f;

  eng_vdwl = 0;
  virial = 0;
  // clear force on own and ghost atoms

  for(i = 0; i < nall; i++) {
    f[i][0] = 0.0;
    f[i][1] = 0.0;
    f[i][2] = 0.0;
  }

  // loop over all neighbors of my atoms
  // store force on both atoms i and j

  for(i = 0; i < nlocal; i++) {
    neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    numneigh = neighbor.numneigh[i];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];

    for(k = 0; k < numneigh; k++) {
      j = neighs[k];
      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx * delx + dely * dely + delz * delz;

      if(rsq < cutforcesq) {
        sr2 = 1.0 / rsq;
        sr6 = sr2 * sr2 * sr2 * sigma6;
        force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon;
        f[i][0] += delx * force;
        f[i][1] += dely * force;
        f[i][2] += delz * force;
        f[j][0] -= delx * force;
        f[j][1] -= dely * force;
        f[j][2] -= delz * force;

        if(EVFLAG) {
          eng_vdwl += (4.0 * sr6 * (sr6 - 1.0)) * epsilon;
          virial += (delx * delx + dely * dely + delz * delz) * force;
        }
      }
    }
  }
}


//optimised version of compute
//  -MPI only
//  -use temporary variable for summing up fi
//  -enables vectorization by:
//     -getting rid of 2d pointers
//     -use pragma simd to force vectorization of inner loop
template<int EVFLAG, int GHOST_NEWTON>
void ForceLJ::compute_halfneigh(Atom &atom, Neighbor &neighbor, int me)
{
  int* neighs;
  int tid = omp_get_thread_num();

  const int nlocal = atom.nlocal;
  const int nall = atom.nlocal + atom.nghost;
  MMD_float* x = &atom.x[0][0];
  MMD_float* f = &atom.f[0][0];

  // clear force on own and ghost atoms
  for(int i = 0; i < nall; i++) {
    f[i * PAD + 0] = 0.0;
    f[i * PAD + 1] = 0.0;
    f[i * PAD + 2] = 0.0;
  }

  // loop over all neighbors of my atoms
  // store force on both atoms i and j
  MMD_float t_energy = 0;
  MMD_float t_virial = 0;

  for(int i = 0; i < nlocal; i++) {
    neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int numneighs = neighbor.numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];

    MMD_float fix = 0.0;
    MMD_float fiy = 0.0;
    MMD_float fiz = 0.0;

#ifdef USE_SIMD
    #pragma simd reduction (+: fix,fiy,fiz)
#endif
    for(int k = 0; k < numneighs; k++) {
      const int j = neighs[k];
      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      if(rsq < cutforcesq) {
        const MMD_float sr2 = 1.0 / rsq;
        const MMD_float sr6 = sr2 * sr2 * sr2 * sigma6;
        const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon;

        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;

        if(GHOST_NEWTON || j < nlocal) {
          f[j * PAD + 0] -= delx * force;
          f[j * PAD + 1] -= dely * force;
          f[j * PAD + 2] -= delz * force;
        }

        if(EVFLAG) {
          const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? 1.0 : 0.5;
          t_energy += scale * (4.0 * sr6 * (sr6 - 1.0)) * epsilon;
          t_virial += scale * (delx * delx + dely * dely + delz * delz) * force;
        }

      }
    }

    f[i * PAD + 0] += fix;
    f[i * PAD + 1] += fiy;
    f[i * PAD + 2] += fiz;

  }

  eng_vdwl += t_energy;
  virial += t_virial;

}

//optimised version of compute
//  -MPI + OpenMP (atomics for fj update)
//  -use temporary variable for summing up fi
//  -enables vectorization by:
//    -getting rid of 2d pointers
//    -use pragma simd to force vectorization of inner loop (not currently supported due to OpenMP atomics
template<int EVFLAG, int GHOST_NEWTON>
void ForceLJ::compute_halfneigh_threaded(Atom &atom, Neighbor &neighbor, int me)
{
  int nlocal, nall;
  int* neighs;
  MMD_float* x, *f;
  int tid = omp_get_thread_num();

  MMD_float t_eng_vdwl = 0;
  MMD_float t_virial = 0;

  nlocal = atom.nlocal;
  nall = atom.nlocal + atom.nghost;
  x = &atom.x[0][0];
  f = &atom.f[0][0];

  #pragma omp barrier
  // clear force on own and ghost atoms

  OMPFORSCHEDULE
  for(int i = 0; i < nall; i++) {
    f[i * PAD + 0] = 0.0;
    f[i * PAD + 1] = 0.0;
    f[i * PAD + 2] = 0.0;
  }

  // loop over all neighbors of my atoms
  // store force on both atoms i and j

  OMPFORSCHEDULE
  for(int i = 0; i < nlocal; i++) {
    neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int numneighs = neighbor.numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];
    MMD_float fix = 0.0;
    MMD_float fiy = 0.0;
    MMD_float fiz = 0.0;

    for(int k = 0; k < numneighs; k++) {
      const int j = neighs[k];
      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      if(rsq < cutforcesq) {
        const MMD_float sr2 = 1.0 / rsq;
        const MMD_float sr6 = sr2 * sr2 * sr2 * sigma6;
        const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon;

        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;

        if(GHOST_NEWTON || j < nlocal) {
          #pragma omp atomic
          f[j * PAD + 0] -= delx * force;
          #pragma omp atomic
          f[j * PAD + 1] -= dely * force;
          #pragma omp atomic
          f[j * PAD + 2] -= delz * force;
        }

        if(EVFLAG) {
          const MMD_float scale = (GHOST_NEWTON || j < nlocal) ? 1.0 : 0.5;
          t_eng_vdwl += scale * (4.0 * sr6 * (sr6 - 1.0)) *  epsilon;
          t_virial += scale * (delx * delx + dely * dely + delz * delz) * force;
        }
      }
    }

    #pragma omp atomic
    f[i * PAD + 0] += fix;
    #pragma omp atomic
    f[i * PAD + 1] += fiy;
    #pragma omp atomic
    f[i * PAD + 2] += fiz;
  }

  #pragma omp atomic
  eng_vdwl += t_eng_vdwl;
  #pragma omp atomic
  virial += t_virial;

  #pragma omp barrier
}

//optimised version of compute
//  -MPI + OpenMP (using full neighborlists)
//  -gets rid of fj update (read/write to memory)
//  -use temporary variable for summing up fi
//  -enables vectorization by:
//    -get rid of 2d pointers
//    -use pragma simd to force vectorization of inner loop
template<int EVFLAG>
void ForceLJ::compute_fullneigh(Atom &atom, Neighbor &neighbor, int me)
{
  int nlocal, nall;
  int* neighs;
  MMD_float* x, *f;
  int tid = omp_get_thread_num();

  MMD_float t_eng_vdwl = 0;
  MMD_float t_virial = 0;
  nlocal = atom.nlocal;
  nall = atom.nlocal + atom.nghost;
  x = &atom.x[0][0];
  f = &atom.f[0][0];
  
  #pragma omp barrier
  // clear force on own and ghost atoms

  OMPFORSCHEDULE
  for(int i = 0; i < nlocal; i++) {
    f[i * PAD + 0] = 0.0;
    f[i * PAD + 1] = 0.0;
    f[i * PAD + 2] = 0.0;
  }

  // loop over all neighbors of my atoms
  // store force on atom i

  OMPFORSCHEDULE
  for(int i = 0; i < nlocal; i++) {
    neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int numneighs = neighbor.numneigh[i];
    const MMD_float xtmp = x[i * PAD + 0];
    const MMD_float ytmp = x[i * PAD + 1];
    const MMD_float ztmp = x[i * PAD + 2];
    MMD_float fix = 0;
    MMD_float fiy = 0;
    MMD_float fiz = 0;

// FAKEELSE is a temporary workaround for the compiler to get rid of mask blending  	
#ifdef FAKEELSE
int c=1;

#ifdef USE_SIMD
    #pragma simd reduction (+: fix,fiy,fiz,t_eng_vdwl,t_virial)
#endif
    for(int k = 0; k < numneighs; k++) {
      const int j = neighs[k];
      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;

      if(rsq < cutforcesq) {
        const MMD_float sr2 = 1.0 / rsq;
        const MMD_float sr6 = sr2 * sr2 * sr2 * sigma6;
        const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon;
        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;

        if(EVFLAG) {
          t_eng_vdwl += sr6 * (sr6 - 1.0) * epsilon;
          t_virial += delx * delx * force + dely * dely * force + delz * delz * force;
        }
      }

     else
     {
      c=2;
     }

    }
     f[i * PAD + 0] += fix + (c & ~3); // (c & ~3) is 0 always
     f[i * PAD + 1] += fiy;
     f[i * PAD + 2] += fiz;

#else

#ifdef USE_SIMD
    #pragma simd reduction (+: fix,fiy,fiz,t_eng_vdwl,t_virial)
#endif
    for(int k = 0; k < numneighs; k++) {
      const int j = neighs[k];
	  
      const MMD_float delx = xtmp - x[j * PAD + 0];
      const MMD_float dely = ytmp - x[j * PAD + 1];
      const MMD_float delz = ztmp - x[j * PAD + 2];
      const MMD_float rsq = delx * delx + dely * dely + delz * delz;
	  
      if(rsq < cutforcesq) {
        const MMD_float sr2 = 1.0 / rsq;
        const MMD_float sr6 = sr2 * sr2 * sr2 * sigma6;
        const MMD_float force = 48.0 * sr6 * (sr6 - 0.5) * sr2 * epsilon;
        fix += delx * force;
        fiy += dely * force;
        fiz += delz * force;

        if(EVFLAG) {
          t_eng_vdwl += sr6 * (sr6 - 1.0) * epsilon;
          t_virial += delx * delx * force + dely * dely * force + delz * delz * force;				  
        }
      }
    }
     f[i * PAD + 0] += fix;
     f[i * PAD + 1] += fiy;
     f[i * PAD + 2] += fiz;
#endif
}

  t_eng_vdwl *= 4.0;
  t_virial *= 0.5;

  #pragma omp atomic
  eng_vdwl += t_eng_vdwl;
  #pragma omp atomic
  virial += t_virial;
  #pragma omp barrier
}


#ifdef KNC_FORCE_INTRINSIC_SWGS
//optimised version of compute with INTRINSIC
//Ashish Jha, ashish.jha@intel.com, Intel Corporation

//  -MPI + OpenMP (using full neighborlists)
//  -gets rid of fj update (read/write to memory)
//  -use temporary variable for summing up fi
//  -enables vectorization by:
//    -get rid of 2d pointers
//    -use pragma simd to force vectorization of inner loop
template<int EVFLAG>
void ForceLJ::compute_fullneigh_intrinsic_SWGS(Atom &atom, Neighbor &neighbor, int me)
{
  int nlocal, nall;
  int* neighs;
  MMD_float* x, *f;
  int tid = omp_get_thread_num();

  MMD_float t_eng_vdwl = 0;
  MMD_float t_virial = 0;
  nlocal = atom.nlocal;
  nall = atom.nlocal + atom.nghost;
  x = &atom.x[0][0];
  f = &atom.f[0][0];

 __mmask8 k11 = _mm512_int2mask(0x11);
 __mmask8 k07 = _mm512_int2mask(0x07);
 __mmask8 k70 = _mm512_int2mask(0x70);
 __mmask8 k77 = _mm512_int2mask(0x77);
 __mmask8 kF0 = _mm512_int2mask(0xF0);
 __mmask8 kFF = _mm512_int2mask(0xFF);
 
__declspec(align(64)) int permMASK[16]={0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15};//{15,14,7,6,13,12,5,4,11,10,3,2,9,8,1,0};
  const double one = 1.0;
  const double zerop5 = 0.5;
  const double fortyEightp0 = 48.0;
__m512d z_cutforcesq = _mm512_set_1to8_pd(cutforcesq);
__m512d z_one = _mm512_set_1to8_pd(one);
__m512d z_0p5 = _mm512_set_1to8_pd(zerop5);
__m512d z_48p0 = _mm512_set_1to8_pd(fortyEightp0);
__m512d z_1p0 = z_one;
__m512i z_permMASKi =  _mm512_load_epi32(&permMASK[0]);
__m512d z_t_eng_vdwl = _mm512_setzero_pd();
__m512d z_t_virial_x = _mm512_setzero_pd();
__m512d z_t_virial_y = _mm512_setzero_pd();
__m512d z_t_virial_z = _mm512_setzero_pd();

  // clear force on own and ghost atoms
  #pragma omp barrier

OMPFORSCHEDULE
  for(int i = 0; i < nlocal; i++) {
    f[i * PAD + 0] = 0.0;
    f[i * PAD + 1] = 0.0;
    f[i * PAD + 2] = 0.0;
  }

OMPFORSCHEDULE
  for(int i = 0; i < nlocal; i++) {
	
    neighs = &neighbor.neighbors[i * neighbor.maxneighs];
    const int numneighs = neighbor.numneigh[i];

	__declspec(align(64)) double rsqBuf1[8]={0};
	
    const int numneighR = numneighs % 8;
    const int numneighL = numneighs - numneighR; 

	__m512d z_fx = _mm512_setzero_pd();
	__m512d z_fy = _mm512_setzero_pd();
	__m512d z_fz = _mm512_setzero_pd();
	__m512d z_ZERO = _mm512_setzero_pd();

	__m512d z_xyz_tmp = _mm512_extloadunpacklo_pd(_mm512_undefined_pd(), &x[PAD*i+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
	__m512i z_xyz_tmpi = _mm512_permute4f128_epi32(_mm512_castpd_si512(z_xyz_tmp), _MM_PERM_BABA); 
	__m512d z_xtmp = _mm512_castsi512_pd(_mm512_swizzle_epi64(z_xyz_tmpi, _MM_SWIZ_REG_AAAA));
	__m512d z_ytmp = _mm512_castsi512_pd(_mm512_swizzle_epi64(z_xyz_tmpi, _MM_SWIZ_REG_BBBB)); 
	__m512d z_ztmp = _mm512_castsi512_pd(_mm512_swizzle_epi64(z_xyz_tmpi, _MM_SWIZ_REG_CCCC)); 
	
	__m512d z_f_xyz = _mm512_extloadunpacklo_pd(_mm512_undefined_pd(), &f[PAD*i+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);		
	__m512i z_f_xyzi = _mm512_permute4f128_epi32(_mm512_castpd_si512(z_f_xyz), _MM_PERM_BABA); 
	__m512d z_f_x = _mm512_castsi512_pd(_mm512_swizzle_epi64(z_f_xyzi, _MM_SWIZ_REG_AAAA));
	__m512d z_f_y = _mm512_castsi512_pd(_mm512_swizzle_epi64(z_f_xyzi, _MM_SWIZ_REG_BBBB));
	__m512d z_f_z = _mm512_castsi512_pd(_mm512_swizzle_epi64(z_f_xyzi, _MM_SWIZ_REG_CCCC));	
	

    int k = 0;	
    for (k = 0; k < numneighL; k+=8) {
		const int j0 = neighs[k+0];
		const int j1 = neighs[k+1];
		const int j2 = neighs[k+2];
		const int j3 = neighs[k+3];
		const int j4 = neighs[k+4];
		const int j5 = neighs[k+5];
		const int j6 = neighs[k+6];
		const int j7 = neighs[k+7];

		__m512d j04_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);	
		j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		__m512d j15_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
		j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz,k70,  &x[PAD*j5+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		__m512d j26_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
		j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz,k70, &x[PAD*j6+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		__m512d j37_xyz = _mm512_mask_extloadunpacklo_pd(_mm512_undefined_pd(), k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
		j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz,k70, &x[PAD*j7+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
				
		__m512i j04_26_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j04_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j26_xyz), _MM_PERM_CCAA);
		__m512i j15_37_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j15_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j37_xyz), _MM_PERM_CCAA);
		__m512i j04_26_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j26_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j04_xyz), _MM_PERM_DDBB);
		__m512i j15_37_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j37_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j15_xyz), _MM_PERM_DDBB);

		__m512d j04152637_x =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_xyi, _mm512_int2mask(0xAA), j15_37_xyi, _MM_SWIZ_REG_CDAB));
		__m512d j04152637_y =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j15_37_xyi, _mm512_int2mask(0x55), j04_26_xyi, _MM_SWIZ_REG_CDAB));
		__m512d j04152637_z =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_zi, _mm512_int2mask(0xAA), j15_37_zi, _MM_SWIZ_REG_CDAB));

		__m512d z_delx = _mm512_sub_pd(z_xtmp, j04152637_x);		
		__m512d z_dely = _mm512_sub_pd(z_ytmp, j04152637_y);		
		__m512d z_delz = _mm512_sub_pd(z_ztmp, j04152637_z);		
		
		__m512d z_rsqx = _mm512_mul_pd(z_delx, z_delx);
		__m512d z_rsqy = _mm512_mul_pd(z_dely, z_dely);
		__m512d z_rsqz = _mm512_mul_pd(z_delz, z_delz);
				
		__m512d z_rsq = _mm512_add_pd(z_rsqx, z_rsqy);
		z_rsq = _mm512_add_pd(z_rsq, z_rsqz); 

		__mmask8 k_rsqLTcutforcesq = _mm512_cmplt_pd_mask(z_rsq,z_cutforcesq);      
		int mask_k_rsqLTcutforcesq = _mm512_mask2int(k_rsqLTcutforcesq);
	
		__m512d z_sr2 = _mm512_mask_div_pd(z_ZERO,k_rsqLTcutforcesq,z_one,z_rsq);
		__m512d z_sr6 = _mm512_mul_pd(z_sr2,z_sr2);
		z_sr6 = _mm512_mul_pd(z_sr2,z_sr6);
		
		__m512d z_sr6m0p5 = _mm512_mask_sub_pd(z_ZERO,k_rsqLTcutforcesq,z_sr6,z_0p5);
		__m512d z_force = _mm512_mul_pd(z_sr6, z_sr2);
		__m512d z_48Xsr6m0p5 = _mm512_mul_pd(z_48p0,z_sr6m0p5);
		z_force = _mm512_mul_pd(z_force,z_48Xsr6m0p5);
			
		z_fx = _mm512_fmadd_pd(z_delx, z_force, z_fx);
		z_fy = _mm512_fmadd_pd(z_dely, z_force, z_fy);
		z_fz = _mm512_fmadd_pd(z_delz, z_force, z_fz);

		if(EVFLAG) {
			__m512d z_sr6m1p0 = _mm512_mask_sub_pd(z_ZERO,k_rsqLTcutforcesq,z_sr6,z_1p0);
			z_t_eng_vdwl = _mm512_fmadd_pd(z_sr6, z_sr6m1p0, z_t_eng_vdwl);
			
			z_t_virial_x = _mm512_fmadd_pd(z_rsqx, z_force, z_t_virial_x);
			z_t_virial_y = _mm512_fmadd_pd(z_rsqy, z_force, z_t_virial_y);
			z_t_virial_z = _mm512_fmadd_pd(z_rsqz, z_force, z_t_virial_z);			
		} //if(EVFLAG)
			
    } //k full vec loop

	if(numneighR) {
	  int CMP_MASK = 0x0;
	  __m512d j04_xyz = z_ZERO;
	  __m512d j15_xyz = z_ZERO;
	  __m512d j26_xyz = z_ZERO;
	  __m512d j37_xyz = z_ZERO;	

	  if (numneighR == 1) {
		CMP_MASK = 0x01;
		const int j0 = neighs[k+0];	
		j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	  } else if (numneighR == 2) {
		CMP_MASK = 0x03;
		const int j0 = neighs[k+0];
		const int j1 = neighs[k+1];		
		j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	  } else if (numneighR == 3) {
		CMP_MASK = 0x07;
		const int j0 = neighs[k+0];
		const int j1 = neighs[k+1];
		const int j2 = neighs[k+2];
		j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	  } else if (numneighR == 4) {
		CMP_MASK = 0x0F;
		const int j0 = neighs[k+0];
		const int j1 = neighs[k+1];
		const int j2 = neighs[k+2];
		const int j3 = neighs[k+3];
		j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	  } else if (numneighR == 5) {
		CMP_MASK = 0x1F;
		const int j0 = neighs[k+0];
		const int j1 = neighs[k+1];
		const int j2 = neighs[k+2];
		const int j3 = neighs[k+3];
		const int j4 = neighs[k+4];
		j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
		j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	  } else if (numneighR == 6) {
		CMP_MASK = 0x3F;
		const int j0 = neighs[k+0];
		const int j1 = neighs[k+1];
		const int j2 = neighs[k+2];
		const int j3 = neighs[k+3];
		const int j4 = neighs[k+4];
		const int j5 = neighs[k+5];
		j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
		j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
		j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz,k70,  &x[PAD*j5+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	  } else if (numneighR == 7) {
		CMP_MASK = 0x7F;
		const int j0 = neighs[k+0];
		const int j1 = neighs[k+1];
		const int j2 = neighs[k+2];
		const int j3 = neighs[k+3];
		const int j4 = neighs[k+4];
		const int j5 = neighs[k+5];
		const int j6 = neighs[k+6];
		j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz, k07,  &x[PAD*j0+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
		j04_xyz = _mm512_mask_extloadunpacklo_pd(j04_xyz,k70,  &x[PAD*j4+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz, k07,  &x[PAD*j1+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
		j15_xyz = _mm512_mask_extloadunpacklo_pd(j15_xyz,k70,  &x[PAD*j5+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz, k07,  &x[PAD*j2+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);
		j26_xyz = _mm512_mask_extloadunpacklo_pd(j26_xyz,k70, &x[PAD*j6+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

		j37_xyz = _mm512_mask_extloadunpacklo_pd(j37_xyz, k07,  &x[PAD*j3+0], _MM_UPCONV_PD_NONE, _MM_HINT_NONE);

	  } //if k remainder loop
	
	  __m512i j04_26_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j04_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j26_xyz), _MM_PERM_CCAA);
	  __m512i j15_37_xyi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j15_xyz), _mm512_int2mask(0xF0F0), _mm512_castpd_si512(j37_xyz), _MM_PERM_CCAA);
	  __m512i j04_26_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j26_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j04_xyz), _MM_PERM_DDBB);
	  __m512i j15_37_zi = _mm512_mask_permute4f128_epi32(_mm512_castpd_si512(j37_xyz), _mm512_int2mask(0x0F0F), _mm512_castpd_si512(j15_xyz), _MM_PERM_DDBB);

	  __m512d j04152637_x =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_xyi, _mm512_int2mask(0xAA), j15_37_xyi, _MM_SWIZ_REG_CDAB));
	  __m512d j04152637_y =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j15_37_xyi, _mm512_int2mask(0x55), j04_26_xyi, _MM_SWIZ_REG_CDAB));
	  __m512d j04152637_z =  _mm512_castsi512_pd(_mm512_mask_swizzle_epi64(j04_26_zi, _mm512_int2mask(0xAA), j15_37_zi, _MM_SWIZ_REG_CDAB));		  
	  
	  __m512d z_delx = _mm512_mask_sub_pd(z_ZERO, _mm512_int2mask(CMP_MASK), z_xtmp, j04152637_x);
	  __m512d z_dely = _mm512_mask_sub_pd(z_ZERO, _mm512_int2mask(CMP_MASK), z_ytmp, j04152637_y);
	  __m512d z_delz = _mm512_mask_sub_pd(z_ZERO, _mm512_int2mask(CMP_MASK), z_ztmp, j04152637_z);
  
		__m512d z_rsqx = _mm512_mul_pd(z_delx, z_delx);
		__m512d z_rsqy = _mm512_mul_pd(z_dely, z_dely);
		__m512d z_rsqz = _mm512_mul_pd(z_delz, z_delz);
				
		__m512d z_rsq = _mm512_add_pd(z_rsqx, z_rsqy);
		z_rsq = _mm512_add_pd(z_rsq, z_rsqz); 
		
      __mmask8 k_rsqLTcutforcesq = _mm512_mask_cmplt_pd_mask(_mm512_int2mask(CMP_MASK), z_rsq,z_cutforcesq);
		
		int mask_k_rsqLTcutforcesq = _mm512_mask2int(k_rsqLTcutforcesq);
	
		__m512d z_sr2 = _mm512_mask_div_pd(z_ZERO,k_rsqLTcutforcesq,z_one,z_rsq);
		__m512d z_sr6 = _mm512_mul_pd(z_sr2,z_sr2);
		z_sr6 = _mm512_mul_pd(z_sr2,z_sr6);
		
		__m512d z_sr6m0p5 = _mm512_mask_sub_pd(z_ZERO,k_rsqLTcutforcesq,z_sr6,z_0p5);
		__m512d z_force = _mm512_mul_pd(z_sr6, z_sr2);
		__m512d z_48Xsr6m0p5 = _mm512_mul_pd(z_48p0,z_sr6m0p5);
		z_force = _mm512_mul_pd(z_force,z_48Xsr6m0p5);
			
		z_fx = _mm512_fmadd_pd(z_delx, z_force, z_fx);
		z_fy = _mm512_fmadd_pd(z_dely, z_force, z_fy);
		z_fz = _mm512_fmadd_pd(z_delz, z_force, z_fz);

		if(EVFLAG) {
			__m512d z_sr6m1p0 = _mm512_mask_sub_pd(z_ZERO,k_rsqLTcutforcesq,z_sr6,z_1p0);
			z_t_eng_vdwl = _mm512_fmadd_pd(z_sr6, z_sr6m1p0, z_t_eng_vdwl);

			z_t_virial_x = _mm512_fmadd_pd(z_rsqx, z_force, z_t_virial_x);
			z_t_virial_y = _mm512_fmadd_pd(z_rsqy, z_force, z_t_virial_y);
			z_t_virial_z = _mm512_fmadd_pd(z_rsqz, z_force, z_t_virial_z);				
		} //if(EVFLAG)
	} //if(numneighR)
	
	__m512d z_fx_tmp = _mm512_add_pd(z_fx, _mm512_swizzle_pd(z_fx, _MM_SWIZ_REG_CDAB));
	__m512d z_fy_tmp = _mm512_add_pd(z_fy, _mm512_swizzle_pd(z_fy, _MM_SWIZ_REG_CDAB)); 
	__m512d z_fz_tmp = _mm512_add_pd(z_fz, _mm512_swizzle_pd(z_fz, _MM_SWIZ_REG_CDAB)); 
	
	z_fx_tmp = _mm512_add_pd(z_fx_tmp, _mm512_swizzle_pd(z_fx_tmp, _MM_SWIZ_REG_BADC));
	z_fy_tmp = _mm512_add_pd(z_fy_tmp, _mm512_swizzle_pd(z_fy_tmp, _MM_SWIZ_REG_BADC));
	z_fz_tmp = _mm512_add_pd(z_fz_tmp, _mm512_swizzle_pd(z_fz_tmp, _MM_SWIZ_REG_BADC));
	
	__m512i z_fx_tmpi = _mm512_permute4f128_epi32(_mm512_castpd_si512(z_fx_tmp), _MM_PERM_DCDC);
	__m512i z_fy_tmpi = _mm512_permute4f128_epi32(_mm512_castpd_si512(z_fy_tmp), _MM_PERM_DCDC);
	__m512i z_fz_tmpi = _mm512_permute4f128_epi32(_mm512_castpd_si512(z_fz_tmp), _MM_PERM_DCDC);
	
	z_fx_tmp = _mm512_add_pd(z_fx_tmp, _mm512_castsi512_pd(z_fx_tmpi));
	z_fy_tmp = _mm512_add_pd(z_fy_tmp, _mm512_castsi512_pd(z_fy_tmpi));
	z_fz_tmp = _mm512_add_pd(z_fz_tmp, _mm512_castsi512_pd(z_fz_tmpi));
	
	__m512d z_fxyz = _mm512_castsi512_pd(_mm512_mask_or_epi64(_mm512_castpd_si512(z_fx_tmp), _mm512_int2mask(0x02), _mm512_castpd_si512(z_fy_tmp), _mm512_castpd_si512(z_fy_tmp))); 
	z_fxyz = _mm512_castsi512_pd(_mm512_mask_or_epi64(_mm512_castpd_si512(z_fxyz), _mm512_int2mask(0x04), _mm512_castpd_si512(z_fz_tmp), _mm512_castpd_si512(z_fz_tmp))); 
	
	_mm512_mask_extpackstorelo_pd(&f[PAD*i+0], k07, z_fxyz, _MM_DOWNCONV_PD_NONE, _MM_HINT_NONE);
  } //i loop

//  if(EVFLAG) {
	  t_eng_vdwl = _mm512_reduce_add_pd(z_t_eng_vdwl);	  
	  z_t_virial_x = _mm512_add_pd(z_t_virial_x, z_t_virial_y);
	  z_t_virial_x = _mm512_add_pd(z_t_virial_x, z_t_virial_z);
	  t_virial = _mm512_reduce_add_pd(z_t_virial_x);
	  
	  t_eng_vdwl *= 4.0;
	  t_virial *= 0.5;

  #pragma omp atomic
  eng_vdwl += t_eng_vdwl;
  #pragma omp atomic
  virial += t_virial;
//  }
  #pragma omp barrier
}

#endif
