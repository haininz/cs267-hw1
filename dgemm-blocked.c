#include <immintrin.h>
#include <string.h>
#include <stdio.h>

const char* dgemm_desc = "Simple blocked dgemm.";

#define BLOCK_L1 256
#define BLOCK_L2 512

#define min(a,b) (((a)<(b))?(a):(b))

static inline void calc_16(int lda, int K, double* a, double* b, double* c)
{
  __m256d ac0, ac1;

  __m256d br00, br01, br02, br03;
  __m256d br10, br11, br12, br13;

  __m256d cc00, cc01, cc02, cc03;
  __m256d cc10, cc11, cc12, cc13;

  double* cc1_ptr = c + lda;
  double* cc2_ptr = cc1_ptr + lda;
  double* cc3_ptr = cc2_ptr + lda;

  cc00 = _mm256_loadu_pd(c);
  cc01 = _mm256_loadu_pd(cc1_ptr);
  cc02 = _mm256_loadu_pd(cc2_ptr);
  cc03 = _mm256_loadu_pd(cc3_ptr);

  cc10 = _mm256_setzero_pd();
  cc11 = _mm256_setzero_pd();
  cc12 = _mm256_setzero_pd();
  cc13 = _mm256_setzero_pd();

  int K_mod_2 = K % 2;
  int K_2 = K - K_mod_2;

  for (int i = 0; i < K_2; i += 2) 
  {
    ac0 = _mm256_loadu_pd(a);
    a += 4;

    br00 = _mm256_broadcast_sd(b++);
    br01 = _mm256_broadcast_sd(b++);
    br02 = _mm256_broadcast_sd(b++);
    br03 = _mm256_broadcast_sd(b++);

    cc00 = _mm256_fmadd_pd(ac0, br00, cc00);
    cc01 = _mm256_fmadd_pd(ac0, br01, cc01);
    cc02 = _mm256_fmadd_pd(ac0, br02, cc02);
    cc03 = _mm256_fmadd_pd(ac0, br03, cc03);

    ac1 = _mm256_loadu_pd(a);
    a += 4;

    br10 = _mm256_broadcast_sd(b++);
    br11 = _mm256_broadcast_sd(b++);
    br12 = _mm256_broadcast_sd(b++);
    br13 = _mm256_broadcast_sd(b++);

    cc10 = _mm256_fmadd_pd(ac1, br10, cc10);
    cc11 = _mm256_fmadd_pd(ac1, br11, cc11);
    cc12 = _mm256_fmadd_pd(ac1, br12, cc12);
    cc13 = _mm256_fmadd_pd(ac1, br13, cc13);
  }

  if (K_mod_2 != 0)
  {
    ac0 = _mm256_loadu_pd(a);

    br00 = _mm256_broadcast_sd(b++);
    br01 = _mm256_broadcast_sd(b++);
    br02 = _mm256_broadcast_sd(b++);
    br03 = _mm256_broadcast_sd(b++);

    cc00 = _mm256_fmadd_pd(ac0, br00, cc00);
    cc01 = _mm256_fmadd_pd(ac0, br01, cc01);
    cc02 = _mm256_fmadd_pd(ac0, br02, cc02);
    cc03 = _mm256_fmadd_pd(ac0, br03, cc03);
  }

  cc00 = _mm256_add_pd(cc00, cc10);
  cc01 = _mm256_add_pd(cc01, cc11);
  cc02 = _mm256_add_pd(cc02, cc12);
  cc03 = _mm256_add_pd(cc03, cc13);


  _mm256_storeu_pd(c, cc00);
  _mm256_storeu_pd(cc1_ptr, cc01);
  _mm256_storeu_pd(cc2_ptr, cc02);
  _mm256_storeu_pd(cc3_ptr, cc03);

}

static inline void copy_a (int lda, const int K, double* a_src, double* a_dst) 
{
  /* For each 4xK block-row of A */
  for (int i = 0; i < K; ++i) 
  {
    *a_dst++ = a_src[0];
    *a_dst++ = a_src[1];
    *a_dst++ = a_src[2];
    *a_dst++ = a_src[3];
    a_src += lda;
  }
}

static inline void copy_b (int lda, const int K, double* b_src, double* b_dst) 
{
  double* b_ptr1 = b_src + lda;
  double* b_ptr2 = b_ptr1 + lda;
  double* b_ptr3 = b_ptr2 + lda;

  for (int i = 0; i < K; ++i) 
  {
    *b_dst++ = *b_src++;
    *b_dst++ = *b_ptr1++;
    *b_dst++ = *b_ptr2++;
    *b_dst++ = *b_ptr3++;
  }
}
/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  double A_block[M * K], B_block[K * N];
  double *a_ptr, *b_ptr, *c_ptr;
  double cip = 0;

  int Nmax = N - 3;
  int Mmax = M - 3;
  int M_mod_4 = M % 4;
  int N_mod_4 = N % 4;
  int i = 0, j = 0;

  for (j = 0 ; j < Nmax; j += 4) 
  {
    b_ptr = &B_block[j * K];
    copy_b(lda, K, B + j * lda, b_ptr);
    
    for (i = 0; i < Mmax; i += 4) 
    {
      a_ptr = &A_block[i * K];

      if (j == 0)
      {
        copy_a(lda, K, A + i, a_ptr);
      } 

      c_ptr = C + i + j * lda;
      calc_16(lda, K, a_ptr, b_ptr, c_ptr);
    }
  }

  if (M_mod_4 != 0) 
  {
    for (i; i < M; ++i)
    {
      for (int p = 0; p < N; ++p) 
      {
        cip = C[i + p * lda];

        for (int k = 0; k < K; ++k)
        {
          cip += A[i + k * lda] * B[k + p * lda];
        }

        C[i + p * lda] = cip;
      }
    }
  }
  if (N_mod_4 != 0) 
  {
    Mmax = M - M_mod_4;
    for (j; j < N; ++j)
    { 
      for (int p = 0; p < Mmax; ++p) 
      {
        cip = C[p + j * lda];

        for (int k = 0; k < K; ++k)
        {
          cip += A[p + k * lda] * B[k + j * lda];
        }

        C[p + j * lda] = cip;
      }
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. 
 * Optimization: Two levels of blocking. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  for (int t = 0; t < lda; t += BLOCK_L2) 
  {
    for (int s = 0; s < lda; s += BLOCK_L2) 
    {
      for (int r = 0; r < lda; r += BLOCK_L2) 
      {
        int end_k = t + min(BLOCK_L2, lda - t);
        int end_j = s + min(BLOCK_L2, lda - s);
        int end_i = r + min(BLOCK_L2, lda - r);
        for (int k = t; k < end_k; k += BLOCK_L1) 
        {
          for (int j = s; j < end_j; j += BLOCK_L1) 
          {
            for (int i = r; i < end_i; i += BLOCK_L1) 
            {
              int K = min(BLOCK_L1, end_k - k);
              int N = min(BLOCK_L1, end_j - j);
              int M = min(BLOCK_L1, end_i - i);

              do_block(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
            }
          }
        }
      }
    }
  }
}

