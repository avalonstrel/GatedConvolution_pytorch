#include <stdio.h>
#include <curand_kernel.h>

__host__ __device__ unsigned int XY_TO_INT(int x, int y) {//r represent the number of 10 degree, x,y - 11 bits, max = 2047, r - max = 36, 6 bits
	return (((y) << 11) | (x));
}
__host__ __device__ int INT_TO_X(unsigned int v) {
	return (v)&((1 << 11) - 1);
}
__host__ __device__ int INT_TO_Y(unsigned int v) {
	return (v >> 11)&((1 << 11) - 1);
}

__device__ int get_max(int x,int y){

	if(x>y)
		return x;
	return y;
}

__device__ int get_min(int x,int y){

	if(x<y)
		return x;
	return y;
}


__device__ float compute_distance(float *a, 
		float *aa,
		float *b, 
		float *bb,
		int rows,
		int cols,
		int channels,
		int patch_size,
		int ax,
		int ay, 
		int bx,
		int by){
	int num_points = 0;
	float pixel_sum1 = 0;
	float pixel_sum2 = 0;
	float temp_distance = 0; 
	int curr_pix_ax = 0;
	int curr_pix_ay = 0;

	int curr_pix_bx = 0;
	int curr_pix_by = 0;


	for(int y = -patch_size/2 ; y <= patch_size/2 ; y++ ){
		for(int x = -patch_size/2 ; x <= patch_size/2 ; x++){

			curr_pix_ax = ax + x; 
			curr_pix_ay = ay + y; 

			curr_pix_bx = bx + x; 
			curr_pix_by = by + y; 

			if ( curr_pix_ax >= 0 && curr_pix_ax < cols && curr_pix_ay >= 0 && curr_pix_ay < rows 
					&&
					curr_pix_bx >= 0 && curr_pix_bx < cols && curr_pix_by >= 0 && curr_pix_by < rows ){

				for(int ch = 0 ; ch < channels ; ch++){

					temp_distance =  a[channels*(curr_pix_ay*cols + curr_pix_ax ) +ch] 
						- b[channels*(curr_pix_by*cols + curr_pix_bx ) +ch] ;
					pixel_sum1 += temp_distance * temp_distance;


					temp_distance = aa[channels*(curr_pix_ay*cols + curr_pix_ax ) +ch] 
						-  bb[channels*(curr_pix_by*cols + curr_pix_bx ) +ch] ;
					pixel_sum2 += temp_distance * temp_distance;
				}
				num_points ++;
			}
		}
	}
	if(num_points ==0){
		printf("HMM");
	}
	return (pixel_sum1+pixel_sum2) /(float) num_points;

}

__device__ void compare_and_update(float *a, 
		float *aa,
		float *b, 
		float *bb,
		int rows, 
		int cols , 
		int channels, 
		int patch_size,
		int *nnf,
		float *nnd,
		int x, 
		int y,
		int bx_new,
		int by_new,
		int *best_x,
		int *best_y,
		float *best_d)
{

	float dist_new = compute_distance(a,aa,b,bb,rows,cols,channels,patch_size,x,y,bx_new,by_new);

	if(dist_new < *best_d){
		*best_d = dist_new; 
		*best_y = by_new;
		*best_x = bx_new;
	}




}


__device__ float get_rand(curandState *state){
	return curand_uniform(state);

}

__device__ void InitcuRand(curandState *state) {//random number in cuda, between 0 and 1

	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	curand_init(i, j, 0, state);

}
	extern "C" 
__global__ void patch_match(float *a, 
		float *aa,
		float *b, 
		float *bb,
		int *nnf,
		unsigned int *nnf_t,
		float *nnd,
		int rows, 
		int cols , 
		int channels, 
		int patch_size, 
		int iters,
		int jump_size,
		int search_radius)
{


	int xmin, xmax, ymin, ymax;


	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;

	if(row >= rows || col >= cols)
		return;

	curandState state;
	InitcuRand(&state);

	int init_x = nnf[2*(row*cols + col) + 0 ];
	int init_y = nnf[2*(row*cols + col) + 1 ];


	nnd[row*cols + col] = compute_distance(a,aa,b,bb,rows,cols,channels,patch_size,col,row,init_x,init_y) ; 



	for(int i = 0 ; i < iters; i++){
			int best_x = nnf[2*(row*cols + col) + 0 ];
			int best_y = nnf[2*(row*cols + col) + 1 ];
			float best_d = nnd[row*cols + col];
		for(int jump = jump_size ; jump >0 ; jump /=2){

			//test up 
			if (row - jump >=0){

				int test_x = nnf[2*((row-jump)*cols + col) + 0 ];
				int test_y = nnf[2*((row-jump)*cols + col) + 1 ] + jump;

				if(test_y < rows)
				{

					compare_and_update(a, 
							aa,
							b, 
							bb,
							rows, 
							cols , 
							channels, 
							patch_size,
							nnf,
							nnd,
							col,
							row,
							test_x,
							test_y,
							&best_x,
							&best_y,
							&best_d);

					nnf[2*(row*cols + col) + 0] = best_x;
					nnf[2*(row*cols + col) + 1] = best_y;
					nnd[1*(row*cols + col) ] = best_d;
				}
			}



			if (row + jump < rows){

				int test_x = nnf[2*((row+jump)*cols + col) + 0 ];
				int test_y = nnf[2*((row+jump)*cols + col) + 1 ] - jump;

				if(test_y >=0)
				{

					compare_and_update(a, 
							aa,
							b, 
							bb,
							rows, 
							cols , 
							channels, 
							patch_size,
							nnf,
							nnd,
							col,
							row,
							test_x,
							test_y,
							&best_x,
							&best_y,
							&best_d);

					nnf[2*(row*cols + col) + 0] = best_x;
					nnf[2*(row*cols + col) + 1] = best_y;
					nnd[1*(row*cols + col) ] = best_d;
				}
			}

			/*//test left*/
			if (col - jump >=0){

				int test_x = nnf[2*(row*cols + col-jump) + 0 ] +jump;
				int test_y = nnf[2*(row*cols + col-jump) + 1 ];

				if(test_x < cols)
				{

					compare_and_update(a, 
							aa,
							b, 
							bb,
							rows, 
							cols , 
							channels, 
							patch_size,
							nnf,
							nnd,
							col,
							row,
							test_x,
							test_y,
							&best_x,
							&best_y,
							&best_d);

					nnf[2*(row*cols + col) + 0] = best_x;
					nnf[2*(row*cols + col) + 1] = best_y;
					nnd[1*(row*cols + col) ] = best_d;
				}
			}

			//test right
			if (col + jump < cols){

				int test_x = nnf[2*(row*cols + col +jump) + 0 ] -jump;
				int test_y = nnf[2*(row*cols + col +jump) + 1 ];

				if(test_x >=0)
				{

					compare_and_update(a, 
							aa,
							b, 
							bb,
							rows, 
							cols , 
							channels, 
							patch_size,
							nnf,
							nnd,
							col,
							row,
							test_x,
							test_y,
							&best_x,
							&best_y,
							&best_d);

					nnf[2*(row*cols + col) + 0] = best_x;
					nnf[2*(row*cols + col) + 1] = best_y;
					nnd[1*(row*cols + col) ] = best_d;
				}
			}

			int rs_start = search_radius;
			if (rs_start > get_max(cols, rows)) {
				rs_start = get_max(cols, rows);
			}
			for (int mag = rs_start; mag >= 1; mag /= 2) {
				xmin = get_max(best_x - mag, 0), xmax = get_min(best_x + mag + 1, cols);
				ymin = get_max(best_y - mag, 0), ymax = get_min(best_y + mag + 1, rows);
				int test_x  = xmin + (int)(get_rand(&state)*(xmax - xmin)) % (xmax - xmin);
				int test_y = ymin + (int)(get_rand(&state)*(ymax - ymin)) % (ymax - ymin);
				compare_and_update(a, 
						aa,
						b, 
						bb,
						rows, 
						cols , 
						channels, 
						patch_size,
						nnf,
						nnd,
						col,
						row,
						test_x,
						test_y,
						&best_x,
						&best_y,
						&best_d);
			}           

			nnf[2*(row*cols + col) + 0] = best_x;
			nnf[2*(row*cols + col) + 1] = best_y;
			nnd[1*(row*cols + col) ] = best_d;

			__syncthreads();

		}




	}



}
