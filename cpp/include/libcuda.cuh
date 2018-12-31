/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

void cu_malloc_carr(char **data_ptr, unsigned long fsize){
	cudaHostAlloc((void**)data_ptr, fsize * sizeof(char), cudaHostAllocMapped);
}

