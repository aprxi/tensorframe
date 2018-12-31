/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <iostream>
#include <string>
#include <cstring>

#include "libio.h"
#include "io__fileops.h"


// NOTE: requires approximately 6GB GPU memory
// TODO: make this flexible (e.g. custom option or auto detected)
#define CHUNKSIZE 1024 * 1024 * 1536
#define BUFSIZE 128
#define REWIND_SIZE 32

using namespace std;

void get_column_names(FILE *f, unsigned long *offset, unsigned long *columns, char sep, char eol){
// TODO: function does not yet contain proper eror checking. Eexample is set in cpp__write_csvfile().
	char buf[BUFSIZE];
	int l;

	while(fgets(buf, BUFSIZE, f) != NULL){
		l = strlen(buf);

		for(int i=0; buf[i]; i++) *columns += (unsigned long) (buf[i] == sep);

		// Check if end of line criterium is met
		if(buf[l - 1] == eol){
			*columns += 1;
			*offset += (unsigned long) l;
			break;
		};
		*offset += (unsigned long) l;
	}
}

void get_offsets(FILE *f, unsigned long **offsets_ptr, uint64_t fsize, char eol, unsigned long *no_chunks){
// TODO: function does not yet contain proper eror checking. Example is set in cpp__write_csvfile().

	uint64_t *offsets = *offsets_ptr;
	uint64_t chunks = lldiv(fsize, (uint64_t) CHUNKSIZE).quot + 1; //Expected number of chunksd
	uint64_t new_offset;;

    uint64_t idx;
	uint8_t line_length;

    char buf[REWIND_SIZE];

	// Ensure file is set to first offset
	fseek(f, offsets[0], SEEK_SET);

	// Allocate memory to fit expected number of chunks, and one to hold last offset (=fsize position)
	offsets = (unsigned long*) realloc((void*) offsets, sizeof(unsigned long) * (chunks + 1));

	// Append offsets as long as current offset + chunksize does not exceed filesize
    for(idx=1; (offsets[idx-1] + (uint64_t) CHUNKSIZE) < fsize; idx++){
		// If number of chunks exceeds expected number chunks, update memory allocation to hold the offset numbers
		if(idx > chunks) {offsets = (unsigned long*) realloc((void*) offsets, sizeof(unsigned long) * (idx+1));}

		// Initialise new eol at previous offset + chunksize
		new_offset = offsets[idx - 1] + (uint64_t) CHUNKSIZE - (unsigned long) REWIND_SIZE;

		// Initialise index offset at 0
		offsets[idx] = 0;

		// Find last EOL char within added chunk
		while(fseek(f, new_offset, SEEK_SET) == 0){
            // Read last line
            if(fgets(buf, REWIND_SIZE, f) == NULL){ break;}

			line_length = strlen(buf);
			if(line_length > 0 && buf[line_length - 1] == eol){
				offsets[idx] = new_offset + line_length;
				break;
			}
            // Else. Line did not contain EOL or EOF. Reset new offset to prepare for next round.
			new_offset = new_offset - (unsigned long) REWIND_SIZE;

            // Stop EOL search if we move beyond last offset or hit EOF
			if(new_offset < offsets[idx-1]){ break;}
		}

		//No EOL found in chunk. Stop searching because we cant have lines larger than chunksize
		if(offsets[idx] == 0){ break;}
	}

	if(idx > chunks){ offsets = (unsigned long*) realloc((void*) offsets, sizeof(unsigned long) * (idx+1));}

	// Set last offset to EOF (fsize + 1 for NULL)
	offsets[idx] = fsize + 1;

	*offsets_ptr = offsets;
	*no_chunks = (unsigned long) idx;
	return;
}


const int lc_int8(void *data, char *buf_ptr){ return sprintf(buf_ptr, "%i,", *((int8_t*) data));};
const int lc_int16(void *data, char *buf_ptr){ return sprintf(buf_ptr, "%i,", *((int16_t*) data));};
const int lc_int32(void *data, char *buf_ptr){ return sprintf(buf_ptr, "%i,", *((int32_t*) data));};
const int lc_int64(void *data, char *buf_ptr){ return sprintf(buf_ptr, "%li,", *((int64_t*) data));};

const int lc_uint8(void *data, char *buf_ptr){ return sprintf(buf_ptr, "%u,", *((uint8_t*) data));};
const int lc_uint16(void *data, char *buf_ptr){ return sprintf(buf_ptr, "%u,", *((uint16_t*) data));};
const int lc_uint32(void *data, char *buf_ptr){ return sprintf(buf_ptr, "%u,", *((uint32_t*) data));};
const int lc_uint64(void *data, char *buf_ptr){ return sprintf(buf_ptr, "%lu,", *((uint64_t*) data));};

// Note on floats. This code part is currently only used to write random test csv files.
// Therefore we dont need an actual half but can suffice (/cheat) with a regular float (writing a half requires some tricks)
// TODO: fix this so we can use this code more generically.
const int lc_fp16(void *data, char *buf_ptr){ return sprintf(buf_ptr, "%.8f,", *((float*) data));};
const int lc_fp32(void *data, char *buf_ptr){ return sprintf(buf_ptr, "%.12f,", *((float*) data));};
const int lc_fp64(void *data, char *buf_ptr){ return sprintf(buf_ptr, "%.16f,", *((double*) data));};


void cpp__write_csvfile(char *outputfile, tf__frame *frame){
    char *buf;
    char *buf_ptr;

    uint64_t column;
    uint64_t offset;

    int no_bytes;
    FILE *pFile;

    // Allocate 32 bytes per field, 1 for separator (|| newline)
    if(!(buf = (char*) malloc(frame->no_columns * sizeof(char) * 33))){ throw 200;};

    if(!(pFile = fopen(outputfile, "wb"))){ 
        cerr << "E:cant open file: " + string(outputfile) << endl;
        free(buf);
        throw 2;
    }

    // Write column names
    // TODO: add support to pass custom names
    buf_ptr = buf;
    for(column = 0; column < frame->no_columns; column++){
        if((no_bytes = (int) sprintf(buf_ptr, "COL_%lu,", column)) < 0){ free(buf); throw 100;};
        buf_ptr += no_bytes;
    }
    if(sprintf(buf_ptr - 1, "\n") != 1){ free(buf); throw 100;};   //overwrite last separator with newline
    no_bytes = (buf_ptr - buf);
    if(fwrite(buf, sizeof(char), no_bytes, pFile) != no_bytes){
        cerr << "E:failure in writing to file: " + string(outputfile) << endl;
        fclose(pFile);
        free(buf);
        throw 4;
    }

    // Write rows
    for(uint64_t row = 0; row < frame->no_rows; row++){
        // reset  to start location
        buf_ptr = buf;  // + 1;

        // Collect all column data for this line
        for(column = 0; column < frame->no_columns; column++){
            offset = ((uint64_t) frame->ptr_data[column]) + ((uint64_t)(frame->strides[column] * row));
            if((no_bytes = (int) lc_to_string[frame->functionmap[column]]((void*) offset,  buf_ptr)) < 0){ free(buf); throw 100;};
            buf_ptr += no_bytes;
        }

        // Write line to file
        if(sprintf(buf_ptr - 1, "\n") != 1){ free(buf); throw 100;};   //overwrite last separator with newline
        no_bytes = (buf_ptr - buf); // - 1);     //deduct one to strip off last separator
        if(fwrite(buf, sizeof(char), no_bytes, pFile) != no_bytes){
            cerr << "E:failure in writing to file: " + string(outputfile) << endl;
            fclose(pFile);
            free(buf);
            throw 6;
        }
    }

    free(buf);

    // Write a last newline and close file
    if(fclose(pFile) != 0){
        cerr << "E:failure in writing to file: " + string(outputfile) << endl;
        throw 8;
    }
}


void load_csvfile(char *arg_str, struct file_object *fobj){
// TODO: function does not yet contain proper eror checking. Eexample is set in cpp__write_csvfile().
    struct stat st;
	FILE *f;
	char **string_ptr;

    fobj->offsets = (unsigned long*) malloc(sizeof(unsigned long));
	unsigned long *columns = (unsigned long*) malloc(sizeof(unsigned long));

	fobj->offsets[0] = 0;
	*columns = 0;

    stat(arg_str, &st); 
    fobj->fsize = (uint64_t) st.st_size;
  
	f = fopen(arg_str, "r" );
	get_column_names(f, fobj->offsets, columns, ',', '\n');
	get_offsets(f, &fobj->offsets, fobj->fsize, '\n', &fobj->chunks);
	
	string_ptr = &fobj->string;
	cu_malloc_carr(string_ptr, fobj->fsize * 1 + 1);

	fseek(f, 0, SEEK_SET);
	if(fread(fobj->string, fobj->fsize, 1, f) != 1){  exit(1);}
	fclose(f);

	fobj->string[fobj->fsize] = 0;

	// NOTE. Rember to ensure last char in array is a newline. We check on number of columns, ending with newline
	// Allocate space for 2 dimensions (rows and columns)
    fobj->dims = (uint64_t*) malloc(sizeof(uint64_t) * 2);
	fobj->dims[0] = 0;
	fobj->dims[1] = *columns;
	
    free(columns);
};


void cpp__read_csv(char *arg_str, struct file_object *fobj){
    load_csvfile(arg_str, fobj);
}
void cpp__parse_csv(struct file_object *fobj, void *dest, uint8_t *columns){
	cuda_parse_csv(fobj->string, fobj->fsize, fobj->offsets, fobj->chunks, fobj->dims, dest, columns);
}
