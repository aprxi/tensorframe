/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

typedef const int(*threadedFunc_lc_to_string)(void *data, char *buf_ptr);

const int lc_int8(void *data, char *buf_ptr);
const int lc_int16(void *data, char *buf_ptr);
const int lc_int32(void *data, char *buf_ptr);
const int lc_int64(void *data, char *buf_ptr);

const int lc_uint8(void *data, char *buf_ptr);
const int lc_uint16(void *data, char *buf_ptr);
const int lc_uint32(void *data, char *buf_ptr);
const int lc_uint64(void *data, char *buf_ptr);

const int lc_fp16(void *data, char *buf_ptr);
const int lc_fp32(void *data, char *buf_ptr);
const int lc_fp64(void *data, char *buf_ptr);

threadedFunc_lc_to_string lc_to_string[] = {
    NULL,
    lc_int8,
    lc_int16,
    lc_int32,
    lc_int64,
    lc_uint8,
    lc_uint16,
    lc_uint32,
    lc_uint64,
    lc_fp16,
    lc_fp32,
    lc_fp64
};
