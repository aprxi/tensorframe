/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

#include "tf_types_no.h"

typedef struct {
    uint8_t id;
    uint8_t size;
    const char *name;
} lookup_types;

//Conversion table to match C(++) types with those used in PyArray objects.
//Size is important for mapping instructions on our data properly to minimise instructions.

//Imporant note. id is essential to map with an array of CUDA functions (see tf_types.cuh).
//I.e. dont change the order as long as this is not (yet) checked for in code (TODO).

lookup_types TF_TYPES[] = {
    { TF_CHAR,          sizeof(char),       "|b1"},     //
    { TF_INT8,          sizeof(int8_t),     "|i1"},     //Position jump to be flexible on inserting future types
    { TF_INT16,         sizeof(int16_t),    "<i2"},     //while keeping a consistent numerical order
    { TF_INT32,         sizeof(int32_t),    "<i4"},     //
    { TF_INT64,         sizeof(int64_t),    "<i8"},
    { TF_UINT8,         sizeof(uint8_t),    "|u1"},
    { TF_UINT16,        sizeof(uint16_t),   "<u2"},
    { TF_UINT32,        sizeof(uint32_t),   "<u4"},
    { TF_UINT64,        sizeof(uint64_t),   "<u8"},
    { TF_FP16,          sizeof(float) / 2,  "<f2"},     //
    { TF_FP32,          sizeof(float),      "<f4"},     //C++11/IEEE-754
    { TF_FP64,          sizeof(double),     "<f8"},     //
    { 255,          0,                  NULL}       //ADD EOD TO ALLOW ITERATION. Max 256 - 1.
};


const bool _datatype_column_typebyname(char *name, uint8_t *value){
    for(int j=0; (TF_TYPES[j].name); j++){
        if(strncmp(name, TF_TYPES[j].name, 4)==0){ *value = TF_TYPES[j].id; return true;};
    }
    return false;   
}

uint8_t _datatype_column_nobyid(uint8_t id){
    for(uint8_t j=0; (TF_TYPES[j].name); j++){
        if(TF_TYPES[j].id == id){   return j;};
    }
    return 0;   //Default
}
