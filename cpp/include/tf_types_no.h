/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/
// Included in both host (.cpp) as cuda (.cu) code.

#if !defined(TF_DATATYPES_NO)
#define TF_DATATYPES_NO 1
const uint8_t TF_CHAR =     0;
const uint8_t TF_INT8 =     13;     //
const uint8_t TF_INT16 =    14;     //Position jump to be flexible on inserting future types
const uint8_t TF_INT32 =    15;     //while keeping a consistent numerical order
const uint8_t TF_INT64 =    16;     //
const uint8_t TF_UINT8 =    23;
const uint8_t TF_UINT16 =   24;
const uint8_t TF_UINT32 =   25;
const uint8_t TF_UINT64 =   26;
const uint8_t TF_FP16 =     31;
const uint8_t TF_FP32 =     32;
const uint8_t TF_FP64 =     33;
#endif

typedef struct {
    void *memory_ptr;
    uint64_t no_items;
    uint8_t datatype;
} tf_column_obj;

typedef struct {
    uint64_t no_items;
    uint64_t min_value; // Actual stored datatype can be also int64_t or double (any 8 byte format)
    uint64_t max_value; // to be determined by the function that the struct is passed on
    uint64_t offset;    // Start position values in memory block
    uint64_t offset_na;    // Start position NaN values in memory block
    uint8_t datatype;
} tf_column_meta;

