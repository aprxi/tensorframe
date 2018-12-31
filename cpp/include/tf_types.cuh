/* Copyright (C) 2018 Anthony Potappel, The Netherlands. All Rights Reserved.
 * This work is licensed under the terms of the MIT license (for details, see attached LICENSE file).
*/

//Typedefs for CUDA to improve code readability,
//This also prepares for guarantueeing (asserting) types and sizes in later versions.

typedef int8_t int8_cu;
typedef int16_t int16_cu;
typedef int32_t int32_cu;
typedef long long int int64_cu;

typedef uint8_t uint8_cu;
typedef uint16_t uint16_cu;
typedef uint32_t uint32_cu;
typedef unsigned long long int uint64_cu;

typedef half fp16_cu;
typedef float fp32_cu;
typedef double fp64_cu;
typedef float4 fp128_cu;

