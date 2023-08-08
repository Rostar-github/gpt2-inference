#ifndef TYPE_H
#define TYPE_H

namespace cudaTransformer {

typedef enum datatype_enum { FP32, FP16, INT8 } Datatype;
typedef enum memorytype_enum { CPU, GPU , CPU_Pinned, UNI_MEM} Memtype;
typedef enum copytype_enum { CPU2GPU, GPU2CPU} Copytype;


}
#endif