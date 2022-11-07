#ifndef EAI_TINY_API_H
#define EAI_TINY_API_H

#include <stdio.h>
#include <stdlib.h>
#include "eai.h"
#define MAX_IO_COUNT 10
#define BUFFER_ALIGNMENT 256
#define MAX_FILE_PATH_LENGTH 512
#define ALIGN(n, alignment) ((n + alignment - 1) & (~(alignment - 1)))
#define EAI_FLAGS_LPI_MODE                      0x00000001

#ifdef __cplusplus
extern "C"
{
#endif
    struct eai_sample_context
    {
        const char *model_name;
        const char *output_path;
        int input_file_count;
        const char *input_file[MAX_IO_COUNT];
        FILE *io_file[2][MAX_IO_COUNT];
        uint8_t *model_buffer;
        size_t model_size;
        uint8_t *scratch_buffer;
        size_t scratch_buffer_size;
        eaih_t eai_handle;
        int tensor_count[2]; // 0: input tensor, 1: output tensor
        eai_tensor_info_t tensors[2][MAX_IO_COUNT];
        eai_buffer_info_t eai_buffers[2][MAX_IO_COUNT];
        eai_batch_info_t eai_batch;
        int max_execution_count;
        int use_enpu;
        int allocate_io_buf;
        char *output_name[MAX_IO_COUNT];
        const char *root_folder;
        uint32_t flags; //Eai config flags to be used during init (eg used for lpi mode etc)
        uint8_t *cpd_last_input_address;
    };
    int load_model(struct eai_sample_context *context);
    int init_eai(struct eai_sample_context *context);
    int deinit(struct eai_sample_context *context);
    int get_model_io(struct eai_sample_context *context);
    void print_model_io(struct eai_sample_context *context);
    int fill_io_batch(struct eai_sample_context *context);
    int generate_output_file_name(struct eai_sample_context *context, char *full_path, int index);
    int initialize_o(struct eai_sample_context *context);
    int initialize_i(struct eai_sample_context *context);
    int save_outputs(struct eai_sample_context *context);
#ifdef __cplusplus
}
#endif
#endif // EAI_TINY_API_H