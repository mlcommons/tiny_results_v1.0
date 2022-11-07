#include "eai_tiny_api.h"
#include "platform.h"
#include "eai_log.h"
#include <string.h>


static const char *data_type_string[] = {
    "",      "float32", "uint8", "int8",    "uint16",  "int16",  "int32",
    "int64", "",        "",      "float16", "float64", "uint32", "uint64",
};


int load_model(struct eai_sample_context *context)
{
    FILE *fp = NULL;
    int ret = 0;

    if (!context->model_name)
    {
        return -1;
    }

    context->model_size = 0;
    fp = fopen(context->model_name, "rb");
    if (fp == NULL)
    {
        EAI_LOG("Unable to open model file: %s\n", context->model_name);
        return -1;
    }

    do {
        fseek(fp, 0, SEEK_END);
        context->model_size = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        context->model_buffer = (uint8_t *)malloc(context->model_size);
        if (!context->model_buffer)
        {
            ret = -1;
            break;
        }

        if (!ret && fread(context->model_buffer, 1, context->model_size, fp) != context->model_size)
        {
            ret = -1;
            break;
        }
    } while (0);

    fclose(fp);
    return ret;
}

uint32_t get_eai_init_flags(struct eai_sample_context *context) {
    uint32_t eai_init_flags = 0x0;
    if (context->flags & EAI_FLAGS_LPI_MODE)
    {
        eai_init_flags |= EAI_INIT_FLAGS_LPI;
    }
    return eai_init_flags;
}

int init_eai(struct eai_sample_context *context)
{
    EAI_RESULT eai_ret = EAI_SUCCESS;
    EAI_MLA_USAGE_TYPE mla_usage = context->use_enpu ? EAI_MLA_USAGE_TYPE_YES : EAI_MLA_USAGE_TYPE_NO;
    eai_memory_info_t scratch_memory;
	eai_client_perf_config_t client_perf_config;
	client_perf_config.fps        = 10;
	client_perf_config.ftrt_ratio = 0x8000;
    uint32_t eai_init_flags = get_eai_init_flags(context);
    if (eai_ret != EAI_SUCCESS) {
        EAI_LOG("eai_init_ex fail, result = %d\n", eai_ret);
        return -1;
    }
    
    if (!context->model_buffer)
    {
        return -1;
    }

	eai_ret = eai_init_ex(&context->eai_handle, context->model_buffer, context->model_size, eai_init_flags, NULL); // flags contain user set flags such as enable/disable lpi mode
    if (eai_ret != EAI_SUCCESS)
    {
        EAI_LOG("eai_init fail, result = %d\n", eai_ret);
        return -1;
    }

    if (mla_usage == EAI_MLA_USAGE_TYPE_YES){
        // Needs to be set before MLA_USAGE, since it passes down to ENPU
        if ((eai_ret = eai_set_property(context->eai_handle, (EAI_PROP) EAI_PROP_CLIENT_PERF_CFG, &client_perf_config)) != EAI_SUCCESS){
            EAI_LOG("Failed to set property EAI_PROP_CLIENT_PERF_CFG : result = %d \n", eai_ret);
            return -1;
        }
    }

    eai_ret = eai_set_property(context->eai_handle, EAI_PROP_MLA_USAGE, &mla_usage);
    if (eai_ret != EAI_SUCCESS && eai_ret != EAI_MLA_NOT_AVAILABLE)
    {
        EAI_LOG("eai_set_property(EAI_PROP_MLA_USAGE) fail, result = %d\n", eai_ret);
        return -1;
    }

    if (mla_usage == EAI_MLA_USAGE_TYPE_YES) {
        eai_enpu_ctx_t ctx;
        eai_ret = eai_get_property(context->eai_handle, EAI_PROP_ENPU_INFO, &ctx);
        if (eai_ret != EAI_SUCCESS || ctx.lpmla_drv_handle == NULL) {
            EAI_LOG("eai_set_property(EAI_PROP_ENPU_INFO) fail, result = %d\n", eai_ret);
            return -1;
        }
        EAI_LOG("eai_set_property(EAI_PROP_ENPU_INFO) driver handle = %p\n", ctx.lpmla_drv_handle);
    }

    eai_ret = eai_preapply(context->eai_handle);
    if (eai_ret != EAI_SUCCESS)
    {
        EAI_LOG("eai_preapply fail, result = %d\n", eai_ret);
        return -1;
    }

    // get scratch buffer info
    eai_ret = eai_get_property(context->eai_handle, EAI_PROP_SCRATCH_MEM, &scratch_memory);
    if (eai_ret != EAI_SUCCESS)
    {
        EAI_LOG("eai_get_property(EAI_PROP_SCRATCH_MEM) FAIL. result = %d\n", eai_ret);
        return -1;
    }

    context->scratch_buffer_size = scratch_memory.memory_size;
    context->scratch_buffer = (uint8_t *)malloc(scratch_memory.memory_size);

    scratch_memory.addr = context->scratch_buffer;
    // set scratch buffer for eai api
    eai_ret = eai_set_property(context->eai_handle, EAI_PROP_SCRATCH_MEM, &scratch_memory);
    if (eai_ret != EAI_SUCCESS)
    {
        EAI_LOG("eai_set_property(EAI_PROP_SCRATCH_MEM) FAIL. result = %d\n", eai_ret);
        return -1;
    }

    eai_ret = eai_apply(context->eai_handle);
    if (eai_ret != EAI_SUCCESS)
    {
        EAI_LOG("eai_apply FAIL. result = %d\n", eai_ret);
        return -1;
    }

    return eai_ret;
}

int get_model_io(struct eai_sample_context *context)
{
    EAI_RESULT eai_ret = EAI_SUCCESS;

    for (int i = 0; i < 2 && eai_ret == EAI_SUCCESS; i++)
    {
        eai_ports_info_t ports_info;
        ports_info.input_or_output = i;
        eai_ret = eai_get_property(context->eai_handle, EAI_PROP_PORTS_NUM, &ports_info);
        if (eai_ret != EAI_SUCCESS)
        {
            EAI_LOG("Failed eai_get_property(EAI_PROP_PORTS_NUM - inputs). result = %d\n", eai_ret);
            break;
        }
        context->tensor_count[i] = ports_info.size;

        for (unsigned int j = 0; j < ports_info.size; j++)
        {
            context->tensors[i][j].index = j;
            context->tensors[i][j].input_or_output = i;

            eai_ret = eai_get_property(context->eai_handle, EAI_PROP_TENSOR_INFO, &(context->tensors[i][j]));
            EAI_LOG("user scratch pointer context->tensors[%d][%d]: %p\n",i,j,context->tensors[i][j].address)
            if (eai_ret != EAI_SUCCESS)
            {
                EAI_LOG("Failed eai_get_property(EAI_PROP_TENSOR_SIZE_INFO - inputs). result = %d\n", eai_ret);
                break;
            }
            if (!context->allocate_io_buf) {
                if (context->tensors[i][j].address == NULL) {
                    eai_ret = EAI_RESOURCE_FAILURE;
                    EAI_LOG("Input/Output not configured to use scratch, please use: -allocate_io\n");
                    break;
                }
            }
        }
    }
    
    if (eai_ret == EAI_SUCCESS && context->allocate_io_buf) {
        size_t aligned_tensor_size = 0;
        for (int i = 0; i < 2 && eai_ret == EAI_SUCCESS; i++) {
            for (int j = 0; j < context->tensor_count[i] && eai_ret == EAI_SUCCESS; j++) {
                aligned_tensor_size = context->tensors[i][j].tensor_size + BUFFER_ALIGNMENT; // Allocate extra memory for alignment since this buffer is being allocated from the user side
                context->tensors[i][j].address = malloc_align(BUFFER_ALIGNMENT, aligned_tensor_size);

                if (context->tensors[i][j].address == NULL) {
                    EAI_LOG("Failed to allocate buffer for I/O\n");
                    eai_ret = EAI_RESOURCE_FAILURE;
                    break;
                }

                //todo: register io buffer to the runtime if runtime is in the root pd
            }
        }
    }
    return (eai_ret == EAI_SUCCESS) ? 0 : -1;
}

void print_model_io(struct eai_sample_context *context)
{
    printf("print model io \r\n");
    for (int i = 0; i < 2; i++)
    {
        if (i == 0) {
            EAI_LOG("input:\n");
        }
        else {
            EAI_LOG("output:\n");
        }

        for (int j = 0; j < context->tensor_count[i]; j++)
        {
            eai_tensor_info_t *tensor = &(context->tensors[i][j]);
            EAI_LOG("date type: %s\n", data_type_string[tensor->element_type]);
            EAI_LOG("dimension:");
            for (unsigned int k = 0; k < tensor->num_dims; k++)
            {
                EAI_LOG(" %lu", (unsigned long)(tensor->dims[k]));
            }
            EAI_LOG("\n");
        }
    }
}

int fill_io_batch(struct eai_sample_context *context)
{
    for (int i = 0; i < 2; i++)
    {
        for (int j = 0; j < context->tensor_count[i]; j++)
        {
            eai_tensor_info_t *tensor = &(context->tensors[i][j]);
            context->eai_buffers[i][j].index = j;
            context->eai_buffers[i][j].element_type = tensor->element_type;
            context->eai_buffers[i][j].addr = tensor->address;
            context->eai_buffers[i][j].buffer_size = tensor->tensor_size;
        }
    }
    context->eai_batch.num_inputs = context->tensor_count[0];
    context->eai_batch.num_outputs = context->tensor_count[1];
    context->eai_batch.inputs = &(context->eai_buffers[0][0]);
    context->eai_batch.outputs = &(context->eai_buffers[1][0]);
    return 0;
}

int deinit(struct eai_sample_context *context) {
    if (!context) {
        return -1;
    }

    if (!context->eai_handle) {
        return 0;
    }

    EAI_RESULT eai_ret = eai_deinit(context->eai_handle);
    if (eai_ret != EAI_SUCCESS) {
        EAI_LOG("fail to deinit eai");
    }

    free(context->scratch_buffer);
    free(context->model_buffer);

    //free io buffers if allocated
    if (context->allocate_io_buf) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < context->tensor_count[0]; j++) {
                if(context->tensors[i][j].address) {
                    free_align(context->tensors[i][j].address);
                }
            }
        }
    }

    // close io files
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < context->tensor_count[i]; j++) {
            fclose(context->io_file[i][j]);
        }
    }
	
    return 0;
}

int generate_output_file_name(struct eai_sample_context *context, char *full_path, int index)
{
    char output_file_name[256];
    snprintf(output_file_name, 256, "output_%d.raw", index);
    if (context->output_path)
    {
        strcpy(full_path, context->output_path);
        if (context->output_path[strlen(context->output_path) - 1] != '/')
        {
            strcat(full_path, "/");
        }
    }
    strcat(full_path, output_file_name);
    return 0;
}

int initialize_o(struct eai_sample_context *context)
{
    int ret = 0;
    for (int i = 0; i < context->tensor_count[1]; i++)
    {
        context->output_name[i] = (char *)malloc(MAX_FILE_PATH_LENGTH);
        if(context->output_name[i] == NULL) {
            ret = -1;
            break;
        }
        context->output_name[i][0] = 0;
        generate_output_file_name(context, context->output_name[i], i);
        context->io_file[1][i] = fopen(context->output_name[i], "wb");
        if (context->io_file[1][i] == NULL)
        {
            EAI_LOG("fail to open output file %s\n", context->output_name[i]);
            ret = -1;
            break;
        }
    }
    return ret;
}

int initialize_i(struct eai_sample_context *context)
{
    int ret = 0;
    for (int i = 0; i < context->tensor_count[0]; i++)
    {
        context->io_file[0][i] = fopen(context->input_file[i], "rb");
        if (context->io_file[0][i] == NULL)
        {
            EAI_LOG("fail to open input file %s\n", context->input_file[i]);
            ret = -1;
            break;
        }
    }
    return ret;
}

int save_outputs(struct eai_sample_context *context)
{
    int ret = 0;
    for (int i = 0; i < context->tensor_count[1]; i++)
    {
        eai_tensor_info_t *tensor = &context->tensors[1][i];
        if (!tensor || !tensor->address) {
            EAI_LOG("invalid i/o tensor!\n");
            ret = -1;
            break;
        }
        size_t write_size = fwrite(tensor->address, 1, tensor->tensor_size, context->io_file[1][i]);

        if (write_size != tensor->tensor_size)
        {
            ret = -1;
            break;
        }
    }
    return ret;
}
