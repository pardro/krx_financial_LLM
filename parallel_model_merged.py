from IPython.display import clear_output
import torch, os, yaml
from huggingface_hub import login
from mergekit.config import MergeConfiguration
from mergekit.merge import MergeOptions, run_merge

from parallel_model_utils import model_load

def create_yaml_file(
    yaml_data,
    file_name
):
    try:
        with open(file_name, 'w') as file:
            yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True)
        print(f"YAML 파일이 성공적으로 생성되었습니다: {file_name}")
    except Exception as e:
        print(f"YAML 파일 생성 중 에러가 발생했습니다: {e}")

def model_merged(
    output_path,
    file_name
):
    with open(file_name, "r", encoding="utf-8") as fp:
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(fp))

    run_merge(
        merge_config,
        out_path=output_path,
        options=MergeOptions(
            lora_merge_cache="/tmp",
            cuda=torch.cuda.is_available(),
            copy_tokenizer=True,
            lazy_unpickle=True,
            low_cpu_memory=True,
        )
    )

def model_hf_upload(
    output_path,
    hf_save_model_name,
    private=True,
    hf_token=None
):
    if hf_token is not None:
        login(token=hf_token)
    model, tokenizer = model_load(output_path, hf_token)

    model.config.use_cache = True
    model.push_to_hub(
        hf_save_model_name,
        private=private,
        use_temp_dir=True
    )

    tokenizer.push_to_hub(
        hf_save_model_name,
        private=private,
        use_temp_dir=True
    )

