import os
import time
import psutil
import torch
import onnxruntime as ort
import onnx
from onnxruntime.transformers import optimizer
import transformers
from transformers import (BertConfig, BertForQuestionAnswering, BertTokenizer)
from transformers import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV1Processor
from metaflow import Flow, S3

DEFAULT_FLOW_NAME = 'OnnxDependenciesTest'
DEFAULT_MODEL_PATH =  "bert-base-cased-squad_opt_cpu_fp32.onnx"


def download_predict_file(local_cache_dir):
    import os
    predict_file_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
    predict_file = os.path.join(local_cache_dir, "dev-v1.1.json")
    if not os.path.exists(predict_file):
        import wget
        print("Start downloading predict file.")
        wget.download(predict_file_url, predict_file)
        print("Predict file downloaded.")
    return predict_file


def load_assets(
    local_cache_dir,
    model_name_or_path,
    total_samples,
    max_seq_length,
    doc_stride,
    max_query_length,
    predict_file=None
):

    print("pytorch:", torch.__version__)
    print("onnxruntime:", ort.__version__)
    print("onnx:", onnx.__version__)
    print("transformers:", transformers.__version__)

    if not os.path.exists(local_cache_dir):
        os.makedirs(local_cache_dir)
    if predict_file is None:
        predict_file = download_predict_file(local_cache_dir)

    config_class, model_class, tokenizer_class = (BertConfig, BertForQuestionAnswering, BertTokenizer)
    config = config_class.from_pretrained(model_name_or_path, cache_dir=local_cache_dir)
    tokenizer = tokenizer_class.from_pretrained(model_name_or_path, do_lower_case=True, cache_dir=local_cache_dir)
    model = model_class.from_pretrained(model_name_or_path, from_tf=False, config=config, cache_dir=local_cache_dir)
    processor = SquadV1Processor()
    examples = processor.get_dev_examples(None, filename=predict_file)

    features, dataset = squad_convert_examples_to_features( 
        examples=examples[:total_samples],
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        return_dataset='pt'
    )
    return dataset, model


def optimize(
    model,
    dataset,
    max_seq_length,
    opset_version,
    enable_overwrite
):

    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    print('Using GPU.' if use_gpu else 'Not using GPU.')

    model.eval()
    model.to(device)

    data = dataset[0]
    inputs = {
        'input_ids':      data[0].to(device).reshape(1, max_seq_length),
        'attention_mask': data[1].to(device).reshape(1, max_seq_length),
        'token_type_ids': data[2].to(device).reshape(1, max_seq_length)
    }

    output_dir = os.path.join(".", "onnx_models")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
    export_model_path = os.path.join(output_dir, 'bert-base-cased-squad_opset{}.onnx'.format(opset_version))

    if enable_overwrite or not os.path.exists(export_model_path):
        with torch.no_grad():
            symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
            torch.onnx.export(model,                                          # model being run
                            args=tuple(inputs.values()),                      # model input (or a tuple for multiple inputs)
                            f=export_model_path,                              # where to save the model (can be a file or file-like object)
                            opset_version=opset_version,                      # the ONNX version to export the model to
                            do_constant_folding=True,                         # whether to execute constant folding for optimization
                            input_names=['input_ids',                         # the model's input names
                                        'input_mask', 
                                        'segment_ids'],
                            output_names=['start', 'end'],                    # the model's output names
                            dynamic_axes={'input_ids': symbolic_names,        # variable length axes
                                            'input_mask' : symbolic_names,
                                            'segment_ids' : symbolic_names,
                                            'start' : symbolic_names,
                                            'end' : symbolic_names})


    path_prefix = "./onnx"
    optimized_fp32_model_path = os.path.join(
        path_prefix,
        'bert-base-cased-squad_opt_{}_fp32.onnx'.format('gpu' if use_gpu else 'cpu')
    )
    opt_model = optimizer.optimize_model(
        export_model_path,
        model_type='bert',
        opt_level=1,        
        only_onnxruntime=False 
    )
    opt_model.save_model_to_file(optimized_fp32_model_path)
    return optimized_fp32_model_path


def download_model(
    flow_name=DEFAULT_FLOW_NAME,
    model_dst_path=DEFAULT_MODEL_PATH,
    run=None
):
    if run is None:
        f=Flow(flow_name)
        # Use Client API + Metaflow tags + run properties to filter desired run
        run=f.latest_successful_run
    with S3(run=run) as s3:
        obj = s3.get('fp32.onnx')
        os.rename(obj.path, model_dst_path)
    print(f'Model downloaded to {model_dst_path}.')
    return model_dst_path


def test_inference(model_path, dataset, max_seq_length, num_samples=100):
    """
    Test inference performance of an ONNX model.
    
    Args:
        model_path (str): Path to the optimized ONNX model
        dataset: Dataset containing input samples
        max_seq_length (int): Maximum sequence length for the model
        num_samples (int, optional): Number of samples to test. Defaults to 100.
    
    Returns:
        float: Average inference time in milliseconds
    """
    import time
    import psutil
    import torch
    import onnxruntime as ort
    from tqdm import tqdm

    use_gpu = torch.cuda.is_available()
    device_name = 'gpu' if use_gpu else 'cpu'
    providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
    sess_options.optimized_model_filepath = os.path.join(
        ".", f"optimized_model_{device_name}.onnx"
    )

    session = ort.InferenceSession(model_path, sess_options, providers=providers)
    latency = []

    for i in tqdm(range(num_samples), desc="Running inference"):
        data = dataset[i]
        ort_inputs = {
            'input_ids': data[0].cpu().reshape(1, max_seq_length).numpy(),
            'input_mask': data[1].cpu().reshape(1, max_seq_length).numpy(),
            'segment_ids': data[2].cpu().reshape(1, max_seq_length).numpy()
        }
        start_time = time.time()
        _ = session.run(None, ort_inputs)
        latency.append(time.time() - start_time)

    avg_inference_time_ms = (sum(latency) * 1000) / num_samples
    print(f"ONNX Runtime {device_name.upper()} Inference time = {avg_inference_time_ms:.2f} ms")
    
    return avg_inference_time_ms


def test_e2e(
    local_cache_dir = 'data',
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad",
    model_path = DEFAULT_MODEL_PATH,
    predict_file = 'data/dev-v1.1.json',
    max_seq_length = 128,
    doc_stride = 128,
    max_query_length = 64,
    total_samples = 100,
    opset_version = 11,
    run=None
):
    dataset, _ = load_assets(
        local_cache_dir=local_cache_dir,
        model_name_or_path=model_name,
        total_samples=total_samples,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        predict_file=predict_file
    )
    model_path = download_model(run=run)
    avg_time = test_inference(
        model_path,
        dataset,
        max_seq_length,
        num_samples=total_samples
    )
    print(f"End-to-end test completed. Average inference time: {avg_time:.2f} ms")
    return avg_time