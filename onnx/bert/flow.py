from metaflow import FlowSpec, step, pypi_base, kubernetes, model, current, card
from metaflow.profilers import gpu_profile
from metaflow.cards import ProgressBar


@pypi_base(
    python="3.11",
    packages={
        "pandas": "2.0.0",
        "s3fs": "2024.9.0",
        "numpy": "1.26.4",
        "pyarrow": "16.0.0",
        "transformers": "4.31.0",
        "torch": "2.4.0",
        "onnx": "1.17.0",
        "optimum": "1.11.1",
        "onnxruntime-gpu": "1.19.0",
        "protobuf": "3.20.3",
        "wget": "3.2",
        "psutil": "6.1.0"
    },
)
class OnnxDependenciesTest(FlowSpec):

    '''
    A Metaflow workflow based on this example:
    https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb
    '''

    model_name_or_path = "bert-large-uncased-whole-word-masking-finetuned-squad"
    local_cache_dir = 'data'
    export_model_path = 'model'
    enable_overwrite = True
    total_samples = 1000
    opset_version=11
    max_seq_length = 128
    doc_stride = 128
    max_query_length = 64

    @model
    @card(type='blank', id='inference_progress')
    @gpu_profile(interval=1)
    @kubernetes(gpu=1, memory=24000)
    @step
    def start(self):
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

        print("pytorch:", torch.__version__)
        print("onnxruntime:", ort.__version__)
        print("onnx:", onnx.__version__)
        print("transformers:", transformers.__version__)

        if not os.path.exists(self.local_cache_dir):
            os.makedirs(self.local_cache_dir)
        predict_file = self.download_predict_file()

        config_class, model_class, tokenizer_class = (BertConfig, BertForQuestionAnswering, BertTokenizer)
        config = config_class.from_pretrained(self.model_name_or_path, cache_dir=self.local_cache_dir)
        tokenizer = tokenizer_class.from_pretrained(self.model_name_or_path, do_lower_case=True, cache_dir=self.local_cache_dir)
        model = model_class.from_pretrained(self.model_name_or_path, from_tf=False, config=config, cache_dir=self.local_cache_dir)
        processor = SquadV1Processor()
        examples = processor.get_dev_examples(None, filename=predict_file)

        features, dataset = squad_convert_examples_to_features( 
            examples=examples[:self.total_samples],
            tokenizer=tokenizer,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            is_training=False,
            return_dataset='pt'
        )

        use_gpu = torch.cuda.is_available()
        device = torch.device("cuda" if use_gpu else "cpu")
        print('Using GPU.' if use_gpu else 'Not using GPU.')

        model.eval()
        model.to(device)

        data = dataset[0]
        inputs = {
            'input_ids':      data[0].to(device).reshape(1, self.max_seq_length),
            'attention_mask': data[1].to(device).reshape(1, self.max_seq_length),
            'token_type_ids': data[2].to(device).reshape(1, self.max_seq_length)
        }

        output_dir = os.path.join(".", "onnx_models")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)   
        export_model_path = os.path.join(output_dir, 'bert-base-cased-squad_opset{}.onnx'.format(self.opset_version))

        if self.enable_overwrite or not os.path.exists(export_model_path):
            with torch.no_grad():
                symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
                torch.onnx.export(model,                                          # model being run
                                args=tuple(inputs.values()),                      # model input (or a tuple for multiple inputs)
                                f=self.export_model_path,                         # where to save the model (can be a file or file-like object)
                                opset_version=self.opset_version,                      # the ONNX version to export the model to
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

        self.exported_model = current.model.save(
            self.export_model_path,
            label="bert_exported",
            metadata={
                "file_name": self.export_model_path,
            },
        )

        self.optimized_fp32_model_path = './onnx/bert-base-cased-squad_opt_{}_fp32.onnx'.format('gpu' if use_gpu else 'cpu')
        opt_model = optimizer.optimize_model(
            self.export_model_path,
            model_type='bert',
            opt_level=1,        
            only_onnxruntime=False 
        )
        opt_model.save_model_to_file(self.optimized_fp32_model_path)

        self.fp32_bert_model = current.model.save(
            self.optimized_fp32_model_path,
            label="bert_fp32",
            metadata={
                "file_name": self.optimized_fp32_model_path,
                "precision": "fp32",
                "optimizer": "onnx"
            }
        )

        ### Inference test ###

        use_gpu = torch.cuda.is_available()
        device_name = 'gpu' if use_gpu else 'cpu'
        providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        model_path = self.optimized_fp32_model_path

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = psutil.cpu_count(logical=True)
        sess_options.optimized_model_filepath = os.path.join(
            ".", "optimized_model_{}.onnx".format(device_name)
        )

        session = ort.InferenceSession(model_path, sess_options, providers=providers)
        total_samples = len(dataset)
        latency = []
        N = 100
        pbar = ProgressBar(max=N, label="Sample")
        current.card['inference_progress'].append(pbar)
        for i in range(N):
            data = dataset[i]
            ort_inputs = {
                'input_ids': data[0].cpu().reshape(1, self.max_seq_length).numpy(),
                'input_mask': data[1].cpu().reshape(1, self.max_seq_length).numpy(),
                'segment_ids': data[2].cpu().reshape(1, self.max_seq_length).numpy()
            }
            start_time = time.time()
            _ = session.run(None, ort_inputs)
            latency.append(time.time() - start_time)
            pbar.update(i)
            current.card['inference_progress'].refresh()

        # Calculate and print average inference time
        self.avg_inference_time_ms = (sum(latency) * 1000) / N
        print(f"ONNX Runtime {device_name.upper()} Inference time = {self.avg_inference_time_ms:.2f} ms")

        self.next(self.end)

    def download_predict_file(self):
        import os
        predict_file_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
        predict_file = os.path.join(self.local_cache_dir, "dev-v1.1.json")
        if not os.path.exists(predict_file):
            import wget
            print("Start downloading predict file.")
            wget.download(predict_file_url, predict_file)
            print("Predict file downloaded.")
        return predict_file

    @model(load="fp32_bert_model")
    @kubernetes(gpu=1, memory=24000)
    @step
    def end(self):
        pass

if __name__ == "__main__":
    OnnxDependenciesTest()