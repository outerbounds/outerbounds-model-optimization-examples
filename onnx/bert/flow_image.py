from metaflow import FlowSpec, step, IncludeFile, kubernetes, S3


class OnnxDependenciesTest(FlowSpec):

    '''
    A Metaflow workflow based on this example:
    https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/transformers/notebooks/PyTorch_Bert-Squad_OnnxRuntime_GPU.ipynb
    '''

    predict_file = IncludeFile(name='predict_file', default='data/dev-v1.1.json')
    model_name_or_path = "bert-large-uncased-whole-word-masking-finetuned-squad"
    local_cache_dir = 'data'
    enable_overwrite = True
    total_samples = 1000
    opset_version=11
    max_seq_length = 128
    doc_stride = 128
    max_query_length = 64
    num_inference_samples = 100

    def _init_data_dir(self):
        import os

        os.makedirs('data', exist_ok=True)
        with open('data/dev-v1.1.json', 'w') as f:
            f.write(self.predict_file)

    @kubernetes(gpu=1, memory=24000, image="docker.io/eddieob/optimum-onnx-torch")
    @step
    def start(self):
        from mymodule import load_assets, optimize

        self._init_data_dir()
        dataset, model = load_assets(
            local_cache_dir=self.local_cache_dir,
            model_name_or_path=self.model_name_or_path,
            total_samples=self.total_samples,
            max_seq_length=self.max_seq_length,
            doc_stride=self.doc_stride,
            max_query_length=self.max_query_length,
            predict_file='data/dev-v1.1.json'
        )
        optimized_fp32_model_path = optimize(
            model,
            dataset,
            self.max_seq_length,
            self.opset_version,
            self.enable_overwrite
        )
        with S3(run=self) as s3:
            s3.put_files([('fp32.onnx', optimized_fp32_model_path)])
        self.next(self.end)

    @kubernetes(gpu=1, memory=24000, image="docker.io/eddieob/optimum-onnx-torch")
    @step
    def end(self):
        from mymodule import test_e2e

        self._init_data_dir()
        self.avg_inference_time_ms = test_e2e(run=self)

if __name__ == "__main__":
    OnnxDependenciesTest()