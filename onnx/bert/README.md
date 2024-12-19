## Bake the image

First, add all dependencies you'd like (besides torch, which is already in the base image) in `reqs.txt`.

Then, build the image:
```bash
docker build -t optimum-onnx-torch-2.5.1 .
```
tag it:
```bash
docker tag optimum-onnx-torch-2.5.1 $REGISTRY/optimum-onnx-torch
```
and push it:
```bash
docker push $REGISTRY/optimum-onnx-torch
```


## Run the workflow

### Option 1: Use your baked image 
```bash
python flow.py run
python flow.py argo-workflows create
python flow.py argo-workflows trigger
```

### Option 2: Use fast bakery
```bash
python flow.py --environment=fast-bakery run
python flow.py --environment=fast-bakery argo-workflows create
python flow.py --environment=fast-bakery argo-workflows trigger
```