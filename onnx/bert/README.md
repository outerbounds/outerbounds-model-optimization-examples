## Run the workflow

To run this example on your Outerbounds deployment:
```bash
python flow.py --environment=fast-bakery run
python flow.py --environment=fast-bakery argo-workflows create
python flow.py --environment=fast-bakery argo-workflows trigger
```