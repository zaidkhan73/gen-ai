from huggingface_hub import InferenceClient

client = InferenceClient(
    api_key="YOUR_HF_TOKEN"
)

models = client.list_deployed_models()

print(models)