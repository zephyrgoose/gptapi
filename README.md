# gptapi

A configuration-as-code approach for interacting with the OpenAI API. Profiles stored as YAML files define models, prompts and parameters so code stays minimal and reproducible.

## Installation

Clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

## API keys

Create a `keys.yaml` file in the project root containing your OpenAI API key:

```yaml
openai_api: YOUR_OPENAI_KEY
```

Profiles may reference a custom credentials file with the `credentials_file` field, otherwise `./keys.yaml` is used by default.

## Profiles

YAML profiles live under the `profiles/` directory. Use the file name (without the extension) as the profile name. For example `profiles/cot.yaml` can be invoked using the profile name `cot`.

## Basic usage

### Running the example script

The `example.py` script reads a prompt from `input.txt` and calls the API using the `cot` profile:

```bash
python example.py
```

### Calling `gptapi()` directly

You can also use the library in your own code:

```python
from gptapi import gptapi

result = gptapi("cot", "Explain chain of thought reasoning")
print(result)
```


