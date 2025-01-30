
from datasets import load_dataset
import pandas as pd
from json import loads, dumps

ds = load_dataset("rungalileo/medical_transcription_40", split='test')

df = pd.DataFrame(ds)

result = df.to_json(orient="split")
parsed = loads(result)

new_df = pd.DataFrame({
    "prompt": ["create medical dataset"],
    "result": [df.to_dict(orient="records")]
})

# Save as CSV (not ideal for nested data)
new_df.to_csv("medical_transcription_with_prompt.csv", index=False)