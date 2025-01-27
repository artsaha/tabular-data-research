from transformers import pipeline, set_seed

# Set up the text generation pipeline
generator = pipeline('text-generation', model='gpt2')
set_seed(42)

# Generate a dictionary-like structure
prompt = """
Generate a Python dictionary with age and height data:
[
  {"age": 25, "height": 175},
  {"age": 30, "height": 180},
"""
output = generator(prompt, max_length=150, num_return_sequences=1)

# Print the generated text
print("Generated Output:")
print(output[0]['generated_text'])
