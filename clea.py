import json
import sys
import re
from pathlib import Path
from typing import Dict, Any, List

def count_tokens(data_str: str) -> int:
    """
    Counts tokens by splitting by whitespace and punctuation.
    """
    if not data_str:
        return 0
    # Improved regex: simpler and covers unicode boundaries better
    tokens = re.findall(r'\w+|[^\w\s]', data_str, re.UNICODE)
    return len(tokens)

def clean_notebook(file_path: str) -> None:
    path = Path(file_path)
    
    if not path.exists():
        print(f"❌ Error: File not found: {path}")
        return

    try:
        # 1. Read content once
        print(f"Processing: {path.name}")
        raw_content = path.read_text(encoding='utf-8')
        
        initial_tokens = count_tokens(raw_content)
        notebook_data = json.loads(raw_content)

        # 2. Clean Global Metadata
        # We must keep kernelspec/language_info so the notebook remains runnable
        if 'metadata' in notebook_data:
            allowed_keys = {'loutre', 'kernelspec', 'language_info'}
            notebook_data['metadata'] = {
                k: v for k, v in notebook_data['metadata'].items() 
                if k in allowed_keys
            }

        # 3. Clean Cells and Outputs
        img_count = 0
        if 'cells' in notebook_data:
            for cell in notebook_data['cells']:
                # Don't delete metadata key, reset to empty dict to maintain schema
                if 'metadata' in cell:
                    cell['metadata'] = {}

                if 'outputs' in cell:
                    for output in cell['outputs']:
                        # Reset output metadata
                        if 'metadata' in output:
                            output['metadata'] = {}
                        
                        # Remove heavy image data
                        if 'data' in output:
                            image_keys = ['image/png', 'image/jpeg', 'image/svg+xml']
                            for key in image_keys:
                                if key in output['data']:
                                    output['data'].pop(key)
                                    img_count += 1

        # 4. Save the file
        # indent=1 is compact, indent=2 is standard Jupyter default. 
        # Using 1 to match your original preference.
        cleaned_json_str = json.dumps(notebook_data, indent=1, ensure_ascii=False)
        path.write_text(cleaned_json_str, encoding='utf-8')

        # 5. Count tokens after cleaning
        final_tokens = count_tokens(cleaned_json_str)
        reduction = initial_tokens - final_tokens

        print(f"✅ Done.")
        print(f"   - Images removed: {img_count}")
        print(f"   - Tokens: {initial_tokens:,} -> {final_tokens:,} (Reduction: {reduction:,})")
        print("-" * 45)

    except json.JSONDecodeError:
        print(f"❌ Error: {path} is not a valid JSON file.")
    except Exception as e:
        print(f"❌ Unexpected Error on {path}: {e}")

# This block allows the script to be run directly
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean.py <path_to_notebook.ipynb>")
    else:
        for input_path in sys.argv[1:]:
            clean_notebook(input_path)