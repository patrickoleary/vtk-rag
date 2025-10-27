# Raw VTK Documentation Files

This directory contains the raw JSONL files used as input for corpus preparation.

## Expected Files

Place the following files in this directory:

- `vtk-python-docs.jsonl` - API documentation (~2,900 classes, ~61 MB)
- `vtk-python-examples.jsonl` - Python examples (~850 examples, ~5 MB)  
  - Should include: `user_queries`, `data_files`, `data_download_info`, `image_url`
- `vtk-python-tests.jsonl` - Python tests (~900 tests, ~4.5 MB)
  - Should include: `user_query`, `data_files`, `data_download_info`, `has_baseline`

## File Format

Each file should be in JSONL format (one JSON object per line).

**Note:** Examples and tests files should be the augmented versions with LLM-generated queries, data file URLs, and baseline image information. These are generated in the source repositories (`vtk-python-examples` and `vtk-python-tests`) and copied here.

## Processing

To process these files into chunks:

```bash
cd ../..
python prepare-corpus/prepare_corpus.py
```

Output will be written to `data/processed/`
