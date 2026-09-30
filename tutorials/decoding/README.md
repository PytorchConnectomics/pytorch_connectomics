# Decoding tutorials

Use `tutorials/basics/decoding_abiss.yaml` for external ABISS decoding and
`tutorials/waterz_decoding.yaml` for waterz agglomeration of saved affinities.
The ABISS setup is documented in `tutorials/basics/README.md`.

```bash
python scripts/main.py --config tutorials/waterz_decoding.yaml --mode test
```

Replace the prediction path in the config and install the optional decoder
package described in `INSTALLATION.md` before running.
