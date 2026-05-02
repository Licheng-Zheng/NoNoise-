# Adding New Metrics to build_metrics

Creating new metrics under the `build_metrics` so it can be computed and put into a nice and clean folder. 

Honestly, I barely have a clue how I add metrics, so I'm writing this down for myself too  : D

---

## General structure

1.  `build_metrics/metrics.py`: The actual functions (Can be wrappers for implementations to keep the file tidy, the SSIM is a wrapper because the code for that is humongous. Also included an example (onesies), that is a very simple function that just returns 1 for all cubes to show that it also works in there)

2. `build_metrics/run_metrics.py`: Teensy bit bloated: loads the data, runs the metrics, and saves the results. Will need to separate this into smaller chunks when I have time. :( (this file will stay like this for the rest of eternity) 

3.  `build_metrics/data_loader.py`: Loads data from `.mat` files, need to add functionality for not .mat files, but for now, just use the conversion code that I created because I that's a lot of unrelated work that will make me forget what I'm doing in the first place. 


Things it (kind of) does
- Provides functions a bunch of different parameters such as the path for the clean and processed datacube to calculate the different metrics on. 
- Uses `**kwargs` which, I think, will allow me to provide arguments that are not required, but I need to figure this out with a function that doesn't require as many parameters (I will figure this out, because this one is important)
- Creates a CSV file with all the different metrics, performance and the name of the model, making it easier to look at and bring into other softwares. Also ensures everyone is using the same metrics (one place where all the functions exist) 

---

## Metric function design
To create a metric (the most important thing, because my SSIM PSNR and onesies (the most important function of all) will not get you very far. )

1. Write metrics as plain Python functions in `metrics.py`
    - Note that you can also (and should) create a wrapper function instead of writing everything in `metrics.py` so it doesn't get too messy (or maybe you should, I don't know how its supposed to be written, but if you don't know and I don't know, let's don't know together and write it how its already written)
2. Put in the required inputs, notes that additional arguments are also provided with `**kwargs`, so you need to specify the parameters that you want here
    - I just implemented this, I have no idea how its going to work, so this is more like during the testing phase. 

## Registering a new metric
You thought you were done by writing the code for the metric? Nuh uh! Before you're done, you need to tell the program that the metric exists and to compute it! 

Open `build_metrics/run_metrics.py` and add your metric to the `metrics_to_run` dictionary. (This is literally the most simple I could make it)
- This will be how it is appended to the csv and printed out, so this is where you control the ordering. 



## Passing optional parameters (metric_context)

The runner assembles a `metric_context` dictionary and passes it to every metric via a safe dispatcher. Metrics only receive the keys they accept (thanks to signature filtering). Add any global options here:

```python
metric_context = {
    "window_size": 11,   # for SSIM
    "normalize": True,   # for SSIM
    "device": None,      # for SSIM ('cuda' or 'cpu')
    "max_val": 1.0,      # for PSNR
    "reduction": "mean", # for SAM (example)
    # "mask": my_mask_array,  # if you load one
}
```

If you introduce a new optional parameter in a metric, add it to `metric_context` to activate or tune it.

---

## Metrics with special input needs

### 1) No-reference metrics (only processed)

For metrics that don’t need `clean`:

- Easiest: keep a consistent signature and ignore the `clean` argument:

```python
def run_nr_sharpness(clean, processed, **kwargs):
    return float(np.std(np.gradient(processed)))
```

- Alternatively, extend the dispatcher to support varying positional requirements. The current approach favors simplicity: all metrics accept `(clean, processed, ...)`.

### 2) Metrics that require extra inputs (e.g., masks, band weights)

- Make the input a required keyword-only parameter (no default) to fail fast when missing:

```python
def run_masked_psnr(clean, processed, *, mask: np.ndarray, **kwargs) -> float:
    # compute PSNR only where mask==True
    ...
```

- Provide the required data through `metric_context` in `run_metrics.py`. If the data must be loaded from disk (e.g., a companion `.mat`), extend `data_loader.py` or load it inside `run_metrics.py` before you build `metric_context`.

---

## How the runner invokes metrics

Key points in `run_metrics.py`:

- `metric_context` is built once and passed to metrics.
- `call_metric` uses introspection to only pass accepted keyword args:

```python
import inspect

def call_metric(metric_func, clean, processed, context):
    sig = inspect.signature(metric_func)
    accepted = {}
    for k, v in context.items():
        if k in sig.parameters:
            param = sig.parameters[k]
            if param.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
                accepted[k] = v
    return metric_func(clean, processed, **accepted)
```

- Printing is dynamic: header and rows are built from `metrics_to_run`. Adding a metric automatically updates the terminal view and the CSV columns.

---

## CSV output

- File: `processed/<dataset_name>/metrics.csv`
- Columns: `Model Output` + metrics in the order defined by `metrics_to_run`.
- Each row corresponds to a processed model’s `.mat` output.

---

## Testing checklist

1. Add your new metric function to `metrics.py`.
2. Register it in `metrics_to_run` in `run_metrics.py`.
3. (Optional) Add parameters to `metric_context` if your metric uses them.
4. Run `run_metrics.py`.
5. Verify:
   - Terminal prints a header including your new metric.
   - Each model row includes a value for your metric.
   - `processed/<dataset>/metrics.csv` contains a column for your metric with values.

---

## Troubleshooting

- TypeError about unexpected keyword argument:
  - Ensure your metric includes `**kwargs`, or add the option name to the function signature.
  - Alternatively, remove the key from `metric_context` if not needed.

- Shape mismatch warnings/skips:
  - Confirm your `clean` and `processed` arrays have the same shape. Update loaders or variable names if necessary.

- NaNs or infinities in results:
  - Check your metric implementation for numerical stability (e.g., denominators, log inputs). Add epsilons if needed.

- Not seeing your metric in the terminal/CSV:
  - Ensure it’s registered in `metrics_to_run` and there are no import errors.

---

## Style guidelines

- Prefer explicit required parameters and keyword-only optional parameters.
- Validate inputs as needed and fail fast with clear error messages.
- Document any non-obvious options in the function docstring.
- Keep metrics deterministic and side-effect free (pure functions) whenever possible.

---

## Example: Complete flow for a new metric

1) Implement in `metrics.py`:

```python
def run_mse(clean, processed, **kwargs) -> float:
    import numpy as np
    return float(np.mean((clean - processed) ** 2))
```

2) Register in `run_metrics.py`:

```python
metrics_to_run = {
    "PSNR": run_psnr,
    "SSIM": run_ssim,
    "MSE": run_mse,
}
```

3) (Optional) Add options to `metric_context` if applicable.

4) Run the script and confirm terminal/CSV include the MSE column and values.

---

By following this guide, your new metrics will integrate cleanly with the dynamic printing and CSV export mechanisms in the `build_metrics` pipeline.