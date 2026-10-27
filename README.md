## Power Grid Representation for Large Language Models: A Case Study on Voltage Control
#### Repository with paper related code

### Generate new data
1. Set up environment and install requirements.
2. Set up the Ollama API settings: export `BASE_URL` and `API_KEY` as environment variables (read in main.py).
3. Set up run configurations in main (model-config, bias, grid) this is done by commenting related lines of code.
4. Run main.py (results will be saved in run_results)

```bash
pip install -r requirements.txt
python main.py
```


### Plotting already existing Data
1. Add the results you want to plot to the plots/raw_data folder.
2. If necessary, adjust the path in plotting_with_raw_data.
3. Set up which plots to generate.
4. Run plotting_with_raw_data.

```bash
pip install -r requirements.txt
python plots/plotting_with_raw_data.py
```


### OPF baseline
1. Install the requirements from /opf
2. Run opf/run_opf_monee.py

```bash
pip install -r opf/requirements.txt
python opf/run_opf_monee.py
```
