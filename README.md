# PALSYN (Private Autoregressive Log Synthesizer)

## Overview

PALSYN (Private Autoregressive Log Synthesizer) is a tool designed to generate process-oriented synthetic event logs.
It addresses the privacy concerns by integrating differential privacy. 
By doing so, it can make it easier for researches to share synthetic data with stakeholders, 
facilitating AI and process mining research. However, legal compliance, such as adherence to GDPR or 
other similar regulations, must be confirmed before sharing data, even if strong differential private guarantees are used. 

A detailed explanation of the algorithm and its workings can be found in our preprint:
[PALSYN: A Method for Synthetic Multi-Perspective Event Log Generation with Differential Private Guarantees](https://www.researchsquare.com/article/rs-6565248/v1)


## Features

- **Process-Oriented Data Generation:** Handles the complexity of process-oriented data (Event Logs).
- **Multiple Perspectives:** Considers various perspectives or attributes of the data, not just control-flow.
- **Differential Privacy:** Ensures privacy by incorporating differential privacy techniques.

## Installation
PALSYN can be installed by cloning the repository and installing the necessary dependencies.


### Installation with Git

To install PALSYN by cloning the repository, follow these steps.

Clone the repository:
```bash
git clone https://github.com/martinkuhn94/PALSYN.git


Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model 
For the training of the model, the stacked layers are configured with 32, 16 and 8 LSTM units respectively, and an embedding dimension of 16. The model trains for 3 epochs with a batch size of 16. The number of clusters for numerical attributes is set to 10, and to speed up the training, only the top 50% quantile of traces by length are considered, in this example. The noise multiplier is set to 0.0, which means that the model is trained without differential privacy. To train the model with differential privacy, set the noise multiplier to a value greater than 0.0. The epsilon value can be retrieved after training the model.
```bash
import pm4py
from PALSYN.synthesizer import DPEventLogSynthesizer


# Read Event Log Road_Traffic_Fine_Management_Process.xes
xes_file_path = "example_logs/Road_Traffic_Fine_Management_Process_short.xes"
event_log = pm4py.read_xes(xes_file_path)

# Initialize Model
palsyn_model = DPEventLogSynthesizer(
    embedding_output_dims=128,
    epochs=5,
    batch_size=128,
    max_clusters=15,
    dropout=0.3,
    trace_quantile=0.9,
    l2_norm_clip=1.0,
    method="Bi-LSTM",
    units_per_layer=[32, 16]
)

# Train Model
palsyn_model.fit(event_log)
palsyn_model.save_model("models/Bi-LSTM_Road_Fines_u=32_e=inf")

```

### Sampling Event Logs 
To sample synthetic event logs, use the following example with a trained model can be used. The sample size is set to 1050, and the batch size is set to 16. The synthetic event log is saved as a XES file.
Pretrained models can be found in the "models" folder.
```bash
import pm4py
from PALSYN.synthesizer import DPEventLogSynthesizer
from PALSYN.postprocessing.log_postprocessing import clean_xes_file
from pm4py.objects.petri_net.importer import importer as pnml_importer


# Load Model
palsyn_model = DPEventLogSynthesizer()
palsyn_model.load("models/LSTM_Sepsis Case_u=32_e=inf_ep=10")

# Load a Petri Net
pnml_path = "petri-net.pnml"
net, im, fm = pnml_importer.apply(pnml_path)

```
Choose between 3 sampling processes: standard PALSYN, conditional PALSYN with Transition-List, conditional PALSYN with Petri Net Simulation
```bash
# Sample with PALSYN
event_log = palsyn_model.sample(sample_size=1050, batch_size=16)
```
```bash
# Sample with C-PALSYN-TL
event_log = palsyn_model.sample(sample_size=1050, batch_size=16, petri_net=(net, im, fm), mode="transtion-list")
```
```bash
# Sample with C-PALSYN-PN
event_log = palsyn_model.sample(sample_size=1050, batch_size=16, petri_net=(net, im, fm), mode="simulation")
```
```bash
event_log_xes = pm4py.convert_to_event_log(event_log)

# Save as XES File
xes_filename = "sepsis=inf.xes"
pm4py.write_xes(event_log_xes, xes_filename)
clean_xes_file(xes_filename, xes_filename)

# Save as XSLX File for quick inspection
df = pm4py.convert_to_dataframe(event_log_xes)
df["time:timestamp"] = df["time:timestamp"].astype(str)
df.to_excel("sepsis=inf.xlsx", index=False)
```

## Future Work
Future work will focus on enhancing the algorithm and making it available on PyPI.

## Contribution

We welcome contributions from the community. If you have any suggestions or issues, please create a GitHub issue or a pull request. 


## License
This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details. 
