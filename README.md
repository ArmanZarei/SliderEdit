# SliderEdit: Continuous Image Editing with Fine-Grained Instruction Control

### [Paper](https://arxiv.org/abs/2511.09715) | [Project Page](https://armanzarei.github.io/SliderEdit/)

![SliderEdit for continuous control over instruction-based image editing models](examples/teasor.png)

We introduce *SliderEdit*, a framework for continuous image editing with fine-grained, interpretable instruction control. Given a multi-part edit instruction, SliderEdit disentangles the individual instructions and exposes each as a globally trained slider, allowing smooth adjustment of its strength.

## ⚙️Setup

1. Clone the repository
    ```bash
    git clone git@github.com:ArmanZarei/SliderEdit.git
    cd SliderEdit
    ```
2. Create and activate the conda environment
    ```bash
    conda env create -f environment.yml
    conda activate slideredit
    pip install -e .
    ```

## 🚀Quick Start

First, load the SliderEdit pipeline:
```python
from slideredit.pipelines import SliderEditFluxKontextPipeline

pipe = SliderEditFluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    torch_dtype=torch.bfloat16
).to("cuda")
```


### GSTLoRA

GSTLoRA is designed for single-instruction editing scenarios or when a single slider is sufficient to control the overall edit intensity.

```python
pipe.load_gstlora("PATH_TO_CKPT")

output_image = pipe(
    image=IMAGE_TO_BE_EDITED,
    prompt="EDIT PROMPT",
    generator=torch.Generator().manual_seed(SEED)
    slider_alpha=STRENGH_VALUE,
).images[0]
```

The parameter `slider_alpha` controls edit strength. Negative values increase intensity, while positive values suppress the effect. We recommend initially sweeping values in the range [-1, 1].

### STLoRA

STLoRA is designed for multi-instruction editing prompts, providing independent control over the strength of each individual instruction.

```python
pipe.load_stlora("PATH_TO_CKPT")

output_image = pipe(
    image=IMAGE_TO_BE_EDITED,
    prompt="Edit_Instruction_1 and Edit_Instruct_2 and ...",
    generator=torch.Generator().manual_seed(SEED)
    subprompts_list=["Edit_Instruction_1", "Edit_Instruct_2", ...],
    slider_alpha_list=[Strength_Value_1, Strength_Value_2, ...],
).images[0]
```

*Parameters:*
- `subprompts_list`: Individual instructions (sub-prompts) from the original edit prompt
- `slider_alpha_list`: Corresponding intensity values for each instruction (we recommend initially sweeping values in the range: [-1, 1])


See [`sample_inference.ipynb`](sample_inference.ipynb) for complete inference examples.

## 🔬Training SliderEdit

### GSTLoRA

The training script for GSTLoRA is available at [`train_gstlora_flux_kontext.py`](training/train_gstlora_flux_kontext.py). An example configuration using an open-source image editing dataset is provided in [`train_gstlora_flux_kontext.yaml`](training/configs/train_gstlora_flux_kontext.yaml).

To launch training:
```bash
python training/train_gstlora_flux_kontext.py --config="training/configs/train_gstlora_flux_kontext.yaml"
```

View training progress and slider visualizations in [this W&B report](https://wandb.ai/armanzarei/slider-edit/reports/SliderEdit-STLoRA-and-GSTLoRA-Sample-Training--VmlldzoxNTgyNjYwNQ?accessToken=kthi3nq5ebkajza4be49aeg1374i0y6a9g7alek69adb4xnawi817civqngzpfpy).

### STLoRA

Training STLoRA with our proposed PPS loss requires multi-instruction edit prompts. (If you instead wish to use the SPPS loss, the same dataset as in the GSTLoRA setting can be reused)

Below, we provide a simple small-scale example illustrating how to construct such a dataset and perform training. In this example, we focus on human face editing by manually combining a set of predefined single-instruction edits to form multi-instruction prompts.

For more details, please refer to the training script [`train_stlora_flux_kontext.py`](training/train_stlora_flux_kontext.py) and the example configuration file [`train_stlora_pps_flux_kontext.yaml`](training/configs/train_stlora_pps_flux_kontext.yaml).


First, download the sample human faces dataset:

```bash
mkdir -p datasets
cd datasets
gdown 183-Jubsu2rFiQmgpBYjFpAxGSwqn2XOM
unzip slideredit_faces_dataset.zip
cd ..
```

Then, launch training:

```bash
python training/train_stlora_flux_kontext.py --config="training/configs/train_stlora_pps_flux_kontext.yaml"
```

View training progress and slider visualizations in [this W&B report](https://wandb.ai/armanzarei/slider-edit/reports/SliderEdit-STLoRA-and-GSTLoRA-Sample-Training--VmlldzoxNTgyNjYwNQ?accessToken=kthi3nq5ebkajza4be49aeg1374i0y6a9g7alek69adb4xnawi817civqngzpfpy).


### Inference

Below are example checkpoints from models trained using the above configurations:

```bash
mkdir -p checkpoints
gdown 1YHrHhSeKovEPGpFgFbv0iPL67YRgg6rG -O checkpoints/ # GSTLoRA iter500
gdown 1PdORTgzFzfGGbNAoPQb0T5su3t82xErY -O checkpoints/ # STLoRA iter1200
```

Example results:
- GSTLoRA

    <p align="center">
    <img alt="Example generated images using the trained GSTLoRA" src="examples/example_gstlora.png" width="80%">
    <br>
    <i>"Make her hair curly"</i>
    </p>
- STLoRA

    <p align="center">
    <img alt="Example generated images using the trained STLoRA" src="examples/example_stlora.png" width="80%">
    <br>
    <i>"<u>make the person fat</u> and <u>make the person laugh</u>"</i>
    </p>

See [`sample_inference.ipynb`](sample_inference.ipynb) for more details.

## 📊Evaluation

Simply instantiate any of the `VLMEvaluator` or `FeatureDistanceEvaluator` classes from the `evaluation` module and use them to compute the corresponding scores or distances. For more details and usage examples, please refer to [`evaluation/example_evaluation.ipynb`](evaluation/example_evaluation.ipynb).

## BibTeX
```bibtex
@article{zarei2025slideredit,
  title={SliderEdit: Continuous Image Editing with Fine-Grained Instruction Control},
  author={Zarei, Arman and Basu, Samyadeep and Pournemat, Mobina and Nag, Sayan and Rossi, Ryan and Feizi, Soheil},
  journal={arXiv preprint arXiv:2511.09715},
  year={2025}
}
```