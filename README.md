# Bridging Design Gaps: A Parametric Data Completion Approach with Graph-Guided Diffusion Models

This repository accompanies the paper presented at the **IDETC 2024** conference (Paper No: IDETC2024-143359).

## Overview

This work introduces a novel generative imputation model designed to complete missing parametric data in engineering designs, specifically demonstrated using a bicycle CAD dataset. The model leverages **Graph Attention Networks (GATs)** informed by assembly graphs and **tabular diffusion models**.

The core idea is to function as an **AI Design Copilot**: given an incomplete set of design parameters, the model suggests multiple, diverse, and accurate ways to complete the design.

![AI Copilot Overview](intro.jpg)

## Key Features

* **Generative Imputation:** Fills missing numerical and categorical parameters in engineering designs.
* **Graph-Guided:** Utilizes assembly graph structures via GNNs/GATs to capture parameter interdependencies.
* **Diffusion-Based:** Employs diffusion models for high accuracy and diverse output generation.
* **AI Copilot:** Provides multiple valid design completions from partial user input.
* **Performance:** Significantly outperforms classical imputation methods (MissForest, hotDeck, PPCA) and the TabCSDI diffusion model in accuracy (RMSE, Error Rate) and diversity.

## Methodology

1.  **Graph Construction:** An assembly graph represents component connections. Feature-specific graphs are built for each input.
2.  **Encoding:** Feature and positional tokenizers (inspired by TabCSDI) encode tabular data. GATs encode assembly graph information.
3.  **Fusion:** Cross-attention merges graph, feature, and positional embeddings.
4.  **Diffusion:** A diffusion model uses the fused embedding to denoise/impute missing parameters.
5.  **Rendering:** Completed parameters can be sent to a CAD engine for visualization.

## Citation

If you find this work useful, please cite our paper:

```bibtex

@proceedings{10.1115/DETC2024-143359,
    author = {Zhou, Rui and Yuan, Chenyang and Permenter, Frank and Zhang, Yanxia and Arechiga, Nikos and Klenk, Matt and Ahmed, Faez},
    title = {Bridging Design Gaps: A Parametric Data Completion Approach With Graph-Guided Diffusion Models},
    volume = {Volume 3A: 50th Design Automation Conference (DAC)},
    series = {International Design Engineering Technical Conferences and Computers and Information in Engineering Conference},
    pages = {V03AT03A008},
    year = {2024},
    month = {08},
    abstract = {This study introduces a generative imputation model leveraging graph attention networks and tabular diffusion models for completing missing parametric data in engineering designs. This model functions as an AI design co-pilot, providing multiple design options for incomplete designs, which we demonstrate using the bicycle design CAD dataset. Through comparative evaluations, we demonstrate that our model significantly outperforms existing classical methods, such as MissForest, hotDeck, PPCA, and tabular generative method TabCSDI in both the accuracy and diversity of imputation options. Generative modeling also enables a broader exploration of design possibilities, thereby enhancing design decision-making by allowing engineers to explore a variety of design completions. The graph model combines GNNs with the structural information contained in assembly graphs, enabling the model to understand and predict the complex interdependencies between different design parameters. The graph model helps accurately capture and impute complex parametric interdependencies from an assembly graph, which is key for design problems. By learning from an existing dataset of designs, the imputation capability allows the model to act as an intelligent assistant that autocompletes CAD designs based on user-defined partial parametric design, effectively bridging the gap between ideation and realization. The proposed work provides a pathway to not only facilitate informed design decisions but also promote creative exploration in design.},
    doi = {10.1115/DETC2024-143359},
    url = {https://doi.org/10.1115/DETC2024-143359},
    eprint = {https://asmedigitalcollection.asme.org/IDETC-CIE/proceedings-pdf/IDETC-CIE2024/88360/V03AT03A008/7402917/v03at03a008-detc2024-143359.pdf},
}
