---
title: 'PyCCEA: A Python package of cooperative co-evolutionary algorithms for feature selection in high-dimensional data'
tags:
  - Cooperative Co-Evolutionary Algorithms
  - Feature Selection
  - Machine Learning
  - Python
authors:
  - name: Pedro Vinícius A. B. Venâncio
    orcid: 0000-0003-4665-562X
    affiliation: "1, 3"
    corresponding: true
  - name: Lucas S. Batista
    orcid: 0000-0002-7444-3440
    affiliation: "2, 3"
affiliations:
 - name: Graduate Program in Electrical Engineering, Universidade Federal de Minas Gerais, Brazil
   index: 1
 - name: Departament of Electrical Engineering, Universidade Federal de Minas Gerais, Brazil
   index: 2
 - name: Operations Research and Complex Systems Laboratory, Universidade Federal de Minas Gerais, Brazil
   index: 3
date: 6 April 2025
bibliography: paper.bib

---

# Summary

Feature selection is a critical preprocessing step in many machine learning pipelines, particularly when dealing with high-dimensional datasets commonly found in domains such as genomics, text mining, and image analysis. However, many feature selection techniques, such as heuristic search methods and evolutionary algorithms, struggle to maintain predictive performance and interpretability as the dimensionality of data increases [@Theng:2024].

Cooperative co-evolutionary algorithms (CCEAs) offer a promising approach for tackling this challenge by dividing the high-dimensional space into multiple tractable low-dimensional subcomponents. Each subcomponent is evolved independently using a subpopulation, and candidate solutions are evaluated based on their collaboration with representatives from other subpopulations [@Ma:2018].

`PyCCEA` implements CCEAs specifically for feature selection in high-dimensional data. It provides modular components for decomposition, collaboration, fitness evaluation, and optimization, allowing users to reproduce existing methods or develop new ones. The framework currently only follows a wrapper-based approach, in which subsets of features are evaluated using machine learning models to guide the search. However, this design is not restrictive, `PyCCEA` can be extended to support other evaluation paradigms, such as filters, hybrid, or embedded methods, which are planned as future features.

All machine learning and data processing components are built on top of well-established libraries such as `scikit-learn` [@Pedregosa:2011] and `Pandas` [@McKinney:2010], ensuring compatibility, flexibility, and ease of use. To the best of our knowledge, `PyCCEA` is the first open-source package dedicated to CCEAs for feature selection, offering a standardized and extensible framework for tackling high-dimensional datasets in a reproducible manner, as shown in Figure \ref{fig:architecture}.

![An overview of the `PyCCEA package`, its modules and underlying Python dependencies. \label{fig:architecture}](figures/architecture.png)

# Statement of need

Despite the growing interest in cooperative co-evolutionary algorithms for feature selection and their promising results [@Song:2020], there is currently no publicly available software package that consolidates these techniques into a reusable and extensible tool. Existing research typically relies on custom [@Rashid:2020a; @Rashid:2020b; @Firouznia:2023] or unpublished implementations [@Song:2020; @Zhou:2024], which makes it difficult to reproduce results, compare methods, or build upon previous work.

`PyCCEA` addresses this gap by providing a well-organized, research-focused implementation of cooperative co-evolutionary algorithms tailored to feature selection. It incorporates widely used strategies from the literature and encourages standardization in experimental design and evaluation. By enabling consistent benchmarking and facilitating the development of new strategies, `PyCCEA` supports both methodological innovation and practical application in high-dimensional machine learning problems. Its release lowers the barrier to entry for researchers and practitioners, accelerating progress in the field of feature selection using evolutionary computation.

Recent research in evolutionary computation and feature selection has already leveraged `PyCCEA` as a foundational framework for implementing and evaluating cooperative co-evolutionary strategies [@Venancio:2025], demonstrating its value in supporting reproducible and extensible scientific contributions. Future improvements aim to broaden its evaluation capabilities and encourage further community-driven development.


# Acknowledgements

This work was supported by Brazilian agencies FAPEMIG (Research Support Foundation of the State of Minas Gerais), CNPq (The National Council for Scientific and Technological Development), CAPES (Coordination for the Improvement of Higher Education Personnel) -- Finance Code 001, and the Operations Research and Complex Systems Laboratory (ORCS Lab./UFMG). Professor Lucas S. Batista is a FAPEMIG-CNPQ scholarship holder (process APQ-06716-24).

# References