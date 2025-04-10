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

`PyCCEA` is a dedicated and extensible package for implementing CCEAs aimed at feature selection in high-dimensional data. As illustrated in Figure \ref{fig:architecture}, it is structured around modular components that encapsulate key elements of the CCEA paradigm, including decomposition strategies, cooperation schemes, fitness evaluation methods, and evolutionary optimizers. This design allows researchers and practitioners to easily replicate existing approaches or prototype novel algorithms by composing interchangeable modules.

The package currently adopts a wrapper-based evaluation approach, where subsets of features are assessed using machine learning models to guide the search process. However, this design choice is not restrictive. `PyCCEA` has been structured to accommodate future extensions supporting other evaluation paradigms, such as filter-based, embedded, or hybrid methods. These alternatives are under consideration for upcoming releases to broaden the range of applicable scenarios and research directions.

In addition to its evolutionary components, `PyCCEA` integrates machine learning and data processing functionalities built on top of widely adopted libraries, including `scikit-learn` [@Pedregosa:2011] and `Pandas` [@McKinney:2010]. In view of this, users can take advantage of built-in support for models (e.g., KNN, SVM, Random Forest), metrics (e.g., accuracy, $F_1$-score), and data operations (e.g., normalization, cross-validation, splitting, preprocessing). This tight integration ensures compatibility with existing machine learning workflows while maintaining flexibility and ease of use.

To further support empirical studies, `PyCCEA` includes a benchmarking module composed of well-known datasets (e.g., DLBCL, lung cancer) [@Kelly:2017] and baseline CCEAs (e.g., CCEAFS, CCFSRFG1). This enables standardized evaluation and comparison of new strategies in a reproducible and transparent manner. To the best of our knowledge, `PyCCEA` is the first open-source package specifically designed for cooperative co-evolution in the context of feature selection, providing a solid foundation for research and practical application in high-dimensional machine learning problems.

![An overview of the `PyCCEA` package, its modules and underlying Python dependencies. \label{fig:architecture}](figures/architecture.png)

# Statement of need

Despite the growing interest in cooperative co-evolutionary algorithms for feature selection and their promising results [@Song:2020], there is currently no publicly available software package that consolidates these techniques into a reusable and extensible tool. Existing research typically relies on custom [@Rashid:2020a; @Rashid:2020b; @Firouznia:2023] or unpublished implementations [@Song:2020; @Zhou:2024], which makes it difficult to reproduce results, compare methods, or build upon previous work.

`PyCCEA` addresses this gap by providing a well-organized, research-focused implementation of cooperative co-evolutionary algorithms tailored to feature selection. It incorporates widely used strategies from the literature and encourages standardization in experimental design and evaluation. By enabling consistent benchmarking and facilitating the development of new strategies, `PyCCEA` supports both methodological innovation and practical application in high-dimensional machine learning problems. Its release lowers the barrier to entry for researchers and practitioners, accelerating progress in the field of feature selection using evolutionary computation.

Recent research has already leveraged `PyCCEA` as a foundational framework for implementing and evaluating cooperative co-evolutionary strategies in high-dimensional classification problems [@Venancio:2025], demonstrating its effectiveness in supporting reproducible and extensible scientific research. Planned enhancements aim to extend the framework’s generalization capabilities, enabling broader applicability to diverse problem domains, including regression and clustering tasks. These developments also aim to foster greater community engagement by making the framework more adaptable, accessible, and conducive to collaborative development.

# Acknowledgements

This work was supported by Brazilian agencies FAPEMIG (Research Support Foundation of the State of Minas Gerais), CNPq (The National Council for Scientific and Technological Development), CAPES (Coordination for the Improvement of Higher Education Personnel) -- Finance Code 001, and the Operations Research and Complex Systems Laboratory (ORCS Lab./UFMG). Professor Lucas S. Batista is a FAPEMIG-CNPQ scholarship holder (process APQ-06716-24).

# References