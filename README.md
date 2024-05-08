# GradNav: Accelerated Exploration of Potential Energy Surfaces with Gradient-Based Navigation

## Introduction
GradNav is an algorithm introduced in our study that enhances the exploration of the potential energy surface (PES), thereby enabling the proper reconstruction of the PES. This approach has been applied to Langevin dynamics within Mueller-type potential energy surfaces and molecular dynamics simulations of the Fs-Peptide protein, demonstrating GradNav's capability to efficiently escape deep energy wells and its reduced reliance on initial conditions. For more detailed information, please refer to our [preprint](https://arxiv.org/abs/2403.10358) and  [published paper](https://doi.org/10.1021/acs.jctc.4c00316).


https://github.com/hoon-ock/landscape-search/assets/93333323/7bc0ad59-3e21-4121-b293-934a7709321a


## Installation
Instructions for the installation will be updated soon. The necessary packages required to run GradNav are listed below:
- `openmm` version 8.0.0
- `scipy` version 1.10.0
- `mdtraj` version 1.9.7

## Data
The datasets used in our study are as follows:
- **Langevin Dynamics**: Simulations of a single particle in Muller-type potentials conducted using OpenMM.
- **Molecular Dynamics**: Trajectories of Fs-peptide proteins are available at [Figshare](https://figshare.com/articles/dataset/Fs_MD_Trajectories/1030363?file=1502287).

## How to Run
All analyses described in the paper are available in the provided Jupyter notebooks:
1. **Single Particle LD Simulation in Muller-Type Potential**:
   - LD simulation and its analysis: `muller_LD.ipynb` and `modified_muller_LD.ipynb`.
   - GradNav application on the Muller potentials case and its analysis: `muller_GradNav.ipynb` and `modified_muller_GradNav.ipynb`.
   - Energy surface reconstruction analysis: `muller_energy_curve.ipynb`.

2. **Fs-Peptide MD Simulation**:
   - Data for ALA9 and ARG20 used in the paper is located in the `data/fspeptide` directory.
   - To process your own data, place the original Fs-peptide trajectories in a directory and run:
     ```
     python postprocess_peptide-md.py <path_to_md_trajectories> <save_path> --residue <residue_type>
     ```
   - To apply GradNav in a pseudo-molecular dynamics manner, run `run_realsys_GradNav.py`.
   - All analysis regarding Fs-peptide can be found in `realsys_analysis.ipynb`.

## Contact
For any inquiries, please reach out to Janghoon Ock at jock@andrew.cmu.edu.

## Citation
Please cite our work using the following BibTeX entry:
```bibtex
@misc{ock2024gradnav,
      title={GradNav: Accelerated Exploration of Potential Energy Surfaces with Gradient-Based Navigation}, 
      author={Janghoon Ock and Parisa Mollaei and Amir Barati Farimani},
      year={2024},
      eprint={2403.10358},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph}
}
