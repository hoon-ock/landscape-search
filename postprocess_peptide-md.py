import mdtraj as md
import itertools as it
from itertools import groupby
import pickle, os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('original_data_path', type=str, help='original data path')
parser.add_argument('save_dir', type=str, help='save directory')
parser.add_argument('--residue', choices=['ALA9', 'ARG20'], default='ALA9', help='residue name')

args = parser.parse_args()
data_path = args.original_data_path
save_dir = args.save_dir
residue = args.residue

# data_path = "/home/hoon/boltzmann/brownian/data/Fs_peptide/traj/"
prot_act = md.load(os.path.join(data_path, 'fs-peptide.pdb'))
top_act = md.load(os.path.join(data_path, 'fs-peptide.pdb')).topology

sig_atm = [('O', 'C', 'CA', 'N', 'CB', 'CG', 'NE2'),
           ('O', 'C', 'CA', 'N', 'CB', 'CG', 'CD', 'NE', 'NH1', 'NH2'), 
           ('O', 'C', 'CA', 'CB', 'N'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG', 'OE2', 'CD', 'OE1'), 
           ('O', 'C', 'CA', 'N'), 
           ('O', 'C', 'CA', 'N', 'CG2', 'CB', 'OG1'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG', 'CZ'), 
           ('O', 'C', 'CA', 'N', 'CB', 'OG'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG', 'OD1', 'OD2'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG1', 'CG2'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG', 'OH'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG', 'CD1', 'CD2'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG', 'CD', 'OE1', 'NE2'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG', 'CD', 'CE', 'NZ'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG1', 'CG2', 'CD1'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG', 'CD1', 'CE2', 'CH2'), 
           ('O', 'C', 'CA', 'N', 'CB', 'SG'), 
           ('O', 'C', 'CA', 'CG'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG', 'ND2', 'OD1'), 
           ('O', 'C', 'CA', 'N', 'CB', 'CG', 'SD', 'CE')]
res = ['HIS','ARG','ALA', 'GLU', 'GLY', 'THR', 'PHE', 'SER', 'ASP', 'VAL', 'TYR', 'LEU', 'GLN', 'LYS', 'ILE', 'TRP', 'CYS', 'PRO', 'ASN', 'MET']

# combinations of three atoms in each residue
cmb_atm = []
for i in range(len(res)):
    c_atm = (res[i], list(it.combinations(sig_atm[i], 3)))
    cmb_atm.append(c_atm)

# residue-atom (i.e HIS7-N)
all_res_inv = []
ina_top = prot_act.topology
for res_ina in ina_top.atoms:
    all_res_inv.append(str(res_ina))

# all atoms of each residue segment eg. [['ACE1-C', 'ACE1-O', 'ACE1-CH3', 'ACE1-H1', 'ACE1-H2', 'ACE1-H3'], ....
res_ina = [list(i) for j, i in groupby(all_res_inv, lambda a: a.split('-')[0])]

# find unique residues
unq_res_inv = []
for r_ina in range(len(res_ina)):
    unq_res_inv.append(res_ina[r_ina][0].split('-')[0])

# get index of angles from all_res_inv
angle_ind = []
for i in range(len(unq_res_inv)):
    b = []
    for j in range(len(cmb_atm)):
        if unq_res_inv[i][:3]==cmb_atm[j][0]:
            c = []
            for k in range(len(cmb_atm[j][1])):
                a = (unq_res_inv[i]+'-'+cmb_atm[j][1][k][0], unq_res_inv[i]+'-'+cmb_atm[j][1][k][1], unq_res_inv[i]+'-'+cmb_atm[j][1][k][2])
                d = []
                for l in range(len(all_res_inv)):
                    if a[0]==all_res_inv[l]:
                        f1 = l
                    if a[1]==all_res_inv[l]:
                        f2 = l
                    if a[2]==all_res_inv[l]:
                        f3 = l
                t = ((f1, f2, f3), a)
                c.append(t)
            b.append(c)
    if len(b)!=0:
        angle_ind.append(b)

if residue == 'ALA9':
    feature_no, angle_no = 7, 2 # 7, 2 (A9) | 18, 104 (R20) 
elif residue == 'ARG20':
    feature_no, angle_no = 18, 104
angle_name = '_'.join(f'{atom}' for atom in angle_ind[feature_no][0][angle_no][1])
#save_path = f'data/fspeptide/{angle_name}.pkl'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, f"{angle_name}.pkl")

result={}
for n in range(28):
    filename = f'trajectory-{n+1}.xtc'
    traj = md.load_xtc(os.path.join(data_path, filename), top= os.path.join(data_path,'fs-peptide.pdb'))
    angle_list = angle_ind[feature_no][0][angle_no][0]
    ind = [int(angle_list[0]), int(angle_list[1]), int(angle_list[2])]
    ang_act = md.compute_angles(traj, [ind], periodic=True, opt=True)
    result.update({n+1: ang_act})

with open(save_path, 'wb') as f:
    pickle.dump(result, f)