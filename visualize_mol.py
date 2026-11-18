# import rdkit
# from rdkit import Chem
# from rdkit.Chem import Draw
# from rdkit.Chem import AllChem
# import torch
# import os
# from torch_geometric.data import Data
#
# ####################################
# # 1. 原子 / 键 映射 & 价态表
# ####################################
#
# # 和你之前一样的价态表，可以根据自己需要再补
# ATOM_VALENCY = {
#     1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1,
#     11: 1, 20: 2, 5: 3, 29: 2, 30: 2, 27: 3, 25: 3, 33: 3, 13: 3,
#     28: 2, 34: 2, 14: 2, 40: 2, 50: 2, 3: 3, 51: 3, 26: 4, 46: 2,
#     80: 2, 83: 3, 22: 2, 67: 3, 32: 2, 78: 2, 44: 2, 45: 3, 24: 4,
#     31: 3, 19: 1, 47: 1, 79: 1, 65: 4, 77: 1, 52: 2, 12: 2, 82: 2,
#     74: 4, 55: 3, 42: 2, 75: 3, 92: 4, 64: 2, 81: 3, 89: 1, 43: 3
# }
#
# # 8 维 one-hot → 原子序号
# atom_types = {
#     0: 6,   # C
#     1: 7,   # N
#     2: 8,   # O
#     3: 9,   # F
#     4: 15,  # P
#     5: 16,  # S
#     6: 17,  # Cl
#     7: 35,  # Br
#     8: 53,  # I
#     9: 1,   # H
#     10: 11, # Na
#     11: 19, # K
#     12: 20, # Ca
#     13: 26, # Fe
#     14: 29, # Cu
#     15: 30  # Zn
# }
#
# # 你原来的 bond_decoder_simple
# bond_decoder_simple = {
#     1: Chem.rdchem.BondType.SINGLE,
#     2: Chem.rdchem.BondType.DOUBLE,
#     3: Chem.rdchem.BondType.TRIPLE,
# }
#
#
# def get_atom_type(feature_vector: torch.Tensor):
#     """Convert atom feature vector to atomic number."""
#     if torch.all(feature_vector == 0):
#         return None
#     idx = int(torch.argmax(feature_vector[:len(atom_types)]).item())
#     return atom_types.get(idx, 6)  # 默认 C
#
#
# def get_bond_type(edge_attr: torch.Tensor):
#     """
#     Convert bond feature vector to RDKit bond type,
#     按照 bond_decoder_simple = {1,2,3} 解释单/双/三键。
#     """
#     if edge_attr is None:
#         return Chem.rdchem.BondType.SINGLE
#
#     # 只看前 3 维
#     if edge_attr.dim() == 1:
#         bf = edge_attr[:3]
#     else:
#         bf = edge_attr[0][:3]
#
#     if torch.all(bf == 0):
#         return Chem.rdchem.BondType.SINGLE
#
#     idx0 = int(torch.argmax(bf).item())  # 0 / 1 / 2
#     idx = idx0 + 1                        # 1 / 2 / 3
#     return bond_decoder_simple.get(idx, Chem.rdchem.BondType.SINGLE)
#
#
# ####################################
# # 2. 从 batch 中取出单个图
# ####################################
#
# def get_subgraph_for_batch(batch_data: Data, batch_id: int):
#     """
#     从带 batch 向量的 Data 中取出某个 batch 对应的子图，并重新编号节点。
#     返回: x_sub, edge_index_sub, edge_attr_sub
#     """
#     node_mask = (batch_data.batch == batch_id)
#     node_idx = node_mask.nonzero(as_tuple=True)[0]
#     if node_idx.numel() == 0:
#         return None, None, None
#
#     x_sub = batch_data.x[node_mask]
#
#     # old -> new 编号
#     mapping = -torch.ones(batch_data.x.size(0), dtype=torch.long, device=batch_data.x.device)
#     mapping[node_idx] = torch.arange(node_idx.size(0), device=batch_data.x.device)
#
#     ei = batch_data.edge_index
#     edge_mask = node_mask[ei[0]] & node_mask[ei[1]]
#     ei_sub = ei[:, edge_mask]
#
#     if ei_sub.size(1) == 0:
#         return x_sub, ei_sub, None
#
#     ei_sub = mapping[ei_sub]
#
#     if batch_data.edge_attr is not None:
#         ea_sub = batch_data.edge_attr[edge_mask]
#     else:
#         ea_sub = None
#
#     return x_sub, ei_sub, ea_sub
#
#
# ####################################
# # 3. 原子额外属性（一般用不到）
# ####################################
#
# def process_special_atoms(mol, atom_idx, feature_vector: torch.Tensor):
#     """处理电荷 / 手性 / 芳香性（如果 feature_vector 足够长才会生效）"""
#     atom = mol.GetAtomWithIdx(atom_idx)
#     L = feature_vector.numel()
#
#     # 电荷：16-18
#     if L > 18:
#         charge_features = feature_vector[16:19]
#         if torch.any(charge_features):
#             charge_idx = int(torch.argmax(charge_features).item())
#             atom.SetFormalCharge(charge_idx - 1)   # -1, 0, +1
#
#     # 手性：19-21
#     if L > 21:
#         chiral_features = feature_vector[19:22]
#         if torch.any(chiral_features):
#             chiral_idx = int(torch.argmax(chiral_features).item())
#             if chiral_idx == 1:
#                 atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
#             elif chiral_idx == 2:
#                 atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
#
#     # 芳香性：第 22 位
#     if L > 22 and int(feature_vector[22].item()) == 1:
#         atom.SetIsAromatic(True)
#
#
# ####################################
# # 4. 单个分子可视化（增加价态过滤）
# ####################################
#
# def visualize_molecule(data: Data, idx: int = 0, output_dir: str = './mol_vis', batch_id=None):
#     """Visualize a single molecule."""
#     try:
#         os.makedirs(output_dir, exist_ok=True)
#
#         # 取第 idx 个图
#         if hasattr(data, 'batch'):
#             x, edge_index, edge_attr = get_subgraph_for_batch(data, idx)
#         else:
#             x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
#
#         if x is None or x.size(0) == 0:
#             print(f"No atoms for batch {idx}")
#             return None
#
#         mol = Chem.RWMol()
#         atom_map = {}
#         atom_atomic_num = {}
#         valence_used = {}
#
#         # 1) 加原子
#         for i in range(x.size(0)):
#             atomic_num = get_atom_type(x[i])
#             if atomic_num is None:
#                 continue
#             a = Chem.Atom(int(atomic_num))
#             atom_idx = mol.AddAtom(a)
#             atom_map[i] = atom_idx
#             atom_atomic_num[i] = int(atomic_num)
#             valence_used[i] = 0
#             process_special_atoms(mol, atom_idx, x[i])
#
#         if not atom_map:
#             print(f"No valid atoms for batch {idx}")
#             return None
#
#         # 2) 加键：根据 ATOM_VALENCY 限制价态
#         if edge_index is not None and edge_index.size(1) > 0:
#             added_bonds = set()
#             for e in range(edge_index.size(1)):
#                 s = int(edge_index[0, e].item())
#                 t = int(edge_index[1, e].item())
#                 if s not in atom_map or t not in atom_map:
#                     continue
#
#                 key = (min(s, t), max(s, t))
#                 if key in added_bonds:
#                     continue
#
#                 bt = get_bond_type(edge_attr[e] if edge_attr is not None else None)
#
#                 # RDKit 键类型 → “价”贡献
#                 if bt == Chem.rdchem.BondType.DOUBLE:
#                     order = 2
#                 elif bt == Chem.rdchem.BondType.TRIPLE:
#                     order = 3
#                 else:
#                     order = 1
#
#                 max_vs = ATOM_VALENCY.get(atom_atomic_num[s], 4)
#                 max_vt = ATOM_VALENCY.get(atom_atomic_num[t], 4)
#
#                 # 如果这条键会让任意一端超价，直接跳过
#                 if valence_used[s] + order > max_vs:
#                     continue
#                 if valence_used[t] + order > max_vt:
#                     continue
#
#                 try:
#                     mol.AddBond(atom_map[s], atom_map[t], bt)
#                     valence_used[s] += order
#                     valence_used[t] += order
#                     added_bonds.add(key)
#                 except Exception:
#                     # 某些奇怪情况继续跳过
#                     continue
#
#         # 3) sanitize & 2D 坐标
#         try:
#             Chem.SanitizeMol(mol)
#         except Exception as e:
#             print(f"[Warn] SanitizeMol failed for batch {batch_id}: {e}")
#         AllChem.Compute2DCoords(mol)
#
#         # 4) 画图（风格和 SMILES 那个类似）
#         drawer = Draw.rdMolDraw2D.MolDraw2DCairo(300, 300)
#         opts = drawer.drawOptions()
#         opts.addStereoAnnotation = True
#         opts.bondLineWidth = 2
#         opts.additionalAtomLabelPadding = 0.25
#
#         Draw.PrepareAndDrawMolecule(drawer, mol)
#         drawer.FinishDrawing()
#
#         name = batch_id if batch_id is not None else idx
#         out_path = os.path.join(output_dir, f"molecule_{name}.png")
#         drawer.WriteDrawingText(out_path)
#         print(f"Saved {out_path}")
#
#         return mol
#
#     except Exception as e:
#         print(f"Error generating molecule visualization: {e}")
#         return None
#
#
# ####################################
# # 5. 批量可视化
# ####################################
#
# def visualize_batch(batch_data: Data, output_dir: str = './mol_vis'):
#     """Visualize all molecules in a batch."""
#     if not hasattr(batch_data, 'batch'):
#         return visualize_molecule(batch_data, idx=0, output_dir=output_dir)
#
#     unique_batch_ids = torch.unique(batch_data.batch).tolist()
#     print("unique_batch_ids", unique_batch_ids)
#     print(f"Visualizing {len(unique_batch_ids)} molecules...")
#
#     for batch_id in unique_batch_ids:
#         print(f"visualize_batch(): Visualizing {batch_id}th molecule...")
#         visualize_molecule(batch_data, idx=batch_id, output_dir=output_dir, batch_id=batch_id)

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
import torch
import os
from torch_geometric.data import Data

####################################
# 1. 原子 / 键 映射 & 价态表
####################################

ATOM_VALENCY = {
    1: 1, 6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1,
    11: 1, 20: 2, 5: 3, 29: 2, 30: 2, 27: 3, 25: 3, 33: 3, 13: 3,
    28: 2, 34: 2, 14: 2, 40: 2, 50: 2, 3: 3, 51: 3, 26: 4, 46: 2,
    80: 2, 83: 3, 22: 2, 67: 3, 32: 2, 78: 2, 44: 2, 45: 3, 24: 4,
    31: 3, 19: 1, 47: 1, 79: 1, 65: 4, 77: 1, 52: 2, 12: 2, 82: 2,
    74: 4, 55: 3, 42: 2, 75: 3, 92: 4, 64: 2, 81: 3, 89: 1, 43: 3
}

atom_types = {
    0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35,
    8: 53, 9: 1, 10: 11, 11: 19, 12: 20, 13: 26, 14: 29, 15: 30
}

bond_decoder_simple = {
    1: Chem.rdchem.BondType.SINGLE,
    2: Chem.rdchem.BondType.DOUBLE,
    3: Chem.rdchem.BondType.TRIPLE,
}

def get_atom_type(feature_vector: torch.Tensor):
    if torch.all(feature_vector == 0):
        return None
    idx = int(torch.argmax(feature_vector[:len(atom_types)]).item())
    return atom_types.get(idx, 6)

def get_bond_type(edge_attr: torch.Tensor):
    if edge_attr is None:
        return Chem.rdchem.BondType.SINGLE
    bf = edge_attr[:3] if edge_attr.dim() == 1 else edge_attr[0][:3]
    if torch.all(bf == 0):
        return Chem.rdchem.BondType.SINGLE
    idx = int(torch.argmax(bf).item()) + 1
    return bond_decoder_simple.get(idx, Chem.rdchem.BondType.SINGLE)


####################################
# 2. 提取 batch 子图
####################################

def get_subgraph_for_batch(batch_data: Data, batch_id: int):
    node_mask = (batch_data.batch == batch_id)
    node_idx = node_mask.nonzero(as_tuple=True)[0]

    if node_idx.numel() == 0:
        return None, None, None

    x_sub = batch_data.x[node_mask]

    mapping = -torch.ones(batch_data.x.size(0), dtype=torch.long, device=batch_data.x.device)
    mapping[node_idx] = torch.arange(node_idx.size(0), device=batch_data.x.device)

    ei = batch_data.edge_index
    edge_mask = node_mask[ei[0]] & node_mask[ei[1]]
    ei_sub = mapping[ei[:, edge_mask]]

    if ei_sub.size(1) == 0:
        return x_sub, ei_sub, None

    ea_sub = batch_data.edge_attr[edge_mask] if batch_data.edge_attr is not None else None

    return x_sub, ei_sub, ea_sub


####################################
# 3. 处理特殊原子属性（可选）
####################################

def process_special_atoms(mol, atom_idx, feature_vector: torch.Tensor):
    atom = mol.GetAtomWithIdx(atom_idx)
    L = feature_vector.numel()

    if L > 18:
        charge_features = feature_vector[16:19]
        if torch.any(charge_features):
            cidx = int(torch.argmax(charge_features).item())
            atom.SetFormalCharge(cidx - 1)

    if L > 21:
        chiral_features = feature_vector[19:22]
        if torch.any(chiral_features):
            cidx = int(torch.argmax(chiral_features).item())
            if cidx == 1:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            elif cidx == 2:
                atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)

    if L > 22 and int(feature_vector[22].item()) == 1:
        atom.SetIsAromatic(True)


####################################
# 4. 可视化单个分子（改进：支持 custom_name）
####################################

def visualize_molecule(
    data: Data,
    idx: int = 0,
    output_dir: str = './mol_vis',
    batch_id=None,
    custom_name: str = None
):
    try:
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(data, 'batch'):
            x, edge_index, edge_attr = get_subgraph_for_batch(data, idx)
        else:
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        if x is None or x.size(0) == 0:
            print(f"No atoms for batch {idx}")
            return None

        mol = Chem.RWMol()
        atom_map = {}
        atom_atomic_num = {}
        valence_used = {}

        for i in range(x.size(0)):
            atomic_num = get_atom_type(x[i])
            if atomic_num is None:
                continue
            atom_idx = mol.AddAtom(Chem.Atom(int(atomic_num)))
            atom_map[i] = atom_idx
            atom_atomic_num[i] = int(atomic_num)
            valence_used[i] = 0
            process_special_atoms(mol, atom_idx, x[i])

        if not atom_map:
            print(f"No valid atoms for batch {idx}")
            return None

        added_bonds = set()
        if edge_index is not None and edge_index.size(1) > 0:
            for e in range(edge_index.size(1)):
                s = int(edge_index[0, e].item())
                t = int(edge_index[1, e].item())
                if s not in atom_map or t not in atom_map:
                    continue

                key = (min(s, t), max(s, t))
                if key in added_bonds:
                    continue

                bt = get_bond_type(edge_attr[e] if edge_attr is not None else None)
                order = 1 if bt == Chem.rdchem.BondType.SINGLE else (2 if bt == Chem.rdchem.BondType.DOUBLE else 3)

                if valence_used[s] + order > ATOM_VALENCY.get(atom_atomic_num[s], 4):
                    continue
                if valence_used[t] + order > ATOM_VALENCY.get(atom_atomic_num[t], 4):
                    continue

                try:
                    mol.AddBond(atom_map[s], atom_map[t], bt)
                    valence_used[s] += order
                    valence_used[t] += order
                    added_bonds.add(key)
                except:
                    continue

        try:
            Chem.SanitizeMol(mol)
        except Exception as e:
            print(f"[Warn] SanitizeMol failed: {e}")

        AllChem.Compute2DCoords(mol)

        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(300, 300)
        opts = drawer.drawOptions()
        opts.addStereoAnnotation = True
        opts.bondLineWidth = 2
        opts.additionalAtomLabelPadding = 0.25

        Draw.PrepareAndDrawMolecule(drawer, mol)
        drawer.FinishDrawing()

        # ---- 改动：文件名不重复 ----
        name = custom_name
        if name is None:
            name = batch_id if batch_id is not None else idx

        out_path = os.path.join(output_dir, f"molecule_{name}.png")
        drawer.WriteDrawingText(out_path)
        print(f"Saved {out_path}")

        return mol

    except Exception as e:
        print(f"Error generating molecule visualization: {e}")
        return None


####################################
# 5. 批量可视化（完全解决重名问题）
####################################

def visualize_batch(batch_data: Data,
                    output_dir: str = './mol_vis',
                    prefix: str = ''):
    """
    prefix 示例：
      prefix='bbbp_step3'
      prefix='clintox_aug12'
    最终文件名：
      molecule_bbbp_step3_b0_n0.png
      molecule_bbbp_step3_b0_n1.png
    """
    os.makedirs(output_dir, exist_ok=True)

    if not hasattr(batch_data, 'batch'):
        return visualize_molecule(
            batch_data, idx=0, output_dir=output_dir,
            batch_id=0, custom_name=f'{prefix}_b0'
        )

    unique_batch_ids = torch.unique(batch_data.batch).tolist()
    print(f"Visualizing {len(unique_batch_ids)} molecules...")

    for local_idx, batch_id in enumerate(unique_batch_ids):
        name = f'{prefix}_b{batch_id}_n{local_idx}'
        print(f"Visualizing {name} ...")

        visualize_molecule(
            batch_data,
            idx=batch_id,
            output_dir=output_dir,
            batch_id=batch_id,
            custom_name=name
        )
