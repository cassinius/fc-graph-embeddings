"""
  NOTES

    "Normal" w2v / d2v doesn't know how to encode URIs..
    enrich_graph_with_str_features(esco_graph, 'job', jobs_data, 'conceptUri')
    enrich_graph_with_str_arr_features(esco_graph, 'job', jobs_data, 'inScheme')

    Hidden labels are usually empty, so we just ignore them
    enrich_graph_with_str_arr_features(esco_graph, 'job', jobs_data, 'hiddenLabels')

    Python Pandas converts empty strings to NaN by default -> WTF !?
"""

from helpers.preprocess import enrich_graph_with_numeric_features, enrich_graph_with_onehot_features, enrich_graph_with_str_features, enrich_graph_with_str_arr_features
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import dgl
from dgl.data.utils import save_graphs


esco_out_file = Path(__file__).parent / "../../trained_models/esco_graph.bin"
esco_out_file_features = Path(__file__).parent / "../../trained_models/esco_graph_feats.bin"


"""
  Input
"""
esco_csv_dir = Path(__file__).parent / '../../data/esco/csvs'

skills_data = pd.read_csv(esco_csv_dir / 'e_skills.csv')
# some descriptions in skills_data are empty, which pandas converts to NaN (seriously!)
skills_data['description'].replace(np.nan, 'n/a', regex=True, inplace=True)

jobs_data = pd.read_csv(esco_csv_dir / 'e_jobs.csv')
broader_skill_data = pd.read_csv(esco_csv_dir / 'e_broaderSkill.csv')
broader_occ_data = pd.read_csv(esco_csv_dir / 'e_broaderJob.csv')
related_skill_data = pd.read_csv(esco_csv_dir / 'e_relatedSkill.csv')
job_skill_data = pd.read_csv(esco_csv_dir / 'e_jobSkills.csv')


"""
  Normalizing IDs for DGL schema of consecutive integers starting from 0
"""
job_ids = np.arange(len(jobs_data))
job_ids_dict = dict(zip(jobs_data._id, job_ids))
skill_ids = np.arange(len(skills_data))
skill_ids_dict = dict(zip(skills_data._id, skill_ids))

# jobs_data['_id'] = jobs_data['_id'].map(jobs_id_dict)
# skills_data['_id'] = skills_data['_id'].map(skills_id_dict)

broader_occ_data['_from'] = broader_occ_data['_from'].map(job_ids_dict)
broader_occ_data['_to'] = broader_occ_data['_to'].map(job_ids_dict)
broader_skill_data['_from'] = broader_skill_data['_from'].map(skill_ids_dict)
broader_skill_data['_to'] = broader_skill_data['_to'].map(skill_ids_dict)
related_skill_data['_from'] = related_skill_data['_from'].map(skill_ids_dict)
related_skill_data['_to'] = related_skill_data['_to'].map(skill_ids_dict)
job_skill_data['_from'] = job_skill_data['_from'].map(job_ids_dict)
job_skill_data['_to'] = job_skill_data['_to'].map(skill_ids_dict)

# print(job_skill_data)


"""
  Construct single graphs
  - job hierarchy
  - skill hierarchy
  - related skills
  - job skills
"""
job_hierarchy_src = broader_occ_data['_from'].to_numpy()
job_hierarchy_dst = broader_occ_data['_to'].to_numpy()
# job_hierarchy_graph = dgl.graph((job_hierarchy_src, job_hierarchy_dst))
# print('Job hierarchy graph:\n', job_hierarchy_graph)

skill_hierarchy_src = broader_skill_data['_from'].to_numpy()
skill_hierarchy_dst = broader_skill_data['_to'].to_numpy()
# skill_hierarchy_graph = dgl.graph((skill_hierarchy_src, skill_hierarchy_dst))
# print('Skill hierarchy graph:\n', skill_hierarchy_graph)

related_skill_src = related_skill_data['_from'].to_numpy()
related_skill_dst = related_skill_data['_to'].to_numpy()
# related_skill_graph = dgl.graph((related_skill_src, related_skill_dst))
# print('Related skills graph:\n', related_skill_graph)

job_skill_src = job_skill_data['_from'].to_numpy()
job_skill_dst = job_skill_data['_to'].to_numpy()
# job_skill_graph = dgl.graph((job_skill_src, job_skill_dst))
# print('Job skills graph:\n', job_skill_graph)


"""
  Construct the DGL heterograph out of single graphs
"""
esco_graph_data = {
    ('job', 'superjob', 'job'): (job_hierarchy_src, job_hierarchy_dst),
    ('job', 'subjob', 'job'): (job_hierarchy_dst, job_hierarchy_src),
    ('skill', 'superskill', 'skill'): (skill_hierarchy_src, skill_hierarchy_dst),
    ('skill', 'subskill', 'skill'): (skill_hierarchy_dst, skill_hierarchy_src),
    ('skill', 'related-to', 'skill'): (related_skill_src, related_skill_dst),
    ('skill', 'related-from', 'skill'): (related_skill_dst, related_skill_src),
    ('job', 'requires', 'skill'): (job_skill_src, job_skill_dst),
    ('skill', 'required', 'job'): (job_skill_dst, job_skill_src)
}
esco_graph = dgl.heterograph(esco_graph_data)
print('ESCO graph:\n', esco_graph)


"""
  For training on TOP JOBS
"""
# Occupations have an 'iscoGroup' field, whereas ISCO groups have a 'code' field
isco_codes = [int(x) if ~np.isnan(x) else int(y) for (x, y) in zip(jobs_data['iscoGroup'], jobs_data['code'])]
# For easier retrieval later...
esco_graph.nodes['job'].data['isco-common'] = torch.tensor(isco_codes).long()

top_jobs = [int(x/1000) for x in isco_codes]
unique_isco_codes = list(set(top_jobs))
print(f"Unique ISCO codes: {unique_isco_codes}")
esco_graph.nodes['job'].data['top-job'] = torch.tensor(top_jobs)

"""
  For training on SKILL-TYPE (skewed: min: 673, max: 10582)
"""
skill_types_dict = {
    'knowledge': 1,
    'skill/competence': 2
}
skill_types = [skill_types_dict.get(typ, 0) for typ in skills_data['skillType']]
unique_skill_types = set(skill_types)
print(f"Unique skill types: {unique_skill_types}, length: {len(unique_skill_types)}")
esco_graph.nodes['skill'].data['type'] = torch.tensor(skill_types).long()


"""
  For training on REUSE-LEVEL
"""
reuse_levels = {
    'cross-sector': 1,
    'occupation-specific': 2,
    'sector-specific': 3,
    'transversal': 4
}
skill_reuse = [reuse_levels.get(level, 0) for level in skills_data['reuseLevel']]
unique_skill_reuse = set(skill_reuse)
print(f"Skill re-use levels (multi-class): {unique_skill_reuse}, length: {len(unique_skill_reuse)}")
esco_graph.nodes['skill'].data['reuse'] = torch.tensor(skill_reuse).long()


def enrich_esco_graph_with_features(G):
    """ Add the other node features to the graph """
    enrich_graph_with_numeric_features(G, 'job', jobs_data, 'iscoGroup')
    enrich_graph_with_onehot_features(G, 'job', jobs_data, 'conceptType')
    enrich_graph_with_str_features(G, 'job', jobs_data, 'preferredLabel')
    enrich_graph_with_str_features(G, 'job', jobs_data, 'description')
    enrich_graph_with_str_arr_features(G, 'job', jobs_data, 'altLabels')

    enrich_graph_with_onehot_features(G, 'skill', skills_data, 'conceptType')
    enrich_graph_with_onehot_features(G, 'skill', skills_data, 'skillType')
    enrich_graph_with_str_features(G, 'skill', skills_data, 'preferredLabel')
    enrich_graph_with_str_features(G, 'skill', skills_data, 'description')
    enrich_graph_with_str_arr_features(G, 'skill', skills_data, 'altLabels')

    print('ESCO graph after FEATURE ENRICHMENT:\n', G)
    print("Node types:", G.ntypes)
    print('Edge types:', G.etypes)
    print('Canonical edge types:', G.canonical_etypes)


def save_esco(G, features: bool):
    """ Saving the graph, wither with or without features """
    if features:
        out_file = esco_out_file_features
        enrich_esco_graph_with_features(G)
    else:
        out_file = esco_out_file

    # no idea which labels to use and why... DGL doc could be a bit more explicit here!
    graph_labels = {"glabel": torch.tensor([0])}
    save_graphs(str(out_file), [G], graph_labels)


save_esco(esco_graph, True)
