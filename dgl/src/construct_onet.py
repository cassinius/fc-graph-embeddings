from helpers.preprocess import enrich_graph_with_numeric_features, enrich_graph_with_onehot_features, enrich_graph_with_str_features, enrich_graph_with_str_arr_features
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import dgl
from dgl.data.utils import save_graphs


onet_out_file = Path(__file__).parent / "../../trained_models/onet_graph.bin"


"""
  Input
"""
onet_csv_dir = Path(__file__).parent / '../../data/onet/csvs'


"""
  Functions for normalizing IDs for DGL schema of consecutive integers starting from zero
"""


def normalize_node_ids(name, data, structure):
    data_ids = np.arange(len(data))
    structure[name] = dict(zip(data._id, data_ids))


def normalize_edge_ids(edges, dir, node_ids):
    edges[dir] = edges[dir].map(node_ids)


"""
    Nodes
"""
nodes = {
    "job": pd.read_csv(onet_csv_dir / 'o_jobs.csv'),
    "skill": pd.read_csv(onet_csv_dir / 'o_skills.csv'),
    "tech_skill": pd.read_csv(onet_csv_dir / 'o_techSkills.csv'),
    "abs": pd.read_csv(onet_csv_dir / 'o_abilities.csv'),
    "act": pd.read_csv(onet_csv_dir / 'o_activities.csv'),
    "alt_title": pd.read_csv(onet_csv_dir / 'o_altTitles.csv'),
    "rep_title": pd.read_csv(onet_csv_dir / 'o_reportedTitles.csv'),
    "dwa": pd.read_csv(onet_csv_dir / 'o_dwa.csv'),
    "iwa": pd.read_csv(onet_csv_dir / 'o_iwa.csv'),
    "know": pd.read_csv(onet_csv_dir / 'o_knowledge.csv'),
    "scale": pd.read_csv(onet_csv_dir / 'o_scale.csv'),
    "task": pd.read_csv(onet_csv_dir / 'o_tasks.csv'),
    "tool": pd.read_csv(onet_csv_dir / 'o_tools.csv')
}
nn_ids = {}

for key, data in nodes.items():
    normalize_node_ids(key, data, nn_ids)

# print("Normalized IDs for scale:\n", nn_ids['scale'])


"""
    Edges
"""
edges = {
    "ab2act": pd.read_csv(onet_csv_dir / 'o_ab2WAct.csv'),
    "broaderJob": pd.read_csv(onet_csv_dir / 'o_broaderJob.csv'),
    "dwa2Iwa": pd.read_csv(onet_csv_dir / 'o_dwa2Iwa.csv'),
    "jobAltTitles": pd.read_csv(onet_csv_dir / 'o_jobAltTitles.csv'),
    "jobRepTitles": pd.read_csv(onet_csv_dir / 'o_jobRepTitles.csv'),
    "jobTasks": pd.read_csv(onet_csv_dir / 'o_jobTasks.csv'),
    "jobTechSkills": pd.read_csv(onet_csv_dir / 'o_jobTechSkills.csv'),
    "jobTools": pd.read_csv(onet_csv_dir / 'o_jobTools.csv'),
    "skill2WAct": pd.read_csv(onet_csv_dir / 'o_skill2WAct.csv'),
    # wrongly named DB table and file...
    "dwa2Task": pd.read_csv(onet_csv_dir / 'o_task2Dwa.csv'),
}

normalize_edge_ids(edges["ab2act"], "_from", nn_ids['abs'])
normalize_edge_ids(edges["ab2act"], "_to", nn_ids['act'])
normalize_edge_ids(edges["broaderJob"], "_from", nn_ids['job'])
normalize_edge_ids(edges["broaderJob"], "_to", nn_ids['job'])
normalize_edge_ids(edges["dwa2Iwa"], "_from", nn_ids['dwa'])
normalize_edge_ids(edges["dwa2Iwa"], "_to", nn_ids['iwa'])
normalize_edge_ids(edges["jobAltTitles"], "_from", nn_ids['job'])
normalize_edge_ids(edges["jobAltTitles"], "_to", nn_ids['alt_title'])
normalize_edge_ids(edges["jobRepTitles"], "_from", nn_ids['job'])
normalize_edge_ids(edges["jobRepTitles"], "_to", nn_ids['rep_title'])
normalize_edge_ids(edges["jobTasks"], "_from", nn_ids['job'])
normalize_edge_ids(edges["jobTasks"], "_to", nn_ids['task'])
normalize_edge_ids(edges["jobTechSkills"], "_from", nn_ids['job'])
normalize_edge_ids(edges["jobTechSkills"], "_to", nn_ids['tech_skill'])
normalize_edge_ids(edges["jobTools"], "_from", nn_ids['job'])
normalize_edge_ids(edges["jobTools"], "_to", nn_ids['tool'])
normalize_edge_ids(edges["skill2WAct"], "_from", nn_ids['skill'])
normalize_edge_ids(edges["skill2WAct"], "_to", nn_ids['act'])
normalize_edge_ids(edges["dwa2Task"], "_from", nn_ids['dwa'])
normalize_edge_ids(edges["dwa2Task"], "_to", nn_ids['task'])

# print("Normalized edge endpoints for Skill->Activity:\n", onet_edges["skill2WAct"])
# print("Normalized edge endpoints for DWA->DWA:\n", edges["dwa2Task"])


"""
    Hypernodes
"""
h_nodes = {
    "jobAbilities": pd.read_csv(onet_csv_dir / 'o_jobAbilities.csv'),
    "jobActivities": pd.read_csv(onet_csv_dir / 'o_jobActivities.csv'),
    "jobKnowledge": pd.read_csv(onet_csv_dir / 'o_jobKnowledge.csv'),
    "jobSkills": pd.read_csv(onet_csv_dir / 'o_jobSkills.csv'),
    "jobTaskRatings": pd.read_csv(onet_csv_dir / 'o_jobTaskRatings.csv')
}
nhn_ids = {}

for key, data in h_nodes.items():
    normalize_node_ids(key, data, nhn_ids)

# print("Normalized Hyper IDs for jobAbilities:\n", list(nhn_ids['jobAbilities']))


"""
  Sources / Destinations for sub-graphs
"""
ab_2_act_src = edges["ab2act"]["_from"].to_numpy()
ab_2_act_dst = edges["ab2act"]["_to"].to_numpy()
broader_job_src = edges["broaderJob"]["_from"].to_numpy()
broader_job_dst = edges["broaderJob"]["_to"].to_numpy()
dwa_2_iwa_src = edges["dwa2Iwa"]["_from"].to_numpy()
dwa_2_iwa_dst = edges["dwa2Iwa"]["_to"].to_numpy()
job_alt_titles_src = edges["jobAltTitles"]["_from"].to_numpy()
job_alt_titles_dst = edges["jobAltTitles"]["_to"].to_numpy()
job_rep_titles_src = edges["jobRepTitles"]["_from"].to_numpy()
job_rep_titles_dst = edges["jobRepTitles"]["_to"].to_numpy()
job_tasks_src = edges["jobTasks"]["_from"].to_numpy()
job_tasks_dst = edges["jobTasks"]["_to"].to_numpy()
job_tech_skills_src = edges["jobTechSkills"]["_from"].to_numpy()
job_tech_skills_dst = edges["jobTechSkills"]["_to"].to_numpy()
job_tools_src = edges["jobTools"]["_from"].to_numpy()
job_tools_dst = edges["jobTools"]["_to"].to_numpy()
skill_2_act_src = edges["skill2WAct"]["_from"].to_numpy()
skill_2_act_dst = edges["skill2WAct"]["_to"].to_numpy()
dwa_2_task_src = edges["dwa2Task"]["_from"].to_numpy()
dwa_2_task_dst = edges["dwa2Task"]["_to"].to_numpy()


"""
    Hyper Edges
"""
h_edges = {
    # Towards the hypernode
    "abs2JobAbilities": pd.read_csv(onet_csv_dir / 'o_abs2JobAbilities.csv'),
    "acts2JobActivities": pd.read_csv(onet_csv_dir / 'o_acts2JobActivities.csv'),
    "know2JobKnowledge": pd.read_csv(onet_csv_dir / 'o_know2JobKnowledge.csv'),
    "skill2JobSkills": pd.read_csv(onet_csv_dir / 'o_skill2JobSkills.csv'),
    "task2JobTaskRatings": pd.read_csv(onet_csv_dir / 'o_task2JobTaskRatings.csv'),
    # Towards the job node
    "jobAbilities2Jobs": pd.read_csv(onet_csv_dir / 'o_jobAbilities2Jobs.csv'),
    "jobActivities2Jobs": pd.read_csv(onet_csv_dir / 'o_jobActivities2Jobs.csv'),
    "jobKnowledge2Jobs": pd.read_csv(onet_csv_dir / 'o_jobKnowledge2Jobs.csv'),
    "jobSkill2Jobs": pd.read_csv(onet_csv_dir / 'o_jobSkill2Jobs.csv'),
    "jobTaskRatings2Jobs": pd.read_csv(onet_csv_dir / 'o_jobTaskRatings2Jobs.csv'),
    # Payload from the Scale
    "scale2JobAbilities": pd.read_csv(onet_csv_dir / 'o_scale2JobAbilities.csv'),
    "scale2JobActivities": pd.read_csv(onet_csv_dir / 'o_scale2JobActivities.csv'),
    "scale2JobKnowledge": pd.read_csv(onet_csv_dir / 'o_scale2JobKnowledge.csv'),
    "scale2JobSkills": pd.read_csv(onet_csv_dir / 'o_scale2JobSkills.csv'),
    "scale2JobTaskRatings": pd.read_csv(onet_csv_dir / 'o_scale2JobTaskRatings.csv'),
}

# Hyper edge endpoint normalization
normalize_edge_ids(h_edges["abs2JobAbilities"], "_from", nn_ids['abs'])
normalize_edge_ids(h_edges["abs2JobAbilities"], "_to", nhn_ids['jobAbilities'])
normalize_edge_ids(h_edges["acts2JobActivities"], "_from", nn_ids['act'])
normalize_edge_ids(h_edges["acts2JobActivities"], "_to", nhn_ids['jobActivities'])
normalize_edge_ids(h_edges["know2JobKnowledge"], "_from", nn_ids['know'])
normalize_edge_ids(h_edges["know2JobKnowledge"], "_to", nhn_ids['jobKnowledge'])
normalize_edge_ids(h_edges["skill2JobSkills"], "_from", nn_ids['skill'])
normalize_edge_ids(h_edges["skill2JobSkills"], "_to", nhn_ids['jobSkills'])
normalize_edge_ids(h_edges["task2JobTaskRatings"], "_from", nn_ids['task'])
normalize_edge_ids(h_edges["task2JobTaskRatings"], "_to", nhn_ids['jobTaskRatings'])
# print("Normalized HE `abs2JobAbilities`:\n", h_edges['abs2JobAbilities'])
# print("Normalized HE `acts2JobActivities`:\n", h_edges['acts2JobActivities'])
# print("Normalized HE `task2JobTaskRatings`:\n", h_edges['task2JobTaskRatings'])

normalize_edge_ids(h_edges["jobAbilities2Jobs"], "_from", nhn_ids['jobAbilities'])
normalize_edge_ids(h_edges["jobAbilities2Jobs"], "_to", nn_ids['job'])
normalize_edge_ids(h_edges["jobActivities2Jobs"], "_from", nhn_ids['jobActivities'])
normalize_edge_ids(h_edges["jobActivities2Jobs"], "_to", nn_ids['job'])
normalize_edge_ids(h_edges["jobKnowledge2Jobs"], "_from", nhn_ids['jobKnowledge'])
normalize_edge_ids(h_edges["jobKnowledge2Jobs"], "_to", nn_ids['job'])
normalize_edge_ids(h_edges["jobSkill2Jobs"], "_from", nhn_ids['jobSkills'])
normalize_edge_ids(h_edges["jobSkill2Jobs"], "_to", nn_ids['job'])
normalize_edge_ids(h_edges["jobTaskRatings2Jobs"], "_from", nhn_ids['jobTaskRatings'])
normalize_edge_ids(h_edges["jobTaskRatings2Jobs"], "_to", nn_ids['job'])

normalize_edge_ids(h_edges["scale2JobAbilities"], "_from", nn_ids['scale'])
normalize_edge_ids(h_edges["scale2JobAbilities"], "_to", nhn_ids['jobAbilities'])
normalize_edge_ids(h_edges["scale2JobActivities"], "_from", nn_ids['scale'])
normalize_edge_ids(h_edges["scale2JobActivities"], "_to", nhn_ids['jobActivities'])
normalize_edge_ids(h_edges["scale2JobKnowledge"], "_from", nn_ids['scale'])
normalize_edge_ids(h_edges["scale2JobKnowledge"], "_to", nhn_ids['jobKnowledge'])
normalize_edge_ids(h_edges["scale2JobSkills"], "_from", nn_ids['scale'])
normalize_edge_ids(h_edges["scale2JobSkills"], "_to", nhn_ids['jobSkills'])
normalize_edge_ids(h_edges["scale2JobTaskRatings"], "_from", nn_ids['scale'])
normalize_edge_ids(h_edges["scale2JobTaskRatings"], "_to", nhn_ids['jobTaskRatings'])


"""
  Sources / Destinations for sub-graphs (hyper-edges)
"""
scale_2_job_ab_src = h_edges["scale2JobAbilities"]["_from"].to_numpy()
scale_2_job_ab_dst = h_edges["scale2JobAbilities"]["_to"].to_numpy()
scale_2_job_act_src = h_edges["scale2JobActivities"]["_from"].to_numpy()
scale_2_job_act_dst = h_edges["scale2JobActivities"]["_to"].to_numpy()
scale_2_job_know_src = h_edges["scale2JobKnowledge"]["_from"].to_numpy()
scale_2_job_know_dst = h_edges["scale2JobKnowledge"]["_to"].to_numpy()
scale_2_job_skill_src = h_edges["scale2JobSkills"]["_from"].to_numpy()
scale_2_job_skill_dst = h_edges["scale2JobSkills"]["_to"].to_numpy()
scale_2_job_tsk_src = h_edges["scale2JobTaskRatings"]["_from"].to_numpy()
scale_2_job_tsk_dst = h_edges["scale2JobTaskRatings"]["_to"].to_numpy()

ab_2_job_ab_src = h_edges["abs2JobAbilities"]["_from"].to_numpy()
ab_2_job_ab_dst = h_edges["abs2JobAbilities"]["_to"].to_numpy()
act_2_job_act_src = h_edges["acts2JobActivities"]["_from"].to_numpy()
act_2_job_act_dst = h_edges["acts2JobActivities"]["_to"].to_numpy()
know_2_job_know_src = h_edges["know2JobKnowledge"]["_from"].to_numpy()
know_2_job_know_dst = h_edges["know2JobKnowledge"]["_to"].to_numpy()
skill_2_job_skill_src = h_edges["skill2JobSkills"]["_from"].to_numpy()
skill_2_job_skill_dst = h_edges["skill2JobSkills"]["_to"].to_numpy()
tsk_2_job_tsk_src = h_edges["task2JobTaskRatings"]["_from"].to_numpy()
tsk_2_job_tsk_dst = h_edges["task2JobTaskRatings"]["_to"].to_numpy()

job_ab_2_job_src = h_edges["jobAbilities2Jobs"]["_from"].to_numpy()
job_ab_2_job_dst = h_edges["jobAbilities2Jobs"]["_to"].to_numpy()
job_act_2_job_src = h_edges["jobActivities2Jobs"]["_from"].to_numpy()
job_act_2_job_dst = h_edges["jobActivities2Jobs"]["_to"].to_numpy()
job_know_2_job_src = h_edges["jobKnowledge2Jobs"]["_from"].to_numpy()
job_know_2_job_dst = h_edges["jobKnowledge2Jobs"]["_to"].to_numpy()
job_skill_2_job_src = h_edges["jobSkill2Jobs"]["_from"].to_numpy()
job_skill_2_job_dst = h_edges["jobSkill2Jobs"]["_to"].to_numpy()
job_tsk_2_job_src = h_edges["jobTaskRatings2Jobs"]["_from"].to_numpy()
job_tsk_2_job_dst = h_edges["jobTaskRatings2Jobs"]["_to"].to_numpy()


"""
  Construct O*NET graph
"""
onet_graph_data = {
    ('job', 'superjob', 'job'): (broader_job_src, broader_job_dst),
    ('job', 'subjob', 'job'): (broader_job_dst, broader_job_src),
    ('job', 'has-alt-title', 'alt_title'): (job_alt_titles_src, job_alt_titles_dst),
    ('alt_title', 'is-alt-title', 'job'): (job_alt_titles_dst, job_alt_titles_src),
    ('job', 'has-rep-title', 'rep_title'): (job_rep_titles_src, job_rep_titles_dst),
    ('rep_title', 'is-rep-title', 'job'): (job_rep_titles_dst, job_rep_titles_src),
    ('job', 'comprises-task', 'task'): (job_tasks_src, job_tasks_dst),
    ('task', 'contained-task', 'job'): (job_tasks_dst, job_tasks_src),
    ('job', 'requires-tech-skill', 'tech_skill'): (job_tech_skills_src, job_tech_skills_dst),
    ('tech_skill', 'tech-for-job', 'job'): (job_tech_skills_dst, job_tech_skills_src),
    ('job', 'requires-tool', 'tool'): (job_tools_src, job_tools_dst),
    ('tool', 'tool-for-job', 'job'): (job_tools_dst, job_tools_src),
    ('ability', 'requires-ability', 'activity'): (ab_2_act_src, ab_2_act_dst),
    ('activity', 'required-by-activity', 'ability'): (ab_2_act_dst, ab_2_act_src),
    ('skill', 'enables-activity', 'activity'): (skill_2_act_src, skill_2_act_dst),
    ('activity', 'enabled-by-skill', 'skill'): (skill_2_act_dst, skill_2_act_src),
    ('dwa', 'describes-task', 'task'): (dwa_2_task_src, dwa_2_task_dst),
    ('task', 'described-by-task', 'dwa'): (dwa_2_task_dst, dwa_2_task_src),
    ('dwa', 'sub-act', 'iwa'): (dwa_2_iwa_src, dwa_2_iwa_dst),
    ('iwa', 'super-act', 'dwa'): (dwa_2_iwa_dst, dwa_2_iwa_src),

    # Hyper Hyper ...

    # ('scale', 'gives-weight-2-ability', 'job_ability'): (scale_2_job_ab_src, scale_2_job_ab_dst),
    # ('job_ability', 'ab-gets-weight-from-scale', 'scale'): (scale_2_job_ab_dst, scale_2_job_ab_src),
    # ('scale', 'gives-weight-2-activity', 'job_activity'): (scale_2_job_act_src, scale_2_job_act_dst),
    # ('job_activity', 'act-gets-weight-from-scale', 'scale'): (scale_2_job_act_dst, scale_2_job_act_src),
    # ('scale', 'gives-weight-2-knowledge', 'job_knowledge'): (scale_2_job_know_src, scale_2_job_know_dst),
    # ('job_knowledge', 'know-gets-weight-from-scale', 'scale'): (scale_2_job_know_dst, scale_2_job_know_src),
    # ('scale', 'gives-weight-2-skill', 'job_skill'): (scale_2_job_skill_src, scale_2_job_skill_dst),
    # ('job_skill', 'skill-gets-weight-from-scale', 'scale'): (scale_2_job_skill_dst, scale_2_job_skill_src),
    # ('scale', 'gives-weight-2-task', 'job_task'): (scale_2_job_tsk_src, scale_2_job_tsk_dst),
    # ('job_task', 'task-gets-weight-from-scale', 'scale'): (scale_2_job_tsk_dst, scale_2_job_tsk_src),

    # ('ability', 'ab-2-job-ab', 'job_ability'): (ab_2_job_ab_src, ab_2_job_ab_dst),
    # ('job_ability', 'job-ab-2-ab', 'ability'): (ab_2_job_ab_dst, ab_2_job_ab_src),
    # ('activity', 'act-2-job-act', 'job_activity'): (act_2_job_act_src, act_2_job_act_dst),
    # ('job_activity', 'job-act-2-act', 'activity'): (act_2_job_act_dst, act_2_job_act_src),
    # ('knowledge', 'know-2-job-know', 'job_knowledge'): (know_2_job_know_src, know_2_job_know_dst),
    # ('job_knowledge', 'job-know-2-know', 'knowledge'): (know_2_job_know_dst, know_2_job_know_src),
    # ('skill', 'skill-2-job-skill', 'job_skill'): (skill_2_job_skill_src, skill_2_job_skill_dst),
    # ('job_skill', 'job-skill-2-skill', 'skill'): (skill_2_job_skill_dst, skill_2_job_skill_src),
    # ('task', 'task-2-job-task', 'job_task'): (tsk_2_job_tsk_src, tsk_2_job_tsk_dst),
    # ('job_task', 'job-task-2-task', 'task'): (tsk_2_job_tsk_dst, tsk_2_job_tsk_src),

    # ('job_ability', 'job-ab-2-job', 'job'): (job_ab_2_job_src, job_ab_2_job_dst),
    # ('job', 'job-2-job-ab', 'job_ability'): (job_ab_2_job_dst, job_ab_2_job_src),
    # ('job_activity', 'job-act-2-job', 'job'): (job_act_2_job_src, job_act_2_job_dst),
    # ('job', 'job-2-job-act', 'job_activity'): (job_act_2_job_dst, job_act_2_job_src),
    # ('job_knowledge', 'job-know-2-job', 'job'): (job_know_2_job_src, job_know_2_job_dst),
    # ('job', 'job-2-job-know', 'job_knowledge'): (job_know_2_job_dst, job_know_2_job_src),
    # ('job_skill', 'job-skill-2-job', 'job'): (job_skill_2_job_src, job_skill_2_job_dst),
    # ('job', 'job-2-job-skill', 'job_skill'): (job_skill_2_job_dst, job_skill_2_job_src),
    # ('job_task', 'job-task-2-job', 'job'): (job_tsk_2_job_src, job_tsk_2_job_dst),
    # ('job', 'job-2-job-task', 'job_task'): (job_tsk_2_job_dst, job_tsk_2_job_src)
}
onet_graph = dgl.heterograph(onet_graph_data)
print('O*NET graph:\n', onet_graph)


"""
  Add node features
"""


"""
  Labels for training on TOP JOBS
"""
top_jobs = nodes['job']['topJob']
unique_top_jobs = set(top_jobs)
top_job_ids_dict = dict(zip(unique_top_jobs, np.arange(len(unique_top_jobs))))
print(f"Unique top jobs: {unique_top_jobs}, length: {len(unique_top_jobs)}")

top_jobs_normalized = list(top_jobs.map(top_job_ids_dict))
print(
    f"Unique top jobs normalized: {set(top_jobs_normalized)}, length: {len(set(top_jobs_normalized))}")
# print(f"Top jobs normalized: {top_jobs_normalized}, length: {len(top_jobs_normalized)}")

onet_graph.nodes['job'].data['top-job'] = torch.tensor(
    top_jobs_normalized).long()


"""
  Labels for training on TASK TYPE

  TODO this is a very skewed classification task (between 58 and 14k elements per class)
"""
task_types_dict = {
    'New': 1,
    'Core': 2,
    'Revision': 3,
    'Supplemental': 4
}
task_types = [task_types_dict.get(typ, 0) for typ in nodes['task']['Task Type']]
unique_task_types = set(task_types)
print(f"Unique task types: {unique_task_types}, length: {len(unique_task_types)}")
onet_graph.nodes['task'].data['type'] = torch.tensor(task_types)


"""
  Labels for training on TASK SOURCE
"""
task_sources_dict = {
    'Analyst': 0,
    'Incumbent': 1,
    'Occupational Expert': 2
}
task_sources = [task_sources_dict.get(src, 0) for src in nodes['task']['Domain Source']]
unique_sources = set(task_sources)
print(f"Unique task sources: {unique_sources}, length: {len(unique_sources)}")
onet_graph.nodes['task'].data['source'] = torch.tensor(task_sources)


"""
  Saving the graph
"""
# no idea which labels to use and why... DGL doc could be a bit more explicit here!
graph_labels = {"glabel": torch.tensor([0])}
save_graphs(str(onet_out_file), [onet_graph], graph_labels)
