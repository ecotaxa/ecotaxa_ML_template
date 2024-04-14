#!/usr/bin/python3
#
# Download data from one or several EcoTaxa project(s) to serve as training and test sets
#
# (c) 2022 Jean-Olivier Irisson, GNU General Public License v3

## Changeable settings ----

# EcoTaxa login info 
# !! replace with your own !!
ecotaxa_user = 'ecotaxa.api.user@gmail.com'
ecotaxa_pass = 'test!'

# numeric id(s) of project(s) to download
# you should have appropriate access rights to it:
# - either the project should be visible to all
# - or you should be explicitely registered to it, as viewer at least
proj_ids = [185]

# path or URL of a taxonomic grouping file
# should have level0, level1, level2 columns where level0 is the current EcoTaxa
# classif and level1 and 2 are increasingly coarse groupings
grouping_url = 'https://docs.google.com/spreadsheets/d/1Rn73hMWGI3N-WzqlRX2X1H1aa94OLZGZIppzhQnOj-8/export?format=csv'
# NB: replace the end by `export?format=csv`
grouping_level = 'level1'

# where to store the raw data (including the images)
data_dir = '~/datasets/raw/'


## Download data ----

import os

import pandas as pd
import urllib.request
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil

import ecotaxa_py_client
from ecotaxa_py_client.api import authentification_api
from ecotaxa_py_client.model.login_req import LoginReq
from ecotaxa_py_client.api import objects_api
from ecotaxa_py_client.model.project_filters import ProjectFilters

# authenticate

print("Log in EcoTaxa")
with ecotaxa_py_client.ApiClient() as client:
    api = authentification_api.AuthentificationApi(client)
    token = api.login(LoginReq(username=ecotaxa_user, password=ecotaxa_pass))

config = ecotaxa_py_client.Configuration(access_token=token, discard_unknown_keys=True)

print("Get data from project(s)")
# get validated objects (and their names + images paths) from a project
def get_objects_df(ecotaxa_py_client, proj_id):
    with ecotaxa_py_client.ApiClient(config) as client:
        objects_instance = objects_api.ObjectsApi(client)
        # only validated
        filters = ProjectFilters(statusfilter="V") 
        # get taxonomic name and image file name
        fields = 'txo.display_name,img.file_name'
        # get objects
        objs = objects_instance.get_object_set(proj_id, project_filters=filters, fields=fields)
    # format result as DataFrame
    df = pd.DataFrame(objs['details'], columns=fields.split(','))
    df['id'] = objs['object_ids']
    return(df)

objs = [get_objects_df(ecotaxa_py_client, proj_id) for proj_id in proj_ids]

# format as a single DataFrame
df = pd.concat(objs).reset_index()

print("Download images")
# compute final path for images
img_dir = os.path.join(os.path.expanduser(data_dir), 'imgs')
os.makedirs(img_dir, exist_ok=True)
# name image according to internal object_id to ensure uniqueness
df['img_path'] = [os.path.join(img_dir, str(this_id)+'.png') for this_id in df['id']]
# TODO: detect extension of original image

# download images (with a nice progress bar)
vault_path = '/remote/ecotaxa/vault'
for i in tqdm(range(df.shape[0])):
  # if the file has not been copied already
  if not os.path.isfile(df['img_path'][i]):
    # copy from vault
    if os.path.isdir(vault_path):
      res = shutil.copyfile(
        src=os.path.join(vault_path, df['img.file_name'][i]),
        dst=df['img_path'][i]
      )
    # or copy through the internet
    else:
      res = urllib.request.urlretrieve(
        url='https://ecotaxa.obs-vlfr.fr/vault/'+df['img.file_name'][i],
        filename=df['img_path'][i]
      )

print("Perform taxonomic regrouping")
# read taxonomic grouping
groups = pd.read_csv(grouping_url, index_col='level0')
groups = groups[['level1', 'level2']]
# add grouping levels
dfg = df.join(groups, on='txo.display_name')
# save to disk
dfg.to_csv(os.path.join(data_dir, 'taxa.csv.gz'))

# choose level
dfg = dfg.rename(columns={grouping_level: 'label'})
dfg = dfg[['id', 'img_path', 'label']]
# drop images that end up with no taxon name
dfg = dfg.dropna(subset=['label'])

# split in train-test, stratified by (regrouped) label
train_df,test_df = train_test_split(dfg, test_size=0.1, stratify=dfg['label'], random_state=1)

# save to disk
train_df.to_csv('io/training_labels.csv', index=False)
test_df.to_csv('io/test_labels.csv', index=False)
