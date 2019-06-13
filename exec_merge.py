#!/bin/env python
import sys
from time import sleep
from time import time
from nbconvert.preprocessors import CellExecutionError
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import subprocess
import requests
import os

############################## exec notebooks
notebooks = [
    '01_overview{}.ipynb',
    '02_bdd100k{}.ipynb',
    '03_mask_rcnn{}.ipynb',
    '04_yolo{}.ipynb',
    '05_traffic_flow_collection{}.ipynb',
    '06_ts_data_prediction{}.ipynb'
]

COLOR_START = '\033[92m'
COLOR_END = '\033[0m'

start = time()
base_path = './'

executed_folder = 'executed/'
if os.path.exists(executed_folder) == False:
    os.mkdir(executed_folder)

for notebook in notebooks:
    read_file_path = base_path + notebook.format('')
    with open(read_file_path) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=1200, kernel_name='python3')

        print('{}executing {}{}'.format(COLOR_START, read_file_path, COLOR_END))
        try:
            ep.preprocess(nb, {'metadata': {'path': base_path}})
        except CellExecutionError as e:
            print(e)
            sys.exit(1)

        write_file_path = base_path + executed_folder + notebook.format('.executed')
        print('{}write {}{}\n\n'.format(COLOR_START, write_file_path, COLOR_END))
        with open(write_file_path, 'wt') as write_file:
            nbformat.write(nb, write_file)
        # sleep(15)


############################## merge note books
notebooks_to_merge = [
    '01_overview{}.ipynb',
    '02_bdd100k{}.ipynb',
    '03_mask_rcnn{}.ipynb',
    '04_yolo{}.ipynb',
    '05_traffic_flow_collection{}.ipynb',
    '06_ts_data_prediction{}.ipynb'
]
executed_notebooks = [executed_folder + nb.format('.executed') for nb in notebooks_to_merge]
output_file = 'zhitu_traffic_flow.ipynb'
args = ['nbmerge'] + executed_notebooks + ['-o', output_file] 
print(args)
subprocess.run(args)


replace_collapsed_args = ['sed', '-i', 's#"collapsed": true,##', output_file]
subprocess.run(replace_collapsed_args)

replace_collapsed_args = ['sed', '-i', 's#"collapsed": true##', output_file]
subprocess.run(replace_collapsed_args)

replace_scrolled_args = ['sed', '-i', 's#"scrolled": true#"scrolled": false#', output_file]
subprocess.run(replace_scrolled_args)



############################## Save ipynb to pdf with chrome
# get jupyter notebook servers
cmd_str = "jupyter notebook list | grep 'http' | grep -vi 'token' | awk '{print $1}'"

sp = subprocess.Popen(cmd_str, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
output = sp.communicate()[0]

nb_list = str(output, "utf-8").strip().split('\n')

if len(nb_list) == 0:
    print("Error! No jupyter notebooks are running!")
    sys.exit(1)

# Just use the first one, call chrome to save ipynb to pdf
url = nb_list[0] + 'notebooks/mask-rcnn-taike-course/' + output_file
pdf_file = output_file.replace('.ipynb', '.pdf')
print(url)
args = [
    'google-chrome-stable',
    '--virtual-time-budget=60000',
    '--run-all-compositor-stages-before-draw',
    '--headless',
    '--print-to-pdf={}'.format(pdf_file),
    url
]
print(args)
print("Don't run google-chrome save as PDF for now")
# subprocess.run(args)

############################## upload the zip file to obs
# import os
# import boto3
# AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')
# S3_ENDPOINT = os.environ.get('S3_ENDPOINT')

# client = boto3.client(
#     's3', 
#     aws_access_key_id=AWS_ACCESS_KEY_ID, 
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     endpoint_url=S3_ENDPOINT
# )

# result = client.upload_file(output_file, 'modelarts-maskrcnn', 'data/' + output_file)
# result = client.upload_file(pdf_file, 'modelarts-maskrcnn', 'data/' + pdf_file)

# if result != None:
#     print(result)


############################## clean up
#for notebook in executed_notebooks:
#    subprocess.run(['rm', notebook])

duration = time() - start
print('{}duration: {} seconds{}'.format(COLOR_START, duration, COLOR_END))
# TDOO clean the pdf and ipynb?
