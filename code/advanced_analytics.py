# credit where credit is due
# https://www.kaggle.com/virtonos/advanced-basketball-analytics

import logging
import os
import time

import pandas as pd

start_time = time.time()

formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
console_handler.setLevel(logging.DEBUG)
logger.debug('started')

input_files = sorted([
    '../input/NCAATourneyCompactResults.csv',
    '../input/RegularSeasonDetailedResults.csv',
    '../input/Teams.csv',
    '../input/NCAATourneySeeds.csv',
    '../input/Conferences.csv',
    '../input/MasseyOrdinals.csv',
    '../input/SampleSubmissionStage1.csv'
])
for input_file in input_files:
    if os.path.isfile(input_file):
        logger.debug('input file %s exists' % input_file)
    else:
        logger.warning('input file %s is missing. Quitting.' % input_file)

df_tourney = pd.read_csv('../input/NCAATourneyCompactResults.csv')
df_season = pd.read_csv('../input/RegularSeasonDetailedResults.csv')
df_teams = pd.read_csv('../input/Teams.csv')
df_conferences = pd.read_csv('../input/Conferences.csv')
df_rankings = pd.read_csv('../input/MasseyOrdinals.csv')
df_sample_sub = pd.read_csv('../input/SampleSubmissionStage1.csv')
df_seeds = pd.read_csv('../input/NCAATourneySeeds.csv')

logger.debug('done')
finish_time = time.time()
elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
